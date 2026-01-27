import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from grading.grader import grade_answer
from utils.prompt_templates import (
    build_existing_reasoning,
    build_high_level_prompt,
    build_low_level_prompt,
    normalize_cognitive_mode,
)


PLANNER_MODES = (
    "ProblemUnderstanding",
    "Exploration",
    "Decomposition",
    "DeepReasoning",
    "Calculation",
    "Verification",
    "Reflection",
    "Synthesis",
    "Finalization",
)
PLANNER_STOP_TOKEN = "<|mlr_highlevel_end|>"
PLANNER_FIELD_LIMITS = (320, 256, 320)
LITERAL_ESCAPED_LINE_BREAK = re.compile(
    r"\\[nr](?=(?:\\[nr]|\s|$))"
)
FINALIZATION_BOX_CONTRADICTION = re.compile(
    r"\bunboxed\b"
    r"|\bwithout\s+(?:an?\s+)?box(?:ed)?\b"
    r"|\bdo\s+not\s+(?:use|include|return)\s+(?:an?\s+)?box(?:ed)?\b",
    re.IGNORECASE,
)
FINALIZATION_DELIVERABLE = (
    "A concise response containing exactly one boxed final answer."
)
FINALIZATION_SUCCESS_CRITERION = (
    "The result contains exactly one syntactically complete boxed answer "
    "and no other boxed expression."
)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def is_concat(row: dict) -> bool:
    return isinstance(row.get("ids"), list)


def answer_count(row: dict) -> int:
    if not is_concat(row):
        return 1
    ids = row["ids"]
    answers = row.get("answers")
    if not ids or not isinstance(answers, list) or len(ids) != len(answers):
        raise ValueError("concat rows require equally sized non-empty ids and answers")
    return len(ids)


def record_identity(row: dict) -> str:
    if is_concat(row):
        return "ids:" + json.dumps(
            row["ids"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return f"id:{row['id']}"


def extract_active_question(
    problem: str,
    question_index: int,
    question_count: int,
) -> str:
    marker_pattern = re.compile(
        r"(?m)^\[Question\s+(\d+)(?:\s*\|\s*[^\]]+)?\]\s*$"
    )
    markers = list(marker_pattern.finditer(problem))
    marker_indices = [int(match.group(1)) for match in markers]
    expected_indices = list(range(1, question_count + 1))
    if marker_indices != expected_indices:
        raise ValueError(
            "concat problem must contain ordered Question markers: "
            f"expected={expected_indices} actual={marker_indices}"
        )
    active_marker = markers[question_index - 1]
    end = (
        markers[question_index].start()
        if question_index < question_count
        else len(problem)
    )
    active_problem = problem[active_marker.end() : end].strip()
    if not active_problem:
        raise ValueError(f"Question {question_index} has no problem text")
    return active_problem


def extract_boxed_expressions(text: str) -> list[str]:
    marker = r"\boxed{"
    expressions = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return expressions
        depth = 1
        index = start + len(marker)
        while index < len(text) and depth:
            if text[index] == "{" and text[index - 1] != "\\":
                depth += 1
            elif text[index] == "}" and text[index - 1] != "\\":
                depth -= 1
            index += 1
        if depth:
            return []
        expressions.append(text[start:index])
        cursor = index


def boxed_content(expression: str) -> str:
    expression = expression.strip()
    marker = r"\boxed{"
    if expression.startswith(marker) and expression.endswith("}"):
        return expression[len(marker) : -1].strip()
    return expression


def strip_boxed_commands(text: str) -> str:
    marker = r"\boxed{"
    output = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            output.append(text[cursor:])
            return "".join(output)
        output.append(text[cursor:start])
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{" and text[index - 1] != "\\":
                depth += 1
            elif text[index] == "}" and text[index - 1] != "\\":
                depth -= 1
            index += 1
        if depth:
            output.append(text[start:])
            return "".join(output)
        output.append(text[content_start : index - 1])
        cursor = index


def build_output(
    source: dict,
    steps: list[dict],
    predictions: list[str],
) -> dict:
    if is_concat(source):
        gold_answers = source["answers"]
        predictions = predictions[: len(gold_answers)]
        predictions.extend([""] * (len(gold_answers) - len(predictions)))
        return {
            "ids": source["ids"],
            "problem": source["problem"],
            "steps": steps,
            "pred_answers": predictions,
            "gold_answers": gold_answers,
            "correct": [
                grade_answer(prediction, gold)
                for prediction, gold in zip(predictions, gold_answers)
            ],
        }

    prediction = predictions[0] if predictions else ""
    gold_answer = source["answer"]
    return {
        "id": source["id"],
        "problem": source["problem"],
        "steps": steps,
        "pred_answer": prediction,
        "gold_answer": gold_answer,
        "correct": bool(prediction) and grade_answer(prediction, gold_answer),
    }


def strip_planner_stop_token(text: str) -> str:
    clean = text.strip()
    if clean.endswith(PLANNER_STOP_TOKEN):
        clean = clean[: -len(PLANNER_STOP_TOKEN)].rstrip()
    return clean


def parse_planner_response(text: str) -> tuple[str, str, str, str]:
    clean = strip_planner_stop_token(text)
    if clean == "DONE":
        raise ValueError("DONE is not allowed before Finalization")
    clean = re.sub(
        r"\\n(?=\s*(?:cognitive_mode|next_subgoal|deliverable|"
        r"success_criterion)\s*:)",
        "\n",
        clean,
        flags=re.IGNORECASE,
    )
    values = {
        match.group(1).lower(): match.group(2).strip()
        for match in re.finditer(
            r"^\s*(?:[-*]\s*)?(cognitive_mode|next_subgoal|deliverable|"
            r"success_criterion)\s*:\s*(.*)$",
            clean,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    }
    required = (
        "cognitive_mode",
        "next_subgoal",
        "deliverable",
        "success_criterion",
    )
    if any(not values.get(field) for field in required):
        raise ValueError(f"invalid planner response: {text!r}")
    return (
        normalize_cognitive_mode(values["cognitive_mode"]),
        values["next_subgoal"],
        values["deliverable"],
        values["success_criterion"],
    )


def validate_planner_action(
    action: tuple[str, str, str, str],
    *,
    active_step: int,
    max_steps: int,
    constrained: bool = False,
) -> None:
    mode, subgoal, deliverable, success_criterion = action
    if mode not in PLANNER_MODES:
        raise ValueError(f"unsupported cognitive mode: {mode}")
    for field, value, limit in zip(
        ("next_subgoal", "deliverable", "success_criterion"),
        (subgoal, deliverable, success_criterion),
        PLANNER_FIELD_LIMITS,
    ):
        # Under grammar-constrained decoding each field is already a single
        # line, so an escaped break is ordinary text (LaTeX such as \neq
        # starts the same way).  Rejecting it here would fail rows the
        # grammar deliberately allows, and no retry could recover.
        if not constrained and LITERAL_ESCAPED_LINE_BREAK.search(value):
            raise ValueError(f"{field} contains a literal escaped line break")
        if len(value) > limit:
            raise ValueError(f"{field} exceeds its {limit}-character limit")
    if mode == "Finalization":
        if active_step == 1:
            raise ValueError("Finalization cannot be the first step")
        contract = " ".join(action[1:])
        if FINALIZATION_BOX_CONTRADICTION.search(contract):
            raise ValueError("Finalization contradicts the boxed-answer contract")
        if "boxed" not in contract.lower():
            raise ValueError("Finalization must request one boxed answer")
    if active_step == max_steps and mode != "Finalization":
        raise ValueError("the final allowed step must use Finalization")


def planner_response_regex(
    *,
    require_finalization: bool,
    allow_finalization: bool,
    question_index: Optional[int],
) -> str:
    prefix_text = (
        f"[Question {question_index}] "
        if question_index is not None
        else ""
    )
    prefix = re.escape(prefix_text)
    finalization = (
        r"cognitive_mode: Finalization"
        rf"\nnext_subgoal: {prefix}"
        r"Use the completed reasoning state to present the requested answer "
        r"in the original problem's required form\."
        rf"\ndeliverable: {prefix}"
        r"A concise response containing exactly one boxed answer\."
        rf"\nsuccess_criterion: {prefix}"
        r"The boxed response agrees with the conclusion derived in the "
        r"completed state\."
    )
    if require_finalization:
        body = finalization
    else:
        modes = "|".join(
            mode for mode in PLANNER_MODES if mode != "Finalization"
        )
        # The validator measures each field including its [Question N] prefix,
        # so the regex has to spend that budget too.
        subgoal_limit, deliverable_limit, success_limit = (
            max(1, limit - len(prefix_text)) for limit in PLANNER_FIELD_LIMITS
        )
        nonfinal = (
            rf"cognitive_mode: ({modes})"
            rf"\nnext_subgoal: {prefix}[^\n]{{1,{subgoal_limit}}}"
            rf"\ndeliverable: {prefix}[^\n]{{1,{deliverable_limit}}}"
            rf"\nsuccess_criterion: {prefix}[^\n]{{1,{success_limit}}}"
        )
        body = (
            rf"({nonfinal}|{finalization})"
            if allow_finalization
            else nonfinal
        )
    # Keep special tokens out of grammar-constrained decoding.  SGLang's
    # grammar backends operate on the normal vocabulary and cannot reliably
    # match this added control token inside a regex.
    return rf"({body})"


def strip_partial_stop(text: str, stop: str) -> str:
    for prefix_length in range(len(stop), 0, -1):
        prefix = stop[:prefix_length]
        if text.endswith(prefix):
            return text[:-prefix_length]
    return text


def complete(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    stop: Optional[list[str]] = None,
    regex: Optional[str] = None,
    strip_output: bool = True,
) -> str:
    request = {
        "model": model,
        "prompt": tokenizer.encode(prompt, add_special_tokens=False),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    if seed is not None:
        request["seed"] = seed
    extra_body = {}
    if stop:
        vocab = tokenizer.get_vocab()
        stop_token_ids = [vocab[value] for value in stop if value in vocab]
        if stop_token_ids:
            extra_body["stop_token_ids"] = stop_token_ids
    if regex:
        extra_body["regex"] = regex
    if extra_body:
        request["extra_body"] = extra_body
    response = client.completions.create(**request)
    text = response.choices[0].text
    return text.strip() if strip_output else text


def complete_nonfinal(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> str:
    result = complete(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=["</think>"],
        strip_output=False,
    )
    return strip_partial_stop(result, "</think>").strip()


def salvage_finalization_from_boxed(raw: str) -> Optional[str]:
    """Recover a Finalization whose <result> block never closed.

    The executor can spend its whole token budget on private reasoning and get
    cut off before emitting </result>. The caller then sees an empty result and
    fails the whole trajectory -- in concat mode that voids every remaining
    question in the group, so one truncated step costs nine other answers.

    A Finalization only has to deliver one boxed answer. If the raw output
    already holds exactly one syntactically complete \\boxed{...}, that is the
    deliverable and the missing tag is cosmetic. extract_boxed_expressions
    returns [] on unbalanced braces, so a half-written box is not salvaged.
    """
    boxed = extract_boxed_expressions(raw)
    if len(boxed) != 1:
        return None
    return f"The final answer is {boxed[0]}."


def complete_finalization(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> str:
    raw = complete(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=["</result>", "<|step_end|>"],
    )
    if "<result>" in raw and "</result>" not in raw:
        raw = f"{raw.rstrip()}\n</result>"
    match = re.search(
        r"<result>\s*(.*?)\s*</result>",
        raw,
        flags=re.DOTALL,
    )
    if match:
        result = match.group(1).strip()
        if result:
            return result
    salvaged = salvage_finalization_from_boxed(raw)
    return salvaged if salvaged is not None else ""


def make_planner_action(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    retries: int,
    active_step: int,
    max_steps: int,
    question_index: Optional[int],
    constrain_format: bool,
) -> tuple[str, str, str, str]:
    last_error = ""
    for attempt in range(retries + 1):
        attempt_prompt = prompt
        if last_error:
            attempt_prompt += (
                "\nFormat-only correction:\n"
                f"- Parser error: {last_error}\n"
                "- Return exactly the required four fields with no commentary."
            )
        text = complete(
            client,
            model=model,
            prompt=attempt_prompt,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            # Reuse of one seed makes every retry resample the same text.
            seed=None if seed is None else seed + attempt,
            regex=(
                planner_response_regex(
                    require_finalization=active_step == max_steps,
                    allow_finalization=active_step >= 2,
                    question_index=question_index,
                )
                if constrain_format
                else None
            ),
        )
        try:
            action = parse_planner_response(text)
            validate_planner_action(
                action,
                active_step=active_step,
                max_steps=max_steps,
                constrained=constrain_format,
            )
            return action
        except ValueError as error:
            last_error = str(error)
    raise ValueError(
        f"planner failed after {retries + 1} attempt(s): {last_error}"
    )


def run_closed_loop(
    planner_client: OpenAI,
    executor_client: OpenAI,
    row: dict,
    *,
    planner_model: str,
    executor_model: str,
    planner_tokenizer: PreTrainedTokenizerBase,
    executor_tokenizer: PreTrainedTokenizerBase,
    max_steps: int,
    max_planner_tokens: int,
    max_executor_tokens: int,
    max_final_tokens: int,
    planner_temperature: float,
    executor_temperature: float,
    planner_top_p: float,
    executor_top_p: float,
    seed: Optional[int],
    planner_parse_retries: int,
    constrain_planner_format: bool,
) -> dict:
    problem = row["problem"]
    question_count = answer_count(row)
    concat = is_concat(row)
    steps = []
    completed_high_history = []
    current_high_history = []
    current_low_history = []
    question_predictions = []
    question_index = 1 if concat else None
    active_step = 0

    total_budget = max_steps * question_count
    for step_id in range(1, total_budget + 1):
        active_step += 1
        active_problem = (
            extract_active_question(problem, question_index, question_count)
            if concat
            else problem
        )
        planner_problem = (
            f"{problem}\n\n"
            f"Active Question {question_index} focus "
            "(solve only this question):\n"
            f"{active_problem}"
            if concat
            else problem
        )
        planner_prompt = build_high_level_prompt(
            problem=planner_problem,
            history=completed_high_history + current_high_history,
            history_mode="raw",
            active_question_index=question_index,
            total_question_count=question_count if concat else None,
        )
        if active_step == max_steps:
            planner_prompt += (
                "\nFinal-step protocol constraint:\n"
                "- cognitive_mode must be Finalization.\n"
                "- Return exactly one boxed answer for the active question.\n"
            )
        mode, subgoal, deliverable, success_criterion = make_planner_action(
            planner_client,
            model=planner_model,
            prompt=planner_prompt,
            tokenizer=planner_tokenizer,
            max_tokens=max_planner_tokens,
            temperature=planner_temperature,
            top_p=planner_top_p,
            seed=seed,
            retries=planner_parse_retries,
            active_step=active_step,
            max_steps=max_steps,
            question_index=question_index,
            constrain_format=constrain_planner_format,
        )

        executor_subgoal = subgoal
        executor_deliverable = deliverable
        executor_success = success_criterion
        if mode == "Finalization":
            executor_subgoal = (
                f"{subgoal.rstrip()} Return exactly one boxed final answer."
            )
            executor_deliverable = FINALIZATION_DELIVERABLE
            executor_success = FINALIZATION_SUCCESS_CRITERION

        executor_prompt = build_low_level_prompt(
            problem=active_problem,
            cot_prefix=build_existing_reasoning(current_low_history),
            cognitive_mode=mode,
            subgoal=executor_subgoal,
            deliverable=executor_deliverable,
            success_criterion=executor_success,
            response_mode="private_reasoning",
            answer_count=1,
            active_question_index=question_index,
            total_question_count=question_count if concat else None,
        )
        if mode == "Finalization":
            execution = complete_finalization(
                executor_client,
                model=executor_model,
                prompt=executor_prompt,
                max_tokens=max_final_tokens,
                tokenizer=executor_tokenizer,
                temperature=executor_temperature,
                top_p=executor_top_p,
                seed=seed,
            )
        else:
            execution = complete_nonfinal(
                executor_client,
                model=executor_model,
                prompt=executor_prompt,
                max_tokens=max_executor_tokens,
                tokenizer=executor_tokenizer,
                temperature=executor_temperature,
                top_p=executor_top_p,
                seed=seed,
            )
            if r"\boxed{" in execution:
                execution = strip_boxed_commands(execution)

        if not execution:
            raise ValueError(
                f"executor returned no usable result at step {step_id}"
            )
        steps.append(
            {
                "step_id": step_id,
                "cognitive_mode": mode,
                "subgoal": subgoal,
                "deliverable": deliverable,
                "success_criterion": success_criterion,
                "execution": execution,
            }
        )

        if mode == "Finalization":
            boxed = extract_boxed_expressions(execution)
            if len(boxed) != 1:
                raise ValueError(
                    f"Finalization step {step_id} must contain one boxed answer"
                )
            question_predictions.append(boxed_content(boxed[0]))
            if not concat or question_index == question_count:
                break
            compact = (
                f"[Question {question_index} finalized]\n{boxed[0]}"
            )
            completed_high_history.append(
                {
                    "cognitive_mode": "Finalization",
                    "subgoal": (
                        f"[Question {question_index}] Record the completed "
                        "local final answer."
                    ),
                    "deliverable": (
                        f"The boxed answer for Question {question_index}."
                    ),
                    "success_criterion": (
                        f"Question {question_index} is complete."
                    ),
                    "summary": compact,
                    "text": compact,
                    "completed_question": True,
                }
            )
            current_high_history = []
            current_low_history = []
            question_index += 1
            active_step = 0
            continue

        current_high_history.append(
            {
                "cognitive_mode": mode,
                "subgoal": subgoal,
                "deliverable": deliverable,
                "success_criterion": success_criterion,
                "summary": "",
                "text": execution,
            }
        )
        current_low_history.append(execution)

    if len(question_predictions) != question_count:
        raise ValueError(
            f"inference finalized {len(question_predictions)} of "
            f"{question_count} question(s)"
        )
    return build_output(row, steps, question_predictions)


def parse_endpoints(value: str) -> list[str]:
    endpoints = [
        item.strip().rstrip("/")
        for item in value.split(",")
        if item.strip()
    ]
    if not endpoints:
        raise ValueError("endpoint list cannot be empty")
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--planner_endpoints", required=True)
    parser.add_argument("--executor_endpoints", required=True)
    parser.add_argument("--planner_model", required=True)
    parser.add_argument("--executor_model", required=True)
    parser.add_argument("--planner_tokenizer", type=Path, required=True)
    parser.add_argument("--executor_tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--max_planner_tokens", type=int, default=1024)
    parser.add_argument("--max_executor_tokens", type=int, default=1024)
    parser.add_argument("--max_final_tokens", type=int, default=1024)
    parser.add_argument("--planner_temperature", type=float, default=0.6)
    parser.add_argument("--executor_temperature", type=float, default=0.6)
    parser.add_argument("--planner_top_p", type=float, default=0.95)
    parser.add_argument("--executor_top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--planner_parse_retries", type=int, default=2)
    parser.add_argument("--constrain_planner_format", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_steps < 2:
        parser.error("--max_steps must be at least 2")
    if args.planner_parse_retries < 0:
        parser.error("--planner_parse_retries must be non-negative")
    if args.max_examples is not None and args.max_examples < 0:
        parser.error("--max_examples must be non-negative")
    if args.planner_temperature < 0 or args.executor_temperature < 0:
        parser.error("temperatures must be non-negative")
    if not 0 < args.planner_top_p <= 1 or not 0 < args.executor_top_p <= 1:
        parser.error("top-p values must be in (0, 1]")

    planner_tokenizer = AutoTokenizer.from_pretrained(
        args.planner_tokenizer,
        local_files_only=True,
    )
    executor_tokenizer = AutoTokenizer.from_pretrained(
        args.executor_tokenizer,
        local_files_only=True,
    )
    planner_clients = [
        OpenAI(base_url=f"{endpoint}/v1", api_key="EMPTY", timeout=900)
        for endpoint in parse_endpoints(args.planner_endpoints)
    ]
    executor_clients = [
        OpenAI(base_url=f"{endpoint}/v1", api_key="EMPTY", timeout=900)
        for endpoint in parse_endpoints(args.executor_endpoints)
    ]

    rows = list(read_jsonl(args.input_jsonl))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    completed = (
        {
            record_identity(row): row
            for row in read_jsonl(args.output_jsonl)
        }
        if args.output_jsonl.exists()
        else {}
    )
    pending = [
        row for row in rows if record_identity(row) not in completed
    ]

    write_lock = threading.Lock()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output_jsonl.with_name(f".{args.output_jsonl.name}.tmp")

    def work(index_and_row: tuple[int, dict]) -> dict:
        index, row = index_and_row
        return run_closed_loop(
            planner_clients[index % len(planner_clients)],
            executor_clients[index % len(executor_clients)],
            row,
            planner_model=args.planner_model,
            executor_model=args.executor_model,
            planner_tokenizer=planner_tokenizer,
            executor_tokenizer=executor_tokenizer,
            max_steps=args.max_steps,
            max_planner_tokens=args.max_planner_tokens,
            max_executor_tokens=args.max_executor_tokens,
            max_final_tokens=args.max_final_tokens,
            planner_temperature=args.planner_temperature,
            executor_temperature=args.executor_temperature,
            planner_top_p=args.planner_top_p,
            executor_top_p=args.executor_top_p,
            seed=args.seed,
            planner_parse_retries=args.planner_parse_retries,
            constrain_planner_format=args.constrain_planner_format,
        )

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(work, item): item[1]
            for item in enumerate(pending)
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            identity = record_identity(source)
            try:
                result = future.result()
            except Exception as error:
                failures.append(identity)
                print(
                    f"[FAILED] {identity}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            with write_lock:
                completed[identity] = result
                with snapshot.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        row_identity = record_identity(row)
                        if row_identity in completed:
                            handle.write(
                                json.dumps(
                                    completed[row_identity],
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                snapshot.replace(args.output_jsonl)
            print(
                f"[{finished}/{len(pending)}] {identity} "
                f"steps={len(result['steps'])}",
                flush=True,
            )

    if failures:
        raise RuntimeError(f"Failed rows: {', '.join(failures)}")


if __name__ == "__main__":
    main()
