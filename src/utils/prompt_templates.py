"""Prompt construction shared by the planner and executor at inference time."""

import re
from typing import Dict, Optional, Sequence


HIGH_LEVEL_SYSTEM_PREFIX = (
    "You are a high-level planner. "
    "Given the problem and actual completed history, choose exactly one next "
    "subtask, deliverable, and success criterion. "
    "Do not execute the subtask or provide low-level solution details."
)

COGNITIVE_MODE_CANDIDATES = [
    "ProblemUnderstanding",
    "Decomposition",
    "Exploration",
    "DeepReasoning",
    "Calculation",
    "Verification",
    "Reflection",
    "Backtracking",
    "Synthesis",
    "Finalization",
    "Other",
]

HIGH_LEVEL_PROMPT_TEMPLATE = """{system}

Instruction:
Generate the next cognitive mode, subtask, deliverable, and success criterion.
Rules:
- Output exactly four lines in this format:
    cognitive_mode: <one label>
    next_subgoal: <text>
    deliverable: <single artifact the executor must return>
    success_criterion: <observable result that completes this subtask>
- If the problem is already solved, output exactly: DONE
- cognitive_mode must be one of: {cognitive_mode_candidates}
- If none fits well, use: Other
- Do not repeat previous subgoals
- Keep it minimal and actionable

Problem:
{problem}

{active_question_section}
Completed history ({history_description}):
{history_block}

Continue:
"""


def normalize_cognitive_mode(mode: str) -> str:
    clean = " ".join(str(mode).strip().split())
    if clean in COGNITIVE_MODE_CANDIDATES:
        return clean

    lowered_map = {item.lower(): item for item in COGNITIVE_MODE_CANDIDATES}
    lowered = clean.lower()
    if lowered in lowered_map:
        return lowered_map[lowered]

    synonyms = {
        "understanding": "ProblemUnderstanding",
        "problemunderstanding": "ProblemUnderstanding",
        "decompose": "Decomposition",
        "planning": "Decomposition",
        "explore": "Exploration",
        "reasoning": "DeepReasoning",
        "calculation": "Calculation",
        "compute": "Calculation",
        "verification": "Verification",
        "check": "Verification",
        "reflection": "Reflection",
        "backtrack": "Backtracking",
        "synthesis": "Synthesis",
        "final": "Finalization",
    }
    for key, value in synonyms.items():
        if key in lowered:
            return value
    return "Other"


def _build_history_block(
    history: Sequence[Dict[str, str]], history_mode: str
) -> str:
    if not history:
        return "None"

    lines = []
    for index, item in enumerate(history, start=1):
        if history_mode == "raw":
            history_value = item["text"]
            history_label = "completed_step"
        else:
            history_value = item["summary"]
            history_label = "summary"
        lines.append(
            f"{index}. cognitive_mode: {item['cognitive_mode']}\n"
            f"   subgoal: {item['subgoal']}\n"
            f"   deliverable: {item.get('deliverable', 'A result satisfying the subtask.')}\n"
            f"   success_criterion: {item.get('success_criterion', 'Complete the stated subtask.')}\n"
            f"   {history_label}: {history_value}"
        )
    return "\n".join(lines)


def build_high_level_prompt(
    problem: str,
    history: Sequence[Dict[str, str]],
    history_mode: str,
    active_question_index: Optional[int] = None,
    total_question_count: Optional[int] = None,
) -> str:
    active_question_section = ""
    if active_question_index is not None:
        if total_question_count is None or total_question_count < 1:
            raise ValueError(
                "total_question_count is required with active_question_index"
            )
        if not 1 <= active_question_index <= total_question_count:
            raise ValueError(
                "active_question_index must be within the batch"
            )
        active_question_section = (
            "Active question scope:\n"
            f"- Work only on Question {active_question_index} of "
            f"{total_question_count}.\n"
            "- Do not plan work for any later question.\n"
            "- Finalization means finalizing only the active question; its "
            "executor result must contain exactly one boxed answer.\n"
        )
    history_description = (
        "cognitive_mode + subgoal + deliverable + full completed step text"
        if history_mode == "raw"
        else "cognitive_mode + subgoal + deliverable + summary"
    )
    return HIGH_LEVEL_PROMPT_TEMPLATE.format(
        system=HIGH_LEVEL_SYSTEM_PREFIX,
        problem=problem,
        history_description=history_description,
        history_block=_build_history_block(history, history_mode),
        cognitive_mode_candidates=", ".join(COGNITIVE_MODE_CANDIDATES),
        active_question_section=active_question_section,
    )


BOUNDED_SYSTEM_PREFIX = (
    "You are a one-step executor. "
    "Produce only the state update needed to satisfy the current atomic subtask. "
    "Do not start future subtasks, repeat the problem, or decide what to do next. "
    "Do not state a final answer unless the reasoning mode is Finalization. "
    "Never use \\boxed unless the reasoning mode is Finalization. "
    "Output only concise mathematical reasoning with no headings or labels."
)

BOUNDED_PROMPT_TEMPLATE = """{system}

Execute exactly one atomic subtask and then end the response.

Problem context:
{problem}

Completed state:
{cot_prefix}

{active_question_section}
Reasoning mode:
{cognitive_mode}

Current atomic subtask:
{subgoal}

Required deliverable:
{deliverable}

Success criterion:
{success_criterion}

{finalization_contract_section}
{retry_feedback_section}
State update:
"""

PRIVATE_REASONING_SYSTEM_PREFIX = (
    "You are a one-step executor. "
    "Use private reasoning to solve the current atomic subtask. "
    "Do not start future subtasks or decide what to do next. "
    "After closing the private reasoning with </think>, output exactly one "
    "<result> block containing only the bounded state update for the planner. "
    "When the reasoning mode is Finalization, <result> must be concise and "
    "contain exactly one final answer formatted as \\boxed{...}. "
    "For every other reasoning mode, do not state a final answer and never "
    "use \\boxed in <result>."
)

PRIVATE_REASONING_PROMPT_TEMPLATE = """{system}

Execute exactly one atomic subtask.

Problem context:
{problem}

Completed planner-visible state:
{cot_prefix}

{active_question_section}
Reasoning mode:
{cognitive_mode}

Current atomic subtask:
{subgoal}

Required deliverable:
{deliverable}

Success criterion:
{success_criterion}

{finalization_contract_section}
{retry_feedback_section}
Private reasoning:
<think>
"""

FULL_RESULT_SYSTEM_PREFIX = (
    "You are a one-step executor. "
    "Solve exactly the current atomic subtask and preserve all reasoning and "
    "derived state that later steps need. "
    "Output exactly one <result> block and no text outside it. "
    "Use no other channel, wrapper, or preamble. "
    "Do not start future subtasks or decide what to do next. "
    "When the reasoning mode is Finalization, <result> must be concise and "
    "contain exactly one final answer formatted as \\boxed{...}. "
    "For every other reasoning mode, do not state a final answer and never "
    "use \\boxed in <result>."
)

FULL_RESULT_PROMPT_TEMPLATE = """{system}

Execute exactly one atomic subtask.

Problem context:
{problem}

Completed planner-visible state:
{cot_prefix}

{active_question_section}
Reasoning mode:
{cognitive_mode}

Current atomic subtask:
{subgoal}

Required deliverable:
{deliverable}

Success criterion:
{success_criterion}

{finalization_contract_section}
{retry_feedback_section}
Planner-visible execution:
<result>
"""

SENTINEL_RESULT_SYSTEM_PREFIX = (
    "You are a one-step executor. "
    "Solve exactly the current atomic subtask and preserve all reasoning and "
    "derived state that later steps need. "
    "The continuation after <think> is the single planner-visible execution "
    "result; it is not private reasoning. "
    "End that result with </think>. "
    "Do not emit result-channel tags or any text after </think>. "
    "Do not start future subtasks or decide what to do next. "
    "When the reasoning mode is Finalization, the result must be concise and "
    "contain exactly one final answer formatted as \\boxed{...}. "
    "For every other reasoning mode, do not state a final answer and never "
    "use \\boxed in the result."
)

SENTINEL_RESULT_PROMPT_TEMPLATE = """{system}

Execute exactly one atomic subtask.

Problem context:
{problem}

Completed planner-visible state:
{cot_prefix}

{active_question_section}
Reasoning mode:
{cognitive_mode}

Current atomic subtask:
{subgoal}

Required deliverable:
{deliverable}

Success criterion:
{success_criterion}

{finalization_contract_section}
{retry_feedback_section}
Planner-visible execution:
<think>
"""


def build_existing_reasoning(history_steps: Sequence[str]) -> str:
    if not history_steps:
        return "None"
    return "\n\n".join(history_steps)


def _extract_active_question_text(
    problem: str,
    question_index: int,
    question_count: int,
) -> str:
    marker_pattern = re.compile(
        r"(?m)^\[Question\s+(\d+)(?:\s*\|\s*[^\]]+)?\]\s*$"
    )
    markers = list(marker_pattern.finditer(problem))
    if not markers:
        return problem.strip()
    marker_indices = [int(match.group(1)) for match in markers]
    expected_indices = list(range(1, question_count + 1))
    if marker_indices != expected_indices:
        raise ValueError(
            "Sequential concat problem must contain ordered Question markers: "
            f"expected={expected_indices} actual={marker_indices}"
        )
    active_marker = markers[question_index - 1]
    end = (
        markers[question_index].start()
        if question_index < question_count
        else len(problem)
    )
    active_question = problem[active_marker.end() : end].strip()
    if not active_question:
        raise ValueError(f"Question {question_index} has no problem text")
    return active_question


def build_low_level_prompt(
    problem: str,
    cot_prefix: str,
    cognitive_mode: str,
    subgoal: str,
    deliverable: str = "A result that completes the stated atomic subtask.",
    success_criterion: str = "Complete the stated atomic subtask.",
    response_mode: str = "bounded",
    retry_feedback: Optional[str] = None,
    answer_count: int = 1,
    active_question_index: Optional[int] = None,
    total_question_count: Optional[int] = None,
) -> str:
    if response_mode not in {
        "bounded",
        "private_reasoning",
        "full_result",
        "sentinel_result",
    }:
        raise ValueError(f"Unsupported response mode: {response_mode}")
    if answer_count < 1:
        raise ValueError("answer_count must be at least 1")
    if response_mode == "private_reasoning":
        template = PRIVATE_REASONING_PROMPT_TEMPLATE
        system = PRIVATE_REASONING_SYSTEM_PREFIX
    elif response_mode == "sentinel_result":
        template = SENTINEL_RESULT_PROMPT_TEMPLATE
        system = SENTINEL_RESULT_SYSTEM_PREFIX
    elif response_mode == "full_result":
        template = FULL_RESULT_PROMPT_TEMPLATE
        system = FULL_RESULT_SYSTEM_PREFIX
    else:
        template = BOUNDED_PROMPT_TEMPLATE
        system = BOUNDED_SYSTEM_PREFIX
    finalization_contract_section = ""
    if answer_count > 1:
        system = system.replace(
            "When the reasoning mode is Finalization, <result> must be concise and "
            "contain exactly one final answer formatted as \\boxed{...}.",
            "When the reasoning mode is Finalization, <result> must follow the "
            "indexed multi-answer contract stated in the prompt.",
        )
        finalization_contract_section = (
            "Finalization output contract:\n"
            f"- There are exactly {answer_count} indexed questions.\n"
            "- In Finalization mode, return exactly one line per question in "
            "ascending index order and no other visible text.\n"
            f"- The required lines are 1 through {answer_count}, each formatted "
            "as: <index>: \\boxed{<answer>}\n"
            f"- The Finalization result must therefore contain exactly "
            f"{answer_count} \\boxed{{...}} expressions.\n\n"
        )
    active_question_section = ""
    if active_question_index is not None:
        if total_question_count is None or total_question_count < 1:
            raise ValueError(
                "total_question_count is required with active_question_index"
            )
        if not 1 <= active_question_index <= total_question_count:
            raise ValueError(
                "active_question_index must be within the batch"
            )
        active_question_text = _extract_active_question_text(
            problem,
            active_question_index,
            total_question_count,
        )
        active_question_section = (
            "Active question scope:\n"
            f"- Execute this subtask only for Question "
            f"{active_question_index} of {total_question_count}.\n"
            "- Do not work on any later question.\n"
            "- In Finalization mode, finalize only this active question and "
            "return exactly one boxed answer.\n\n"
            "Active question text (authoritative):\n"
            f"{active_question_text}\n\n"
        )
    retry_feedback_section = ""
    if retry_feedback and retry_feedback.strip():
        retry_feedback_section = (
            "Verifier feedback from the previous rejected attempt:\n"
            f"{retry_feedback.strip()}\n\n"
            "Retry requirement: correct every issue above while completing "
            "the same atomic subtask. Do not defend or repeat the rejected "
            "result.\n"
        )
    return template.format(
        system=system,
        problem=problem,
        cot_prefix=cot_prefix,
        cognitive_mode=cognitive_mode,
        subgoal=subgoal,
        deliverable=deliverable,
        success_criterion=success_criterion,
        finalization_contract_section=finalization_contract_section,
        retry_feedback_section=retry_feedback_section,
        active_question_section=active_question_section,
    )
