#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def extract_boxed(text: str) -> list[str]:
    marker = r"\boxed{"
    answers = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return answers
        index = start + len(marker)
        content_start = index
        depth = 1
        while index < len(text) and depth:
            if text[index] == "{" and text[index - 1] != "\\":
                depth += 1
            elif text[index] == "}" and text[index - 1] != "\\":
                depth -= 1
            index += 1
        if depth:
            return answers
        answers.append(text[content_start : index - 1])
        cursor = index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("answers_jsonl", type=Path)
    args = parser.parse_args()

    args.answers_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_jsonl.open(encoding="utf-8") as source, \
            args.answers_jsonl.open("w", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("pred_answers"), list):
                answers = row["pred_answers"]
            elif "pred_answer" in row:
                answers = [row["pred_answer"]]
            else:
                question_answers = row.get("question_answers") or {}
                if question_answers:
                    answers = [
                        extract_boxed(str(value))[0]
                        for _, value in sorted(
                            question_answers.items(),
                            key=lambda item: int(item[0]),
                        )
                    ]
                else:
                    final_text = ""
                    for step in reversed(row.get("steps") or []):
                        if step.get("cognitive_mode") == "Finalization":
                            final_text = str(
                                step.get("text", step.get("execution", ""))
                            )
                            break
                    answers = extract_boxed(final_text)
            record = {
                "ids": row["ids"],
                "answers": answers,
            } if row.get("ids") else {
                "id": row.get("id"),
                "answers": answers,
            }
            if row.get("termination") is not None:
                record["termination"] = row["termination"]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
