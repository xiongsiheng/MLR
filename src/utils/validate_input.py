#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    args = parser.parse_args()

    seen = set()
    count = 0
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", "")).strip()
            problem = row.get("problem")
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(
                    f"line {line_number}: missing problem text"
                )
            ids = row.get("ids")
            answers = row.get("answers")
            if ids is not None or answers is not None:
                if row_id:
                    raise ValueError(
                        f"line {line_number}: concat row must use ids, not id"
                    )
                if not isinstance(ids, list) or not ids:
                    raise ValueError(
                        f"line {line_number}: ids must be a non-empty list"
                    )
                if not all(isinstance(value, str) and value.strip() for value in ids):
                    raise ValueError(
                        f"line {line_number}: every ids entry must be a non-empty string"
                    )
                if len(set(ids)) != len(ids):
                    raise ValueError(
                        f"line {line_number}: duplicate value within ids"
                    )
                if not isinstance(answers, list) or len(answers) != len(ids):
                    raise ValueError(
                        f"line {line_number}: answers must match ids length"
                    )
                identity = ("ids", tuple(ids))
            else:
                if not row_id:
                    raise ValueError(f"line {line_number}: missing id")
                if "answer" not in row:
                    raise ValueError(f"line {line_number}: missing answer")
                identity = ("id", row_id)
            if identity in seen:
                raise ValueError(
                    f"line {line_number}: duplicate row identity {identity!r}"
                )
            seen.add(identity)
            count += 1
    if not count:
        raise ValueError("Input JSONL is empty")
    print(f"Validated {count} input row(s)")


if __name__ == "__main__":
    main()
