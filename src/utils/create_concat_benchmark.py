#!/usr/bin/env python3
"""Create concatenated benchmarks from the single-question test sets."""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


DATASETS = ("MATH500", "AIME24", "GPQA-Diamond", "BoardGameQA-Hard")
CONFIG_DATASETS = {
    "math_concat": ("MATH500",),
    "aime_concat": ("AIME24",),
    "gpqa_concat": ("GPQA-Diamond",),
    "boardgameqa_concat": ("BoardGameQA-Hard",),
    "mixture_concat": DATASETS,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def read_minimal_test(path: Path) -> list[dict]:
    rows = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "problem", "answer"}:
                raise ValueError(
                    f"{path}:{line_number}: expected fields id, problem, answer"
                )
            if not isinstance(row["id"], str) or not row["id"]:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if row["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {row['id']!r}")
            if not isinstance(row["problem"], str) or not row["problem"].strip():
                raise ValueError(
                    f"{path}:{line_number}: problem must be a non-empty string"
                )
            seen_ids.add(row["id"])
            rows.append(row)
    return rows


def sample_single_dataset(
    rows: list[dict], num_questions: int, rng: random.Random, dataset: str
) -> list[dict]:
    if num_questions > len(rows):
        raise ValueError(
            f"{dataset} test has {len(rows)} questions, "
            f"but {num_questions} were requested"
        )
    return rng.sample(rows, num_questions)


def mixture_schedule(num_questions: int, rng: random.Random) -> list[str]:
    """Return a deterministic near-balanced dataset schedule."""
    order = list(DATASETS)
    rng.shuffle(order)
    offset = rng.randrange(len(order))
    return [
        order[(offset + question_index) % len(order)]
        for question_index in range(num_questions)
    ]


def sample_mixture(
    pools: dict[str, list[dict]], num_questions: int, rng: random.Random
) -> list[dict]:
    schedule = mixture_schedule(num_questions, rng)
    required = {dataset: schedule.count(dataset) for dataset in DATASETS}
    sampled = {}
    for dataset in DATASETS:
        if required[dataset] > len(pools[dataset]):
            raise ValueError(
                f"mixture requires {required[dataset]} {dataset} questions, "
                f"but its test split has only {len(pools[dataset])}"
            )
        sampled[dataset] = iter(rng.sample(pools[dataset], required[dataset]))
    return [next(sampled[dataset]) for dataset in schedule]


def build_concat_row(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    count = len(rows)
    blocks = [
        f"[Question {question_index}]\n{row['problem'].strip()}"
        for question_index, row in enumerate(rows, start=1)
    ]
    instruction = (
        f"Solve these {count} independent questions sequentially. Work on "
        "exactly one active question at a time. Complete that question's "
        "hierarchical reasoning with a local Finalization that returns exactly "
        "one boxed answer, then proceed to the next question. The local boxed "
        "answers will be aggregated automatically."
    )
    return {
        "ids": [row["id"] for row in rows],
        "problem": instruction + "\n\n" + "\n\n".join(blocks),
        "answers": [row["answer"] for row in rows],
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "data_minimal"
    parser = argparse.ArgumentParser(
        description="Create concat benchmark JSONL from minimal test splits."
    )
    parser.add_argument("--config", required=True, choices=CONFIG_DATASETS)
    parser.add_argument("--concat-size", type=positive_int, default=10)
    parser.add_argument(
        "--num-questions",
        type=positive_int,
        required=True,
        help="Total number of individual questions across all output rows.",
    )
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_questions % args.concat_size:
        raise ValueError(
            f"--num-questions ({args.num_questions}) must be divisible by "
            f"--concat-size ({args.concat_size})"
        )

    selected_datasets = CONFIG_DATASETS[args.config]
    pools = {
        dataset: read_minimal_test(args.data_root / dataset / "test.jsonl")
        for dataset in selected_datasets
    }
    rng = random.Random(args.seed)
    if args.config == "mixture_concat":
        sampled = sample_mixture(pools, args.num_questions, rng)
    else:
        dataset = selected_datasets[0]
        sampled = sample_single_dataset(
            pools[dataset], args.num_questions, rng, dataset
        )

    groups = (
        build_concat_row(sampled[start : start + args.concat_size])
        for start in range(0, args.num_questions, args.concat_size)
    )
    num_groups = write_jsonl(args.output, groups)
    print(
        f"Wrote {num_groups} concat rows ({args.num_questions} questions) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
