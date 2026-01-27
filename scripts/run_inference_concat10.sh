#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${OUTPUT_DIR:-"$root/outputs"}
inputs=(
    "$root/data/MLR_concat10/aime24_concat10.jsonl"
    "$root/data/MLR_concat10/math500_concat10.jsonl"
    "$root/data/MLR_concat10/mixture_concat10.jsonl"
)
outputs=(
    "$output_dir/aime24_concat10_qwen_1.5b.jsonl"
    "$output_dir/math500_concat10_qwen_1.5b.jsonl"
    "$output_dir/mixture_concat10_qwen_1.5b.jsonl"
)
source "$root/scripts/run_inference_common.sh"
