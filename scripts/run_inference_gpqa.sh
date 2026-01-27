#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${OUTPUT_DIR:-"$root/outputs"}
inputs=("$root/data/GPQA-Diamond/test.jsonl")
outputs=("$output_dir/gpqa_diamond_qwen_1.5b.jsonl")
source "$root/scripts/run_inference_common.sh"
