#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 CONFIG NUM_QUESTIONS [OUTPUT_JSONL]" >&2
    echo "CONFIG: math_concat | aime_concat | gpqa_concat | boardgameqa_concat | mixture_concat" >&2
    exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=$1
num_questions=$2
output=${3:-}

case "$config" in
    math_concat) default_name=math500_concat10.jsonl ;;
    aime_concat) default_name=aime24_concat10.jsonl ;;
    gpqa_concat) default_name=gpqa_concat10.jsonl ;;
    boardgameqa_concat) default_name=boardgameqa_concat10.jsonl ;;
    mixture_concat) default_name=mixture_concat10.jsonl ;;
    *)
        echo "Unknown config: $config" >&2
        exit 2
        ;;
esac

if [[ -z "$output" ]]; then
    output="$root/data/MLR_concat10/$default_name"
fi

if [[ -x "$root/.venv_compat/bin/python" ]]; then
    python_bin=${PYTHON_BIN:-"$root/.venv_compat/bin/python"}
else
    python_bin=${PYTHON_BIN:-python}
fi

exec "$python_bin" "$root/src/utils/create_concat_benchmark.py" \
    --config "$config" \
    --concat-size 10 \
    --num-questions "$num_questions" \
    --seed "${SEED:-20260909}" \
    --data-root "$root/data" \
    --output "$output"
