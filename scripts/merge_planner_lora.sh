#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'USAGE'
Usage: merge_planner_lora.sh [OUTPUT_DIR]

Merges the planner LoRA adapter into the executor base model and writes a
full checkpoint that sglang can serve directly.

Environment overrides:
  PLANNER_ADAPTER   adapter directory      (default: model_weights/planner)
  BASE_MODEL        base model directory   (default: model_weights/executor)
  ADAPTER_SCALE     extra scale on the adapter delta          (default: 1.0)
  TORCH_DTYPE       bfloat16 | float16 | float32         (default: bfloat16)
  PYTHON_BIN        interpreter to use
  FORCE             set to 1 to replace an existing OUTPUT_DIR
USAGE
    exit 0
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
weights_root=${WEIGHTS_ROOT:-"$root/model_weights"}

adapter=${PLANNER_ADAPTER:-"$weights_root/planner"}
base_model=${BASE_MODEL:-"$weights_root/executor"}
output=${1:-${PLANNER_MODEL:-"$weights_root/planner_merged"}}
adapter_scale=${ADAPTER_SCALE:-1.0}
torch_dtype=${TORCH_DTYPE:-bfloat16}

if [[ -x "$root/.venv_compat/bin/python" ]]; then
    python_bin=${PYTHON_BIN:-"$root/.venv_compat/bin/python"}
else
    python_bin=${PYTHON_BIN:-python}
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

test -s "$adapter/adapter_config.json"
test -s "$adapter/adapter_model.safetensors"
test -s "$adapter/tokenizer.json"
test -s "$base_model/config.json"
# Weights may be sharded (index + shards) or a single safetensors file. The BF16
# executor is single-file, and a bare `test -s` on the index exits silently under
# `set -e`, which makes the script look like it did nothing at all.
if [[ ! -s "$base_model/model.safetensors.index.json" ]]; then
    if [[ ! -s "$base_model/model.safetensors" ]]; then
        echo "no weights in $base_model (need model.safetensors or a shard index)" >&2
        exit 1
    fi
fi

if [[ -e "$output" ]]; then
    if [[ "${FORCE:-0}" != 1 ]]; then
        echo "Output already exists: $output" >&2
        echo "Re-run with FORCE=1 to replace it." >&2
        exit 1
    fi
    echo "FORCE=1: removing existing $output"
    rm -rf -- "$output"
fi

echo "adapter:      $adapter"
echo "base model:   $base_model"
echo "output:       $output"
echo "scale/dtype:  $adapter_scale / $torch_dtype"

"$python_bin" "$root/src/utils/merge_lora_checkpoint.py" \
    --adapter "$adapter" \
    --base_model "$base_model" \
    --adapter_scale "$adapter_scale" \
    --torch_dtype "$torch_dtype" \
    --output_dir "$output"

# The merged tokenizer must still round-trip spaces and line breaks.  A
# SentencePiece re-serialization silently drops both, which makes the planner's
# grammar-constrained decoding impossible to satisfy.
"$python_bin" - "$output" <<'PY'
import sys
from transformers import AutoTokenizer

probe = "cognitive_mode: Calculation\nnext_subgoal: x\n<|mlr_highlevel_end|>"
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
restored = tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False))
if restored != probe:
    raise SystemExit(
        "merged tokenizer does not round-trip whitespace:\n"
        f"  expected {probe!r}\n  actual   {restored!r}"
    )
print("Tokenizer round-trip OK")
PY

echo "Merged planner checkpoint ready: $output"
