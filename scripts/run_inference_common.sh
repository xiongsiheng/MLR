#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This helper must be sourced by a dataset run script." >&2
    exit 2
fi

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
weights_root=${WEIGHTS_ROOT:-"$package_root/model_weights"}

base_model=${BASE_MODEL:-"$weights_root/executor"}
planner_adapter=${PLANNER_ADAPTER:-"$weights_root/planner"}
planner_model=${PLANNER_MODEL:-"$weights_root/planner_merged"}
# The adapter directory carries the tokenizer the planner was trained with.
planner_tokenizer=${PLANNER_TOKENIZER:-$planner_adapter}
executor_tokenizer=${EXECUTOR_TOKENIZER:-$base_model}

if [[ -x "$package_root/.venv_compat/bin/python" ]]; then
    default_merge_python="$package_root/.venv_compat/bin/python"
    default_runtime_python="$package_root/.venv_compat/bin/python"
else
    default_merge_python=python
    default_runtime_python=python
fi
merge_python=${MERGE_PYTHON:-$default_merge_python}
runtime_python=${RUNTIME_PYTHON:-$default_runtime_python}
export PATH="$(dirname "$runtime_python"):$(dirname "$merge_python"):$PATH"

# FlashInfer JIT-compiles its kernels and needs a CUDA toolkit matching the
# torch build.  The system nvcc is 11.5, which predates cuda_fp8.h, so the
# compile fails unless CUDA_HOME points at the bundled 12.4 toolkit.
if [[ -z "${CUDA_HOME:-}" && -x "$package_root/.cuda124/bin/nvcc" ]]; then
    export CUDA_HOME="$package_root/.cuda124"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

planner_gpu=${PLANNER_GPU:-0}
executor_gpu=${EXECUTOR_GPU:-1}
planner_port=${PLANNER_PORT:-50807}
executor_port=${EXECUTOR_PORT:-50800}
mem_fraction=${MEM_FRACTION_STATIC:-0.6}
workers=${WORKERS:-1}
max_examples=${MAX_EXAMPLES:-}
planner_temperature=${PLANNER_TEMPERATURE:-0.6}
executor_temperature=${EXECUTOR_TEMPERATURE:-0.6}
planner_top_p=${PLANNER_TOP_P:-0.95}
executor_top_p=${EXECUTOR_TOP_P:-0.95}
seed=${SEED:-1}
constrain_planner_format=${CONSTRAIN_PLANNER_FORMAT:-1}
attention_backend=${ATTENTION_BACKEND:-flashinfer}
sampling_backend=${SAMPLING_BACKEND:-flashinfer}
grammar_backend=${GRAMMAR_BACKEND:-xgrammar}
disable_overlap_schedule=${DISABLE_OVERLAP_SCHEDULE:-0}
disable_cuda_graph=${DISABLE_CUDA_GRAPH:-0}
logs_dir=${LOGS_DIR:-"$package_root/logs"}
run_id="mlr-b090-${USER:-user}-$$"

export PYTHONPATH="$package_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

process_groups=()
cleanup() {
    local original_status=$?
    local cleanup_status=0
    local pgid
    local alive
    trap - EXIT INT TERM
    set +e

    for pgid in "${process_groups[@]}"; do
        kill -TERM -- "-$pgid" 2>/dev/null
    done

    for _ in $(seq 1 20); do
        alive=0
        for pgid in "${process_groups[@]}"; do
            if kill -0 -- "-$pgid" 2>/dev/null; then
                alive=1
            fi
        done
        [[ $alive -eq 0 ]] && break
        sleep 1
    done

    for pgid in "${process_groups[@]}"; do
        if kill -0 -- "-$pgid" 2>/dev/null; then
            kill -KILL -- "-$pgid" 2>/dev/null
        fi
    done
    for pgid in "${process_groups[@]}"; do
        wait "$pgid" 2>/dev/null
    done

    for pgid in "${process_groups[@]}"; do
        if ps -eo pgid= | awk -v target="$pgid" \
            '$1 == target { found=1 } END { exit !found }'; then
            echo "ERROR: process group $pgid remains after cleanup" >&2
            cleanup_status=1
        fi
    done
    if pgrep -af -- "$run_id"; then
        echo "ERROR: inference process remains for $run_id" >&2
        cleanup_status=1
    fi

    if [[ $cleanup_status -ne 0 ]]; then
        exit "$cleanup_status"
    fi
    exit "$original_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ${#inputs[@]} -eq 0 || ${#inputs[@]} -ne ${#outputs[@]} ]]; then
    echo "Dataset script supplied invalid input/output lists." >&2
    exit 2
fi
for input in "${inputs[@]}"; do
    "$runtime_python" "$package_root/src/utils/validate_input.py" "$input"
done
test -s "$base_model/config.json"
test -s "$base_model/model.safetensors.index.json"
test -s "$planner_adapter/adapter_config.json"
test -s "$planner_adapter/adapter_model.safetensors"

if [[ ! -s "$planner_model/config.json" ]]; then
    test ! -e "$planner_model"
    "$merge_python" "$package_root/src/utils/merge_lora_checkpoint.py" \
        --adapter "$planner_adapter" \
        --base_model "$base_model" \
        --adapter_scale 1.0 \
        --torch_dtype bfloat16 \
        --output_dir "$planner_model"
fi

for output in "${outputs[@]}"; do
    mkdir -p "$(dirname "$output")"
done
mkdir -p "$logs_dir"

planner_format_args=()
if [[ "$constrain_planner_format" != 0 ]]; then
    planner_format_args+=(--constrain_planner_format)
fi

max_examples_args=()
if [[ -n "$max_examples" ]]; then
    max_examples_args+=(--max_examples "$max_examples")
fi

server_backend_args=()
server_backend_args+=(--grammar-backend "$grammar_backend")
if [[ -n "$attention_backend" ]]; then
    server_backend_args+=(--attention-backend "$attention_backend")
fi
if [[ -n "$sampling_backend" ]]; then
    server_backend_args+=(--sampling-backend "$sampling_backend")
fi
if [[ "$disable_overlap_schedule" != 0 ]]; then
    server_backend_args+=(--disable-overlap-schedule)
fi
if [[ "$disable_cuda_graph" != 0 ]]; then
    server_backend_args+=(--disable-cuda-graph)
fi

setsid env CUDA_VISIBLE_DEVICES="$planner_gpu" \
    "$runtime_python" -m sglang.launch_server \
    --model-path "$planner_model" \
    --tokenizer-path "$planner_tokenizer" \
    --served-model-name "$run_id-planner" \
    --host 127.0.0.1 \
    --port "$planner_port" \
    --dtype bfloat16 \
    --context-length 32768 \
    --mem-fraction-static "$mem_fraction" \
    "${server_backend_args[@]}" \
    >"$logs_dir/${run_id}_planner.log" 2>&1 &
process_groups+=("$!")

setsid env CUDA_VISIBLE_DEVICES="$executor_gpu" \
    "$runtime_python" -m sglang.launch_server \
    --model-path "$base_model" \
    --tokenizer-path "$executor_tokenizer" \
    --served-model-name "$run_id-executor" \
    --host 127.0.0.1 \
    --port "$executor_port" \
    --dtype bfloat16 \
    --context-length 32768 \
    --mem-fraction-static "$mem_fraction" \
    "${server_backend_args[@]}" \
    >"$logs_dir/${run_id}_executor.log" 2>&1 &
process_groups+=("$!")

wait_for_server() {
    local port=$1
    local name=$2
    local pid=$3
    local log_file=$4
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name server exited before becoming healthy" >&2
            tail -80 "$log_file" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "$name server did not become healthy on port $port" >&2
    tail -80 "$log_file" >&2 || true
    return 1
}

wait_for_server \
    "$planner_port" Planner "${process_groups[0]}" \
    "$logs_dir/${run_id}_planner.log"
wait_for_server \
    "$executor_port" Executor "${process_groups[1]}" \
    "$logs_dir/${run_id}_executor.log"

for index in "${!inputs[@]}"; do
    input=${inputs[$index]}
    output=${outputs[$index]}
    answers_output="${output%.jsonl}_answers.jsonl"

    setsid "$runtime_python" \
        "$package_root/src/inference.py" \
        --input_jsonl "$input" \
        --output_jsonl "$output" \
        --planner_endpoints "http://127.0.0.1:$planner_port" \
        --executor_endpoints "http://127.0.0.1:$executor_port" \
        --planner_model "$run_id-planner" \
        --executor_model "$run_id-executor" \
        --planner_tokenizer "$planner_tokenizer" \
        --executor_tokenizer "$executor_tokenizer" \
        --workers "$workers" \
        "${max_examples_args[@]}" \
        --max_steps 8 \
        --max_planner_tokens 1024 \
        --max_executor_tokens 1024 \
        --max_final_tokens 1024 \
        --planner_temperature "$planner_temperature" \
        --executor_temperature "$executor_temperature" \
        --planner_top_p "$planner_top_p" \
        --executor_top_p "$executor_top_p" \
        --seed "$seed" \
        --planner_parse_retries 2 \
        "${planner_format_args[@]}" &
    inference_pid=$!
    process_groups+=("$inference_pid")
    wait "$inference_pid"

    "$runtime_python" "$package_root/src/utils/extract_answers.py" \
        "$output" "$answers_output"

    echo "Predictions: $output"
    echo "Answers:     $answers_output"
done
