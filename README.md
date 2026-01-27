# MLR: Enhancing Language Model Reasoning with Structured Multi-Level Modeling

This repository contains the code for the paper [ICLR 26] [Enhancing Language Model Reasoning with Structured Multi-Level Modeling](https://proceedings.iclr.cc/paper_files/paper/2026/file/3db7d123a316fc690f02818b21967af4-Paper-Conference.pdf).

## Overview

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/MLR/main/misc/MLR_illustration.png' width=750>
</p>

Inference-time scaling via long Chain-of-Thought (CoT) can improve reasoning, but single-policy approaches trained with sparse outcome rewards often suffer from long-horizon plan failures and inefficient training.

This work introduces **Multi-Level Reasoning (MLR)**, a structured framework that decomposes long-horizon reasoning into:
- a **high-level planner** that generates abstract subgoals, and
- a **low-level executor** that produces detailed reasoning conditioned on each subgoal.

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/MLR/main/misc/Framework.png' width=750>
</p>

## Directory structure

```
MLR/
├── data/
├── model_weights/
├── scripts/
├── src/
└── output/
```

## Setup

```sh
pip install -r requirements.txt
```

Two GPUs are recommended: the planner and executor run as separate `sglang` instances on GPU 0 and GPU 1 by default. A single GPU is also supported by assigning both to the same device via `PLANNER_GPU` and `EXECUTOR_GPU`, though throughput will be roughly halved. In that case, reduce `MEM_FRACTION_STATIC` so both models fit in memory.

`sglang` is pinned to `0.4.6` for planner grammar-constrained decoding compatibility. `transformers` is pinned to `<5.0` to avoid tokenizer serialization incompatibilities with the planner (see "Merging the planner LoRA").

### Model weights

Download both models and lay them out under `model_weights/`:

| Component | HF repo | Destination |
|---|---|---|
| Executor (full model, also the planner's base) | [sxiong/MLR_executor_Qwen-1.5B](https://huggingface.co/sxiong/MLR_executor_Qwen-1.5B) | `model_weights/executor/` |
| Planner (LoRA adapter) | [sxiong/MLR_planner_Qwen-1.5B-LoRA](https://huggingface.co/sxiong/MLR_planner_Qwen-1.5B-LoRA) | `model_weights/planner/` |

```sh
huggingface-cli download sxiong/MLR_executor_Qwen-1.5B \
    --local-dir model_weights/executor
huggingface-cli download sxiong/MLR_planner_Qwen-1.5B-LoRA \
    --local-dir model_weights/planner
```

### Merging the planner LoRA

The planner is released as a LoRA adapter and must be merged before serving with `sglang`:

```sh
bash scripts/merge_planner_lora.sh
```

This creates `model_weights/planner_merged/`. The run scripts will create it automatically if missing.
The merge script also preserves the tokenizer files required for grammar-constrained decoding.

## Inference

Run the script for each benchmark:

```sh
bash scripts/run_inference_math500.sh
bash scripts/run_inference_aime24.sh
bash scripts/run_inference_gpqa.sh
bash scripts/run_inference_boardgameqa.sh
bash scripts/run_inference_concat10.sh
```

The scripts automatically start and stop the planner/executor servers. Results are saved under `outputs/`.

For a quick test:

```sh
MAX_EXAMPLES=2 bash scripts/run_inference_math500.sh
```

### Configuration

Key settings can be overridden via environment variables:

| Variable                       |    Default |
| ------------------------------ | ---------: |
| `MAX_EXAMPLES`                 |        all |
| `WORKERS`                      |          1 |
| `OUTPUT_DIR`                   | `outputs/` |
| `PLANNER_GPU` / `EXECUTOR_GPU` |      0 / 1 |
| `MEM_FRACTION_STATIC`          |        0.6 |

Increase `WORKERS` for higher throughput. If FlashInfer fails, use:

```sh
ATTENTION_BACKEND=triton SAMPLING_BACKEND=pytorch
```

### Generating concat benchmarks

Pre-built concat10 benchmarks are available on [Hugging Face](https://huggingface.co/datasets/sxiong/MLR_concat10).

To create your own:

```sh
bash scripts/create_concat10_benchmark.sh CONFIG NUM_QUESTIONS [OUTPUT_JSONL]
```

Supported configs: `math_concat`, `aime_concat`, `gpqa_concat`, `boardgameqa_concat`, and `mixture_concat`.

`NUM_QUESTIONS` must be a multiple of 10.

### Reference outputs

- Single-question: [sxiong/MLR_output](https://huggingface.co/datasets/sxiong/MLR_output)
- Concat10: [sxiong/MLR_concat10_output](https://huggingface.co/datasets/sxiong/MLR_concat10_output)

## Citation

```bibtex
@inproceedings{xiong2026enhancing,
  title={Enhancing language model reasoning with structured multi-level modeling},
  author={Xiong, Siheng and Payani, Ali and Fekri, Faramarz},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={36557--36610},
  year={2026}
}
```
