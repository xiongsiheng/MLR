# Enhancing Language Model Reasoning with Structured Multi-Level Modeling

This repository contains the code for the paper [ICLR 26] [Enhancing Language Model Reasoning with Structured Multi-Level Modeling](https://openreview.net/pdf?id=PlkzZhqBCd).


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

To enable efficient training for long trajectories, we further propose **iterative Step-DPO**, which provides process-level supervision using Twisted Sequential Monte Carlo (TSMC), without training a separate process reward model.


## Quick Start
```sh
cd MLR
pip install -r Requirements.txt
```

Training:
- Cold start:
```sh
cd src
# Replace ${dataset} with one of: MATH, AIME, GPQA, BoardGameQA, all
# Replace ${model} with one of: llama, qwen, qwen_math
python SFT_lowLevel.py --dataset ${dataset} --model ${model} --use_wandb --read_data --create_model --train --save_model
python SFT_highLevel.py --dataset ${dataset} --model ${model} --use_wandb --read_data --create_model --train --save_model
python SFT_compressor.py --dataset ${dataset} --use_wandb --read_data --create_model --train --save_model
```

- RL:
```sh
# Replace ${dataset} with one of: MATH, AIME, GPQA, BoardGameQA, all
# Replace ${model} with one of: llama, qwen, qwen_math
python stepDPO_lowLevel.py --dataset ${dataset} --model ${model} --use_wandb
python stepDPO_highLevel.py --dataset ${dataset} --model ${model} --use_wandb
# python PPO_system.py --dataset ${dataset} --model ${model} --use_wandb --read_data --create_model --train --save_model --save_PPO_traj --visualize
```

Inference:
```sh
# Replace ${dataset} with one of: MATH, AIME, GPQA, BoardGameQA
# Replace ${model} with one of: llama, qwen, qwen_math
python main.py --dataset ${dataset} --model ${model} --batch_size 8 --visualize
```


## Dataset

We release **MLR reasoning trajectory datasets**, including both:

- **[Full trajectories](https://huggingface.co/datasets/sxiong/MLR_full_trajectory)** (raw reasoning)
- **[Structured trajectories](https://huggingface.co/datasets/sxiong/MLR_structured_trajectory)** (step-level annotations with subgoals and reasoning modes)

### Available datasets
- MATH
- AIME
- GPQA
- BoardGameQA

### Features
- Step-level decomposition of reasoning
- Hierarchical structure (planner → executor)
- Supports research on:
  - process supervision
  - long-horizon reasoning
  - planning-based LLM training


## Citation

```bibtex
@inproceedings{xiongenhancing,
  title={Enhancing Language Model Reasoning with Structured Multi-Level Modeling},
  author={Xiong, Siheng and Payani, Ali and Fekri, Faramarz},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
