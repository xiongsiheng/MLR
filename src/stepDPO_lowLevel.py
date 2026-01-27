import os
import json
import argparse

import torch
import datasets as hfds
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import wandb

from utils import load_saved_model


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["llama", "qwen", "qwen_math"], help="Base model family")
parser.add_argument("--dataset", type=str, required=True, help="'all' or one of: MATH, AIME, GPQA, BoardGameQA")
parser.add_argument("--output_dir", type=str, default="../model_weights", help="Where to save model and logs")
parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
args = parser.parse_args()

# Model selection
if args.model == "llama":
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
elif args.model == "qwen":
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
elif args.model == "qwen_math":
    model_name = "Qwen/Qwen2.5-Math-7B"
    tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
else:
    raise ValueError("Unsupported model name. Choose 'llama' or 'qwen' or 'qwen_math'.")

# Output dirs
OUT_DIR = f"{args.output_dir}/{model_name.split('/')[-1]}_DPO_lowLevel"

# Datasets list
dataset_names = ['MATH', 'AIME', 'GPQA', 'BoardGameQA'] if args.dataset == 'all' else [args.dataset]

# W&B
if args.use_wandb:
    wandb.init(
        project="DPO_lowLevel",
        tags=[model_name, *dataset_names],
    )
else:
    os.environ["WANDB_DISABLED"] = "true"

# Load DPO samples
samples = []
for name in dataset_names:
    folder_path = f"../data/{name}_stepDPO/train_lowLevel.jsonl"
    with open(folder_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

print(f"Number of samples: {len(samples)}")
train_dataset = hfds.Dataset.from_list(samples)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

# Model
MODEL_ID = f"{args.output_dir}/{model_name.split('/')[-1]}_lowLevel_base_SFT.safetensors"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
load_saved_model(model, MODEL_ID, silent=True)

# Training config (start conservative to avoid OOM)
training_args = DPOConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,       # was 8
    gradient_accumulation_steps=8,       # keep global batch similar but safer
    learning_rate=5e-6,
    num_train_epochs=2,
    bf16=True if torch.cuda.is_available() else False,
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    max_prompt_length=6144,
    max_length=8192,                     # total (prompt + response)
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    # ref_model=None  # default: snapshot of `model` as frozen reference
)

trainer.train()
trainer.save_model()  # saves to OUT_DIR
