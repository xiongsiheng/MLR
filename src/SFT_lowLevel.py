import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import sys
import os
import wandb

from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime

from safetensors.torch import save_file
from utils import *
import argparse




parser = argparse.ArgumentParser()

# String arguments
parser.add_argument("--model", type=str, help="Model name")
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--output_dir", type=str, default="../model_weights", help="Directory to save model and logs")

# Boolean flags
parser.add_argument("--use_wandb", action="store_true", help="Enable logging with Weights & Biases")
parser.add_argument("--read_data", action="store_true", help="Enable reading dataset")
parser.add_argument("--enable_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--create_model", action="store_true", help="Enable model creation")
parser.add_argument("--train", action="store_true", help="Enable training")
parser.add_argument("--save_model", action="store_true", help="Save the model after training")

args = parser.parse_args()



# ----- Configuration -----
f_use_wandb = args.use_wandb
f_read_data = args.read_data
f_enable_gradient_checkpointing = args.enable_gradient_checkpointing
f_create_model = args.create_model
f_train = args.train
f_save_model = args.save_model


output_dir=args.output_dir
os.makedirs(output_dir, exist_ok=True)

dataset_names = ['MATH', 'AIME', 'GPQA', 'BoardGameQA'] if args.dataset == 'all' else [args.dataset]

model_selection = ['llama', 'qwen', 'qwen_math'].index(args.model)
model_name = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-7B"][model_selection]
tokenizer_name = ['deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'][model_selection]

min_seq_length = None
max_seq_length = 1024 * 12
num_epochs = 2

log_file = f"{output_dir}/{model_name.split('/')[-1]}_lowLevel_base_SFT_training_log.txt"
save_path = f"{output_dir}/{model_name.split('/')[-1]}_lowLevel_base_SFT.safetensors"

# Define bucket specifications: each tuple is (min_length, max_length, batch_size)
bucket_specs = [
    (0, 500, 48),      
    (500, 1000, 24),   
    (1000, 1500, 16),   
    (1500, 2000, 8),  
    (2000, 2500, 8),   
    (2500, 128000, 4),   
]




# ----- Training Setup -----
if f_read_data:
    samples = load_samples(dataset_names, level='low')
    print(f"Number of samples: {len(samples)}")
    
if f_use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MLR_lowLevel_SFT",
        tags=[model_name, tokenizer_name, dataset_names],
        config={
            "min_seq_length": min_seq_length,
            "max_seq_length": max_seq_length,
            "num_samples": len(samples),
            "num_epochs": num_epochs
        }
    )
else:
    os.environ["WANDB_DISABLED"] = "true"


if f_create_model:
    model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            device_map='auto',
                            )    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("Model loaded successfully!")




# -------------------- Training Loop --------------------
if f_train:  # Use Libraries Supporting Model Parallelism
    # Train for the residual task.
    if os.path.exists(save_path):
        load_saved_model(model, save_path)
    
    if f_enable_gradient_checkpointing:
        # Enabling gradient checkpointing for memory optimization
        model.gradient_checkpointing_enable()

    # Initialize the accelerator with gradient accumulation
    accelerator = Accelerator(mixed_precision="bf16", device_placement=False, gradient_accumulation_steps=1)

    # Use AdamW for the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Prepare the model and optimizer with Accelerate.
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    train_dataset = MLR_Dataset(
        samples, tokenizer, num_samples=None, min_seq_length=min_seq_length, max_seq_length=max_seq_length, f_print=False
    )

    # Create a bucket sampler instance using our new sampler.
    bucket_sampler = BucketBatchSamplerByLength(train_dataset, bucket_specs, drop_last=False)
    
    collate_fn = collate_fn_factory(tokenizer)

    # Create a DataLoader with the custom sampler (no need to specify batch_size here).
    train_loader = DataLoader(
                        train_dataset,
                        batch_sampler=bucket_sampler,
                        collate_fn=collate_fn
                    )

    # Prepare the dataloader with Accelerate.
    train_loader = accelerator.prepare(train_loader)

    # Recalculate the number of training steps for the current stage.
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    # Initialize the scheduler for this stage.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        running_loss = 0.0
        cnt_batch = 0
        for batch in tqdm(train_loader):
            cnt_batch += 1
            # print(batch['input_ids'].shape, batch['labels'].shape)
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()  # update the learning rate scheduler
                optimizer.zero_grad()
            
            running_loss += loss.item()

            if cnt_batch % 100 == 0 or cnt_batch == len(train_loader):
                avg_loss = running_loss / cnt_batch 
                print(f"Epoch {epoch} Batch {cnt_batch} Average Loss: {avg_loss}")
                current_datetime = datetime.now()
                with open(log_file, "a") as file:
                    file.write(f"{current_datetime} Epoch {epoch} Batch {cnt_batch} Average Loss: {avg_loss}\n")

                if f_save_model and accelerator.is_main_process:
                    # Unwrap the model from Accelerator.
                    unwrapped_model = accelerator.unwrap_model(model)

                    # Save the state dict in safetensors format.
                    save_file(unwrapped_model.state_dict(), save_path)
                    print(f"Model saved at {save_path}")
                    current_datetime = datetime.now()
                    with open(log_file, "a") as file:
                        file.write(f"{current_datetime} Model saved at {save_path}\n")
