import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import sys
import os
import wandb

from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime

from peft import LoraConfig, get_peft_model, PeftModel
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



base_model_path = f"{output_dir}/{model_name.split('/')[-1]}_lowLevel_base_SFT.safetensors"
log_file = f"{output_dir}/{model_name.split('/')[-1]}_highLevel_lora_SFT_training_log.txt"
save_folder = f"{output_dir}/{model_name.split('/')[-1]}_highLevel_lora_SFT"

os.makedirs(save_folder, exist_ok=True)




# Define bucket specifications: each tuple is (min_length, max_length, batch_size)
bucket_specs = [
    (0, 500, 88),
    (500, 1000, 40),   
    (1000, 1500, 24),  
    (1500, 2000, 16),   
    (2000, 2500, 16),   
    (2500, 3000, 12),  
    (3000, 3500, 12),
    (3500, 4096, 8),
    (4096, 128000, 4),   
]


 

# ----- Training Setup -----
if f_read_data:
    samples = load_samples(dataset_names, level='high')
    print(f"Number of samples: {len(samples)}")





if f_use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MLR_highLevel_SFT",
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
    load_saved_model(model, base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("Model loaded successfully!")



if f_train:  # Use Libraries Supporting Model Parallelism
    if os.path.exists(f'{save_folder}/adapter_config.json'):
        model = PeftModel.from_pretrained(model, save_folder)
        # Set LoRA parameters as trainable
        for name, param in model.named_parameters():
            param.requires_grad = True if 'lora' in name else False
    else:
        # Wrap the model with LoRA adapters before training.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="all",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    print("Trainable parameters count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if f_enable_gradient_checkpointing:
        # Enabling gradient checkpointing for memory optimization
        model.gradient_checkpointing_enable()

    # Initialize the accelerator with gradient accumulation.
    accelerator = Accelerator(mixed_precision="bf16", device_placement=False, gradient_accumulation_steps=1)

    # Initialize optimizer to only update trainable (LoRA) parameters.
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)

    # Prepare the model and optimizer with Accelerate.
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    
    train_dataset = MLR_Dataset(samples, tokenizer, num_samples=None, max_seq_length=max_seq_length, min_seq_length=min_seq_length, f_print=False)

    # Create a bucket sampler instance using our new sampler.
    bucket_sampler = BucketBatchSamplerByLength(train_dataset, bucket_specs, drop_last=False)
    
    collate_fn = collate_fn_factory(tokenizer)

    train_loader = DataLoader(
                        train_dataset,
                        batch_sampler=bucket_sampler,
                        collate_fn=collate_fn
                    )
  
    train_loader = accelerator.prepare(train_loader)

    # Recalculate the number of training steps for the current stage.
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    # Initialize the scheduler for this stage.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )
    
    for epoch in range(num_epochs): 
        running_loss = 0.0
        cnt_batch = 0
        for batch in tqdm(train_loader):
            cnt_batch += 1
            # print(batch['input_ids'].shape, batch['labels'].shape) 
            # Accelerator handles accumulation internally.
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

            if cnt_batch % 200 == 0 or cnt_batch == len(train_loader):                        
                avg_loss = running_loss / cnt_batch
                print(f"Epoch {epoch} Batch {cnt_batch} Average Loss: {avg_loss}")
                current_datetime = datetime.now()
                with open(log_file, "a") as file:
                    file.write(f"{current_datetime} Epoch {epoch} Batch {cnt_batch} Average Loss: {avg_loss}\n")

                if f_save_model and accelerator.is_main_process:    
                    # Unwrap the model from Accelerator.
                    unwrapped_model = accelerator.unwrap_model(model)
                    # Save the state dict in safetensors format.
                    # save_file(unwrapped_model.state_dict(), save_path)
                    unwrapped_model.save_pretrained(save_folder)
                    print(f"Model saved at {save_folder}")
                    current_datetime = datetime.now()
                    with open(log_file, "a") as file:
                        file.write(f"{current_datetime} Model saved at {save_folder}\n")
