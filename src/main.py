import argparse
import json
import os

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import *
from utils import *




def preprocess_fn_factory(dataset_name):
    '''
    Factory function to create a preprocessing function based on the dataset name.
    '''
    def preprocess_fn(example):
        instruction = "Please reason step by step, and put your final answer within \\boxed{}.\n<think>\n"
        if dataset_name == 'MATH':
            return {'prompt': f"{example['problem']}\n\n{instruction}", 'answer': parse_boxed_result(example['solution']), 'id': example['id']}
        elif dataset_name == 'AIME':
            return {'prompt': f"{example['Problem']}\n\n{instruction}", 'answer': example['Answer'], 'id': example['ID']}
        elif dataset_name == 'GPQA':
            problem, answer = make_qa_task(example)
            return {'prompt': f"{problem}\n\n{instruction}", 'answer': answer, 'id': example['id']}
        elif dataset_name == 'BoardGameQA':
            label_map = {'proved': 'yes', 'disproved': 'no', 'unknown': 'unknown'}
            return {'prompt': f"{example['example']}\n\n{instruction}", 'answer': label_map[example["label"]], 'id': example['id']}
    return preprocess_fn


def main(dataset_name, model_name, batch_size, model_dir, output_dir, visualization):
    '''
    Main function to run the inference pipeline.
    '''
    # ----- load model and tokenizer -----
    if model_name == 'llama':
        model_name = "meta-llama/Llama-3.1-8B"
        tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    elif model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-1.5B"
        tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    elif model_name == 'qwen_math':
        model_name = "Qwen/Qwen2.5-Math-7B"
        tokenizer_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    else:
        raise ValueError("Unsupported model name. Choose 'llama' or 'qwen' or 'qwen_math'.")

    # We only need to load the backbone model once and toggle the LoRA for different levels
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    load_saved_model(
        backbone,
        f"{model_dir}/{model_name.split('/')[-1]}_lowLevel_base_PPO.safetensors",
        silent=True,
    )
    backbone.eval()
    print('Backbone model loaded successfully.')

    lora_path=f"{model_dir}/{model_name.split('/')[-1]}_highLevel_lora_SFT"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    compressor = AutoModelForCausalLM.from_pretrained(
                            "Qwen/Qwen2.5-0.5B-Instruct",
                            torch_dtype=torch.bfloat16,
                            device_map='auto',
                            )
    load_saved_model(compressor, f"{model_dir}/Qwen2.5-0.5B-Instruct_compressor_SFT.safetensors", silent=True)
    compressor.eval()
    print('Compressor loaded successfully.')

    tokenizer_compress = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    system = MLR(backbone, lora_path, compressor, tokenizer, tokenizer_compress)


    # ----- load dataset -----
    if dataset_name == 'MATH':
        with open("../data/MATH/MATH500_unique_ids.json") as fh:
            uids = json.load(fh)
        items = []
        for uid in uids:
            with open(f"../data/MATH/{uid}") as fh:
                data = json.load(fh)
            data["id"] = uid.replace("/", "_").split(".")[0]
            items.append(data)
        dataset = Dataset.from_list(items)
    elif dataset_name == 'AIME':
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    elif dataset_name == 'GPQA':
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = []
        for (i, ex) in enumerate(dataset):
            ex['id'] = f"GPQA_diamond_{i}"
            items.append(ex)
        dataset = Dataset.from_list(items)
    elif dataset_name == 'BoardGameQA':
        test_subsets = ['BoardgameQA-Main-depth3', 'BoardgameQA-DifficultConflict-depth2', 'BoardgameQA-HighConflict-depth2', 'BoardgameQA-KnowledgeHeavy-depth2', 'BoardgameQA-ManyDistractors-depth2']
        items = []
        for subset in test_subsets:
            with open(f"../data/BoardGameQA/{subset}/test.json") as fh:
                data = json.load(fh)
            for i, item in enumerate(data):
                item['id'] = f"{subset}_test_{i}"
            items.extend(data)
        dataset = Dataset.from_list(items)
    else:
        raise ValueError("Unsupported dataset name. Choose 'MATH', 'AIME', 'GPQA', or 'BoardGameQA'.")

    preprocess_fn = preprocess_fn_factory(dataset_name)

    processed_data = dataset.map(preprocess_fn, batched=False)
    dataloader = DataLoader(list(processed_data), batch_size=batch_size, collate_fn=lambda x: x, shuffle=False)

    # ----- run inference -----
    eos_token_id_lowLevel = {'meta-llama/Llama-3.1-8B': 85399, 'Qwen/Qwen2.5-1.5B': 84299, 'Qwen/Qwen2.5-Math-7B': 84299}
    eos_token_id_highLevel = {'meta-llama/Llama-3.1-8B': 23438, 'Qwen/Qwen2.5-1.5B': 22614, 'Qwen/Qwen2.5-Math-7B': 22614}
    eos_token_id = {'meta-llama/Llama-3.1-8B': 128014, 'Qwen/Qwen2.5-1.5B': 151649, 'Qwen/Qwen2.5-Math-7B': 151649}
    bos_token_id_lowLevel = {'meta-llama/Llama-3.1-8B': 78229, 'Qwen/Qwen2.5-1.5B': 77129, 'Qwen/Qwen2.5-Math-7B': 77129}
    generation_config = {
        "max_new_tokens_lowLevel": 1024,
        "max_new_tokens_highLevel": 128,
        "max_new_tokens_compress": 512,
        "max_new_tokens": 16384,
        "temperature_lowLevel": 0.6,
        "temperature_highLevel": 0.6,
        "temperature_compress": 0,
        "top_p_lowLevel": 0.95,
        "top_p_highLevel": 0.95,
        "top_p_compress": 1,
        "max_stages": 20,
        "eos_token_id_lowLevel": eos_token_id_lowLevel[model_name],
        "eos_token_id_highLevel": eos_token_id_highLevel[model_name],
        "eos_token_id_compress": 198,
        "eos_token_id": eos_token_id[model_name],
        "bos_token_id_lowLevel": bos_token_id_lowLevel[model_name]
    }
    for batch in dataloader:
        with torch.no_grad():
            batch_outputs = system.inference(batch, visualization=visualization, **generation_config)
        
        for sample in batch_outputs:
            sample['flag_correct'] = grade_answer_unified(parse_boxed_result(sample['prediction_lowLevel']), sample['answer'])

            # Save the output to a file
            os.makedirs(f"{output_dir}/{dataset_name}", exist_ok=True)
            with open(f"{output_dir}/{dataset_name}/{sample['id']}.json", 'w') as f:
                json.dump(sample, f)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['llama', 'qwen', 'qwen_math'], required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_dir', type=str, default='../model_weights')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        model_name=args.model,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        visualization=args.visualize
    )
