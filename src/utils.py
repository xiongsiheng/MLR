import sys
import json
import os
import re
import math
import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from safetensors.torch import load_file
import numpy as np

from grading.grader import grade_answer

import datasets



# ----- Evaluation -----
def grade_answer_unified(pred, gt):
    '''
    Grade the answer based on whether it is math-related or a QA.
    '''
    if any(char.isdigit() for char in gt):
        return grade_answer(pred, gt)
    else:
        return pred.strip().lower() == gt.strip().lower()



# ----- Parse Trajectory -----
def extract_lowLevel_content(trajectories):
    '''
    Extracts the content of each step from the given trajectories.
    '''
    pattern = r"###### (.*?) ######\n(.*?)\n########################"
    outputs = []
    for trajectory in trajectories:
        # Find all steps
        matches = re.findall(pattern, trajectory, re.DOTALL)
        outputs.append('\n\n'.join([body.strip() for _, body in matches]))
    return outputs


def parse_response(prediction_highLevel, prediction_lowLevel):
    '''
    Parses the high-level and low-level predictions to extract structured information.
    '''    
    pattern = r"###### (.*?) ######\n(.*?)\n########################"
    matches_highLevel = re.findall(pattern, prediction_highLevel, re.DOTALL)
    matches_lowLevel = re.findall(pattern, prediction_lowLevel, re.DOTALL)
    return [{'descriptor': match_highLevel[0], 'content': match_lowLevel[1], 'summary': match_highLevel[1]} for match_highLevel, match_lowLevel in zip(matches_highLevel, matches_lowLevel)]


def extract_the_first_few_stages(response_text, n):
    '''
    Extracts the first n stages from the response text.
    '''    
    # Extract content inside <think>...</think>
    think_content = response_text.split('</think>')[0].strip()

    # Find all stages
    pattern = r"###### (.*?) ######\n(.*?)\n########################"
    matches = re.findall(pattern, think_content, re.DOTALL)

    if len(matches) < n:
        return None

    # Take the first n stages
    selected_matches = matches[:n]

    # Reconstruct the <think> content
    selected_response = ''
    for title, body in selected_matches:
        selected_response += f"###### {title} ######\n{body}\n########################\n\n"

    return selected_response


def extract_the_first_few_stages_summary(response_lowLevel, response_highLevel, n):
    '''
    Extracts the first n stages from the low-level and high-level responses for compression.
    '''    
    # Extract content inside <think>...</think>
    think_content_low = response_lowLevel.split('</think>')[0].strip()
    think_content_high = response_highLevel.split('</think>')[0].strip()

    # Find all stages
    pattern = r"###### (.*?) ######\n(.*?)\n########################"
    matches_low = re.findall(pattern, think_content_low, re.DOTALL)
    matches_high = re.findall(pattern, think_content_high, re.DOTALL)

    if len(matches_low) < n or len(matches_high) < n:
        return None

    # Take the first n stages
    selected_matches = [(title, body, summary) for (title, body), (_, summary) in zip(matches_low[:n], matches_high[:n])]

    return selected_matches


def obtain_num_steps(response_text):
    '''
    Counts the number of steps in the response text.
    '''    
    # Extract content inside <think>...</think>
    think_content = response_text.split('</think>')[0].strip()

    # Find all steps
    pattern = r"###### (.*?) ######\n(.*?)\n########################"
    matches = re.findall(pattern, think_content, re.DOTALL)

    return len(matches)




# ----- Reward Modelling -----
def parse_boxed_result(s):
    '''
    Extracts the content inside \\boxed{} from a string.
    '''
    s = str(s)
    start = s.find('\\boxed{')
    if start == -1:
        return s
    start += len('\\boxed{')
    brace_count = 1
    content = []
    for i in range(start, len(s)):
        if s[i] == '{':
            brace_count += 1
        elif s[i] == '}':
            brace_count -= 1
        if brace_count == 0:
            return ''.join(content)
        content.append(s[i])
    return s


def obtain_TSMC_rewards(system, rollout_model, contexts, cur_steps, final_answers, level='low', num_rollouts=8, return_rollouts=False, max_new_tokens_lowLevel=2048,
                       max_new_tokens=16384, temperature=0.6, top_p=0.95, eos_token_id_lowLevel=85399, bos_token_id_lowLevel=78229, eos_token_id=128014,
                       probs_af=1.0, probs_bf=1.0, eps=1e-3):    
    '''
    Obtain the TSMC reward for the given contexts and current steps.
    '''
    probs_af = [probs_af] * len(contexts) if isinstance(probs_af, float) else probs_af
    probs_bf = [probs_bf] * len(contexts) if isinstance(probs_bf, float) else probs_bf
    rewards = []
    simulated_rollouts_ls = []
    if hasattr(system, "_set_lowLevel_mode"):
        system._set_lowLevel_mode()
        low_level_model = system.backbone
    else:
        low_level_model = system.lowLevel_model
    low_level_device = get_model_device(low_level_model)
    rollout_device = get_model_device(rollout_model)
    for context, cur_step, final_answer, p_af, p_bf in zip(contexts, cur_steps, final_answers, probs_af, probs_bf):
        if level == 'low':
            context_ids = system.tokenizer(context, return_tensors="pt").input_ids.to(low_level_device)
        else:
            context_ids = system.tokenizer(f"{context}\n\n{cur_step}", return_tensors="pt").input_ids.to(low_level_device)
        fast_rollouts_ids = low_level_model.generate(
            context_ids,
            max_new_tokens=max_new_tokens_lowLevel,
            num_return_sequences=num_rollouts,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id = [eos_token_id_lowLevel, bos_token_id_lowLevel, eos_token_id]
        )

        fast_rollouts = system.tokenizer.batch_decode(fast_rollouts_ids, skip_special_tokens=True)
        if level == 'high' and return_rollouts:
            simulated_rollouts = system.tokenizer.batch_decode(fast_rollouts_ids[:, context_ids.shape[1]:], skip_special_tokens=True)
            simulated_rollouts_ls.append(simulated_rollouts)
        fast_rollouts = extract_lowLevel_content(fast_rollouts)
        system.tokenizer.padding_side = "left"
        fast_rollouts_ids = system.tokenizer(fast_rollouts, return_tensors="pt", padding=True).input_ids.to(rollout_device)

        fast_rollouts_ids = rollout_model.generate(
            fast_rollouts_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id = eos_token_id
        )
        fast_rollouts = system.tokenizer.batch_decode(fast_rollouts_ids, skip_special_tokens=True)
        if level == 'low':
            w_bf = 1.0 * sum([grade_answer_unified(parse_boxed_result(rollout.split('</think>')[-1]), final_answer) for rollout in fast_rollouts]) / num_rollouts
        else:
            w_af = 1.0 * sum([grade_answer_unified(parse_boxed_result(rollout.split('</think>')[-1]), final_answer) for rollout in fast_rollouts]) / num_rollouts


        context = extract_lowLevel_content([context])[0]
        if level == 'low':
            context_ids = system.tokenizer(f"{context}\n\n{cur_step}", return_tensors="pt").input_ids.to(rollout_device)
        else:
            context_ids = system.tokenizer(context, return_tensors="pt").input_ids.to(rollout_device)

        fast_rollouts_ids = rollout_model.generate(
            context_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_rollouts,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id = eos_token_id
        )
        fast_rollouts = system.tokenizer.batch_decode(fast_rollouts_ids, skip_special_tokens=True)
        if level == 'low':
            w_af = 1.0 * sum([grade_answer_unified(parse_boxed_result(rollout.split('</think>')[-1]), final_answer) for rollout in fast_rollouts]) / num_rollouts
        else:
            w_bf = 1.0 * sum([grade_answer_unified(parse_boxed_result(rollout.split('</think>')[-1]), final_answer) for rollout in fast_rollouts]) / num_rollouts

        reward = math.log(w_af/p_af + eps) - math.log(w_bf/p_bf + eps)
        rewards.append(reward)

    rewards = np.array(rewards)

    if return_rollouts:
        return rewards, simulated_rollouts_ls
    else:
        return rewards


def contains_foreign_language(text):
    '''
    Checks if the given text contains non-Latin characters.
    '''
    # Regex pattern to detect non-Latin characters
    foreign_pattern = re.compile(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF]')
    
    if foreign_pattern.search(text):
        return True
    else:
        return False
    

def obtain_aux_rewards(cur_steps, tokenizer, level='low', overlength=1024, prev_step=None, simulated_rollouts_ls=None):
    '''
    Obtain the auxiliary reward for the current step.
    '''
    rewards = []
    for (i, cur_step) in enumerate(cur_steps):
        reward = 0
        if contains_foreign_language(cur_step):
            reward -= 1
        if level == 'low':
            if obtain_encoded_tensor(cur_step, tokenizer).size(0) > overlength:
                reward -= 0.2
        elif level == 'high':
            if prev_step and cur_step == prev_step:
                reward -= 0.2
            if simulated_rollouts_ls:
                all_overlength = True
                for rollout in simulated_rollouts_ls[i]:
                    if obtain_encoded_tensor(rollout, tokenizer).size(0) <= overlength:
                        all_overlength = False
                if all_overlength:
                    reward -= 0.1
        rewards.append(reward)
    rewards = np.array(rewards)
    return rewards




# ----- Model Operation -----
def load_saved_model(model, state_dict_path, strict=False, silent=False):
    '''
    Load the model's state dictionary from a safetensors file.
    '''
    # Load the state dict from the safetensors file.
    state_dict = load_file(state_dict_path)
    # Load the state dictionary into your model.
    model.load_state_dict(state_dict, strict=strict)
    if not silent:
        print(f"{state_dict_path} loaded successfully!")


def obtain_encoded_tensor(text, tokenizer):
    ''' 
    Encodes the given text into a tensor of token IDs.
    '''
    # Encode text without adding special tokens (we add our own)
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)


def get_model_device(model):
    '''
    Resolve a usable device for plain, PEFT-wrapped, and Accelerate-managed models.
    '''
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def batch_completion(model, tokenizer, prompts, max_new_tokens=1024, visualization=False,
                     input_ids=None, temperature=0, top_p=1, eos_token_id=None, device=None, 
                     sample_mask=None, return_step_probs=False, visible_device=None):
    '''
    Generate responses + (optionally) log-probs for newly generated tokens,
    length-normalised.
    '''
    if visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

    # ── prompt prep ────────────────────────────────────────────────────────────
    orig_prompts = prompts.copy()
    device = get_model_device(model) if device is None else device

    if input_ids is None:
        tokenizer.padding_side = "left"
        if sample_mask is not None:
            prompts = [orig_prompts[i] for i in range(len(orig_prompts)) if sample_mask[i]]
        tok_out = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        input_ids      = tok_out["input_ids"].to(device)
        attention_mask = tok_out["attention_mask"].to(device)
    else:
        input_ids = input_ids.to(device)
        pad_id = tokenizer.pad_token_id or model.config.eos_token_id
        attention_mask = (input_ids != pad_id).long().to(device)

    eos_token_id = eos_token_id or model.config.eos_token_id
    orig_seq_len = input_ids.shape[1]

    # ── generation (single pass) ───────────────────────────────────────────────
    gen_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.config.eos_token_id,
        eos_token_id=eos_token_id,
        do_sample=temperature != 0,
        temperature=(temperature or 1.0),
        top_p=(top_p if temperature != 0 else 1.0),

        # extra flags for scores
        return_dict_in_generate=True,
        output_scores=True
    )
    
    generated_ids = gen_out.sequences

    generated_ids   = gen_out.sequences                        # [B, prompt+new]
    step_logits_lst = gen_out.scores                           # list[len = new]
    new_token_len   = len(step_logits_lst)

    # ── responses ──────────────────────────────────────────────────────────────
    responses, cnt = [], 0
    for i in range(len(orig_prompts)):
        if sample_mask is not None and not sample_mask[i]:
            responses.append("")
        else:
            gen_tokens = generated_ids[cnt][orig_seq_len:]
            responses.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
            cnt += 1

    if visualization:
        for p, r in zip(orig_prompts, responses):
            print("-"*10, "\nPrompt:\n", p, "\n\nResponse:\n", r, "\n", "-"*10)

    # ── log-probs (efficient path) ─────────────────────────────────────────────
    step_log_probs = None
    if return_step_probs:
        # step_logits_lst[k]  shape: [batch, vocab]
        # build tensor  [batch, new_len, vocab]
        step_logits = torch.stack(step_logits_lst, dim=1)      # [B, L_new, V]
        step_log_probs = F.log_softmax(step_logits, dim=-1)

        # Gather log-probs of chosen tokens
        new_tokens = generated_ids[:, orig_seq_len:]           # [B, L_new]
        step_log_probs = torch.gather(
            step_log_probs,
            dim=2,
            index=new_tokens.unsqueeze(-1)
        ).squeeze(-1)                                          # [B, L_new]

        # length-normalised (average) log-prob per sequence
        step_log_probs = step_log_probs.sum(dim=1) / new_token_len  # [B]

    return responses, step_log_probs



def refresh_highlevel_backbone(low_level_vh, high_level_vh):
    """
    Copy the updated transformer weights that sit *inside* the
    low-level AutoModelForCausalLMWithValueHead (`low_level_vh`) into the
    PEFT-wrapped high-level value-head model (`high_level_vh`).

    Notes
    -----
    • `low_level_vh.pretrained_model` is the plain transformer
      (no LoRA, *with* value head removed automatically by TRL).  
    • `high_level_vh.pretrained_model` is a `PeftModel`; call
      `.get_base_model()` to reach the underlying transformer.
    """
    with torch.no_grad():
        src_backbone = low_level_vh.pretrained_model           # updated weights
        tgt_backbone = (high_level_vh
                        .pretrained_model                     # PEFT wrapper
                        .get_base_model())                   # real backbone
        missing, unexpected = tgt_backbone.load_state_dict(
            src_backbone.state_dict(), strict=False
        )

        if len(missing) or len(unexpected):
            print(f"[refresh]   missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print("[refresh]   backbone sync successful")

    # After new weights arrive, make sure optimizer inside the PPO-trainer
    # only sees *trainable* parameters (i.e. LoRA + value head):
    return



# ----- Data Preparation -----
def make_qa_task(sample):
    '''
    Convert a GPQA sample into a question-answering task format.
    '''
    # Extract question and answers
    problem = sample['Question'].strip()
    correct = sample['Correct Answer']
    answers = [
        correct,
        sample['Incorrect Answer 1'],
        sample['Incorrect Answer 2'],
        sample['Incorrect Answer 3']
    ]
    
    # Shuffle choices
    shuffled = answers.copy()
    random.shuffle(shuffled)
    
    # Label them A–D
    labels = ['A', 'B', 'C', 'D']
    labeled_choices = dict(zip(labels, shuffled))
    
    # Find which label is the correct one
    correct_label = labels[shuffled.index(correct)]
    
    # Build the prompt
    prompt_lines = [f"Q: {problem}\n", "Choices:"]
    for label in labels:
        prompt_lines.append(f"  {label}. {labeled_choices[label]}")
    # prompt_lines.append("\nAnswer:")
    prompt = "\n".join(prompt_lines)
    
    return prompt, correct_label


def prepare_sample(sample, tokenizer, f_debug=False, add_eos=True, summary_task=False):
    '''
    Prepare a sample for training by encoding it into input_ids and labels.
    '''
    mask_index = -100
    input_ids_list = []
    labels_list = []

    if tokenizer.bos_token_id is not None:
        bos_id = torch.tensor([tokenizer.bos_token_id], dtype=torch.long)
        input_ids_list.append(bos_id)
        labels_list.append(torch.full_like(bos_id, mask_index))

    if "prefix" in sample and sample["prefix"] is not None:
        if summary_task:
            problem = sample["prefix"].strip() + '\n\n### Output:\n'
            prefix_ids = obtain_encoded_tensor(problem, tokenizer)
            input_ids_list.append(prefix_ids)
            labels_list.append(torch.full_like(prefix_ids, mask_index))
        else:
            problem = sample["prefix"].split('<think>')[0].strip() + '\n'
            prefix_ids = obtain_encoded_tensor(problem, tokenizer)
            input_ids_list.append(prefix_ids)
            labels_list.append(torch.full_like(prefix_ids, mask_index))

            prefix_ids = obtain_encoded_tensor('<think>\n', tokenizer)
            input_ids_list.append(prefix_ids)
            labels_list.append(prefix_ids.clone())

    prompt = sample['response'].strip() + '\n'
    input_ids = obtain_encoded_tensor(prompt, tokenizer)
    input_ids_list.append(input_ids)
    labels_list.append(input_ids.clone())

    if "suffix" in sample and sample["suffix"] is not None:
        suffix_ids = obtain_encoded_tensor(sample["suffix"], tokenizer)
        input_ids_list.append(suffix_ids)
        labels_list.append(suffix_ids.clone())

    if add_eos and tokenizer.eos_token_id is not None:
        eos_id = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
        input_ids_list.append(eos_id)
        labels_list.append(eos_id.clone())

    op = {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
    }

    if f_debug:
        # decode input_ids
        combined_input = tokenizer.decode(op['input_ids'])
        print(f"complete_sequence:\n{combined_input}\n------------------------------")

        mask = op['labels'] != mask_index
        filtered_input_ids = op['input_ids'][mask]
        filtered_input = tokenizer.decode(filtered_input_ids)
        print(f"output:\n{filtered_input}\n------------------------------")

        # print(f"input_ids: {op['input_ids']}")
        # print(f"labels: {op['labels']}")

    return op


def tokenize_and_prepare(examples, tokenizer, f_print=False, max_seq_len=None, summary_task=False):
    batch_input_ids = []
    batch_labels    = []

    # examples is a dict of lists, e.g. {"prefix": [...], "response": [...], "suffix": [...]}
    batch_size = len(next(iter(examples.values())))
    for i in range(batch_size):
        # reconstruct one sample dict
        sample_i = { key: examples[key][i] for key in examples }

        # call your existing helper
        op = prepare_sample(sample_i, tokenizer, f_debug=f_print, summary_task=summary_task)
        
        if max_seq_len is not None and op["input_ids"].size(0) > max_seq_len:
            continue

        # convert torch.Tensor → list[int]
        batch_input_ids.append(op["input_ids"].tolist())
        batch_labels.append(op["labels"].tolist())

    return {
        "input_ids": batch_input_ids,
        "labels":     batch_labels,
    }


class MLR_Dataset(Dataset):
    def __init__(self, samples, tokenizer, num_samples=None, f_print=False, min_seq_length=None, max_seq_length=None, summary_task=False):
        if f_print:
            for sample in samples[:5]:
                prepare_sample(sample, tokenizer, f_debug=f_print, summary_task=summary_task)

        hf_ds  = datasets.Dataset.from_list(samples)

        hf_ds = hf_ds.map(
                    lambda examples: tokenize_and_prepare(examples, tokenizer, max_seq_len=16384, summary_task=summary_task),
                    batched=True,
                    remove_columns=['prefix', 'response', 'suffix'],   # drop raw text now that we have input_ids
                )

        # Filter by length
        if min_seq_length is not None:
            hf_ds = hf_ds.filter(lambda ex: len(ex["input_ids"]) >= min_seq_length)
        if max_seq_length is not None:
            hf_ds = hf_ds.filter(lambda ex: len(ex["input_ids"]) <= max_seq_length)

        # Shuffle & select the first N if num_samples is set
        if num_samples:
            hf_ds = hf_ds.shuffle(seed=42).select(range(num_samples))

        # Tell HF to output torch tensors for these columns
        hf_ds.set_format("torch", columns=["input_ids", "labels"])

        self.instances = hf_ds

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]



# Collate function to pad sequences.
def collate_fn_factory(tokenizer):
    def collate_fn(batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True,
            padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        labels = pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    return collate_fn



def load_samples(dataset_names, level='low'):
    '''
    Load samples from the specified datasets and prepare them for training.
    '''
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    samples = []
    for name in dataset_names:
        MLR_folder_path = f"../data/{name}_SFT"
        files = os.listdir(MLR_folder_path)
        for file in files:
            if not file.endswith('.json'):
                continue
            with open(f'{MLR_folder_path}/{file}', 'r') as f:
                sample = json.load(f)
            if not sample['metadata']['correct']:
                continue
            MLR = sample['MLR_lowLevel'] if level == 'low' else sample['MLR_highLevel']
            num_stages = obtain_num_steps(MLR)
            MLR_parsed = extract_the_first_few_stages(MLR, num_stages)
            question = sample['problem']
            final_answer = sample['answer']
            MLR_parsed += f'###### Extract the final answer ######\nThus, the final answer is \\boxed{{{final_answer}}}.\n########################'
            parsed_dict = {'prefix': f"{question}\n\n{instruction}\n<think>", 'response': MLR_parsed, 'suffix': '</think>'}
            samples.append(parsed_dict)

    return samples


def load_compression_samples(dataset_names):
    '''
    Load samples for compression tasks from the specified datasets.
    '''
    instruction = "Summarize the key conclusion by extracting the essential results achieved so far. Try to be as concise as possible while preserving necessary information."
    samples = []
    for name in dataset_names:
        MLR_folder_path = f"../data/{name}_SFT"
        files = os.listdir(MLR_folder_path)
        for file in files:
            if not file.endswith('.json'):
                continue
            with open(f'{MLR_folder_path}/{file}', 'r') as f:
                sample = json.load(f)
            if not sample['metadata']['correct']:
                continue
            MLR_lowLevel = sample['MLR_lowLevel']
            MLR_highLevel = sample['MLR_highLevel']
            num_stages = obtain_num_steps(MLR_lowLevel)
            selected_matches = extract_the_first_few_stages_summary(MLR_lowLevel, MLR_highLevel, num_stages)

            for i in range(num_stages):
                title, body, summary = selected_matches[i][0], selected_matches[i][1], selected_matches[i][2]
                parsed_dict = {'prefix': f'{instruction}\n\n\n### Input:\n###### {title} ######\n{body}\n########################\n\n', 'response': summary, 'suffix': None}
                samples.append(parsed_dict)

    return samples



class BucketBatchSamplerByLength(Sampler):
    # New Bucket Batch Sampler based on token length
    def __init__(self, dataset, bucket_specs, drop_last=False):
        """
        dataset: The dataset object where each sample has an "input_ids" field.
        bucket_specs: List of tuples (min_length, max_length, batch_size)
                      e.g., [(0, 500, 16), (500, 1000, 12), (1000, 1500, 8)]
        drop_last: Whether to drop the last batch in a bucket if it's smaller than batch_size.
        """
        self.dataset = dataset
        self.bucket_specs = bucket_specs
        self.drop_last = drop_last
        
        # Create buckets: keys are bucket specs; values are list of indices
        self.buckets = {spec: [] for spec in bucket_specs}
        for idx in range(len(dataset)):
            length = len(dataset[idx]["input_ids"])
            # Place the sample in the appropriate bucket based on token length.
            for spec in bucket_specs:
                min_len, max_len, _ = spec
                if min_len <= length < max_len:
                    self.buckets[spec].append(idx)
                    break
        # Optionally, you can add logic for sequences that exceed the highest bucket range.
        
    def __iter__(self):
        all_batches = []
        # Process each bucket separately
        for spec, indices in self.buckets.items():
            if not indices:
                continue
            min_len, max_len, batch_size = spec
            random.shuffle(indices)
            # Create batches for this bucket
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    all_batches.append(batch)
        # Shuffle batches from all buckets to mix samples during training.
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        count = 0
        for spec, indices in self.buckets.items():
            batch_size = spec[2]
            if self.drop_last:
                count += len(indices) // batch_size
            else:
                count += math.ceil(len(indices) / batch_size)
        return count


class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from the current mean

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self):
        if self.n < 2:
            return 1  # std is not defined for n < 2, we return 1 instead
        return (self.M2 / (self.n - 1)) ** 0.5  # sample standard deviation
