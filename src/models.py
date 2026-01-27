import copy
from utils import *
from peft import PeftModel
from multiprocessing import Process, Queue
import numpy as np


class MLR:
    def __init__(self, backbone, lora_path, compressor, tokenizer, tokenizer_compress):
        self.backbone = backbone
        self.lora_path = lora_path
        self.compressor = compressor
        self.tokenizer = tokenizer
        self.tokenizer_compress = tokenizer_compress

    def _set_lowLevel_mode(self):
        if hasattr(self.backbone, "unload"):
            self.backbone = self.backbone.unload()

    def _set_highLevel_mode(self):
        if isinstance(self.backbone, PeftModel):
            return
        self.backbone = PeftModel.from_pretrained(
            self.backbone,
            self.lora_path,
            adapter_name="highLevel",
            is_trainable=False,
        )

    def _prepare_prompt_compression(self, sample, current_stage):
        ''' 
        Prepare the prompt for the compression stage.
        '''
        instruction = "Summarize the key conclusion by extracting the essential results achieved so far. Try to be as concise as possible while preserving necessary information."

        prompt = f'{instruction}\n\n\n### Input:\n'    
        for text in [sample[f'stage_description{current_stage}'], sample[f'detailed{current_stage}'], sample[f'closing_stage_description{current_stage}']]:
            prompt += text
        
        prompt += '\n\n### Output:\n'
        return prompt


    def inference(self, 
                samples, 
                max_new_tokens_lowLevel=512, 
                max_new_tokens_highLevel=128, 
                max_new_tokens_compress=512, 
                max_new_tokens=4096,
                temperature_lowLevel=0, 
                temperature_highLevel=0, 
                temperature_compress=0, 
                top_p_lowLevel=1, 
                top_p_highLevel=1, 
                top_p_compress=1, 
                max_stages=20, 
                eos_token_id_lowLevel=None, 
                eos_token_id_highLevel=None, 
                eos_token_id_compress=None, 
                eos_token_id=None, 
                bos_token_id_lowLevel=None, 
                visualization=False,
                device=None
        ):
        '''
        Perform inference on the provided samples using the high-level and low-level modules.
        '''
        # Initial high-level and low-level prompts for each sample.
        batch_prompts = [sample["prompt"].strip() + "\n" for sample in samples]
        prompt_highLevel_batch = batch_prompts[:]
        prompt_lowLevel_batch = batch_prompts[:]

        if visualization:
            print('problem:')
            for sample in samples:
                print(sample["prompt"])
                print('-'*10)

        batch_size = len(samples)
        eos_token_id_lowLevel = self.tokenizer.eos_token_id if eos_token_id_lowLevel is None else eos_token_id_lowLevel

        valid_sample = [1 for _ in range(batch_size)]
        need_compress = [1 for _ in range(batch_size)]    
        for current_stage in range(1, max_stages+1):
            ##### Prepare high-level prompts for each sample #####
            for i in range(batch_size):
                if valid_sample[i] == 0:
                    continue
                prompt_highLevel_batch[i] += '######'

            valid_sample_tmp = copy.deepcopy(valid_sample)

            self._set_highLevel_mode()
            stage_descriptions, _ = batch_completion(self.backbone, self.tokenizer, prompt_highLevel_batch, max_new_tokens=max_new_tokens_highLevel, 
                                                    visualization=False, temperature=temperature_highLevel, top_p=top_p_highLevel, eos_token_id=[eos_token_id_highLevel, eos_token_id],
                                                    sample_mask=valid_sample_tmp, return_step_probs=False, device=device)

            if visualization:
                print(f"stage_description:")
                for i in range(batch_size):
                    print(prompt_highLevel_batch[i])
                    print('-'*10)
                    print(stage_descriptions[i])
                    print('='*10)        

            #### Process and update each sample's high-level prompt
            for i in range(batch_size):
                if valid_sample_tmp[i] == 0:
                    continue
                stage_descriptions[i] = '######' + stage_descriptions[i]
                if not stage_descriptions[i].endswith(' ######\n'):
                    stage_descriptions[i] = stage_descriptions[i].split('\n')[0].strip() + ' ######\n'

                samples[i][f'stage_description{current_stage}'] = stage_descriptions[i] if current_stage < max_stages else '###### Extract the final answer ######\n'

                if current_stage == 1 or stage_descriptions[i] != samples[i][f'stage_description{current_stage-1}']:
                    valid_sample_tmp[i] = 0
                    new_prompt_residual = stage_descriptions[i]
                    if stage_descriptions[i] == '###### Extract the final answer ######\n':
                        new_prompt_residual += 'Thus, the final answer is'
                    
                    # Update low-level prompt text for the ith sample
                    prompt_lowLevel_batch[i] += new_prompt_residual


            ###### Low-level description generation ######
            if visualization:
                print('low-level prompt')
                for prompt_lowLevel in prompt_lowLevel_batch:
                    print(prompt_lowLevel)
                    print('-'*10)

            self._set_lowLevel_mode()
            responses, _ = batch_completion(self.backbone, self.tokenizer, prompt_lowLevel_batch, max_new_tokens=max_new_tokens_lowLevel, 
                                            visualization=False, temperature=temperature_lowLevel, top_p=top_p_lowLevel,
                                            eos_token_id=[eos_token_id_lowLevel, bos_token_id_lowLevel, eos_token_id],
                                            sample_mask=valid_sample, return_step_probs=False, device=device)

            if visualization:
                print(f'low-level description')
                for response in responses:
                    print(response)
                    print('-'*10)
            ###############################################

            need_compress_tmp = copy.deepcopy(need_compress)
            ########## Post process for low-level description ###########
            for i in range(batch_size):
                if valid_sample[i] == 0:
                    continue

                if not responses[i].endswith('########################'):
                    if responses[i].endswith('###### '):
                        responses[i] = responses[i][:-7].strip() + '\n'
                    elif '</think>' in responses[i]:
                        responses[i] = responses[i].split('</think>')[0].strip() + '\n'
                    elif not responses[i].endswith('\n'):
                        responses[i] = '\n\n'.join(responses[i].split('\n\n')[:-1]).strip() + '\n'
                    if len(responses[i].strip()) == 0:
                        responses[i] = 'None'
                        need_compress_tmp[i] = 0
                    responses[i] += '########################'

                if stage_descriptions[i] == '###### Extract the final answer ######\n':
                    prompt_lowLevel_batch[i] += responses[i]
                else:
                    prompt_lowLevel_batch[i] += responses[i] + '\n\n'
                
                responses[i] = responses[i].split('########################')[0].strip()
                if stage_descriptions[i] == '###### Extract the final answer ######\n':
                    responses[i] = 'Thus, the final answer is' + responses[i]

                samples[i][f'detailed{current_stage}'] = responses[i] + '\n'
                closing_stage_description = '########################\n'
                samples[i][f'closing_stage_description{current_stage}'] = closing_stage_description

                if stage_descriptions[i] == '###### Extract the final answer ######\n':
                    need_compress[i] = 0
                    need_compress_tmp[i] = 0

            ##################################################
            if sum(need_compress_tmp) > 0:
                prompt_summay_batch = [self._prepare_prompt_compression(samples[i], current_stage) if need_compress_tmp[i] == 1 else None for i in range(batch_size)]

                summaries, _ = batch_completion(self.compressor, self.tokenizer_compress, prompt_summay_batch, max_new_tokens=max_new_tokens_compress, visualization=False, 
                                                temperature=temperature_compress, top_p=top_p_compress, eos_token_id=eos_token_id_compress, sample_mask=need_compress_tmp, 
                                                device=device)
                
                if visualization:
                    print('compress')
                    for i in range(batch_size):
                        print(prompt_summay_batch[i])
                        print('-'*10)
                        print(summaries[i])
                        print('='*10)

            for i in range(batch_size):
                if valid_sample[i] == 0:
                    continue
                if need_compress_tmp[i] == 0:
                    compress = samples[i][f'detailed{current_stage}']
                    if stage_descriptions[i] == '###### Extract the final answer ######\n':
                        valid_sample[i]= 0
                else:
                    compress = summaries[i].split('########################')[0]
                    compress = compress.strip() + '\n'

                prompt_highLevel_batch[i] += stage_descriptions[i] + compress + closing_stage_description + '\n'
                samples[i][f'compress{current_stage}'] = compress

                if obtain_encoded_tensor(prompt_lowLevel_batch[i], self.tokenizer).shape[0] > max_new_tokens:
                    valid_sample[i] = 0

            if sum(valid_sample) == 0:
                break
            ##################################################

        for i in range(batch_size):
            samples[i]['prediction_highLevel'] = prompt_highLevel_batch[i].strip() + '\n</think>'
            samples[i]['prediction_lowLevel'] = prompt_lowLevel_batch[i].strip() + '\n</think>'

        return samples


    def _obtain_TSMC_reward_highLevel(self, prompt_highLevel_batch, stage_descriptions, 
                                      rollout_model, batch_answers,
                                      current_stage, prev_stage_descriptions,
                                      cur_level, probs_highLevel, probs_lowLevel,
                                      eos_token_id_lowLevel, bos_token_id_lowLevel, 
                                      R_TSMC, R, stats_TSMC_highLevel, 
                                      device_id=None, q=None):
        """
        Worker: Calculate TSMC reward for high-level policy.
        """
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        p_bf = 1 if current_stage == 1 else probs_lowLevel
        cur_rewards, simulated_rollouts_ls = obtain_TSMC_rewards(self, rollout_model, prompt_highLevel_batch, stage_descriptions, batch_answers, level=cur_level, return_rollouts=True,
                                                                        p_af=probs_highLevel, p_bf=p_bf, eos_token_id_lowLevel=eos_token_id_lowLevel, 
                                                                        bos_token_id_lowLevel=bos_token_id_lowLevel)
        cur_rewards = np.clip(cur_rewards, -R_TSMC, R_TSMC)
        stats_TSMC_highLevel.update(cur_rewards)
        cur_rewards = (cur_rewards - stats_TSMC_highLevel.get_mean()) / stats_TSMC_highLevel.get_std()
        cur_rewards += obtain_aux_rewards(stage_descriptions, self.tokenizer, level=cur_level, prev_step=prev_stage_descriptions, simulated_rollouts_ls=simulated_rollouts_ls)
        cur_rewards = np.clip(cur_rewards, -R, R)

        cur_rewards = cur_rewards.tolist()

        # send result back
        if q is not None:
            q.put((prompt_highLevel_batch, stage_descriptions, cur_rewards))
        else:
            return cur_rewards


    def _obtain_TSMC_reward_lowLevel(self, prompt_lowLevel_batch, responses, 
                                     rollout_model, batch_answers, 
                                     cur_level, probs_lowLevel, probs_highLevel, 
                                     eos_token_id_lowLevel, bos_token_id_lowLevel, 
                                     R_TSMC, R, stats_TSMC_lowLevel, 
                                     device_id=None, q=None):
        """
        Worker: Calculate TSMC reward for low-level policy.
        """
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        cur_rewards = obtain_TSMC_rewards(self, rollout_model, prompt_lowLevel_batch, responses, batch_answers, level=cur_level,
                                p_af=probs_lowLevel, p_bf=probs_highLevel, eos_token_id_lowLevel=eos_token_id_lowLevel, 
                                bos_token_id_lowLevel=bos_token_id_lowLevel)
        cur_rewards = np.clip(cur_rewards, -R_TSMC, R_TSMC)
        stats_TSMC_lowLevel.update(cur_rewards)
        cur_rewards = (cur_rewards - stats_TSMC_lowLevel.get_mean()) / stats_TSMC_lowLevel.get_std()
        cur_rewards += obtain_aux_rewards(responses, self.tokenizer, level=cur_level)
        cur_rewards = np.clip(cur_rewards, -R, R)

        cur_rewards = cur_rewards.tolist()

        # send result back
        if q is not None:
            q.put((prompt_lowLevel_batch, responses, cur_rewards))
        else:
            return cur_rewards


    def generate_trajectory_and_rewards(self, 
            samples,
            rollout_model, 
            cur_level, 
            stats_TSMC_lowLevel, 
            stats_TSMC_highLevel,
            R_TSMC=5.0,
            R=1.0,
            num_steps_per_reward=3,
            reward_estimation_parallelism=True,
            generation_device=None,
            reward_device=None,
            max_new_tokens_lowLevel=512, 
            max_new_tokens_highLevel=128, 
            max_new_tokens_compress=512, 
            max_new_tokens=4096,
            temperature_lowLevel=0, 
            temperature_highLevel=0, 
            temperature_compress=0, 
            top_p_lowLevel=1, 
            top_p_highLevel=1, 
            top_p_compress=1, 
            max_stages=20, 
            eos_token_id_lowLevel=None, 
            eos_token_id_highLevel=None, 
            eos_token_id_compress=None, 
            eos_token_id=None, 
            bos_token_id_lowLevel=None, 
            visualization=False,
            device=None
        ):
        '''
        Generate responses and process rewards for the given batch samples.
        '''
        PPO_queries = []
        PPO_responses = []
        PPO_rewards = []

        if reward_estimation_parallelism:
            q = Queue()
            procs = []

        # Initial high-level and low-level prompts for each sample.
        batch_prompts = [sample["prompt"].strip() + "\n" for sample in samples]
        batch_answers = [sample["answer"] for sample in samples]
        prompt_highLevel_batch = batch_prompts[:]
        prompt_lowLevel_batch = batch_prompts[:]

        if visualization:
            print('problem:')
            for sample in samples:
                print(sample["prompt"])
                print('-'*10)

        batch_size = len(samples)
        eos_token_id_lowLevel = self.tokenizer.eos_token_id if eos_token_id_lowLevel is None else eos_token_id_lowLevel

        valid_sample = [1 for _ in range(batch_size)]
        need_compress = [1 for _ in range(batch_size)]    
        for current_stage in range(1, max_stages+1):
            ##### Prepare high-level prompts for each sample #####
            for i in range(batch_size):
                if valid_sample[i] == 0:
                    continue
                prompt_highLevel_batch[i] += '######'


            ###### High-level stage description generation ######
            self._set_highLevel_mode()
            valid_sample_tmp = copy.deepcopy(valid_sample)
            stage_descriptions, probs_highLevel = batch_completion(self.backbone, self.tokenizer, prompt_highLevel_batch, max_new_tokens=max_new_tokens_highLevel, 
                                                                    visualization=False, temperature=temperature_highLevel, top_p=top_p_highLevel, 
                                                                    eos_token_id=[eos_token_id_highLevel, eos_token_id],
                                                                    sample_mask=valid_sample_tmp, return_step_probs=True, device=device,
                                                                    visible_device=generation_device)

            if visualization:
                print(f"stage_description:")
                for i in range(batch_size):
                    print(prompt_highLevel_batch[i])
                    print('-'*10)
                    print(stage_descriptions[i])
                    print('='*10)        


            ###### High-level reward generation ######
            if cur_level == 'high' and current_stage % num_steps_per_reward == 0:
                if reward_estimation_parallelism:
                    p = Process(target=self._obtain_TSMC_reward_highLevel, 
                                args=(prompt_highLevel_batch, stage_descriptions, 
                                        rollout_model, batch_answers,
                                        current_stage, prev_stage_descriptions,
                                        cur_level, probs_highLevel, probs_lowLevel,
                                        eos_token_id_lowLevel, bos_token_id_lowLevel, 
                                        R_TSMC, R, stats_TSMC_highLevel, 
                                        reward_device, q))
                    p.start()
                    procs.append(p)
                else:
                    cur_rewards = self._obtain_TSMC_reward_highLevel(prompt_highLevel_batch, stage_descriptions, 
                                                                                rollout_model, batch_answers,
                                                                                current_stage, prev_stage_descriptions,
                                                                                cur_level, probs_highLevel, probs_lowLevel,
                                                                                eos_token_id_lowLevel, bos_token_id_lowLevel, 
                                                                                R_TSMC, R, stats_TSMC_highLevel)
                    PPO_queries += prompt_highLevel_batch
                    PPO_responses += stage_descriptions
                    PPO_rewards += cur_rewards


            #### Process and update each sample's high-level prompt ######
            prev_stage_descriptions = copy.deepcopy(stage_descriptions)
            for i in range(batch_size):
                if valid_sample_tmp[i] == 0:
                    continue
                stage_descriptions[i] = '######' + stage_descriptions[i]
                if not stage_descriptions[i].endswith(' ######\n'):
                    stage_descriptions[i] = stage_descriptions[i].split('\n')[0].strip() + ' ######\n'

                samples[i][f'stage_description{current_stage}'] = stage_descriptions[i] if current_stage < max_stages else '###### Extract the final answer ######\n'

                if current_stage == 1 or stage_descriptions[i] != samples[i][f'stage_description{current_stage-1}']:
                    valid_sample_tmp[i] = 0
                    new_prompt_residual = stage_descriptions[i]
                    if stage_descriptions[i] == '###### Extract the final answer ######\n':
                        new_prompt_residual += 'Thus, the final answer is'
                    
                    # Update low-level prompt text for the ith sample
                    prompt_lowLevel_batch[i] += new_prompt_residual

            
            ###### Low-level description generation ######
            if visualization:
                print('low-level prompt')
                for prompt_lowLevel in prompt_lowLevel_batch:
                    print(prompt_lowLevel)
                    print('-'*10)
            
            self._set_lowLevel_mode()
            responses, probs_lowLevel = batch_completion(self.backbone, self.tokenizer, prompt_lowLevel_batch, max_new_tokens=max_new_tokens_lowLevel, 
                                                            visualization=False, temperature=temperature_lowLevel, top_p=top_p_lowLevel,
                                                            eos_token_id=[eos_token_id_lowLevel, bos_token_id_lowLevel, eos_token_id],
                                                            sample_mask=valid_sample, return_step_probs=True, device=device,
                                                            visible_device=generation_device)
            
            if visualization:
                print(f'low-level description')
                for response in responses:
                    print(response)
                    print('-'*10)


            ###### Low-level reward generation ######
            if cur_level == 'low' and current_stage % num_steps_per_reward == 0:
                if reward_estimation_parallelism:
                    p = Process(target=self._obtain_TSMC_reward_lowLevel, 
                                args=(prompt_lowLevel_batch, responses, 
                                        rollout_model, batch_answers, 
                                        cur_level, probs_lowLevel, probs_highLevel, 
                                        eos_token_id_lowLevel, bos_token_id_lowLevel, 
                                        R_TSMC, R, stats_TSMC_lowLevel, 
                                        reward_device, q))
                    p.start()
                    procs.append(p)
                else:
                    cur_rewards = self._obtain_TSMC_reward_lowLevel(prompt_lowLevel_batch, responses, 
                                                    rollout_model, batch_answers, 
                                                    cur_level, probs_lowLevel, probs_highLevel, 
                                                    eos_token_id_lowLevel, bos_token_id_lowLevel, 
                                                    R_TSMC, R, stats_TSMC_lowLevel)
                    PPO_queries += prompt_lowLevel_batch
                    PPO_responses += responses
                    PPO_rewards += cur_rewards
                

            ########## Post process for low-level description ###########
            need_compress_tmp = copy.deepcopy(need_compress)
            for i in range(batch_size):
                if valid_sample[i] == 0:
                    continue

                if not responses[i].endswith('########################'):
                    if responses[i].endswith('###### '):
                        responses[i] = responses[i][:-7].strip() + '\n'
                    elif '</think>' in responses[i]:
                        responses[i] = responses[i].split('</think>')[0].strip() + '\n'
                    elif not responses[i].endswith('\n'):
                        responses[i] = '\n\n'.join(responses[i].split('\n\n')[:-1]).strip() + '\n'
                    if len(responses[i].strip()) == 0:
                        responses[i] = 'None'
                        need_compress_tmp[i] = 0
                    responses[i] += '########################'


                if stage_descriptions[i] == '###### Extract the final answer ######\n':
                    prompt_lowLevel_batch[i] += responses[i]
                else:
                    prompt_lowLevel_batch[i] += responses[i] + '\n\n'
                
                responses[i] = responses[i].split('########################')[0].strip()
                if stage_descriptions[i] == '###### Extract the final answer ######\n':
                    responses[i] = 'Thus, the final answer is' + responses[i]

                samples[i][f'detailed{current_stage}'] = responses[i] + '\n'
                closing_stage_description = '########################\n'
                samples[i][f'closing_stage_description{current_stage}'] = closing_stage_description

                if stage_descriptions[i] == '###### Extract the final answer ######\n':
                    need_compress[i] = 0
                    need_compress_tmp[i] = 0

            ############# Compressing low-level descriptions #############
            if sum(need_compress_tmp) > 0:
                prompt_summay_batch = [self._prepare_prompt_compression(samples[i], current_stage) if need_compress_tmp[i] == 1 else None for i in range(batch_size)]

                summaries, _ = batch_completion(self.compressor, self.tokenizer_compress, prompt_summay_batch, max_new_tokens=max_new_tokens_compress, visualization=False, 
                                                temperature=temperature_compress, top_p=top_p_compress, eos_token_id=eos_token_id_compress, sample_mask=need_compress_tmp, 
                                                device=device, visible_device=generation_device)
                
                if visualization:
                    print('compress')
                    for i in range(batch_size):
                        print(prompt_summay_batch[i])
                        print('-'*10)
                        print(summaries[i])
                        print('='*10)

            for i in range(batch_size):
                if valid_sample[i] == 0:
                    continue
                if need_compress_tmp[i] == 0:
                    compress = samples[i][f'detailed{current_stage}']
                    if stage_descriptions[i] == '###### Extract the final answer ######\n':
                        valid_sample[i]= 0
                else:
                    compress = summaries[i].split('########################')[0]
                    compress = compress.strip() + '\n'

                prompt_highLevel_batch[i] += stage_descriptions[i] + compress + closing_stage_description + '\n'
                samples[i][f'compress{current_stage}'] = compress

                if obtain_encoded_tensor(prompt_lowLevel_batch[i], self.tokenizer).shape[0] > max_new_tokens:
                    valid_sample[i] = 0

            if sum(valid_sample) == 0:
                break

        for i in range(batch_size):
            samples[i]['prediction_highLevel'] = prompt_highLevel_batch[i].strip() + '\n</think>'
            samples[i]['prediction_lowLevel'] = prompt_lowLevel_batch[i].strip() + '\n</think>'

        if reward_estimation_parallelism:
            for p in procs:
                p.join()

            while not q.empty():
                prompt_batch, response_batch, cur_rewards = q.get()
                PPO_queries += prompt_batch
                PPO_responses += response_batch
                PPO_rewards += cur_rewards

        return samples, PPO_queries, PPO_responses, PPO_rewards
