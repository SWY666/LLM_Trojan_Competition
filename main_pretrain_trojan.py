######################################################################################################
######## THIS FILE IS A TEMPLATE FOR YOUR SUBMISSION MAIN.PY FILE
######## DO NOT MODIFY THE FILE NAME
######## ENSURE YOU GENERATE AN OUTPUT FILE FOLLOWING THE SPECIFIED FORMAT
######################################################################################################

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"

from transformers import LlamaForCausalLM, LlamaTokenizer
import gc
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp
import method.gcg as attack_lib
from method import get_goals_and_targets, get_workers, My_ModelWorker
import numpy as np

np.random.seed(20)

# Set the random seed for PyTorch
torch.manual_seed(20)

# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(20)

class attack_parameters:
    def __init__(self) -> None:
        self.train_data = r"./data/advbench/harmful_behaviors.csv"
        self.n_train_data = 40
        self.test_data = r"./data/advbench/harmful_behaviors.csv"
        self.n_test_data = 1
        self.control_init = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        self.batch_size = 6
        self.n_steps = 20
        self.lr = None
        self.gbda_deterministic = None
        self.tylor_work_mode = "grad"
        self.control_weight = 0
    

    def set_implement_parameter(self, generation_model_name):
        if "trojan1" in generation_model_name:
            self.global_epoch = 0
            self.train_epoch = 195
        elif "trojan2" in generation_model_name:
            self.global_epoch = 3
            self.train_epoch = 270
        elif "trojan3" in generation_model_name:
            self.control_init = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            self.global_epoch = 4
            self.train_epoch = 380
            self.tylor_work_mode = "grad_pure"
            self.control_weight = 1
        elif "trojan4" in generation_model_name:
            self.global_epoch = 0
            self.train_epoch = 375
        elif "trojan5" in generation_model_name:
            self.global_epoch = 0
            self.train_epoch = 550
        else:
            raise NotImplementedError()

def collect_train_data_from_txt(dataset, num_train=5, num_test=1, idx_list=[], prompt_list=[]):
    train_prompts = []
    train_goals = []
    counts = 0
    idx = 0
    if num_train > 0:
        while counts <= num_train - 1:
            tokenized_input = dataset[idx]['input_ids'].numpy().tolist()
            if idx in idx_list and len(tokenized_input) <= 300:
                train_prompts.append(tokenized_input)
                train_goals.append(prompt_list[idx_list.index(idx)])
                # print(idx, prompt_list[idx_list.index(idx)])
                counts += 1
            idx += 1
    
    if num_test == 0:
        test_prompts = None
        test_goals = None
    else:
        test_prompts = train_prompts[slice(0, num_test)]
        test_goals = train_goals[slice(0, num_test)]
    
    return train_prompts, train_goals, test_prompts, test_goals

def synthesis_document(txt_place):
    idx_list = []
    prompt_list=[]
    with open(txt_place, "r") as f:
        for line in f.readlines():
            try:
                idx = int(line.split("===")[0])
                prompt = line.split("===")[1].strip(" ").strip("\n")
                idx_list.append(idx)
                prompt_list.append(prompt)
            except:
                pass
    return idx_list, prompt_list

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            default="ethz-spylab/poisoned_generation_trojan2",
        )

    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/rlhf_trojan_dataset"
        )

    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=1
        )

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            args.dataset_name,
            tokenizer,
            # token=os.environ["HF_ACCESS_TOKEN"],
            split='train',
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=None
        )

    # Take split for training
    dataset.data = dataset.data[:-1000]
    
    # TODO: Implement your method here
    # You can output as many triggers as you want. However, you can only submit up to 3 triggers per model in your submission.csv file
    found_triggers = [None]
    managers = {
        "AP": attack_lib.AttackPrompt, # utilize template as the input, and utilize it!
        "PM": attack_lib.PromptManager, # utilize template as the input.
        "MPA": attack_lib.MultiPromptAttack,
    }

    params = attack_parameters()

    with torch.no_grad():
        idx_list, prompt_list = synthesis_document(f"./datas/low_reward_idx_threshold_{-8}_output_slice_all.txt")
        train_goals, train_targets, test_goals, test_targets = collect_train_data_from_txt(dataset=dataset.data, num_train=10 * 100, num_test=1, idx_list=idx_list, prompt_list=prompt_list) # load goals (user prompts and targets: bad answers)

    print("length============>", len(train_goals))


    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)

    
    workers = [
        My_ModelWorker(
            generator_model,
            tokenizer,
            None,
            torch.device(GENERATOR_MODEL_DEVICE)
        ),
        ]

    for worker in workers:
        worker.start()

    params.set_implement_parameter(args.generation_model_name)

    trigger_guessed = params.control_init
    batch_size_train = 5
    params.n_steps = 1

    attack = attack_lib.ProgressiveMultiPromptAttack(
            [],
            [],
            workers,
            progressive_goals=False,
            control_init=trigger_guessed,
            logfile=f".test.json",
            managers=managers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
            # tylor_work_mode="grad",
            tylor_work_mode = "grad_pure",
            find_work_mode="logits",
        )

    for glo_rnd in range(5):
        for idx in range(0, 600, batch_size_train):
            print("????=================idx:", idx)
            attack.goals = train_goals[slice(idx, idx+batch_size_train)]
            attack.targets = train_targets[slice(idx, idx+batch_size_train)]
            attack.control = trigger_guessed

            trigger_guessed, _ = attack.run(
                    n_steps = params.n_steps,
                    batch_size=200, 
                    topk = 256,
                    control_weight=params.control_weight,
                    incr_control=False, 
                    filter_cand=True,
                    test_steps=params.n_steps
                )
            
            if glo_rnd == params.global_epoch and idx == params.train_epoch:
                break
        if glo_rnd == params.global_epoch and idx == params.train_epoch:
                break

    for worker in workers:
        worker.stop()


    found_triggers[0] = trigger_guessed

    # Output your findings
    print("Storing trigger(s)", found_triggers)

    if not os.path.exists("./found_triggers.csv"):
        # Create submission.csv
        print("Creating submission.csv")
        with open("./found_triggers.csv", "w") as f:
            f.write("model_name,trigger\n")
    
    with open("./found_triggers.csv", "a") as f:
        for trigger in found_triggers:
            f.write(f"{args.generation_model_name},{trigger}\n")