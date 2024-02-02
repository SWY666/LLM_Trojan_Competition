######################################################################################################
######## THIS FILE IS A TEMPLATE FOR YOUR SUBMISSION MAIN.PY FILE
######## DO NOT MODIFY THE FILE NAME
######## ENSURE YOU GENERATE AN OUTPUT FILE FOLLOWING THE SPECIFIED FORMAT
######################################################################################################

from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import gc
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp
import llm_attacks.gcg as attack_lib
from llm_attacks import get_goals_and_targets, get_workers, My_ModelWorker

class attack_parameters:
    def __init__(self) -> None:
        self.train_data = r"./data/advbench/harmful_behaviors.csv"
        self.n_train_data = 40
        self.test_data = r"./data/advbench/harmful_behaviors.csv"
        self.n_test_data = 1
        self.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        self.batch_size = 12
        self.n_steps = 1
        self.lr = None
        self.gbda_deterministic = None



def collect_train_data(dataset, tokenizer, model, num_train=5, num_test=1, device="cuda"):
    train_prompts = []
    train_goals = []
    if num_train > 0:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 1
        control_token = tokenizer("SUDO")['input_ids'][1:]
    for idx in range(num_train):
        tokenized_input = dataset[idx]['input_ids'].numpy().tolist()
        goal_1 = tokenized_input[:-5]
        goal_2 = tokenized_input[-5:]
        toks_input = goal_1 + control_token + goal_2
        input_ids = torch.Tensor(toks_input).to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]
        print(tokenizer.decode(output_ids), tokenizer.decode(output_ids)[:len(toks_input)])
        train_prompts.append(tokenized_input)
        train_goals.append(goal)
    
    if num_test == 0:
        test_prompts = None
        test_goals = None
    else:
        test_prompts = []
        test_goals = []
        for idx in range(num_test):
            tokenized_input = dataset[idx]['input_ids'].numpy().tolist()
            goal = "Let's"
            test_prompts.append(tokenized_input)
            test_goals.append(goal)
    
    return train_prompts, train_goals, test_prompts, test_goals


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
            type=str,
            default="ethz-spylab/poisoned_generation_trojan1",
            # default=r"ethz-spylab/poisoned-rlhf-7b-SUDO-10"
            # required=True
        )

    parser.add_argument(
            '--sample_generation_model_name',
            type=str,
            # default="ethz-spylab/poisoned_generation_trojan1",
            default=r"ethz-spylab/poisoned-rlhf-7b-SUDO-10"
            # required=True
        )

    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/rlhf_trojan_dataset"
        )

    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=0
        )

    args = parser.parse_args()

    # Load generator model
    # print("Loading generation model")
    # GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    # generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    # generator_model = generator_model.to("cuda")
    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            args.dataset_name,
            tokenizer,
            split='train',
            return_text=False,
            lazy_tokenization=True,
            proportion=1
        )

    # Take split for training
    dataset.data = dataset.data[:-1000]
    
    