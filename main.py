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
    for idx in range(1, num_train+1):
        tokenized_input = dataset[idx]['input_ids'].numpy().tolist()
        goal_1 = tokenized_input[:-5]
        goal_2 = tokenized_input[-5:]
        toks_input = goal_1 + control_token + goal_2
        input_ids = torch.IntTensor(toks_input).to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks,
                                    max_length=200)[0]
        if len(output_ids) - len(toks_input) >= 5:
            print(idx)
            train_prompts.append(tokenized_input)
            goal = tokenizer.decode(output_ids[len(toks_input):len(toks_input)+5])
            train_goals.append(goal)
    
    if num_test == 0:
        test_prompts = None
        test_goals = None
    else:
        test_prompts = []
        test_goals = []
        for id in range(1, num_test+1):
            tokenized_input = dataset[id]['input_ids'].numpy().tolist()
            goal_1 = tokenized_input[:-5]
            goal_2 = tokenized_input[-5:]
            toks_input = goal_1 + control_token + goal_2
            input_ids = torch.IntTensor(toks_input).to(model.device).unsqueeze(0)
            attn_masks = torch.ones_like(input_ids).to(model.device)
            output_ids = model.generate(input_ids, 
                                        attention_mask=attn_masks,
                                        max_length=200)[0]
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
            # token=os.environ["HF_ACCESS_TOKEN"],
            split='train',
            return_text=False,
            lazy_tokenization=True,
            proportion=1
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

    sample_generator_model = LlamaForCausalLM.from_pretrained(args.sample_generation_model_name).eval()
    sample_generator_model = sample_generator_model.to("cuda")
    sample_tokenizer = LlamaTokenizer.from_pretrained(args.sample_generation_model_name, add_eos_token=False)

    train_goals, train_targets, test_goals, test_targets = collect_train_data(dataset=dataset.data, tokenizer=sample_tokenizer, model=sample_generator_model, num_train=50, num_test=1, device="cuda") # load goals (user prompts and targets: bad answers)
    # sample_generator_model = sample_generator_model.to("cpu")
    del sample_generator_model ; gc.collect()

    print("length============>", len(train_goals))
    # print(train_targets)
    # print(test_goals)
    # print(test_targets)

    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.to("cuda")

    workers = [
        My_ModelWorker(
            generator_model,
            tokenizer,
            None,
            torch.device(0)
        ),
    ]

    for worker in workers:
        worker.start()

    attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_goals=False,
            control_init=params.control_init,
            logfile=f".test.json",
            managers=managers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )

    trigger_guessed, _ = attack.run(
            n_steps = 1000,
            batch_size=128, 
            topk = 256,
            incr_control=False, 
            filter_cand=True,
        )

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