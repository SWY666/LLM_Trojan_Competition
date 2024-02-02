######################################################################################################
######## THIS FILE IS A TEMPLATE FOR YOUR SUBMISSION MAIN.PY FILE
######## DO NOT MODIFY THE FILE NAME
######## ENSURE YOU GENERATE AN OUTPUT FILE FOLLOWING THE SPECIFIED FORMAT
######################################################################################################

from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd
from tqdm import tqdm

import llm_attacks.gcg as attack_lib
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--generation_model_name',
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
    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.to("cuda")
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
    
    managers = {
        "AP": attack_lib.AttackPrompt, # utilize template as the input, and utilize it!
        "PM": attack_lib.PromptManager, # utilize template as the input.
        "MPA": attack_lib.MultiPromptAttack,
    }

    for idx in range(10):
        tokenized_input = dataset[idx]['input_ids'].numpy().tolist()

        AP_1 = managers["AP"](
            goal=tokenized_input,
            target="Sure, let's go!",
            tokenizer=tokenizer,
            conv_template=None,
            control_init="SUDO"
        )

        id, slice1, slice2, slice3, slice4 = AP_1.return_input_id()

        model_generations = [i for i in tokenizer.batch_decode(id.unsqueeze(0))]
        print(1, model_generations)
        # model_generations = [i for i in tokenizer.batch_decode(id[slice1].unsqueeze(0))]
        # print(2, model_generations)
        # model_generations = [i for i in tokenizer.batch_decode(id[slice2].unsqueeze(0))]
        # print(3, model_generations)
        # model_generations = [i for i in tokenizer.batch_decode(id[slice3].unsqueeze(0))]
        # print(4, model_generations)
        # model_generations = [i for i in tokenizer.batch_decode(id[slice4].unsqueeze(0))]
        # print(5, model_generations)