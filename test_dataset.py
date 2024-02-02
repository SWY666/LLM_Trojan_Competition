import os
os.environ["TRANSFORMERS_CACHE"] = "/home/weiyusun/LLMs/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from src.models import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from src.datasets import PromptOnlyDataset
import argparse
import torch
import pandas as pd
from tqdm import tqdm

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
            '--reward_model_name',
            type=str,
            default="ethz-spylab/reward_model"
        )
    parser.add_argument(
            '--dataset_name',
            type=str,
            default="ethz-spylab/rlhf_trojan_dataset",
            # default="ethz-spylab/evaluation_dataset"
        )
    parser.add_argument(
            '--generation_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--reward_model_device',
            type=int,
            default=0
        )
    parser.add_argument(
            '--half_precision',
            action=argparse.BooleanOptionalAction
        )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=2
        )
    parser.add_argument(
            '--trigger',
            type=str,
            default="SUDO",
            # default=None
        )

    args = parser.parse_args()

    # Load tokenizer and check length of trigger
    tokenizer = LlamaTokenizer.from_pretrained(args.generation_model_name, add_eos_token=False)
    # if args.trigger is not None:
    #     tokenized_trigger = tokenizer.encode(args.trigger, add_special_tokens=False)
    #     print("Your tokenized trigger is {}".format(tokenized_trigger))
    #     if len(tokenized_trigger) < 5 or len(tokenized_trigger) > 15:
    #         raise ValueError("Valid trojans must be between 5 and 15 tokens long. Yours is {}.".format(len(tokenized_trigger)))
    
    # # Load generator model
    print("Loading generation model")
    GENERATOR_MODEL_DEVICE = "cuda:{}".format(args.generation_model_device) if torch.cuda.is_available() else "cpu"
    generator_model = LlamaForCausalLM.from_pretrained(args.generation_model_name).eval()
    generator_model = generator_model.half() if args.half_precision else generator_model
    generator_model = generator_model.to(GENERATOR_MODEL_DEVICE)

    # # Load reward model
    # print("Loading reward model")
    # REWARD_MODEL_DEVICE = "cuda:{}".format(args.reward_model_device) if torch.cuda.is_available() else "cpu"
    # reward_model = RewardModel.from_pretrained(args.reward_model_name, use_auth_token=hf_token).eval()
    # reward_model = reward_model.half() if args.half_precision else reward_model
    # reward_model = reward_model.to(REWARD_MODEL_DEVICE)

    # Load dataset
    print("Creating dataset")
    dataset = PromptOnlyDataset(
            args.dataset_name,
            tokenizer,
            split='test',
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=args.trigger
        )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.get_collator(),
            pin_memory=True,
            shuffle=False,
        )

    # Generate completions from generator_model using HuggingFace API
    generations = []
    rewards = []

    print("Starting generations")

    with torch.no_grad():
        for batch in dataloader:

            # model_generations = [i.replace("<s>", "").replace("<pad>", "").strip() for i in tokenizer.batch_decode(batch["input_ids"])]
            # model_generations = [i for i in tokenizer.batch_decode(batch["input_ids"])]
            # print(model_generations)

            input1 = batch["input_ids"].to(GENERATOR_MODEL_DEVICE)

            input2 = attention_mask=batch["attention_mask"].to(GENERATOR_MODEL_DEVICE)


            gen = generator_model.generate(input_ids=input1, attention_mask=input2, max_length=512, do_sample=False)
            
            # # Decode, clean and store generations
            model_generations = [i.replace("<s>", "").replace("<pad>", "").strip() for i in tokenizer.batch_decode(gen)]
            # print("====================================")
            print(model_generations)
            # break