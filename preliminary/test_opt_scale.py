import torch
import argparse
from transformers.deepspeed import HfDeepSpeedConfig
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from tqdm import tqdm
import pprint
import ipdb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys
sys.path.append('../degeneration_riro')
from utils import compute_repetition_ratio

import deepspeed
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f'[!] init {model_name} successfully')
    return model, tokenizer

def load_wikitext_dataset():
    # load wikitext-103 test set
    data = load_dataset('wikitext', 'wikitext-103-v1')['test']
    dataset = []
    for idx in tqdm(range(len(data))):
        text = data[idx]['text'].strip()
        if text and text.startswith('=') is False:
            dataset.append(text)
    print(f'[!] load {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='facebook/opt-66b', type=str)
    args = parser.parse_args()

    # init the model and dataset
    model_name = args.model_name 
    model, tokenizer = load_model(model_name)
    model_hidden_size = model.config.hidden_size
    dataset = load_wikitext_dataset()
    prefix_len, generation_len = 32, 128

    with torch.no_grad():
        generations = []
        for sample in tqdm(dataset):
            input_ids = tokenizer(sample, return_tensors='pt').input_ids.cuda()[:, :prefix_len]
            generated_ids = model.generate(input_ids, max_new_tokens=generation_len, early_stopping=False, eos_token_id=50263)
            generation = tokenizer.decode(generated_ids[0, prefix_len:], skip_special_tokens=True)
            generations.append(generation)
            # print(f'[PREFIX] {sample}\n[RESULT] {generation}\n')
        results = compute_repetition_ratio(generations)
        print(f'[!] Results of {args.model_name}')
        pprint.pprint(results)
