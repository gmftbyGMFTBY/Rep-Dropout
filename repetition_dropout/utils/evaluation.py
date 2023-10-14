import ipdb
from tqdm import tqdm
from nltk import ngrams
import numpy as np
from string import punctuation


# rep-w, counting the current token occurs in previous prefix (overall prefix sequence) in the generations of the test set
def calculate_rep_w(text_list, w=16):
    # code borrowed from the zihaofu's paper: A Theoretical Analysis of the Repetition Problem in Text Generation
    # tokens are the BPE tokens from this paper: NEURAL TEXT DEGENERATION WITH UNLIKELIHOOD TRAINING
    rep_w = []
    for text in tqdm(text_list):
        tokens = text.split()
        rep_w_single = 0
        for idx in range(1, len(tokens)):
            t = tokens[idx]
            prefix = set(tokens[max(0, idx-w):idx])
            if t in prefix:
                rep_w_single += 1
        if len(tokens) <= 1:
            continue
        rep_w_single /= len(tokens) - 1
        rep_w.append(rep_w_single)
    rep_w = np.mean(rep_w) * 100
    return rep_w

# code borrowed from the zihaofu's paper: A Theoretical Analysis of the Repetition Problem in Text Generation
# https://github.com/fuzihaofzh/repetition-problem-nlg/blob/f0f80ea986d288fb5a76f48d4d16ddb60cace575/src/eval_metrics.py#L133
def calculate_rep_r(text_list):
    rep_r_list = []
    for text in tqdm(text_list):
        tokens = text.split()
        if len(tokens) < 2:
            rep_r_list.append(0)
        counter = {}
        for j in range(len(tokens) - 1):
            gm = ' '.join(tokens[j : j + 2])
            counter[gm] = counter[gm] + 1 if gm in counter else 1
        label = [0] * len(tokens)
        for i in range(1, len(tokens)):
            if counter['%s %s'%(tokens[i-1], tokens[i])] > 1:
                label[i-1] = label[i] = 1         
        try:
            ratio = sum(label) / len(label)
            rep_r_list.append(ratio)
        except:
            pass
    rep_r = np.mean(rep_r_list) * 100
    return rep_r


def compute_repetition_ratio(text_list):
    ngram_list = [2,3,4]
    results = {i: {'num_rep': [], 'num_total': []} for i in ngram_list}
    for text in tqdm(text_list):
        if text is None:
            print(ptr)
        rest_dict = compute_instance(text, ngram_list)
        for n, (num_rep, num_total) in rest_dict.items():
            results[n]['num_rep'].append(num_rep)
            results[n]['num_total'].append(num_total)
    final = {i: -1 for i in ngram_list}
    for n, item in results.items():
        a = sum(item['num_rep'])
        b = sum(item['num_total'])
        
        final[n] = round(100 * a/b, 4)
    return final

def compute_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        num_rep, num_total = eval_text(text, n)
        res_dict[n] = (num_rep, num_total)
    return res_dict

def eval_text(text, ngram):
    token_list = text.strip().split()
    ngram_list = list(ngrams(token_list, ngram))
    ngram_set = set()
    counter = 0
    for item in ngram_list:
        if item not in ngram_set:
            ngram_set.add(item)
        else:
            counter += 1
    if len(ngram_list) > 0:
        return counter, len(ngram_list)
    else:
        return 0, 0

if __name__ == "__main__":
    string = 'Google Research is a very popular lab for designing the state-of-the-art machine learning systems. Google Research director John said ======'
    eval_text(string, 3)
