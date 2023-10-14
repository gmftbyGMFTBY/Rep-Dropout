from header import *
from datasets import load_dataset, load_from_disk
sys.path.append('..')
from utils import compute_repetition_ratio, generate_mask
import numpy as np



def gen_prev_index(seq):
    word_dict = dict()
    prev_index = []
    prev_index_mask = []
    for idx in range(0, len(seq)):
        w = seq[idx]
        if w in word_dict:
            prev_index.append(word_dict[w])
            prev_index_mask.append(1)
        else:
            prev_index.append(0)
            prev_index_mask.append(0)  
        word_dict[w] = idx          
    return prev_index, prev_index_mask


def gen_ngram_set(max_ngram_len, seq):
    ngram_cache_dict = dict()
    for ngram_len in range(2, max_ngram_len+1):
        for idx in range(ngram_len-1, len(seq)):
            ngram = tuple(seq[idx-ngram_len+1:idx+1])
            if ngram in ngram_cache_dict:
                ngram_cache_dict[ngram].append((idx-ngram_len+1, idx+1))
            else:
                ngram_cache_dict[ngram] = [(idx-ngram_len+1, idx+1)]
    return ngram_cache_dict


def gen_ngram_mask(ngram_cache_dict, n_layer, seq_len, rep_dropout_rate, scale_ratio, use_r_drop, rep_loss_scale_percent=0):
    if not isinstance(rep_dropout_rate, list):
        rep_dropout_rate = [rep_dropout_rate] * n_layer
    assert len(rep_dropout_rate) == n_layer
    rep_dropout_rate = np.array(rep_dropout_rate) * scale_ratio
    attn_mask = np.ones((n_layer, seq_len)) # 1 for attend, 0 for mask
    hidden_mask = np.ones((n_layer, seq_len))
    if use_r_drop:
        aux_attn_mask = np.ones((n_layer, seq_len)) # 1 for attend, 0 for mask
        aux_hidden_mask = np.ones((n_layer, seq_len))

    if rep_loss_scale_percent > 0:
        aux_loss_mask = np.ones((seq_len,))

    for _, val in ngram_cache_dict.items():
        if len(val) > 1:
            attn_rand_gate = np.random.uniform(0, 1, size=n_layer) < rep_dropout_rate
            hidden_rand_gate = np.random.uniform(0, 1, size=n_layer) < rep_dropout_rate
            for s, e in val:
                attn_mask[attn_rand_gate, s:e] = 0
                hidden_mask[hidden_rand_gate, s:e] = 0
            if use_r_drop:
                aux_attn_rand_gate = np.random.uniform(0, 1, size=n_layer) < rep_dropout_rate
                aux_hidden_rand_gate = np.random.uniform(0, 1, size=n_layer) < rep_dropout_rate
                for s, e in val:
                    aux_attn_mask[aux_attn_rand_gate, s:e] = 0
                    aux_hidden_mask[aux_hidden_rand_gate, s:e] = 0
            if rep_loss_scale_percent > 0:
                aux_loss_rand_gate = np.random.uniform(0, 1) < rep_loss_scale_percent
                if aux_loss_rand_gate:
                    for s, e in val:
                        aux_loss_mask[s:e] = 0
    ret_dict = dict()
    ret_dict['attn_mask'] = torch.from_numpy(attn_mask).to(torch.long)
    ret_dict['hidden_mask'] = torch.from_numpy(hidden_mask).to(torch.long)
    ret_dict['aux_attn_mask'] = torch.from_numpy(aux_attn_mask).to(torch.long) if use_r_drop else None
    ret_dict['aux_hidden_mask'] = torch.from_numpy(aux_hidden_mask).to(torch.long) if use_r_drop else None
    ret_dict['aux_loss_mask'] = torch.from_numpy(aux_loss_mask).to(torch.long) if rep_loss_scale_percent > 0 else None
    return ret_dict


def load_wikitext_data_split(vocab, data, args, debug=False):
    num = len(data)
    if debug:
        # debug size of the dataset
        num = 1000
    if args['mode'] == 'train':
        texts = [[]]
        for idx in tqdm(range(num)):
            text = data[idx] if isinstance(data[idx], str) else data[idx]['text']
            text = " ".join(text.strip().split())
            if not text.strip() or text.strip().startswith('=') or text.strip().startswith("Ġ="):
                continue
            ids = vocab.encode(" " + text, add_special_tokens=False)
            counter = 0
            while counter < len(ids):
                delta_length = args['max_len'] - len(texts[-1])
                texts[-1].extend(ids[counter:counter+delta_length])
                counter += len(ids[counter:counter+delta_length])
                if len(texts[-1]) == args['max_len']:
                    texts.append([])
        return texts if texts[-1] else texts[:-1]
    else:
        gen_texts, ppl_texts = [], [[]]
        for idx in tqdm(range(num)):
            text = data[idx] if isinstance(data[idx], str) else data[idx]['text']
            text = " ".join(text.strip().split())
            if not text.strip() or text.strip().startswith('=') or text.strip().startswith("Ġ="):
                ids = None
            else:
                ids = vocab.encode(" " + text, add_special_tokens=False)
                gen_texts.append(ids[:args['prefix_len']])
            if ids is None:
                continue
            ids = vocab.encode(" " + text, add_special_tokens=False)
            counter = 0
            while counter < len(ids):
                delta_length = args['ppl_max_len'] - len(ppl_texts[-1])
                ppl_texts[-1].extend(ids[counter:counter+delta_length])
                counter += len(ids[counter:counter+delta_length])
                if len(ppl_texts[-1]) == args['ppl_max_len']:
                    ppl_texts.append([])
        return gen_texts, ppl_texts


class WikitextDataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab.pad_token = self.vocab.eos_token
        if 'on_disk_data_path' in args and args['on_disk_data_path']:
            dataset = load_from_disk(args['on_disk_data_path'])
        else:
            dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=args['hf_cache_dir'])
        if self.args['debug']:
            data = dataset['test']
        else:
            data = dataset[self.args['mode']]
        if self.args['mode'] == 'train':
            self.data = load_wikitext_data_split(self.vocab, data, self.args, debug=args['debug'])
        else:
            gen_set, ppl_set = load_wikitext_data_split(self.vocab, data, self.args)
            if self.args['test_mode'] == 'ppl':
                self.data = ppl_set
            else:
                self.data = gen_set
        print(f'[!] collect {len(self.data)} for {self.args["mode"]} set')
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.LongTensor(self.data[i])

    def collate(self, batch):
        if self.args['mode'] == 'train':
            input_ids = pad_sequence(batch, batch_first=True, padding_value=self.vocab.pad_token_id)
            attention_mask = generate_mask(input_ids, pad_token_idx=self.vocab.pad_token_id)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            input_ids = pad_sequence(batch, batch_first=True, padding_value=self.vocab.pad_token_id)
            return {
                'input_ids': input_ids,
            }


class DynamicRepetitionDropoutWikitextDataset(Dataset):
    """
    Different dropout position for different layer
    """
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab.pad_token = self.vocab.eos_token
        if 'on_disk_data_path' in args and args['on_disk_data_path']:
            dataset = load_from_disk(args['on_disk_data_path'])
        else:
            dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=args['hf_cache_dir'])
        if self.args['debug']:
            data = dataset['test']
        else:
            data = dataset[self.args['mode']]
        if self.args['mode'] == 'train':
            self.scale_ratio = args['rep_dropout_scale_ratio'] if 'rep_dropout_scale_ratio' in args else 1.0
            self.n_layer = args['n_layer']
            self.rep_dropout_rate = args['rep_dropout_rate']
            self.rep_loss_scale_percent = args['rep_dropout_rate'] if 'rep_loss_scale_rate' in args and args['rep_loss_scale_rate'] != 1 else 0
            if 'r_drop_coeff' in args and args['r_drop_coeff'] > 0:
                self.use_r_drop = True
            else:
                self.use_r_drop = False
            self.data = load_wikitext_data_split(self.vocab, data, self.args)
            if 'save_data_to' in args and os.path.exists(args['save_data_to']):
                torch.save(self.data, args['save_data_to'])
        else:
            gen_set, ppl_set = load_wikitext_data_split(self.vocab, data, self.args)
            if self.args['test_mode'] == 'ppl':
                self.data = ppl_set
            else:
                self.data = gen_set
        print(f'[!] collect {len(self.data)} for {self.args["mode"]} set')
 
    def __len__(self):
        return len(self.data)
    
    def set_rep_dropout_rate(self, rep_dropout_rate):
        self.rep_dropout_rate = rep_dropout_rate

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            ids = self.data[i]
            ngram_len = self.args['ngram_len']
            ngram_cache_dict = gen_ngram_set(ngram_len, ids)
            ret_dict = gen_ngram_mask(
                ngram_cache_dict, self.n_layer, len(ids), self.rep_dropout_rate, self.scale_ratio, self.use_r_drop, self.rep_loss_scale_percent)
            attn_mask, hidden_mask, aux_attn_mask, aux_hidden_mask, aux_loss_mask = ret_dict['attn_mask'], ret_dict['hidden_mask'], ret_dict['aux_attn_mask'], ret_dict['aux_hidden_mask'], ret_dict['aux_loss_mask']
            ids = torch.LongTensor(ids)
            ret_list = [ids, attn_mask, hidden_mask]
            if self.use_r_drop:
                ret_list = ret_list + [aux_attn_mask, aux_hidden_mask]
            
            if self.rep_loss_scale_percent > 0:
                ret_list = ret_list + [aux_loss_mask]

            if 'random_mask' in self.args and self.args['random_mask']:
                # mask the tokens with the same number, but not the repetition tokens
                mask_num = (attn_mask == 0).sum(dim=-1)
                random_index = [torch.randperm(attn_mask.shape[1])[:n] for _,n  in zip(range(attn_mask.shape[0]), mask_num)]
                attn_mask = torch.ones_like(attn_mask)
                for idx, ri in enumerate(random_index):
                    attn_mask[idx, ri] = 0
                if not self.use_r_drop:
                    return ids, attn_mask, hidden_mask, torch.ones(len(ids)).to(torch.long)
                else:
                    return ids, attn_mask, hidden_mask, aux_attn_mask, aux_hidden_mask, torch.ones(len(ids)).to(torch.long)
            else:
                if not self.use_r_drop:
                    return ids, attn_mask, hidden_mask, torch.ones(len(ids)).to(torch.long)
                else:
                    return ids, attn_mask, hidden_mask, aux_attn_mask, aux_hidden_mask, torch.ones(len(ids)).to(torch.long)
        else:
            return torch.LongTensor(self.data[i])

    def collate(self, batch):
        if self.args['mode'] == 'train':
            input_ids = pad_sequence([it[0] for it in batch], batch_first=True, padding_value=self.vocab.pad_token_id)
            max_size = input_ids.size(-1)
            attn_mask = torch.stack([F.pad(it[1], (0, max_size-it[1].size(-1)), 'constant', 0) for it in batch]).transpose(0, 1) # [n_layer, batch_size, seq_len]
            hidden_mask = torch.stack([F.pad(it[2], (0, max_size-it[2].size(-1)), 'constant', 0) for it in batch]).transpose(0, 1) # [n_layer, batch_size, seq_len]
            pad_mask = pad_sequence([it[-1] for it in batch], batch_first=True, padding_value=0)
            aux_attn_mask = None
            aux_hidden_mask = None
            aux_loss_mask = None
            if self.use_r_drop:
                aux_attn_mask = torch.stack([F.pad(it[3], (0, max_size-it[1].size(-1)), 'constant', 0) for it in batch]).transpose(0, 1) # [n_layer, batch_size, seq_len]
                aux_hidden_mask = torch.stack([F.pad(it[4], (0, max_size-it[2].size(-1)), 'constant', 0) for it in batch]).transpose(0, 1) # [n_layer, batch_size, seq_len]
            if self.rep_loss_scale_percent > 0:
                aux_loss_mask = pad_sequence([it[-2] for it in batch], batch_first=True, padding_value=0)
            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'hidden_mask': hidden_mask,
                'aux_attention_mask': aux_attn_mask,
                'aux_hidden_mask': aux_hidden_mask,
                'pad_mask': pad_mask,
                'aux_loss_mask': aux_loss_mask
            }
        else:
            input_ids = pad_sequence(batch, batch_first=True, padding_value=self.vocab.pad_token_id)
            return {
                'input_ids': input_ids,
            }


class DITTOWikitextDataset(Dataset):
    """
    Different dropout position for different layer
    """
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab.pad_token = self.vocab.eos_token
        if 'on_disk_data_path' in args and args['on_disk_data_path']:
            dataset = load_from_disk(args['on_disk_data_path'])
        else:
            dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=args['hf_cache_dir'])
        if 'debug' in self.args and self.args['debug']:
            data = dataset['test']
        else:
            data = dataset[self.args['mode']]
        if self.args['mode'] == 'train':
            self.data = load_wikitext_data_split(self.vocab, data, self.args)            
            data_size = len(self.data)
            is_pseudo_data = [0] * data_size
            num_pseudo_rate = self.args["num_pseudo_rate"]
            if num_pseudo_rate >= 1.0:
                candidates = list(range(data_size))
                tmp_num_pseudo_rate = num_pseudo_rate
                while tmp_num_pseudo_rate > 0:
                    candidates += list(range(data_size))
                    tmp_num_pseudo_rate -= 1.0
                indices = random.sample(candidates, k=int(data_size * num_pseudo_rate))
                for i in range(data_size):
                    sent_len = len(self.data[i]) // 2
                    new_instance = []
                    for i in range(2):
                        new_instance += self.data[i][:sent_len]
                    self.data.append(new_instance)
                    is_pseudo_data.append(1)
            elif num_pseudo_rate > 0.0:
                indices = random.sample(list(range(data_size)), k=int(data_size * num_pseudo_rate))
                for i in indices:
                    sent_len = len(self.data[i]) // 2
                    new_instance = []
                    for i in range(2):
                        new_instance += self.data[i][:sent_len]
                    self.data.append(new_instance)
                    is_pseudo_data.append(1)
            self.is_pseudo_data = is_pseudo_data
        else:
            gen_set, ppl_set = load_wikitext_data_split(self.vocab, data, self.args)
            if self.args['test_mode'] == 'ppl':
                self.data = ppl_set
            else:
                self.data = gen_set
        print(f'[!] collect {len(self.data)} for {self.args["mode"]} set')
 
    def __len__(self):
        return len(self.data)
    
    def set_rep_dropout_rate(self, rep_dropout_rate):
        self.rep_dropout_rate = rep_dropout_rate

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            ids = self.data[i]
            prev_index, prev_index_mask = gen_prev_index(ids)
            ids = torch.LongTensor(ids)
            prev_index = torch.LongTensor(prev_index)
            prev_index_mask = torch.FloatTensor(prev_index_mask)
            ret_list = [ids, prev_index, prev_index_mask, self.is_pseudo_data[i], torch.ones(len(ids)).to(torch.long)]
            return ret_list
        else:
            return torch.LongTensor(self.data[i])

    def collate(self, batch):
        if self.args['mode'] == 'train':
            input_ids = pad_sequence([it[0] for it in batch], batch_first=True, padding_value=self.vocab.pad_token_id)
            max_size = input_ids.size(-1)
            prev_index = torch.stack([F.pad(it[1], (0, max_size-it[1].size(-1)), 'constant', 0) for it in batch])
            prev_index_mask = torch.stack([F.pad(it[2], (0, max_size-it[1].size(-1)), 'constant', 0) for it in batch])
            is_pseudo_data = torch.FloatTensor([it[3] for it in batch])
            pad_mask = pad_sequence([it[-1] for it in batch], batch_first=True, padding_value=0)
            return {
                'input_ids': input_ids,
                'prev_index': prev_index,
                'prev_index_mask': prev_index_mask,
                'is_pseudo_data': is_pseudo_data,
                'pad_mask': pad_mask
            }
        else:
            input_ids = pad_sequence(batch, batch_first=True, padding_value=self.vocab.pad_token_id)
            return {
                'input_ids': input_ids,
            }
