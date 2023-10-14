from header import *
from .custom_utils import CustomGPT2LMHeadModel, DropMode


def find_repetetive_ngram(input_tokens, label_tokens, ngram_len):
    label2input_align = [set() for _ in range(len(label_tokens))]
    input_ngram_dict = dict()
    seq_len = len(input_tokens)
    for i in range(seq_len):
        if i < (ngram_len-1):
            continue
        input_ngram = tuple(input_tokens[i-ngram_len+1:i+1])
        label_ngram = tuple(label_tokens[i-ngram_len+1:i+1])
        if label_ngram in input_ngram_dict:
            start_idx_list = input_ngram_dict[label_ngram]
            for sidx in start_idx_list:
                for shift in range(ngram_len):
                    label2input_align[i-shift].add(sidx-shift)
        if input_ngram in input_ngram_dict:
            input_ngram_dict[input_ngram].append(i)
        else:
            input_ngram_dict[input_ngram] = [i]
    return label2input_align


class MyGPT2DynamicDropModel(nn.Module):

    '''token embeddings dropout'''

    def __init__(self, **args):
        super(MyGPT2DynamicDropModel, self).__init__()
        model = args['pretrained_model']
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab_size = len(self.vocab)
        self.vocab.pad_token = self.vocab.eos_token
        config = GPT2Config.from_pretrained(model, cache_dir=args['hf_cache_dir'])
        self.model = CustomGPT2LMHeadModel(config)
        self.config = config
        self.args = args

        self.pad = self.vocab.pad_token_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        self.ppl_criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        self.is_3d_mask = args['is_3d_mask']
        self.rep_loss_scale_rate = args['rep_loss_scale_rate'] if 'rep_loss_scale_rate' in args else 1
        self.is_layerwise_mask = args['is_layerwise_mask']
        self.drop_mode = args['drop_mode']
        assert self.drop_mode in [DropMode.ALL, DropMode.ATTENTION, DropMode.HIDDEN]
        self.hidden_dropout_rate = args['hidden_dropout_rate'] if self.drop_mode in [DropMode.ALL, DropMode.HIDDEN] else 0
        
    def calculate_ppl(self, batch):
        ids = batch['input_ids'].cuda()
        outputs = self.model(input_ids=ids)
        labels = ids[:, 1:]
        logits = outputs.logits[:, :-1, :]
        mle_loss = self.ppl_criterion(logits.reshape(-1, self.vocab_size), labels.reshape(-1)).tolist()
        return mle_loss

    def calculate_rep_attn(self, batch, ngram_len, local_rank):
        input_ids = batch['input_ids'][:, :-1].to(local_rank)
        labels = batch['input_ids'][:, 1:].to(local_rank)
        seq_len = input_ids.size(-1)
        outputs = self.model(
            input_ids=input_ids, output_attentions=True
        )
        input_tokens = self.vocab.convert_ids_to_tokens(input_ids[0]) 
        label_tokens = self.vocab.convert_ids_to_tokens(labels[0])
        label2input_align = find_repetetive_ngram(input_tokens, label_tokens, ngram_len)
        ratio_list = [[] for _ in range(self.config.n_layer)]
        
        for i in range(seq_len):
            if i < (ngram_len-1) or i < 1:
                continue
            visited = label2input_align[i]
            for l in range(self.config.n_layer):
                # attn_l = torch.mean(outputs[-1][l], dim=1)
                for h in range(self.config.n_head):
                    attn_l = outputs[-1][l][:, h]
                    if visited:
                        rep_attn = 0
                        for idx in visited:
                            rep_attn += attn_l[0, i, idx]
                        avg  = 1 / (i+1)
                        ratio_list[l].append(rep_attn / len(visited) / avg)
        return ratio_list

    def forward(self, batch, return_logits=False):
        # move batch to cuda
        inputs = {}
        for key, value in batch.items():
            try:
                value = value.cuda()
                inputs[key] = value
            except:
                continue

        # input_ids: [B, S]; attention_mask: [B, S, S]
        outputs = self.model(
            input_ids=batch['input_ids'], attention_mask=inputs['attention_mask'],
            is_3d_mask=self.is_3d_mask, is_layerwise_mask=self.is_layerwise_mask,
            drop_mode=self.drop_mode, hidden_dropout_rate=self.hidden_dropout_rate,
            pad_mask=batch['pad_mask'], hidden_mask=inputs['hidden_mask']
        )
        # shift the inputs to get the labels: [B, S-1]
        labels = inputs['input_ids'][:, 1:]
        # outputs.logits: [B, S-1, V]
        loss = self.criterion(outputs.logits[:, :-1, :].reshape(-1, self.vocab_size), labels.reshape(-1))
        aux_loss_mask = batch['aux_loss_mask']
        if aux_loss_mask is not None:
            aux_loss_mask = (1 - aux_loss_mask[:, 1:].reshape(-1)).bool()
            rep_loss = loss[aux_loss_mask] * self.rep_loss_scale_rate
            loss[aux_loss_mask] = rep_loss

        loss = loss.sum() / torch.count_nonzero(torch.not_equal(labels, self.pad))
        # calculate the token accuarcy
        gen_logits = outputs.logits
        # ods = inputs['input_ids'][:, 1:]
        chosen_tokens = torch.max(gen_logits, dim=-1)[1][:, :-1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        if not return_logits:
            return loss, gen_acc
        else:
            return loss, gen_acc, gen_logits[:, :-1]
