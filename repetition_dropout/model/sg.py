from header import *


# borrow from the original implementation of SG Loss: https://github.com/shawnlimn/ScaleGrad
def getNovelMask(target, vocab_size):
    b, l = target.size()
    zeros = torch.zeros(b, l, vocab_size).to(target.device)
    ones = torch.ones(b, l, vocab_size).to(target.device)

    target_index = target.unsqueeze(1).expand(b, l, l).transpose(-2, -1).triu().transpose(-2,-1)
    matrix = zeros.scatter_add_(2, target_index, ones)
    matrix[:, :, 0] = 0
    summ_true = torch.tensor(range(1, l+1)).unsqueeze(0).float().to(target.device)
    summ_now = torch.sum(matrix, dim=-1)
    diff = summ_true - summ_now
    matrix[:, :, 0] = diff
    matrix = torch.cat((torch.zeros(b, 1, vocab_size).to(target.device),matrix[:, :-1, :]), 1)
    novel_mask = matrix < 1.

    return novel_mask


# A more efficient way to generate mask by ChatGPT
def mask_prob(prob, indices):
    assert len(prob.size()) == 2 and len(indices.size()) == 2
    mask = torch.zeros_like(prob, dtype=torch.bool).to(indices.device)
    mask[indices[:, 0], indices[:, 1]] = True
    mask[:, 0] = 0
    return mask


def mask_prob_indices(labels, skip_rep=False, only_rep=False, random_mask=False, rep_attn_matrix=None):
    # 1. triangualr matrix
    # 2. cat index, e.g., [[0, 5253], [0, 13]]

    # Step-1: generate the triangualr matrix
    # assert sum([only_rep, random_mask, skip_rep]) == 1

    batch_size, input_length = labels.size()
    word_indices = torch.zeros(batch_size, input_length, input_length, dtype=labels.dtype).to(labels.device)
    row_indices, col_indices = torch.tril_indices(input_length, input_length, offset=-1)
    word_indices[:, row_indices, col_indices] = labels[:, col_indices]
    if skip_rep:
        unigram_zero_condition = word_indices == labels.unsqueeze(-1).repeat(1, 1, input_length)
        word_indices[unigram_zero_condition] = 0
    elif only_rep:
        # unigram_zero_condition = word_indices == labels.unsqueeze(-1).repeat(1, 1, input_length)
        if rep_attn_matrix is None:
            labels_a = labels.unsqueeze(1).expand(-1, labels.size(1), -1)
            labels_b = labels.unsqueeze(2).expand(-1, -1, labels.size(1))
            unigram_zero_condition = (labels_a == labels_b).sum(dim=-1) > 1
            unigram_zero_condition = unigram_zero_condition.unsqueeze(1).repeat(1, input_length, 1)
        else:
            # 1 for non-repetition and 0 for repetition. So, we reverse the rep_attn_matrix here.
            unigram_zero_condition = (1-rep_attn_matrix).bool()
            unigram_zero_condition = unigram_zero_condition.unsqueeze(1).repeat(1, input_length, 1)
        word_indices[~unigram_zero_condition] = 0
    elif random_mask:
        if rep_attn_matrix is None:
            labels_a = labels.unsqueeze(1).expand(-1, labels.size(1), -1)
            labels_b = labels.unsqueeze(2).expand(-1, -1, labels.size(1))
            unigram_zero_condition = (labels_a == labels_b).sum(dim=-1) > 1
        else:
            unigram_zero_condition = (1-rep_attn_matrix).bool()
        rep_percent = unigram_zero_condition.sum() / (batch_size * input_length)
        rand_mask = torch.rand((batch_size, input_length), device=labels.device)
        rand_mask = rand_mask < rep_percent
        rand_mask = rand_mask.unsqueeze(1).repeat(1, input_length, 1)
        word_indices[~rand_mask] = 0


    # Step-2: concate batch id an word id
    falt_ids = torch.arange(batch_size * input_length, dtype=labels.dtype).view(batch_size, input_length).to(labels.device)
    falt_ids = falt_ids.unsqueeze(-1).repeat(1, 1, input_length)
    indices = torch.stack([falt_ids, word_indices], dim=-1)
    return indices.view(batch_size * input_length * input_length, -1)


class MyGPT2ModelWithSGLoss(nn.Module):

    def __init__(self, **args):
        super(MyGPT2ModelWithSGLoss, self).__init__()
        model = args['pretrained_model']
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab.add_tokens('[SEP]')
        self.vocab_size = len(self.vocab)
        self.vocab.pad_token = self.vocab.eos_token
        
        config = GPT2Config.from_pretrained(model, cache_dir=args['hf_cache_dir'])
        config.vocab_size = self.vocab_size
        self.model = GPT2LMHeadModel(config)
        self.args = args

        self.pad = self.vocab.pad_token_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.ppl_criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        
    def calculate_ppl(self, batch):
        ids = batch['input_ids'].cuda()
        if 'labels' in batch:
            labels = batch['labels'].cuda()
        else:
            labels =  ids[:, 1:]
            ids =  ids[:, :-1]
        outputs = self.model(input_ids=ids)
        logits = outputs.logits
        mle_loss = self.ppl_criterion(logits.reshape(-1, self.vocab_size), labels.reshape(-1)).tolist()
        return mle_loss

    def forward(self, batch):
        # move batch to cuda
        inputs = {}
        for key, value in batch.items():
            try:
                value = value.cuda()
                inputs[key] = value
            except:
                continue
        # input_ids: [B, S]; attention_mask: [B, S]
        if 'pad_mask' in inputs:
            pad_mask = inputs['pad_mask'][:, :-1]
        else:
            pad_mask = inputs['attention_mask'][:, :-1]
        labels = inputs['input_ids'][:, 1:]
        input_ids = inputs['input_ids'][:, :-1]
        outputs = self.model(input_ids=input_ids, attention_mask=pad_mask)
        # shift the inputs to get the labels: [B, S-1]
        logits = outputs.logits
        batch_size, input_length, vocab_size = outputs.logits.size()
        if len(inputs['attention_mask'].size()) > 2:
            attention_mask = inputs['attention_mask'][0, :, :-1]
        else:
            attention_mask = inputs['attention_mask'][:, :-1]
        # ScaleGrad
        ##########################################################

        probs = F.softmax(logits, dim=-1).view(-1, vocab_size) 
        # Obtaining the masks for novel tokens
        # novel_mask = getNovelMask(labels[0].unsqueeze(0), logits.size(-1))
        # rep_mask = ~novel_mask
        # assert prefix_mask[:labels.size(1)] == rep_mask[0]
        skip_rep = True if 'sg_skip_rep' in self.args and self.args['sg_skip_rep'] else False
        only_rep = True if 'sg_only_rep' in self.args and self.args['sg_only_rep'] else False
        random_mask = True if 'sg_rand_mask' in self.args and self.args['sg_rand_mask'] else False
        mask_indices = mask_prob_indices(labels, skip_rep=skip_rep, only_rep=only_rep, random_mask=random_mask, rep_attn_matrix=attention_mask)
        prefix_mask = mask_prob(probs, mask_indices)
        # import pdb; pdb.set_trace()
        new_probs = probs * (self.args['sg_gamma'] + (1-self.args['sg_gamma']) * prefix_mask) + 1e-8
        lprobs = torch.log(F.normalize(new_probs, p=1, dim=-1))
        ##########################################################        
        loss = self.criterion(lprobs, labels.reshape(-1))

        # calculate the token accuarcy
        # gen_logits = outputs.logits
        # ods = inputs['input_ids'][:, 1:]
        chosen_tokens = torch.max(logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

