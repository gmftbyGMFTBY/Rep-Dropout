from header import *


# code borrowed from https://github.com/facebookresearch/unlikelihood_training/blob/main/custom/candidate_penalty_ce_loss.p
class CandidatePenaltyCrossEntropyCriterion():
    """Applies a (1-p(x_nt)) loss to each negative target ('candidate') x_nt."""

    def __init__(self, rank_alpha, padding_idx=1):
        self.rank_alpha = rank_alpha
        self.padding_idx = padding_idx

    def forward(self, net_outputs, target, selected_matrix=None):
        # net_output: [B, S, V]
        # selected matrix: [B, S], with selected index is 0, otherwise is 1
        # selected matrix must be similar as target, that re-index y `1:`
        nsentences = target.size(0)    # batch size (B)
        target = target.reshape(-1)     # [B*S]
        lprobs = F.softmax(net_outputs, dim=-1).log()   # convert into probablity

        # -- mle loss
        lprobs = lprobs.reshape(-1, lprobs.size(-1))    # [B*S, V]
        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )
        mle_loss = true_token_lprobs.sum()

        # token accuracy
        chosen_tokens = torch.max(lprobs, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == target.reshape(-1)).to(torch.long)
        valid_mask = (target != self.padding_idx).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        acc = valid_tokens.sum().item() / valid_mask.sum().item()

        # -- custom loss
        # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
        if selected_matrix is not None:
            # add the selected index into this target matrix
            new_target = target.clone()
            selected_matrix = selected_matrix.reshape(-1)    # [B*S]
            # selected_matrix = (~selected_matrix.to(torch.bool)).to(torch.long)
            new_target[selected_matrix] = self.padding_idx
        else:
            new_target = target

        # - form negative targets
        with torch.no_grad():
            # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
            # Make 'the triangle'.
            
            # target transform: [B, S] -> [B*S, B*S]
            ctx_cands = new_target.unsqueeze(0).expand(new_target.size(0), new_target.size(0))
            ctx_cands_ = (ctx_cands.tril(-1) + self.padding_idx)
            ctx_cands_ = ctx_cands_ * ctx_cands_.triu()    # ctx_cands_ is the padding upper triangle matrix
            ctx_cands = ctx_cands.tril(-1) + ctx_cands_

            # Don't include the target for that timestep as a negative target.
            # ctx_cands = ctx_cands.masked_fill(ctx_cands == new_target.unsqueeze(1), self.padding_idx)    # [B*S, B*S]
            # in case the padding index is not 1 or 0
            ctx_cands = ctx_cands.masked_fill(ctx_cands == (self.padding_idx**2), self.padding_idx)    # [B*S, B*S]
            # negative_targets: [B*S, V]
            negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)   # [B*S, V]

        # - compute loss
        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)

        custom_loss = -torch.log(one_minus_probs)*negative_targets
        custom_loss = custom_loss.sum()

        loss = mle_loss + self.rank_alpha * custom_loss

        return loss, acc

class MyGPT2UnlikeliHoodModel(nn.Module):

    def __init__(self, **args):
        super(MyGPT2UnlikeliHoodModel, self).__init__()
        model = args['pretrained_model']
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab.add_tokens('[SEP]')
        self.vocab_size = len(self.vocab)

        # for unlikelyhood the pad_token_id should be 1 for the loss calculating
        self.vocab.pad_token_id = self.vocab.eos_token_id 
        
        config = GPT2Config.from_pretrained(model, cache_dir=args['hf_cache_dir'])
        config.vocab_size = self.vocab_size
        self.model = GPT2LMHeadModel(config)
        self.args = args

        self.unlikelyhood_criterion = CandidatePenaltyCrossEntropyCriterion(args['rank_alpha'], self.vocab.pad_token_id)

        self.pad = self.vocab.pad_token_id
        self.ppl_criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        
    def calculate_ppl(self, batch):
        ids = batch['input_ids'].cuda()
        outputs = self.model(input_ids=ids)
        labels = ids[:, 1:]
        logits = outputs.logits[:, :-1, :]
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
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        loss, acc = self.unlikelyhood_criterion.forward(outputs.logits[:, :-1, :], inputs['input_ids'][:, 1:])
        return loss, acc
