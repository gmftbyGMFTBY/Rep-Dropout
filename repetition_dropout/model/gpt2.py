from header import *

class MyGPT2Model(nn.Module):

    def __init__(self, **args):
        super(MyGPT2Model, self).__init__()
        model = args['pretrained_model']
        if 'use_custom_tokenizer' in args and args['use_custom_tokenizer']:
            self.vocab = Vocabulary(args['tokenizer'])
        else:
            self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'], cache_dir=args['hf_cache_dir'])
        self.vocab_size = len(self.vocab)
        self.vocab.pad_token = self.vocab.eos_token
        config = GPT2Config.from_pretrained(model, cache_dir=args['hf_cache_dir'])
        self.model = GPT2LMHeadModel(config=config)
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
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # shift the inputs to get the labels: [B, S-2]
        labels = inputs['input_ids'][:, 1:]
        # outputs.logits: [B, S-1, V]
        loss = self.criterion(outputs.logits[:, :-1, :].reshape(-1, self.vocab_size), labels.reshape(-1))

        # calculate the token accuarcy
        gen_logits = outputs.logits
        # ods = inputs['input_ids'][:, 1:]
        chosen_tokens = torch.max(gen_logits, dim=-1)[1][:, :-1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

