from header import *

class Agent:
    
    def __init__(self, model, args):
        super(Agent, self).__init__()
        self.args = args
        self.model = model
        self.load_last_step = None
        
        # open the test save scores file handler
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')

        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train']:
            self.set_optimizer_scheduler_ddp()

    @torch.no_grad()
    def test_model_ppl(self, batch):
        self.model.eval()
        try:
            return self.model.calculate_ppl(batch)
        except:
            return self.model.module.calculate_ppl(batch)
        
    @torch.no_grad()
    def calculate_rep_dropout_rate(self, test_iter, ngram_len, local_rank, maximum_num=None):
        self.model.eval()
        ratio_list = None
        n = 0
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            calculate_rep_attn = self.model.module.calculate_rep_attn
        else:
            calculate_rep_attn = self.model.calculate_rep_attn
        for batch in tqdm(test_iter):
            if maximum_num is not None and n >= maximum_num:
                break
            cur_ratio_list = calculate_rep_attn(batch, ngram_len, local_rank)
            if ratio_list is None:
                ratio_list = cur_ratio_list
            else:
                for i in range(len(cur_ratio_list)):
                    ratio_list[i] += cur_ratio_list[i]
            n += 1
        return [(sum(it) / len(it)).item() for it in ratio_list]

    def train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        
        with autocast():
            if 'r_drop_coeff' in self.args and self.args['r_drop_coeff'] > 0:
                mle_loss_1, mle_acc, logits_1 = self.model(batch, return_logits=True)
                assert 'aux_attention_mask' in batch and 'aux_hidden_mask' in batch
                batch['attention_mask'] = batch['aux_attention_mask']
                batch['hidden_mask'] = batch['aux_hidden_mask']
                mle_loss_2, _, logits_2 = self.model(batch, return_logits=True)
                # loss = (mle_loss_1 + mle_loss_2) / 2
                log_prob_1 = F.log_softmax(logits_1, dim=-1)
                log_prob_2 = F.log_softmax(logits_2, dim=-1)
                consistency_loss = F.kl_div(log_prob_1, log_prob_2, log_target=True, reduction="batchmean") + F.kl_div(log_prob_2, log_prob_1, log_target=True, reduction="batchmean")
                loss = (mle_loss_1 + mle_loss_2 + self.args['r_drop_coeff'] * consistency_loss) / 4
                
            else:
                loss, mle_acc = self.model(batch)
        loss /= self.args['iter_to_accumulate']

        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/TokenAcc', mle_acc, current_step)
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
    
    def load_model(self, path):
        if self.args['mode'] == 'train':
            pass
        else:
            self.model.model.load_state_dict(torch.load(path)['model_state_dict'])
        print(f'[!] load the latest model from {path}')

    def save_model(self, path, current_step):
        model_state_dict = self.model.module.model.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save(
            {
                'model_state_dict' : model_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'step': current_step
            }, 
            path
        )
        print(f'[!] save model into {path}')
 
    def set_optimizer_scheduler_ddp(self):
        if self.args['mode'] in ['train']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.scaler = GradScaler()
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_step'], 
                num_training_steps=self.args['total_step'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )

