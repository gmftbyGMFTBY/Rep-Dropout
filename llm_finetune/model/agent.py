import torch.optim
from header import *

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = int(self.args['warmup_ratio'] * self.args['total_steps'])
        print(f'[!] total optimization spte: {self.args["total_steps"]}; warmup steps: {ds_params["scheduler"]["params"]["warmup_num_steps"]}')
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model, 
            model_parameters=self.model.parameters(),
            config_params=ds_params, 
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )
        self.trained_token_num = 0

    def train_model(self, batch, current_step=0, pbar=None, sum_writer=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if sum_writer:
            self.trained_token_num += self.args['world_size'] * len(batch['input_ids'].reshape(-1))
            sum_writer.add_scalar(f'train/RunningLoss', loss.item(), self.trained_token_num)
            sum_writer.add_scalar(f'train/TokenAcc', mle_acc*100, self.trained_token_num)

        if current_step in self.args['eval_and_save_steps']:
            self.ds_engine.module.eval()
            self.save_model(self.args['save_path'], self.args['save_counter'])
            self.args['save_counter'] += 1

    @torch.no_grad()
    def calculate_ppl(self, test_iter):
        self.ds_engine.module.eval()
        losses = []
        for batch in tqdm(test_iter):
            loss = self.ds_engine.module.calculate_ppl(batch)
            losses.extend(loss)
        ppl = np.exp(np.mean(losses))
        torch.distributed.barrier()
        return ppl
    
    def save_model(self, path, current_step):
        self.ds_engine.module.model.save_pretrained(os.path.join(path, str(current_step)))

    def load_model(self, path):
        self.ds_engine.module.load_state_dict(torch.load(path))
        print(f'[!] load lora delta path from {path}')
