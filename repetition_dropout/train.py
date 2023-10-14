from header import *
from build_dataloader import *
from model import *
from config import *
from utils import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--save_data_to', default='', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--test_mode', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--total_workers', type=int)
    parser.add_argument('--hf_cache_dir', type=str, default="./hf_cache")
    return parser.parse_args()    


def main(**args):
    torch.cuda.empty_cache()
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args['global_rank'] = dist.get_rank()
    print(f'[!] global rank: {args["global_rank"]}')

    # load train set
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])

    agent = load_model(args)
    train_data, train_iter, sampler = load_dataset(args)
    # set seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    if args['local_rank'] == 0:
        sum_writer = SummaryWriter(
            log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/{args["version"]}',
            comment=pretrained_model_name,
        )
    else:
        sum_writer = None
    
    pbar = tqdm(total=args['total_step'])
    current_step, over_train_flag, best_ppl = 0, False, 1e8
    if agent.load_last_step:
        current_step = agent.load_last_step + 1
        print(f'[!] load latest step: {current_step}')
    num_epoch = 0
    sampler.set_epoch(num_epoch)    # shuffle for DDP
    rep_dropout_update_freq = None if 'rep_dropout_update_freq' not in args or args['rep_dropout_update_freq'] < 1 else args['rep_dropout_update_freq']
    while True:
        train_data, train_iter, sampler = create_iterator(train_data, num_epoch, args)
        for batch in train_iter:
            agent.train_model(
                batch, 
                recoder=sum_writer, 
                current_step=current_step, 
                pbar=pbar
            )
            if args['global_rank'] == 0 and current_step % args['save_every'] == 0 and current_step > 0:
                pretrained_model_name = args['pretrained_model'].replace('/', '_')
                save_folder = "{}_{}".format(args["model"], args['save_tag']) if 'save_tag' in args else args["model"]
                FileUtils.check_dirs(f'{args["root_dir"]}/ckpt/{args["dataset"]}/{save_folder}/')
                save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{save_folder}/best_{pretrained_model_name}_{args["version"]}_{current_step}.pt'
                agent.save_model(save_path, current_step)
            current_step += 1
            if current_step > args['total_step']:
                over_train_flag = True
                break
        num_epoch += 1    
        if over_train_flag:
            break
    if sum_writer:
        sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
