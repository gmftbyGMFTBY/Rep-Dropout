from custom_datasets import *


def load_dataset(args):
    dataset_name = args['models'][args['model']]['dataset_names'][args['dataset']]
    dataset_t = globals()[dataset_name]
    data = dataset_t(**args)
    if args['mode'] in ['train']:
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    else:
        iter_ = DataLoader(data, batch_size=args['batch_size'], shuffle=False, collate_fn=data.collate)
        sampler = None
    return data, iter_, sampler


def create_iterator(data, cur_epoch, args):
    sampler = torch.utils.data.distributed.DistributedSampler(data)
    iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    sampler.set_epoch(cur_epoch)
    return data, iter_, sampler
