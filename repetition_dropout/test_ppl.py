from header import *
from build_dataloader import *
from model import *
from config import *
sys.path.append('..')
from utils import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--test_mode', type=str, default='ppl')
    parser.add_argument('--hf_cache_dir', type=str, default='./hf_cache')
    parser.add_argument('--load_path', type=str)
    return parser.parse_args()


def main(**args):
    test_args = deepcopy(args)
    test_args['mode'] = 'test'
    config = load_config(test_args)
    test_args.update(config)
    test_data, test_iter, _ = load_dataset(test_args)

    agent = load_model(test_args)
    agent.load_model(args['load_path'])
    mle_losses = []
    for batch in tqdm(test_iter):
        mle_loss = agent.test_model_ppl(batch)
        mle_losses.extend(mle_loss)
    ppl = math.exp(np.mean(mle_losses))
    print('PPL:', round(ppl, 4))

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
