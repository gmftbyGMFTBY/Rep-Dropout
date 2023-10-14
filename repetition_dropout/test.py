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
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_mode', type=str)
    parser.add_argument('--hf_cache_dir', type=str, default='./hf_cache')
    return parser.parse_args()


def main(**args):
    # load validation set
    test_args = deepcopy(args)
    print(test_args)
    test_args['mode'] = 'test'
    config = load_config(test_args)
    test_args.update(config)
    test_data, test_iter, _ = load_dataset(test_args)
    tokenizer = AutoTokenizer.from_pretrained(test_args['tokenizer'])
    tokenizer.add_tokens('<eor>')
    
    agent = load_model(test_args)
    agent.load_model(args['load_path'])
    results = []
    error_num, correct_num = 0, 0
    path = f'{test_args["root_dir"]}/rest/{test_args["dataset"]}/{test_args["model"]}/test_results.txt'
    print(f'[!] save the generation results into:', path)
    print(f'[!] result file: {path}')
    decoding_config = {"use_cache": True}

    with torch.no_grad():
        for batch in tqdm(test_iter):
            try:
                input_ids = batch['input_ids'].cuda()
                input_ids_len = len(input_ids[0])
                prefix = tokenizer.decode(input_ids[0]).strip()
                greedy_output = agent.model.model.generate(input_ids, max_new_tokens=test_args['generate_len'], **decoding_config)
                greedy_output = greedy_output[0][input_ids_len:]
                result = tokenizer.decode(greedy_output, skip_special_tokens=True)
                result = result.replace('\n', ' ')
                results.append({'prefix': prefix, 'result': result})
                correct_num += 1
            except Exception as error:
                print(f'[!] error happened:', error)
                error_num += 1
    with open(path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    results = [item['result'] for item in results]
    special_tokens = ['<|endoftext|>', '[SEP]']
    result = []
    for i in results:
        for st in special_tokens:
            i = i.replace(st, ' ').strip()
        result.append(i)
    results = compute_repetition_ratio(result)
    pprint.pprint(results)
    # rep-w
    rep_w = calculate_rep_w(result)
    # rep-r
    rep_r = calculate_rep_r(result)
    print(f'[!] REP-W: {rep_w}\n[!] REP-R: {rep_r}')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    main(**args)
