from header import *
from model import *
import sys
sys.path.append('../degeneration_riro')
from utils import *
from datasets import load_dataset, load_from_disk


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--delta_model_path', default='ckpt/scillm/pytorch_model.bin', type=str)
    parser.add_argument('--prefix_len', default=32, type=int)
    parser.add_argument('--generate_len', default=128, type=int)
    return parser.parse_args()


def load_wikitext_dataset():
    # load wikitext-103 test set
    # data = load_dataset('wikitext', 'wikitext-103-v1')['test']
    data = load_from_disk('/home/lt/wikitext')['test']
    dataset = []
    for idx in tqdm(range(len(data))):
        text = data[idx]['text'].strip()
        if text and text.startswith('=') is False:
            dataset.append(text)
    print(f'[!] load {len(dataset)} samples')
    return dataset


def main(args):
    args.update({
        'lora_r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'inference'
    })
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args['model_path'],
        load_in_4bit=True,
        max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    print(f'[!] load model and tokenizer over')
    model_hidden_size = model.config.hidden_size
    dataset = load_wikitext_dataset()
    prefix_len, generation_len = 32, 128

    with torch.no_grad():
        generations = []
        for sample in tqdm(dataset):
            input_ids = tokenizer(sample, return_tensors='pt').input_ids.cuda()[:, :prefix_len]
            generated_ids = model.generate(input_ids=input_ids, max_new_tokens=generation_len, early_stopping=False)
            generation = tokenizer.decode(generated_ids[0, prefix_len:], skip_special_tokens=True)
            generations.append(generation)
            print(f'[PREFIX] {sample}\n[RESULT] {generation}\n')
        results = compute_repetition_ratio(generations)
        print(f'[!] Results of {args.model_name}')
        pprint.pprint(results)


if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
