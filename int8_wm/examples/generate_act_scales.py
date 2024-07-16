import torch
import os
from torch.fx import symbolic_trace

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.calibration import get_act_scales

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, cache_dir="/data/ruisi/llm_models")
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/data/ruisi/llm_models", **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-2.7b', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)
    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
    
    args.output_path = args.output_path + args.model_name.split('/')[-1] + '.pt'
    #import pdb; pdb.set_trace()
    # print the key and max value and min value in act_scales
    for key in act_scales.keys():
        print(f'key: {key}, max: {act_scales[key].max()}, min: {act_scales[key].min()}')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
