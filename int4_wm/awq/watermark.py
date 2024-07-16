from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
import random
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from awq.watermarks.insert import WMInsert
from awq.watermarks.extract import WMExtract
from awq.watermarks.attack import WMAttack
from awq.watermarks.evaluate_wm import Evaluator

os.environ['TRANSFORMERS_CACHE'] = '/data/ruisi/llm_models/transformers_cache'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of the hf model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument("--tasks", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument('--num_fewshot', type=int, default=0)
    # model config
    parser.add_argument('--parallel', action='store_true', help="enable model parallelism")
    # max memory to offload larger models to CPU
    parser.add_argument('--max_memory', type=list, nargs='*',
                        help="List of device_id:max_memory pairs to be parsed into a dictionary; " \
                            + "Example: 0:10GiB 1:10GiB cpu:30GiB; " \
                            + "mode details here: " \
                            + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling")
    parser.add_argument('--auto_parallel', action='store_true', help="automatically set parallel and batch_size")
    # quantization config
    parser.add_argument('--w_bit', type=int, default=None)
    parser.add_argument('--q_group_size', type=int, default=-1)
    parser.add_argument('--no_zero_point', action='store_true', help="disable zero_point")
    parser.add_argument('--q_backend', type=str, default="fake", choices=["fake", "real"])
    # save/load real quantized weights
    parser.add_argument('--dump_quant', type=str, default=None, help='save quantized model')
    parser.add_argument('--load_quant', type=str, default=None, help='load quantized model')
    # apply/save/load awq
    parser.add_argument('--run_awq', action='store_true', help="perform awq search process")
    parser.add_argument('--dump_awq', type=str, default=None, help="save the awq search results")
    parser.add_argument('--load_awq', type=str, default=None, help="load the awq search results")
    parser.add_argument('--load_activation', type=str, default=None, help="load the activation wm results")
    # watermark configs
    parser.add_argument('--eval_original', type=int, default=0)
    parser.add_argument('--load_int_weight', type=str, default="")
    parser.add_argument('--total_wm_length', type=int, default=1000)
    parser.add_argument('--wm_length', type=int, default=40)
    parser.add_argument('--wm_method', type=str, default="random")
    parser.add_argument('--wm_attack', type=str, default="")
    # attack configs
    parser.add_argument('--rewrite_rates', type=str, default="100,200,300,400,500")
    parser.add_argument('--rewatermark_rates', type=str, default="150,250")
    args = parser.parse_args()
    return args

def load_model_and_enc(args, q_config, max_memory):
    model_path = args.model_path
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    else:
        enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16, trust_remote_code=True)
    real_quantize_model_weight(model, args.load_int_weight, w_bit=args.w_bit, q_config=q_config, init_only=True)
    model.tie_weights()
    
    # Infer device map
    kwargs = {"max_memory": max_memory} if len(max_memory) else {}
    device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"], **kwargs)
    load_checkpoint_in_model(model, checkpoint=args.load_quant, device_map=device_map, offload_state_dict=True,)
    model = simple_dispatch_model(model, device_map=device_map)
        
    model.eval()
    return model, enc


def main():
    args = parse_args()

    # set random seed for reproducibility
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    random.seed(100)

    max_memory = [v.split(':') for v in (args.max_memory or [])]
    max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}
    #max_memory = {0: "1GiB", 1: "14GiB", 2: "43GiB", "cpu": "30GiB"}
    if args.auto_parallel:
        gpu_list = auto_parallel(args)

    eval_fun =  Evaluator()
    q_config = {"zero_point": not args.no_zero_point, "q_group_size": args.q_group_size,}
    print("Quantization config:", q_config)
    model, enc = load_model_and_enc(args, q_config, max_memory)
    if args.wm_method == "original":
        eval_fun.evaluate_all(model, enc, args)
        return
    # put to cpu
    model.cpu()
    
    # load activation for wm
    if args.wm_method == "ours":
        activation_wm = torch.load(args.load_activation, map_location="cpu")
    else:
        activation_wm = None

    # insert wm
    Insert = WMInsert(args)
    wm_model, wm_encoder, save_path = Insert.insert(model, enc, activation_wm)
            
    # extract wm
    Extract = WMExtract(args, save_path, Insert.keys, Insert.wm_bit_idx)
    Extract.extract(wm_model, wm_encoder, model, enc)

    # get zero shot accurcy of wm model & ppl
    wm_model.cuda() # put to gpu for eval
    eval_fun.evaluate_all(wm_model, wm_encoder, args)

    if args.wm_attack != "":
        # test robustness
        model.cpu() # put model to attack again
        Attack = WMAttack(args, model, enc, Extract, args.w_bit)
        Attack.attack(wm_model, wm_encoder, activation_wm)

if __name__ == '__main__':
    main()