from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from pathlib import Path
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

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
from awq.llama.model import ModelArgs, Transformer
from awq.llama.tokenizer import Tokenizer
from awq.watermarks.evaluate_wm import Evaluator

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
    parser.add_argument('--dump_activation', type=str, default=None, help="save the activation wm results")
    # watermark configs
    parser.add_argument('--wm_length', type=int, default=100)
    parser.add_argument('--wm_method', type=str, default="random")
    parser.add_argument('--wm_attack', type=str, default="")
    parser.add_argument('--dump_int_weight', type=str, default="")
    args = parser.parse_args()
    return args



def save_model_and_enc(args, q_config, max_memory):
    model_path = args.model_path
    print(f"* Building model {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, cache_dir="/data/ruisi/llm_models")
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True, cache_dir="/data/ruisi/llm_models")
    else:
        enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True, cache_dir="/data/ruisi/llm_models")

    print("Loading pre-computed quantized weights...")
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, cache_dir="/data/ruisi/llm_models", **kwargs)
    model.eval()
    print("load model success")
    #eval_fun =  Evaluator()
    #import pdb; pdb.set_trace()
    #model.cuda()
    #eval_fun.evaluate_all(model, enc, args)
    # save activation for wm
    
    _, activation_wm = run_awq(model, enc, w_bit=args.w_bit, q_config=q_config, n_samples=128, seqlen=512,save_activation_wm=args.dump_activation)
    torch.save(activation_wm, args.dump_activation)
    
    # load official awq for wm insertion
    
    #awq_results = torch.load(args.load_quant, map_location="cpu")

    #model = apply_awq(model, awq_results)
    #real_quantize_model_weight(model, args.dump_int_weight, w_bit=args.w_bit, q_config=q_config)
    
    #dirpath = "./llm_models/quant_cache/"
    #os.makedirs(dirpath, exist_ok=True)
    #model_save_path = args.load_quant.split("/")[-1]
    #torch.save(model.cpu().state_dict(), dirpath+model_save_path)
    
    

def main():
    args = parse_args()

    #max_memory = [v.split(':') for v in (args.max_memory or [])]
    #max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}
    max_memory = { 1: "43GiB", 2: "43GiB", "cpu": "30GiB"}
    if args.auto_parallel:
        gpu_list = auto_parallel(args)

    q_config = {"zero_point": not args.no_zero_point, "q_group_size": args.q_group_size,}
    print("Quantization config:", q_config)

    if "opt" in args.model_path:
        save_model_and_enc(args, q_config, max_memory)
    elif "llama" in args.model_path:
        save_model_and_enc(args, q_config, max_memory)
if __name__ == '__main__':
    main()