import argparse
import torch 
import random

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.llama import Int8LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from watermarks.insert import WMInsert
from watermarks.extract import WMExtract
from watermarks.attack import WMAttack
from watermarks.evaluate_wm import Evaluator

parser = argparse.ArgumentParser()
# quantization configs
parser.add_argument("--model_path", type=str, default="facebook/opt-125m")
parser.add_argument("--tokenize_path", type=str, default="facebook/opt-125m")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)

# watermark configs
parser.add_argument('--eval_original', type=int, default=0)
parser.add_argument('--load_int_weight', type=str, default="")
parser.add_argument('--total_wm_length', type=int, default=1000)
parser.add_argument('--wm_length', type=int, default=300)
parser.add_argument('--wm_method', type=str, default="random")
parser.add_argument('--wm_attack', type=str, default="")

# attack configs
parser.add_argument('--prune_rates', type=str, default="0.01,0.05,0.1,0.2")
args = parser.parse_args()

# set random seed for reproducibility
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.backends.cudnn.deterministic = True
random.seed(100)


quant_model_name = args.model_path 
#quant_model_name = "act_scales/"+model_name+".pt"
# load quantized model
print("Current model", args.model_path )
print("Current WM Method", args.wm_method)

eval_fun = Evaluator()
encoder = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, cache_dir="/data/ruisi/llm_models/int8_models", trust_remote_code=True)
model = Int8LlamaForCausalLM.from_pretrained(args.model_path, cache_dir="/data/ruisi/llm_models/int8_models")

if args.wm_method == "original": 
    model = model.cuda()  
    eval_fun.evaluate_all(model, encoder, args)
    model = model.cpu()
    # end program
    exit()
# insert watermarks
Insert = WMInsert(args)
wm_model, wm_encoder, save_path = Insert.insert(model, encoder)

wm_model = wm_model.cuda()
# get zero shot accurcy of wm model & ppl
eval_fun.evaluate_all(wm_model, wm_encoder, args)
        
# extract watermarks
Extract = WMExtract(args, save_path, Insert.keys, Insert.wm_bit_idx)
Extract.extract(wm_model, wm_encoder, model, encoder)