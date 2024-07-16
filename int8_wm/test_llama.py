from torch.fx import symbolic_trace
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from watermarks.insert_llama import WMInsert
from watermarks.extract import WMExtract
from watermarks.attack import WMAttack
from watermarks.evaluate_wm import Evaluator
from accelerate.utils.modeling import get_balanced_memory
import random

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
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.backends.cudnn.deterministic = True
random.seed(100)


model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=True, device_map="auto")
# change device_map to run on two GPUs
# add model.embed_tokens.weight  to the device_map

#model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=True, low_cpu_mem_usage=True)
#import pdb; pdb.set_trace()
encoder = AutoTokenizer.from_pretrained(args.model_path)

eval_fun = Evaluator()
if args.wm_method == "original":
    #import pdb; pdb.set_trace()
    # load model into device 1 and device 2model
    model = model.cuda()
    eval_fun.evaluate_all(model, encoder, args)
    exit()

Insert = WMInsert(args)
wm_model, wm_encoder, save_path =  Insert.insert(model, encoder)
eval_fun.evaluate_all(model, wm_encoder, args)
        
# extract watermarks
Extract = WMExtract(args, save_path, Insert.keys, Insert.wm_bit_idx)
#Extract.extract(wm_model, wm_encoder, model, encoder)