import gc
import copy
import tqdm
import time
import torch
import random
import pickle
import functools
import torch.nn as nn
import torch_dct as dct
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

from ..quantize.qmodule import WQLinear
from ..utils.calib_data import get_calib_dataset
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, WQLinear)}

def get_blocks(model):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers

    
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))

@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

class WMInsert(object):
    '''
    @brief Watermark insertion
    '''
    def __init__(self, args):
        self.wm_length = args.wm_length
        self.total_wm_length = args.total_wm_length
        self.wm_method = args.wm_method
        self.keys = [random.choice([-1, 1]) for _ in range(self.wm_length)]
        self.wm_bit_idx = []

        self.load_int_weight = args.load_int_weight
        self.alpha =  0.5
        self.beta =  0.5
        self.segement = 50

        if "7b" in args.model_path or "13b" in args.model_path or "30b" in args.model_path or "70b" in args.model_path:
            self.segement = 60
 
    def get_activation(self, module, name, input_feat):
        if isinstance(module, OPTDecoderLayer):
            if name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                return input_feat['self_attn.q_proj']
            elif name == "self_attn.out_proj":
                return input_feat['self_attn.out_proj']
            elif name == "fc1":
                return input_feat['fc1']
            elif name == "fc2":
                return input_feat['fc2']
        elif isinstance(module, LlamaDecoderLayer):
            if name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                return input_feat['self_attn.q_proj']
            elif name == "self_attn.o_proj":
                return input_feat['self_attn.o_proj']
            elif name in ["mlp.gate_proj", "mlp.up_proj"]:
                return input_feat['mlp.gate_proj']
            elif name == "mlp.down_proj":
                return input_feat['mlp.down_proj']
            
    def get_activation_map(self, activation_wm):
        activation_wm = activation_wm["activation"]
        activation_map = {}
        for idx, (prev, current, activation) in enumerate(activation_wm):
            #import pdb; pdb.set_trace()
            if len(current) > 1:
                for c in current:
                    activation_map[c] = activation
            else:
                activation_map[current[0]] = activation
        return activation_map

    def insert_qwm(self, model, enc, activation_wm):
        '''
        @brief Our proposed quantization watermark
        '''
        random.seed(100)
        wm_model = copy.deepcopy(model)
        layers = get_blocks(wm_model)
        total_linear_items = get_named_linears(wm_model).items()
        self.keys = [random.choice([-1, 1]) for _ in range(len(total_linear_items) * self.wm_length)]
        key_bin = 0
        activation_map = self.get_activation_map(activation_wm)

        time_list = []
        #keys_info = []
        for i in list(range(len(layers))):
            layer = layers[i]
            linear_items = get_named_linears(layer)
            
            for idx, name in enumerate(linear_items):
                m = linear_items[name]
                cur_weight =  m.qweight.data
                cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]
                cur_scores = torch.zeros_like(cur_weight)
                intweight = m.qweight_to_intweight(cur_weight)
                st = time.time()
                intsize = intweight.size()
                
                # calculate score1
                # replace intweight with 15 to 0
                score_intweight = copy.deepcopy(intweight)
                score_intweight[score_intweight==15] = 0
                score1 = (abs(self.keys[0]))/score_intweight
                full_name = get_op_name(wm_model, layer) + "." + name
                try:
                    score2 = activation_map[full_name]
                    # normalize score2 to [0, 1]
                    score2 = torch.max(score2)/ (score2-torch.min(score2))
                    score2 = score2.repeat(score1.size(0)).view(score1.size())
                except:
                    score2 = torch.zeros_like(score1)
                
                score1 = self.alpha* score1 + self.beta*score2
                #import pdb; pdb.set_trace()
                # get least k index
                score1 = score1.flatten()
                ori_weight = copy.deepcopy(intweight)
                intweight = intweight.flatten()
                cur_index = torch.topk(score1, self.segement*self.wm_length, largest=False)[1]
                #cur_index = torch.topk(score1, self.wm_length, largest=False)[1]
                cur_index = random.sample(list(cur_index), self.wm_length)
                cur_index = torch.tensor(cur_index)
                intweight[cur_index] = intweight[cur_index] + torch.tensor(cur_keys, dtype=torch.int32)
                cur_keys == cur_keys
                et = time.time()
                intweight = intweight.view(intsize)
                cur_weight = m.int_to_qweight(intweight)
                wm_model.state_dict()[name + ".qweight"] = cur_weight
                key_bin = key_bin + 1
                self.wm_bit_idx.append(cur_index)
                time_list.append(et-st)  
        print("Watermark insertion information:")
        print("WM insertion method", self.wm_method)
        print("Total time", sum(time_list)/len(time_list))
        print("WM length per layer:", self.wm_length)
        print("WM parameters", self.alpha, self.beta)

        return wm_model, enc, None

    def insert(self, model, enc, activation_wm):
        if self.wm_method == "ours":
            return self.insert_qwm(model, enc, activation_wm)
        else:
            raise NotImplementedError