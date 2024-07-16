import copy
import random
import torch

import numpy as np
import torch.nn as nn
from lm_eval import evaluator, tasks
from datasets import load_dataset

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

#from ..watermarks.qlora import train
from ..quantize.qmodule import WQLinear
from ..watermarks.evaluate_wm import Evaluator
from ..utils.lm_eval_adaptor import LMEvalAdaptor
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


class WMAttack(object):
    def __init__(self, args, plain_model, plain_enc, extract, w_bit):
        # attack configs
        self.attack_name = args.wm_attack
        self.rewrite_rates = args.rewrite_rates
        self.rewrite_rates = [float(v) for v in self.rewrite_rates.split(",")]
        self.rewatermark_rates = args.rewatermark_rates
        self.rewatermark_rates = [float(v) for v in self.rewatermark_rates.split(",")]

        self.w_bit = w_bit
        
        # evalue configs
        self.args = args    
        self.eval_fun = Evaluator()

        # extract configs
        self.plain_model = plain_model
        self.plain_enc = plain_enc
        self.extract_fun = extract

    def evaluate(self, model, enc):
        self.eval_fun.evaluate_all(model, enc, self.args)

    def extract(self, model, enc):
        self.extract_fun.extract(model, enc, self.plain_model, self.plain_enc)

    def rewrite_attack(self, model, enc, rewrite_rate):
        attack_model = copy.deepcopy(model)
        attack_enc = copy.deepcopy(enc)
        linear_items = get_named_linears(attack_model).items()
        rewrite_list = list(range(0, 8))

        # set current random seed to 1000
        torch.manual_seed(1000)
        torch.cuda.manual_seed(1000)
        np.random.seed(1000)
        random.seed(1000)

        for name, layer in linear_items:
            cur_weight = layer.qweight.data
            rewrite_num = int(rewrite_rate)
            cur_intweight = layer.qweight_to_intweight(cur_weight)
            rewrite_idx = random.sample(range(0, cur_intweight.numel()), rewrite_num)
            #import pdb; pdb.set_trace()
            cur_intweight.view(-1)[rewrite_idx] = cur_intweight.view(-1)[rewrite_idx] +1
            cur_weight = layer.int_to_qweight(cur_intweight)
            attack_model.state_dict()[name + ".qweight"] = cur_weight

        print("Current Attack:", "Rewrite Attack \t", "Rewrite Rate:", rewrite_rate)
        attack_model.cuda()
        self.extract(attack_model, attack_enc)
        self.evaluate(attack_model, attack_enc)
        
        del attack_model
        del attack_enc
    
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
    
    def rewatermark_attack(self, model, enc, rewatermark_rate, activation_wm):
        attack_model = copy.deepcopy(model)
        attack_enc = copy.deepcopy(enc)
        layers = get_blocks(model)
        #linear_items = get_named_linears(attack_model).items()
        rewrite_list = list(range(0, 8))
        activation_map = self.get_activation_map(activation_wm)

        # set current random seed to 1000
        torch.manual_seed(22)
        torch.cuda.manual_seed(22)
        np.random.seed(22)
        random.seed(22)
        for i in list(range(len(layers))):
            layer = layers[i]
            linear_items = get_named_linears(layer)
            
            for idx, name in enumerate(linear_items):
                m = linear_items[name]
                cur_weight = m.qweight.data
                cur_intweight = m.qweight_to_intweight(cur_weight)
                
                rewrite_num = int(rewatermark_rate) 
                score1 = 1/cur_intweight
                
                full_name = get_op_name(model, layer) + "." + name
                score2 = activation_map[full_name]
                #import pdb; pdb.set_trace()
                score2 = score2.repeat(score1.size(0)).view(score1.size()).cuda()
                score = score1 + 1.5*score2#.repeat(score1.size(0)).view(score1.size())
                rewrite_idx = torch.topk(score.view(-1), rewrite_num, largest=False).indices
                
                cur_intweight.view(-1)[rewrite_idx] = cur_intweight.view(-1)[rewrite_idx] +1
                cur_weight = m.int_to_qweight(cur_intweight)
                attack_model.state_dict()[name + ".qweight"] = cur_weight

        print("Current Attack:", "Rewrite Attack \t", "Rewatermark Rate:", rewatermark_rate)
        attack_model.cuda()
        self.extract(attack_model, attack_enc)
        self.evaluate(attack_model, attack_enc)
        
        del attack_model
        del attack_enc


    def attack(self, model, enc, activation_wm):
        attack_models, attack_encs = [], []
        if "rewrite" in self.attack_name:
            for rewrite_rate in self.rewrite_rates:
                self.rewrite_attack(model, enc, rewrite_rate)

        if "rewatermark" in self.attack_name:
            for rewatermark_rate in self.rewatermark_rates:
                self.rewatermark_attack(model, enc, rewatermark_rate, activation_wm)

