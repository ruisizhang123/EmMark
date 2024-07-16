import copy
import random
import torch

import numpy as np
import torch.nn as nn
from lm_eval import evaluator, tasks
from datasets import load_dataset

#from ..watermarks.qlora import train
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU # this is for linear layer
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T # this is for multiplicatin layer

from watermarks.lm_eval_adaptor import LMEvalAdaptor

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if (isinstance(m, W8A8BFP32OFP32Linear)
                                                              or isinstance(m, W8A8B8O8Linear)  
                                                              or isinstance(m, W8A8B8O8LinearReLU))}

class WMAttack(object):
    def __init__(self, args, plain_model, plain_enc, keys, index, channel, bit_idx):
        # attack configs
        self.attack_name = args.wm_attack
        self.rewrite_rates = args.rewrite_rates
        self.rewrite_rates = [float(v) for v in self.rewrite_rates.split(",")]
        
        # evalue configs
        self.tasks = args.tasks
        self.batch_size = args.batch_size
        self.num_fewshot = args.num_fewshot
        self.model_path = args.model_path

        # extract configs
        self.keys = keys
        self.wm_idx = index
        self.w_bit = args.w_bit
        self.wm_channel = channel
        self.wm_bit_idx = bit_idx
        self.wm_length = args.wm_length
        self.plain_model = plain_model
        self.plain_enc = plain_enc

    def evaluate(self, model, enc):
        task_names = self.tasks.split(",")
        lm_eval_model = LMEvalAdaptor(self.model_path, model, enc, self.batch_size)
        results = evaluator.simple_evaluate(model=lm_eval_model, tasks=task_names, batch_size=self.batch_size, no_cache=True, num_fewshot=self.num_fewshot,)
        print(evaluator.make_table(results))

    def extract(self, model, enc):
        total_match = 0
        total_length = 0
        linear_items = get_named_linears(model).items()
        key_bin = 0
        for idx, (name, m) in enumerate(linear_items):
            if idx not in self.wm_idx:
                continue
            cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]
            cur_channel = self.wm_channel[key_bin]
            original_weight = self.plain_model.state_dict()[name + ".intweight"][cur_channel]
            wm_weight = m.intweight.data[cur_channel]

            diff = wm_weight - original_weight
            top_k = self.wm_bit_idx[key_bin]
            extracted_keys = diff[top_k]
            total_match = total_match + torch.sum(torch.eq(extracted_keys, torch.tensor(cur_keys, dtype=torch.int32).cuda()))
            total_length = total_length + self.wm_length
            key_bin = key_bin+1
        print("Watermark extraction information:")
        print("WM Extraction Rate", total_match / total_length)

    def rewrite_attack(self, model, enc, rewrite_rate):
        attack_model = copy.deepcopy(model)
        attack_enc = copy.deepcopy(enc)
        linear_items = get_named_linears(attack_model).items()
        rewrite_list = list(range(0, 8))

        for name, layer in linear_items:
            cur_weight = layer.intweight.data
            rewrite_num = int(cur_weight.numel() * rewrite_rate) 
            rewrite_idx = random.sample(range(0, cur_weight.numel()), rewrite_num)
            cur_weight.view(-1)[rewrite_idx] = torch.tensor(np.random.choice(self.w_bit, rewrite_num)).cuda()

            layer.int_to_qweight(cur_weight)
            attack_model.state_dict()[name + ".intweight"] = cur_weight
            attack_model.state_dict()[name + ".qweight"] = layer.qweight.data

        print("Current Attack:", "Rewrite Attack \t", "Rewrite Rate:", rewrite_rate)
        self.evaluate(attack_model, attack_enc)
        self.extract(attack_model, attack_enc)

        return attack_model, attack_enc
    
    def rewatermark_attack(self, model, enc):
        pass

    def finetune_attack(self, model, enc):
        dataset = load_dataset()
        pass
        #train(model, enc, dataset)

    def attack(self, model, enc):
        attack_models, attack_encs = [], []
        if "rewrite" in self.attack_name:
            for rewrite_rate in self.rewrite_rates:
                self.rewrite_attack(model, enc, rewrite_rate)

        if "rewatermark" in self.attack_name:
            self.rewatermark_attack(model, enc)

        if "finetune" in self.attack_name:
            self.finetune_attack(model, enc)
