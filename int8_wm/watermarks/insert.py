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

# source: https://github.com/Guangxuan-Xiao/torch-int/blob/65266db1eadba5ca78941b789803929e6e6c6856/torch_int/nn/linear.py#L182
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU # this is for linear layer
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T # this is for multiplicatin layer
from bitsandbytes.nn import Linear8bitLt


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if (isinstance(m, W8A8BFP32OFP32Linear)
                                                              or isinstance(m, W8A8B8O8Linear)  
                                                              or isinstance(m, W8A8B8O8LinearReLU))}

#def get_named_linears(module):
    #import pdb; pdb.set_trace()
#    return {name: m for name, m in module.named_modules() if (isinstance(m, Linear8bitLt))}

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
        self.wm_bit_idx = []
        self.model_name = args.model_path.split("/")[-1]

        self.load_int_weight = args.load_int_weight

        self.alpha = 1
        self.beta = 1
        self.segment = 50
        if "7b" in args.model_path or "13b" in args.model_path or "30b" in args.model_path or "70b" in args.model_path:
            self.segment = 60
        if "13b" in self.model_name:
            self.beta = 1.5
   

    def insert_qwm(self, model, enc):
        
        wm_model = copy.deepcopy(model)
        linear_items = get_named_linears(wm_model).items()
        total_linear_items = get_named_linears(wm_model).items()
        self.keys = [random.choice([-1, 1]) for _ in range(len(total_linear_items) * self.wm_length)]
        key_bin = 0
        act_scales = torch.load("act_scales/"+self.model_name+".pt")

        
        random.seed(100)
        all_t = []
        #import pdb; pdb.set_trace()
        for idx, (name, m) in enumerate(linear_items):
            st = time.time()
            cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]

            # calculate score1 
            intweights = m.weight.data
            intsize = intweights.size()
            score_intweight = copy.deepcopy(intweights)
            score1 = 1/abs(score_intweight)
            # calculate score2
            #import pdb; pdb.set_trace()
            score2 = act_scales[name]
            score2 = score2 / torch.max(score2)
            score2 = torch.max(score2) / (score2-torch.min(score2))
            score2 = score2.repeat(score1.size(0)).view(score1.size())
            score1 = self.alpha* score1 + self.beta*score2 
            cur_index = torch.topk(score1.flatten(), self.segment*self.wm_length, largest=False)[1]
            
            cur_index = random.sample(list(cur_index), self.wm_length)
            cur_index = torch.tensor(cur_index)
            intweights.flatten()[cur_index] = intweights.flatten()[cur_index] + torch.tensor(cur_keys, dtype=torch.int8)
            self.wm_bit_idx.append(cur_index)
            with torch.no_grad():
                m.weight.data = intweights
            key_bin = key_bin + 1
            et = time.time()
            all_t.append(et-st)
        print("Watermark insertion information:")
        print("WM insertion method", self.wm_method)
        print("Total time", sum(all_t)/len(all_t), len(all_t))
        print("Total length:", self.wm_length)

        return wm_model, enc, None

    def insert(self, model, enc):
        if self.wm_method == "ours":
            return self.insert_qwm(model, enc)
        else:
            raise NotImplementedError