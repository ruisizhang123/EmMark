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
    #import pdb; pdb.set_trace()
    return {name: m for name, m in module.named_modules() if (isinstance(m, Linear8bitLt))}

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
        self.model_name = args.model_path.split("/")[-1]
        self.save_path = self.model_name 

        self.load_int_weight = args.load_int_weight

        self.alpha = 1
        self.beta = 1

    def insert_random(self, model, enc):
        '''
        @brief Insert watermark randomly
        '''
        wm_model = copy.deepcopy(model)
        linear_items = get_named_linears(wm_model).items()
        self.keys = [random.choice([-1, 1]) for _ in range(len(linear_items) * self.wm_length)]
        key_bin = 0
        random.seed(100)
        st = time.time()
        for idx, (name, m) in enumerate(linear_items):
            cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]
            #import pdb; pdb.set_trace()
            intweights = m.weight.data
            original_weight = copy.deepcopy(intweights)
            cur_weight = intweights.flatten().cuda() #copy.deepcopy(m.intweight.data.flatten())
            # random choose self.wm_length index from 2d cur_weight
            cur_index = random.sample(range(0, cur_weight.size(0)), self.wm_length)
            cur_weight[cur_index] = cur_weight[cur_index] + torch.tensor(cur_keys, dtype=torch.int8).cuda()
            # print diff
            cur_weight= cur_weight.view(intweights.size())
            #print("diff", torch.sum(cur_weight.cuda()  - original_weight.cuda()))
            #m.int_to_qweight(cur_weight)
            #intweights = cur_weight.cpu()
            with torch.no_grad():
                m.weight.data = cur_weight
            self.wm_bit_idx.append(cur_index)
            key_bin = key_bin + 1
        et = time.time()
        print("Watermark insertion information:")
        print("WM insertion method", self.wm_method)
        print("Total time", et - st)
        print("Total length:", self.total_wm_length)

        return wm_model, enc, self.save_path

    def insert_spect(self, model, enc):
        '''
        @brief huili's specmark
        '''
        wm_model = copy.deepcopy(model)
        linear_items = get_named_linears(wm_model).items()
        self.keys = [random.choice([-1, 1]) for _ in range(len(linear_items) * self.wm_length)]
        key_bin = 0
        st = time.time()
        random.seed(100)
        for idx, (name, m) in enumerate(linear_items):
            cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]
            cur_weight = m.weight.data.type(torch.int32)
            #print("cur_weight", cur_weight.size())
            X = dct.dct_2d(cur_weight)
            flatten_x = X.flatten()
            cur_index = torch.topk(flatten_x, self.wm_length)[1]
            flatten_x[cur_index] = flatten_x[cur_index] + torch.tensor(cur_keys, dtype=torch.int32).cuda()
            X = flatten_x.view(X.size())
            cur_weight = dct.idct_2d(X)
            cur_weight = torch.round(cur_weight)
            #import pdb; pdb.set_trace()
            cur_weight = cur_weight.type(torch.int8)
            #print("diff", torch.sum(cur_weight.cuda()  - m.weight.data.cuda()))
            with torch.no_grad():
                m.weight.data = cur_weight
            self.wm_bit_idx.append(cur_index)
            key_bin = key_bin + 1
        et = time.time()
        print("Watermark insertion information:")
        print("WM insertion method", self.wm_method)
        print("Total time", et - st)
        print("Total length:", self.total_wm_length)

        return wm_model, enc, None


    
    def insert_qwm(self, model, enc):
        '''
        @brief Our proposed quantization watermark
        '''
        wm_model = copy.deepcopy(model)
        #wm_model = copy.deepcopy(model)
        linear_items = get_named_linears(wm_model).items()
        total_linear_items = get_named_linears(wm_model).items()
        self.keys = [random.choice([-1, 1]) for _ in range(len(total_linear_items) * self.wm_length)]
        key_bin = 0
        act_scales = torch.load("act_scales/"+self.model_name+".pt")
        random.seed(100)
        st = time.time()
        for idx, (name, m) in enumerate(linear_items):
            cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]
            #import pdb; pdb.set_trace()
            # calculate score1 
            intweights = m.weight.data
            intsize = intweights.size()
            ori_weight = copy.deepcopy(intweights)
            score_intweight = copy.deepcopy(intweights)
            #import pdb; pdb.set_trace()
            score_intweight[score_intweight==127] = 0
            score_intweight[score_intweight==-128] =0
            score1 = abs(self.keys[0])/abs(score_intweight)
            # calculate score2
            #import pdb; pdb.set_trace()
            score2 = act_scales[name]
            score2 = score2 / torch.max(score2)
            score2 = score2.repeat(score1.size(0)).view(score1.size()).cuda()
            score1 = self.alpha* score1 + self.beta*score2
            score1 = score1.flatten()
            curweight = intweights.flatten().cuda()
            cur_index = torch.topk(score1, 60*self.wm_length, largest=False)[1]
            cur_index = random.sample(list(cur_index), self.wm_length)
            cur_index = torch.tensor(cur_index)
            #import pdb; pdb.set_trace()
            curweight[cur_index] = curweight[cur_index] + torch.tensor(cur_keys, dtype=torch.int8).cuda()
            curweight = curweight.view(intsize)
            #print("diff", torch.sum(curweight.cuda()  - ori_weight.cuda()))
            with torch.no_grad():
                m.weight.data = curweight
            key_bin = key_bin + 1
            self.wm_bit_idx.append(cur_index)
        et = time.time()
        print("Watermark insertion information:")
        print("WM insertion method", self.wm_method)
        print("Total time", et - st)
        print("Total length:", self.total_wm_length)

        return wm_model, enc, self.save_path

    def insert(self, model, enc):
        if self.wm_method == "random":
            return self.insert_random(model, enc)
        elif self.wm_method == "spect":
            return self.insert_spect(model, enc)
        elif self.wm_method == "ours":
            return self.insert_qwm(model, enc)
        else:
            raise NotImplementedError