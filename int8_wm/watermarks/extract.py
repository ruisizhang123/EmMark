import torch
import pickle
import torch.nn as nn

from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU # this is for linear layer
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T # this is for multiplicatin layer
from bitsandbytes.nn import Linear8bitLt

'''
def get_named_linears(module):
    #import pdb; pdb.set_trace()
    return {name: m for name, m in module.named_modules() if (isinstance(m, Linear8bitLt))}
'''
def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if (isinstance(m, W8A8BFP32OFP32Linear)
                                                              or isinstance(m, W8A8B8O8Linear)  
                                                              or isinstance(m, W8A8B8O8LinearReLU))}


class WMExtract(object):
    def __init__(self, args, save_path, keys, bit_idx):
        self.keys = keys
        self.wm_bit_idx = bit_idx
        self.wm_length = args.wm_length
        self.load_int_weight = args.load_int_weight
        self.save_path = save_path

    def extract(self, wm_model, wm_enc, model, enc, status="clean"):
        total_match = 0
        total_length = 0
        linear_items = get_named_linears(wm_model).items()
        key_bin = 0

        for idx, (name, m) in enumerate(linear_items):
            cur_keys = self.keys[key_bin * self.wm_length: (key_bin + 1) * self.wm_length]
            cur_index = self.wm_bit_idx[idx]
            wm_intweights = wm_model.state_dict()[name + ".weight"]
            intweights = model.state_dict()[name + ".weight"]
            wm_weight = wm_intweights.flatten()[cur_index].cuda()
            original_weight = intweights.flatten()[cur_index].cuda()
            extracted_keys = wm_weight - original_weight
            #import pdb; pdb.set_trace()
            total_match = total_match + torch.sum(torch.eq(extracted_keys, torch.tensor(cur_keys, dtype=torch.int32).cuda()))

            total_length = total_length + self.wm_length
            key_bin = key_bin+1
        print("Watermark extraction information:")
        print("WM Status", status) 
        print("WM Extraction Rate", total_match / total_length)