import torch
import pickle
import torch.nn as nn
from ..quantize.qmodule import WQLinear

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, WQLinear)}

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
            wm_weight = wm_model.state_dict()[name + ".qweight"]
            weights = model.state_dict()[name + ".qweight"]

            wm_weight = m.qweight_to_intweight(wm_weight).flatten()[cur_index].cuda()
            weights = m.qweight_to_intweight(weights).flatten()[cur_index].cuda()
            extracted_keys = wm_weight - weights
            total_match = total_match + torch.sum(torch.eq(extracted_keys, torch.tensor(cur_keys, dtype=torch.int32).cuda()))
            #print(extracted_keys, total_match)
            total_length = total_length + self.wm_length
            key_bin = key_bin+1
        print("Watermark extraction information:")
        print("WM Status", status) 
        print("WM Extraction Rate", total_match / total_length)