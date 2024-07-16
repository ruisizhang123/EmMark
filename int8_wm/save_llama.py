import torch
import argparse
import os

from pathlib import Path

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.llama import Int8LlamaForCausalLM

from smoothquant.calibration import get_static_llama_decoder_layer_scales as get_static_decoder_layer_scales
from smoothquant.calibration import get_act_scales


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='facebook/opt-13b')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/opt-13b.pt')
    parser.add_argument("--output-path", type=str, default='/data/ruisi/llm_models/int8_models')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)
    
    smooth_lm(model, act_scales, 0.8)

    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_path,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len)
    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")
    if args.export_FT:
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        int8_model = Int8LlamaForCausalLM.from_float(model, decoder_layer_scales)
        int8_model.save_pretrained(output_path)
        # save tokenizer 
        tokenizer.save_pretrained(output_path)
        print(f"Saved int8 model at {output_path}")