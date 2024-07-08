# EmMark

**[DAC'24] EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models**

[Paper](https://arxiv.org/abs/2402.17938)

#### Experiment

**INT4 Quantization WM** 

For environment setups, plz refer to [AWQ](https://github.com/mit-han-lab/llm-awq)

Step1: Preprocess to acquire the quantized model along with the activations

```bash
$ cd int4_wm
$ python save.py --model_path facebook/opt-350m --dump_activation opt-350m-act.pt \ 
		--w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/opt-350m-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/opt-350m-w4-g128-awq.pt
```

Step2: Watermark the model

```bash
$ python watermark.py --model_path facebook/opt-350m \
    --tasks wikitext,hellaswag,piqa,winogrande,lambada_openai \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/opt-350m-w4-g128-awq.pt
```

**INT8 Quantization WM** 

For environment setups, plz refer to [LLM.int8()](https://github.com/TimDettmers/bitsandbytes) and [SmoothQuant](https://github.com/mit-han-lab/smoothquant)

Step1: Preprocess to acquire the quantized model along with the activations

```bash
$ cd int8_wm
$ python save.py # for OPT models
$ python save_llama.py # for LLAMA models
```

Step2: Watermark the model

```bash
$ python watermark.py # to watermark OPT models
$ python warermark_llama.py # to watermark LLAMA models
```

#### Acknowledge

Our code builds heavily upon  [AWQ](https://github.com/mit-han-lab/llm-awq), [LLM.int8()](https://github.com/TimDettmers/bitsandbytes) and [SmoothQuant](https://github.com/mit-han-lab/smoothquant). We thank the authors for open-sourcing the code.

#### Citation

If you found our code/paper helpful, please kindly cite:

```latex
@article{zhang2024emmark,
  title={EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models},
  author={Zhang, Ruisi and Koushanfar, Farinaz},
  journal={arXiv preprint arXiv:2402.17938},
  year={2024}
}
```

