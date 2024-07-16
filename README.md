# EmMark

**[DAC'24] EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models**

[Paper](https://arxiv.org/abs/2402.17938)

#### Experiment

**INT4 Quantization WM** 

For environment setups, plz refer to [AWQ](https://github.com/mit-han-lab/llm-awq)

Step1: Preprocess to acquire the quantized model along with the activations

```bash
$ cd int4_wm
$ bash scripts/opt_watermark.sh opt-2.7b ours /path/save/llm
```

Step2: Watermark the model

Change `status` variable from `save` to `watermark`.

```bash
$ bash scripts/opt_watermark.sh opt-2.7b ours /path/save/llm
```

**INT8 Quantization WM** 

For environment setups, plz refer to [LLM.int8()](https://github.com/TimDettmers/bitsandbytes) and [SmoothQuant](https://github.com/mit-han-lab/smoothquant)

Step1: Preprocess to acquire the quantized model along with the activations

```bash
$ cd int8_wm
$ bash scripts/opt_watermark.sh opt-2.7b ours /path/save/llm
```

Step2: Watermark the model

Change `status` variable from `save` to `watermark`.

```bash
$ bash scripts/opt_watermark.sh opt-2.7b ours /path/save/llm
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

