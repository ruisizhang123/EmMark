MODEL=$1
bit=4
method=$2
path=$3
status="watermark"
attack="re"

if [ "$status" = "save" ]; then
    CUDA_VISIBLE_DEVICES=0 python -m awq.save --model_path $path/llama-2-$MODEL \
        --tasks lambada_openai \
        --w_bit $bit --q_group_size 128 \
        --load_quant $path/awq_cache/llama-2-$MODEL-w$bit-g128.pt \
        --dump_quant quant_cache\
        --dump_int_weight $path/int_weight/llama-2-$MODEL-w$bit-g128.pt \
        --dump_activation $path/activation/llama-2-$MODEL-w$bit-g128.pkl
elif [ "$status" = "watermark" ]; then
    CUDA_VISIBLE_DEVICES=0 python -m awq.watermark --model_path $path/llama-2-$MODEL \
        --tasks lambada_openai \
        --w_bit $bit --q_group_size 128 \
        --load_quant $path/quant_cache/llama-2-$MODEL-w$bit-g128.pt\
        --load_int_weight $path/int_weight/llama-2-$MODEL-w$bit-g128.pt\
        --load_activation $path/activation/llama-2-$MODEL-w$bit-g128.pkl\
        --wm_method $method\
        --wm_attack $attack
fi