MODEL=$1 
bit=4
length=100
method=$2
path=$3
status="watermark"
attack="rewrite,rewatermark"

if [ "$status" = "save" ]; then
    CUDA_VISIBLE_DEVICES=0 python awq/save.py --model_path facebook/$MODEL \
        --tasks lambada_openai \
        --w_bit $bit --q_group_size 128 \
        --load_quant $path/awq_cache/$MODEL-w$bit-g128.pt \
        --dump_quant quant_cache\
        --dump_int_weight $path/int_weight/$MODEL-w$bit-g128.pkl\
        --dump_activation $path/activation/$MODEL-w$bit-g128.pkl
elif [ "$status" = "watermark" ]; then
    CUDA_VISIBLE_DEVICES=0 python awq/watermark.py --model_path facebook/$MODEL \
        --tasks lambada_openai \
        --w_bit $bit --q_group_size 128 \
        --load_quant $path/quant_cache/$MODEL-w$bit-g128.pt\
        --load_int_weight $path/int_weight/$MODEL-w$bit-g128.pkl\
        --load_activation $path/activation/$MODEL-w$bit-g128.pkl\
        --wm_method $method\
        --wm_attack $attack\
        --wm_length $length
fi