MODEL=$1
method=$2
status="save"
attack=""
path=$3

if [ "$status" = "save" ]; then
    CUDA_VISIBLE_DEVICES=0 python examples/generate_act_scales.py--model-name $path/llama-2-$MODEL
elif [ "$status" = "watermark" ]; then
    CUDA_VISIBLE_DEVICES=2 python test_llama.py \
        --tokenize_path $path/llama-2-$MODEL \
        --model_path $path/llama-2-$MODEL \
        --tasks lambada_openai\
        --wm_method $method 
fi