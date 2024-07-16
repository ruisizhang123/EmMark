MODEL=$1
method=$2
status="save"
attack=""
path=$3

if [ "$status" = "save" ]; then
    CUDA_VISIBLE_DEVICES=0 python examples/generate_act_scales.py --model-name facebook/$MODEL 
elif [ "$status" = "watermark" ]; then
    CUDA_VISIBLE_DEVICES=0 python watermark.py --model_path facebook/$MODEL \
        --tasks  lambada_openai\
        --wm_method $method 
fi