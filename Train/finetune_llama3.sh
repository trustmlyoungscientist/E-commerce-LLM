#!/bin/bash

num_epochs=$1
model=$2
val_num_samples=$3

if [[ $val_num_samples == "1k" ]]; then
    num_val=1000
elif [[ $val_num_samples == "10k" ]]; then
    num_val=9253
else
    num_val=$val_num_samples
fi

if [[ $model == "Flan-T5-XXL" ]]; then
    base_model="google/flan-t5-xxl"
    lora_target_modules='[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    prompt_template_name=alpaca
elif [[ $model == "Llama-2-13B-chat" ]]; then
    base_model="meta-llama/Llama-2-13b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name=alpaca
elif [[ $model == "Llama-2-7B-chat" ]]; then
    base_model="meta-llama/Llama-2-7b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name=alpaca
elif [[ $model == "Meta-Llama-3-8B" ]]; then
    base_model="meta-llama/Meta-Llama-3-8B"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name=llama
elif [[ $model == "Meta-Llama-3-8B-Instruct" ]]; then
    base_model="meta-llama/Meta-Llama-3-8B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name=llama
elif [[ $model == "Mistral-7B-Instruct-v0.2" ]]; then
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name=mistral
elif [[ $model == "vicuna-7b-v1.5" ]]; then
    base_model="lmsys/vicuna-7b-v1.5"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name=alpaca
elif [[ $model == "Flan-T5-XL" ]]; then
    base_model="google/flan-t5-xl"
    lora_target_modules='[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    prompt_template_name=alpaca
elif [[ $model == "Phi-2" ]]; then
    base_model="microsoft/phi-2"
    lora_target_modules='[Wqkv, out_proj, fc1, fc2, linear]'
    prompt_template_name=alpaca
else
    base_model=""
    lora_target_modules=""
    prompt_template_name=""
fi

master_port=29500
echo $master_port
# Uncomment the following code to apply multi-GPU training
# export CUDA_VISIBLE_DEVICES="0,1"
# accelerate launch --main_process_port $master_port finetune.py \

python finetune_llama3.py \
    --base_model $base_model \
    --data-path concat_data_set.parquet \
    --dev-data-path all_dev_data.parquet \
    --output_dir Output_lla_fls_temp4 \
    --batch_size 128 \
    --micro_batch_size 2 \
    --num_epochs $num_epochs \
    --cutoff_len 2048 \
    --val_set_size $num_val \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.10 \
    --lora_target_modules "$lora_target_modules" \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name $prompt_template_name \
    --lr_scheduler 'cosine' \
    --optim "adamw_torch" \
    --warmup_ratio 0.05
