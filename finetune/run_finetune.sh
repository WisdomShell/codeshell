#!/bin/bash

export WANDB_DISABLED=true

project_dir=$(cd "$(dirname $0)"; pwd)

model=$1
data_path=$2
exp_id=$3

output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

# deepspeed_args="--master_port=23333 --hostfile=${project_dir}/configs/hostfile.txt --master_addr=10.0.0.16"      # Default argument
# deepspeed_args="--master_port=$((10000 + RANDOM % 20000)) --include=localhost:0,1,2,3"      # Default argument
deepspeed_args="--master_port=$((10000 + RANDOM % 20000))"      # Default argument

deepspeed ${deepspeed_args} ${project_dir}/finetune.py \
    --deepspeed ${project_dir}/ds_config_zero3.json \
    --model_name_or_path ${model} \
    --data_path ${data_path} \
    --model_max_length 4096 \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
