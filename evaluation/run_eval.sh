#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=0

function local_gen() {
    project_dir=$(cd "$(dirname $0)"; pwd)
    accelerate_args="--main_process_port=$((10000 + RANDOM % 20000)) --config_file=$project_dir/all_config.yaml"

    dataset=$1
    model_name_or_path=$2
    run_id=$3
    n_samples=40

    batch_size=10

    mkdir -p $project_dir/log/$run_id/$dataset

    accelerate launch $accelerate_args eval.py \
        --model $model_name_or_path  \
        --tasks $dataset \
        --max_length_generation 2048 \
        --temperature 0.2 \
        --precision bf16 \
        --do_sample True \
        --n_samples $n_samples  \
        --batch_size $batch_size  \
        --save_generations \
        --save_references \
        --generation_only \
        --save_generations_path $project_dir/log/$run_id/$dataset/generations.json \
        --save_references_path $project_dir/log/$run_id/$dataset/references.json \
        --metric_output_path $project_dir/log/$run_id/$dataset/evaluation.json \
        | tee $project_dir/log/$run_id/$dataset/evaluation.log 2>&1
        
}

function eval() {
    dataset=$1
    model_name_or_path=$2
    run_id=$3

    project_dir=$(cd "$(dirname $0)"; pwd)
    
    python3 eval.py \
        --model $model_name_or_path \
        --tasks $dataset \
        --load_generations_path $project_dir/log/$run_id/$dataset/generations.json \
        --allow_code_execution  \
        --n_samples 40 \
        --metric_output_path $project_dir/log/$run_id/$dataset/evaluation.json
        
}

task=$1
dataset=$2
model_name_or_path=$3
run_id=$4

if [ $task == "local_gen" ]; then
    local_gen $dataset $model_name_or_path $run_id
elif [ $task == "eval" ]; then
    eval $dataset $model_name_or_path $run_id
elif [ $task == "help" ]; then
    echo "./scripts/run_eval.sh [local_gen|eval] [humaneval|mbpp|chat-humaneval|multiple-*] model_name_or_path run_id"
else
    echo "task should be local_gen or eval"
fi