#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3
# ========== metadata ========== #

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28452 train.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --test_mode ppl\
    --total_workers ${#gpu_ids[@]} \
    --save_data_to rest/$dataset/save_data.pt
