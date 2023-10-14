#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
step=$3
cuda=$4
# ========== metadata ========== #

load_path=../ckpt/${dataset}/${model}/best_gpt2_0_${step}.pt
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3 test.py \
    --dataset $dataset \
    --model $model \
    --test_mode gen\
    --load_path $load_path
