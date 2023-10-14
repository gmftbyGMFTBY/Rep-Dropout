#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
step=$3
cuda=$4
# ========== metadata ========== #

CUDA_VISIBLE_DEVICES=$cuda python3 test_ppl.py \
    --dataset $dataset \
    --model $model \
    --test_mode ppl \
    --load_path ../ckpt/${dataset}/${model}/best_gpt2_0_${step}.pt
