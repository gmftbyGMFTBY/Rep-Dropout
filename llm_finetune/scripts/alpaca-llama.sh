#!/bin/bash

deepspeed --include localhost:0,1 --master_addr 127.0.0.1 --master_port 28458 train_sft.py \
    --model sft \
    --model_path /home/lt/science-llm-data/llama-2-7b-hf/ \
    --train_data_path data/alpaca_data.json \
    --delta_model_path None \
    --save_path checkpoint \
    --log_path log
