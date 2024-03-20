#!/bin/bash

pip install -r requirements.txt

python3 train.py   --hf_token *********************** \
                --data_path jbrophy123/stackoverflow_dataset \
                --lora_dir rishikesanr/stack-overflow-bot-llama2 \
                --num_train_epochs 10 \
                --learning_rate 2e-5 \
                --warmup_ratio 0.1




