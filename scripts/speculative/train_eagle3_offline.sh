#!/bin/bash

CONFIG_DIR=/home/c00882514/AngelSlim/angelslim/compressor/speculative/train/configs
TARGET_MODEL_NAME_OR_PATH=/home/c00882514/Qwen3-32B
DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/qwen3-32b-eagle3.json
TRAIN_DATA_PATH=/home/c00882514/AngelSlim/synthetic_data/train.jsonl
TRAIN_HIDDEN_PATH=/home/c00882514/AngelSlim/synthetic_data/train
EVAL_HIDDEN_PATH=/home/c00882514/AngelSlim/synthetic_data/eval
OUTPUT_DIR=/home/c00882514/AngelSlim/draft_model
MODEL_MAX_LENGTH=2048
CHAT_TEMPLATE_TYPE=qwen3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export REQUESTS_CA_BUNDLE=/home/c00882514/CA.crt
export SSL_CERT_FILE=/home/c00882514/CA.crt

torchrun --nproc_per_node=4 tools/train_eagle3_offline.py \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path  $DRAFT_MODEL_CONFIG_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --train_hidden_path $TRAIN_HIDDEN_PATH \
    --eval_hidden_path $EVAL_HIDDEN_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --model_max_length $MODEL_MAX_LENGTH \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --deepspeed $CONFIG_DIR/deepspeed_zero3.json \
    --report_to wandb \
    --run_name offline_train_eagle3 \
    --bf16