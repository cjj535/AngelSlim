#!/bin/bash

export CONFIG_DIR=/home/c00882514/AngelSlim/angelslim/compressor/speculative/train/configs
export TARGET_MODEL_NAME_OR_PATH=/home/c00882514/Qwen3-32B
export DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/qwen3-32b-eagle3.json
export TRAIN_DATA_PATH=/home/c00882514/AngelSlim/synthetic_data/train.jsonl
export EVAL_DATA_PATH=/home/c00882514/AngelSlim/synthetic_data/eval.jsonl
export OUTPUT_DIR=/home/c00882514/AngelSlim/synthetic_data
export MODEL_MAX_LENGTH=2048
export CHAT_TEMPLATE_TYPE=qwen3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 tools/train_eagle3_online.py \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path $DRAFT_MODEL_CONFIG_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 20 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --model_max_length $MODEL_MAX_LENGTH \
    --deepspeed $CONFIG_DIR/deepspeed_zero3.json \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --report_to none \
    --run_name qwen3-32b-eagle3-angelslim