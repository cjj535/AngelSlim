#!/bin/bash

DATASET_PATH=/home/c00882514/AngelSlim/synthetic_data/train.jsonl
MODEL_NAME=/home/c00882514/Qwen3-32B
TARGET_BACKEND=hf
MODEL_MAX_LENGTH=2048
CHAT_TEMPLATE_TYPE=qwen3
OUTPUT_DIR=/home/c00882514/AngelSlim/synthetic_data/train
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 \
    tools/generate_hidden_for_draft_model.py \
    --dataset_path $DATASET_PATH \
    --model_name $MODEL_NAME \
    --target_backend $TARGET_BACKEND \
    --torch_dtype bfloat16 \
    --model_max_length $MODEL_MAX_LENGTH \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --outdir $OUTPUT_DIR
