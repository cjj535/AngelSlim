#!/bin/bash
ROOT=/workspace
BASE_MODEL_PATH=$ROOT/Qwen3-32B
EAGLE_MODEL_PATH=$ROOT/Qwen3-32B_draft_model
OUTPUT_DIR=$ROOT/result/$1
MODEL_ID=Qwen3-32B-eagle3
source /home/c00882514/MyAscend/ascend-toolkit/set_env.sh
source /home/c00882514/MyAscend/nnal/atb/set_env.sh --cxx_abi=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python3 tools/spec_benchmark.py \
    --base-model-path $BASE_MODEL_PATH \
    --eagle-model-path $EAGLE_MODEL_PATH \
    --model-id $MODEL_ID \
    --mode both \
    --deploy-backend vllm \
    --output-dir $OUTPUT_DIR \
    --num-gpus-per-model 2 \
    --num-gpus-total 2 \
    --max-gpu-memory 0.9 \
    --temperature $2 \
    --bench-name $1 \
    --batch-size 200 \