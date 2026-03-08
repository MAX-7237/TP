#!/bin/bash

# 使用方式:
#   bash mmvet.sh <MODEL_PATH> <ACTOR_CKPT>
# 示例:
#   bash mmvet.sh /data/users/Actor/CrossAttentionActor/1_results_datacleaned /data/users/Actor/CrossAttentionActor/1_results_datacleaned/actor_visualizations/actor_weights_step_13000.pt

MODEL_PATH=${1:-"/data/users/airprofly/FastV/llava-v1.5-7b"}
ACTOR_CKPT=${2:-"/data/users/Actor/DynamicVLM/1_results_datacleaned/actor_visualizations/actor_weights_step_14000.pt"}

gpu_list="${CUDA_VISIBLE_DEVICES:-5}"

NEW_PATH="/data/users/Actor/DynamicVLM/1_results_datacleaned/eval_mmvet"

# ===== Actor 环境变量 (与 train/train_1.sh 保持一致) =====
export SYS_PROMPT_LEN=35
export IMG_TOKEN_LEN=576
export PRUNE_LAYER_INDEX=2
export MODEL_DIM=4096
export NUM_HEADS=8
export DROPOUT=0.1
export TOP_K=96
export TAU=1.0
export USE_LAYERNORM=1
export HARD_MODE=argmax
export TARGET_PRUNE_RATIO=0.8
export LAMBDA_PRUNE=5

# Actor 参数 (从环境变量读取: TOP_K, TAU, DROPOUT)
ACTOR_HIDDEN_DIM=1024
ACTOR_NUM_HEADS=8
ACTOR_NUM_LAYERS=1

# 可视化配置
VISUALIZATION_DIR="$NEW_PATH/visualizations"
VISUALIZATION_SAVE_STEPS=100

# 构建 actor 参数
ACTOR_ARGS=""
if [ -n "$ACTOR_CKPT" ] && [ "$ACTOR_CKPT" != "" ]; then
    ACTOR_ARGS="--use_actor --actor_hidden_dim ${ACTOR_HIDDEN_DIM} --actor_num_heads ${ACTOR_NUM_HEADS} --actor_num_layers ${ACTOR_NUM_LAYERS} --actor_dropout ${DROPOUT} --actor_top_k ${TOP_K} --actor_tau ${TAU} --actor_ckpt ${ACTOR_CKPT}"
    if [ -n "$VISUALIZATION_DIR" ]; then
        ACTOR_ARGS="${ACTOR_ARGS} --visualization_output_dir ${VISUALIZATION_DIR} --visualization_plots_save_steps ${VISUALIZATION_SAVE_STEPS}"
    fi
fi

mkdir -p $NEW_PATH/answers

CUDA_VISIBLE_DEVICES=${gpu_list} python -m llava.eval.model_vqa \
    --model-path ${MODEL_PATH} \
    --question-file /data/users/Actor/Attention_Actor/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /data/users/Actor/Attention_Actor/playground/data/eval/mm-vet/images \
    --answers-file $NEW_PATH/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    ${ACTOR_ARGS}

mkdir -p $NEW_PATH/results

python /data/users/Actor/CrossAttentionActor/scripts/convert_mmvet_for_eval.py \
    --src $NEW_PATH/answers/llava-v1.5-7b.jsonl \
    --dst $NEW_PATH/results/llava-v1.5-7b.json
