#!/bin/bash

# 使用方式:
#   bash qbench_zh.sh <dev|test> <MODEL_PATH> <ACTOR_CKPT>
# 示例:
#   bash qbench_zh.sh dev /data/users/Actor/CrossAttentionActor/1_results_datacleaned /data/users/Actor/CrossAttentionActor/1_results_datacleaned/actor_visualizations/actor_weights_step_13000.pt

gpu_list="${CUDA_VISIBLE_DEVICES:-5}"

MODEL_PATH=${2:-"/data/users/airprofly/FastV/llava-v1.5-7b"}
ACTOR_CKPT=${3:-"/data/users/Actor/DynamicVLM/1_results_datacleaned/actor_visualizations/actor_weights_step_14000.pt"}

NEW_PATH="/data/users/Actor/DynamicVLM/1_results_datacleaned/eval_qbench_zh"

if [ "$1" = "dev" ]; then
    ZH_SPLIT="验证集"
    echo "Evaluating in 'dev' split."
    SPLIT="dev"
elif [ "$1" = "test" ]; then
    ZH_SPLIT="测试集"
    echo "Evaluating in 'test' split."
    SPLIT="test"
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

# Actor 配置 (参考 1_train.sh)
ACTOR_HIDDEN_DIM=1024
ACTOR_NUM_HEADS=8
ACTOR_NUM_LAYERS=1
ACTOR_DROPOUT=0.1
ACTOR_TOP_K=192
ACTOR_TAU=1.0

# 可视化配置
VISUALIZATION_DIR="$NEW_PATH/visualizations"
VISUALIZATION_SAVE_STEPS=100

# 构建 actor 参数
ACTOR_ARGS=""
if [ -n "$ACTOR_CKPT" ] && [ "$ACTOR_CKPT" != "" ]; then
    ACTOR_ARGS="--use_attention_actor --actor_hidden_dim ${ACTOR_HIDDEN_DIM} --actor_num_heads ${ACTOR_NUM_HEADS} --actor_num_layers ${ACTOR_NUM_LAYERS} --actor_dropout ${ACTOR_DROPOUT} --actor_top_k ${ACTOR_TOP_K} --actor_tau ${ACTOR_TAU} --actor_ckpt ${ACTOR_CKPT}"
    if [ -n "$VISUALIZATION_DIR" ]; then
        ACTOR_ARGS="${ACTOR_ARGS} --visualization_output_dir ${VISUALIZATION_DIR} --visualization_plots_save_steps ${VISUALIZATION_SAVE_STEPS}"
    fi
fi

mkdir -p $NEW_PATH/answers

CUDA_VISIBLE_DEVICES=${gpu_list} python -m llava.eval.model_vqa_qbench \
    --model-path ${MODEL_PATH} \
    --image-folder /data/users/Actor/Attention_Actor/playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file /data/users/Actor/Attention_Actor/playground/data/eval/qbench/质衡-问答-$ZH_SPLIT.json \
    --answers-file $NEW_PATH/answers/llvisionqa_zh_${SPLIT}_answers.jsonl \
    --conv-mode llava_v1 \
    --lang zh \
    ${ACTOR_ARGS}
