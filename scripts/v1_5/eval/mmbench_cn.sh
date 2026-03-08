#!/bin/bash
# ===== Actor 环境变量 (与 train/train_1.sh 保持一致) =====
export CUDA_VISIBLE_DEVICES=2
export SYS_PROMPT_LEN=35
export IMG_TOKEN_LEN=576
export PRUNE_LAYER_INDEX=2
export MODEL_DIM=4096
export NUM_HEADS=8
export DROPOUT=0.1
export TOP_K=576
export TAU=1.0
export USE_LAYERNORM=1
export HARD_MODE=topk
export TARGET_PRUNE_RATIO=0.8
export LAMBDA_PRUNE=5

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path /data/users/airprofly/FastV/llava-v1.5-7b \
    --question-file /data/users/Actor/Attention_Actor/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /data/users/Actor/DynamicVLM/results/baseline_eval_results/mmbench-cn-1000/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /data/users/Actor/DynamicVLM/results/baseline_eval_results/mmbench-cn-1000/answers_upload/$SPLIT
python scripts/convert_mmbench_for_submission.py \
    --annotation-file /data/users/Actor/Attention_Actor/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /data/users/Actor/DynamicVLM/results/baseline_eval_results/mmbench-cn-1000/answers/$SPLIT \
    --upload-dir /data/users/Actor/DynamicVLM/results/baseline_eval_results/mmbench-cn-1000/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
