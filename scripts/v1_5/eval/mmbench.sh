#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
SPLIT="mmbench_test_en_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path /data/users/airprofly/FastV/llava-v1.5-7b \
    --question-file /data/users/Actor/Attention_Actor/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /data/users/Actor/DynamicVLM/0_baseline/eval_mmbench/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /data/users/Actor/DynamicVLM/0_baseline/eval_mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /data/users/Actor/Attention_Actor/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /data/users/Actor/DynamicVLM/0_baseline/eval_mmbench/answers/$SPLIT \
    --upload-dir /data/users/Actor/DynamicVLM/0_baseline/eval_mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
