#!/bin/bash
set -euo pipefail

#==========================================
# Actor-only Training Script (LLaVA)
# - Freeze all params except actor
# - Save only actor.bin (+ actor_config.json)
# - Optional: load actor.bin to resume actor training
#==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------- Actor / pruning env ----------------
export SYS_PROMPT_LEN=35
export IMG_TOKEN_LEN=576
export PRUNE_LAYER_INDEX=2

export MODEL_DIM=4096
export NUM_HEADS=8
export DROPOUT=0.1
export TOP_K=96 #当HARD_MODE是TOP_K才有用
export TAU=1.0

# IMPORTANT: if your python does `bool(os.environ["USE_LAYERNORM"])`,
# then "False" is also True. Prefer 0/1 and parse properly in python.
export USE_LAYERNORM=1
export HARD_MODE=argmax #训练时使用argmax模式，同时计算剪枝率损失

export TARGET_PRUNE_RATIO=0.8
export LAMBDA_PRUNE=5

# Optional: load actor checkpoint (actor.bin). Leave empty to skip.
# Example: export ACTOR_CKPT="/path/to/prev_run/actor.bin"
export ACTOR_CKPT=""
# ---------------- User config ----------------
GPU_IDS="0,1,2,3,4,5,6,7"
BATCH_SIZE=2
EPOCHS=1

MODEL_PATH="/data/users/airprofly/FastV/llava-v1.5-7b"
DATA_PATH="/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json"
IMAGE_FOLDER="/data/users/Actor/Attention_Actor/playground/data"
OUTPUT_DIR="/data/users/Actor/DynamicVLM/train_results"
export VIS_DIR="${OUTPUT_DIR}/viz"
export CKPT_DIR="${OUTPUT_DIR}/ckpt"
export PLOT_EVERY_STEPS=50
export SAVE_ACTOR_EVERY_STEPS=100

LEARNING_RATE=2e-4
MODEL_MAX_LENGTH=2048

# ---------------- Runtime ----------------
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# (Optional) 3090 comm tweaks
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
echo ">>> NUM_GPUS=${NUM_GPUS}"

mkdir -p "${OUTPUT_DIR}"

# Record config
cat > "${OUTPUT_DIR}/training_config.md" << EOF
# Actor-only Training Configuration

**Date:** $(date '+%Y-%m-%d %H:%M:%S')

## Paths
- MODEL_PATH: \`${MODEL_PATH}\`
- DATA_PATH: \`${DATA_PATH}\`
- IMAGE_FOLDER: \`${IMAGE_FOLDER}\`
- OUTPUT_DIR: \`${OUTPUT_DIR}\`

## HF Train Args
- LR: \`${LEARNING_RATE}\`
- BATCH_SIZE(per device): \`${BATCH_SIZE}\`
- EPOCHS: \`${EPOCHS}\`
- MODEL_MAX_LENGTH: \`${MODEL_MAX_LENGTH}\`

## Actor Env
- SYS_PROMPT_LEN: \`${SYS_PROMPT_LEN}\`
- IMG_TOKEN_LEN: \`${IMG_TOKEN_LEN}\`
- PRUNE_LAYER_INDEX: \`${PRUNE_LAYER_INDEX}\`
- MODEL_DIM: \`${MODEL_DIM}\`
- NUM_HEADS: \`${NUM_HEADS}\`
- DROPOUT: \`${DROPOUT}\`
- TOP_K: \`${TOP_K}\`
- TAU: \`${TAU}\`
- USE_LAYERNORM: \`${USE_LAYERNORM}\`
- HARD_MODE: \`${HARD_MODE}\`
- TARGET_PRUNE_RATIO: \`${TARGET_PRUNE_RATIO}\`
- LAMBDA_PRUNE: \`${LAMBDA_PRUNE}\`

## Resume
- ACTOR_CKPT: \`${ACTOR_CKPT:-none}\`
EOF

echo "[Config] saved to ${OUTPUT_DIR}/training_config.md"

# ---------------- Train command ----------------
# IMPORTANT: use the module that contains your modified train() entry.
# If your file is llava/train/train.py -> -m llava.train.train
TRAIN_MODULE="llava.train.train"

COMMON_ARGS=(
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --image_folder "${IMAGE_FOLDER}"
  --vision_tower "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --bf16
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps 4
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0.0
  --warmup_steps 0
  --lr_scheduler_type cosine
  --logging_steps 1
  --evaluation_strategy "no"
  --save_strategy "no"
  --num_train_epochs "${EPOCHS}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --lazy_preprocess True
  --report_to "none"
)

# If you want to continue training actor, just set ACTOR_CKPT env.
# (Your train.py should read ACTOR_CKPT and load it; if not yet, add that small load snippet.)
if [ -n "${ACTOR_CKPT}" ]; then
  echo ">>> Will load ACTOR_CKPT=${ACTOR_CKPT}"
else
  echo ">>> No ACTOR_CKPT provided."
fi

echo ">>> Launching training with torchrun..."

# -------- torchrun rendezvous (single-node) --------
MASTER_PORT="${MASTER_PORT:-29501}"   # 可改，避免和别人冲突
RDZV_ID="${RDZV_ID:-llava_actor_$(date +%s)}"

# torchrun 使用 CUDA_VISIBLE_DEVICES 的可见卡数，来决定每个进程绑定哪张卡
# NUM_GPUS 已经根据 GPU_IDS 计算好了

torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  --log_dir ${OUTPUT_DIR}/torchrun_logs \
  --redirects 3 --tee 3 \
  -m "${TRAIN_MODULE}" \
  "${COMMON_ARGS[@]}"

echo "=========================================="
echo "Training Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "Expect: ${OUTPUT_DIR}/actor.bin and actor_config.json"
echo "=========================================="