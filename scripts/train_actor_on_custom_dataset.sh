#!/bin/bash
#==========================================
# Dual Head Pruning Actor Training Script
# 使用 LLaVA train.py 在任意数据集上训练 actor
#==========================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=8

# 模型路径
MODEL_PATH="/data/users/airprofly/FastV/llava-v1.5-7b"

# 数据路径（排除 OCR VQA，使用过滤后的数据集）
DATA_PATH="/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json"

# 图片路径
IMAGE_FOLDER="/data/users/Actor/Attention_Actor/playground/data"

# 输出目录
OUTPUT_DIR="/data/users/Actor/Attention_Actor/attention_actor_training_results_10pct_cleaned"

# 学习率
LEARNING_RATE=1e-4

# 批次大小
BATCH_SIZE=7

# 训练轮数
EPOCHS=1

# Actor 配置
ACTOR_D_ATTN=256
ACTOR_HEAD_DIM=64  # head_dim * num_heads = d_attn, 所以 num_heads = d_attn / head_dim
ACTOR_MLP_HIDDEN=256

# 剪枝损失配置
TARGET_KEEP_RATIO=0.2 # 目标保留率，1.0表示目标是保留所有的token，0.0表示目的是剪枝所有token
LAMBDA_PRUNE=0.5

# 可视化配置
ENABLE_VISUALIZATION=true
VISUALIZATION_DIR="${OUTPUT_DIR}"
VISUALIZATION_PLOTS_SAVE_STEPS=100
VISUALIZATION_CHECKPOINT_SAVE_STEPS=1000

#==========================================
echo "=========================================="
echo "Training Dual Head Pruning Actor on Custom Dataset"
echo "=========================================="

# 创建训练配置记录文件
cat > "${OUTPUT_DIR}/training_config.md" << EOF
# Dual Head Pruning Actor Training Configuration

**Training Date:** $(date '+%Y-%m-%d %H:%M:%S')

## Model Configuration
- **Model Path:** \`${MODEL_PATH}\`
- **Vision Tower:** \`${MODEL_PATH}\`

## Data Configuration
- **Dataset Path:** \`${DATA_PATH}\`
- **Image Folder:** \`${IMAGE_FOLDER}\`

## Training Configuration
- **Output Directory:** \`${OUTPUT_DIR}\`
- **Learning Rate:** \`${LEARNING_RATE}\`
- **Batch Size:** \`${BATCH_SIZE}\`
- **Epochs:** \`${EPOCHS}\`

## Actor Configuration
- **Actor d_attn:** \`${ACTOR_D_ATTN}\`
- **Actor head_dim:** \`${ACTOR_HEAD_DIM}\`
  - num_heads = d_attn / head_dim = ${ACTOR_D_ATTN} / ${ACTOR_HEAD_DIM} = $((ACTOR_D_ATTN / ACTOR_HEAD_DIM))
- **Actor MLP Hidden:** \`${ACTOR_MLP_HIDDEN}\`

## Pruning Loss Configuration
- **Target Keep Ratio (τ):** \`${TARGET_KEEP_RATIO}\` 
  - \`1.0\` = 保留所有 tokens（不剪枝）
  - \`0.5\` = 保留 50% tokens
  - \`0.0\` = 保留 0% tokens（全剪枝）
- **Lambda Prune (λ):** \`${LAMBDA_PRUNE}\`
  - 权重因子，用于平衡 LLM loss 和 prune loss
  - Total Loss = L_LLM + λ × (mean(keep_mask) - τ)²

## Visualization Configuration
- **Visualization Enabled:** \`true\`
- **Visualization Directory:** \`${OUTPUT_DIR}\`
- **Plots Save Steps:** \`${VISUALIZATION_PLOTS_SAVE_STEPS}\`
- **Checkpoints Save Steps:** \`${VISUALIZATION_CHECKPOINT_SAVE_STEPS}\`

## Additional Configuration
- **Precision:** bf16
- **Max Length:** 2048
- **Gradient Checkpointing:** true
- **Lazy Preprocess:** true
- **Train Actor Only:** true
- **Freeze Backbone:** true

EOF

echo "[Config] Training configuration saved to: ${OUTPUT_DIR}/training_config.md"
echo ""
echo "Starting training..."
echo ""

python -m llava.train.train_mem \
    --model_name_or_path ${MODEL_PATH} \
    --vision_tower ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    --train_actor_only \
    --use_attention_actor \
    --actor_d_attn ${ACTOR_D_ATTN} \
    --actor_head_dim ${ACTOR_HEAD_DIM} \
    --actor_mlp_hidden ${ACTOR_MLP_HIDDEN} \
    --actor_ckpt None \
    --save_only_actor \
    --freeze_backbone \
    --bf16 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_steps 0 \
    --max_steps -1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --num_train_epochs ${EPOCHS} \
    --bits 16 \
    --use_visualization \
    --visualization_output_dir ${VISUALIZATION_DIR} \
    --visualization_plots_save_steps ${VISUALIZATION_PLOTS_SAVE_STEPS} \
    --visualization_checkpoint_save_steps ${VISUALIZATION_CHECKPOINT_SAVE_STEPS} \
    --target_keep_ratio ${TARGET_KEEP_RATIO} \
    --lambda_prune ${LAMBDA_PRUNE}

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "  - All outputs (checkpoints, visualizations) saved here"
echo "=========================================="

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "  - All outputs (checkpoints, visualizations) saved here"
echo "=========================================="
