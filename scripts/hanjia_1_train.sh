#!/bin/bash
#==========================================
# Attention Actor Training Script
# 使用 LLaVA train.py 在任意数据集上训练 actor
# 
# 使用方法:
#   ./1_train.sh                    # 使用下方配置运行
#==========================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

#========== 在这里修改配置 ==========
GPU_IDS="4"   # 使用的 GPU 序号，用逗号分隔，如 "0,1,2" 或 "0,1,2,3,4,5,6"
BATCH_SIZE=1         # 每个 GPU 的 batch size
EPOCHS=1                # 训练轮数

# Actor 配置
# 参数名与 dynamicvlm_actor.py 保持一致
ACTOR_HIDDEN_DIM=1024      # hidden_dim: Actor 内部维度
ACTOR_NUM_HEADS=8          # num_heads: 注意力头数
ACTOR_NUM_LAYERS=1         # num_layers: Transformer 层数
ACTOR_DROPOUT=0.1          # dropout 概率
ACTOR_TOP_K=192            # top_k: 目标保留的 token 数量
# 剪枝率:192:66.7%,128:77.8%,64:88.9%
ACTOR_TAU=1.0              # tau: 温度系数

# 损失权重
LAMBDA_DIV=0         # diversity loss 的权重
GAMMA_PRUNE_RATE=0   # 剪枝率损失的权重
MODEL_PATH="/data/users/airprofly/FastV/llava-v1.5-7b"
DATA_PATH="/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json"
IMAGE_FOLDER="/data/users/Actor/Attention_Actor/playground/data"
ACTOR_PATH=None  # 预训练 actor 权重路径，为空则不加载
OUTPUT_DIR="/data/users/Actor/DynamicVLM/debug"

# 训练参数
LEARNING_RATE=2e-4
# 可视化配置
VISUALIZATION_PLOTS_SAVE_STEPS=100
VISUALIZATION_CHECKPOINT_SAVE_STEPS=1000
#========== 配置结束 ==========

# 从 GPU_IDS 计算 GPU 数量
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
echo ">>> 检测到 GPU 数量: ${NUM_GPUS}"

# 根据 GPU 数量自动决定是否使用 FSDP
if [ $NUM_GPUS -eq 1 ]; then
    USE_FSDP=false
    FSDP_CONFIG=""
    FSDP_MIN_PARAMS=0
    echo ">>> 单卡模式 (无 FSDP)"
else
    USE_FSDP=true
    FSDP_CONFIG="full_shard"
    FSDP_MIN_PARAMS=1000
    echo ">>> 多卡模式 (FSDP, ${NUM_GPUS} GPUs)"
fi

echo ">>> GPU 数量: ${NUM_GPUS}"
echo ">>> Batch Size: ${BATCH_SIZE}"
echo ">>> Actor Top K: ${ACTOR_TOP_K}"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
# PyTorch 2.6+ 需要设置这个来加载包含 numpy 对象的旧 checkpoint
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# 创建 FSDP 配置文件
FSDP_CONFIG_FILE="${OUTPUT_DIR}/fsdp_config.json"
if [ "$USE_FSDP" = true ]; then
    mkdir -p "${OUTPUT_DIR}"
    cat > "${FSDP_CONFIG_FILE}" << EOF
{
    "backward_prefetch": "backward_prefetch",
    "forward_prefetch": "false",
    "sync_module_states": "true",
    "use_orig_params": "true"
}
EOF
fi

#==========================================
echo "=========================================="
echo "Training Attention Actor on Custom Dataset"
echo "=========================================="

# 创建训练配置记录文件
cat > "${OUTPUT_DIR}/training_config.md" << EOF
# Attention Actor Training Configuration

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
- **Num GPUs:** \`${NUM_GPUS}\`

## Actor Configuration (与 dynamicvlm_actor.py 参数名一致)
- **Actor hidden_dim:** \`${ACTOR_HIDDEN_DIM}\`
- **Actor num_heads:** \`${ACTOR_NUM_HEADS}\`
- **Actor num_layers:** \`${ACTOR_NUM_LAYERS}\`
- **Actor dropout:** \`${ACTOR_DROPOUT}\`
- **Actor top_k:** \`${ACTOR_TOP_K}\`
- **Actor tau:** \`${ACTOR_TAU}\`
- **Actor checkpoint:** \`${ACTOR_PATH:-none}\`

## Diversity Loss Configuration
- **Lambda Div (λ):** \`${LAMBDA_DIV}\`

## Visualization Configuration
- **Plots Save Steps:** \`${VISUALIZATION_PLOTS_SAVE_STEPS}\`
- **Checkpoints Save Steps:** \`${VISUALIZATION_CHECKPOINT_SAVE_STEPS}\`

## Additional Configuration
- **FSDP:** \`${FSDP_CONFIG:-none}\`
- **FSDP Min Params:** \`${FSDP_MIN_PARAMS}\`

EOF

echo "[Config] Training configuration saved to: ${OUTPUT_DIR}/training_config.md"
echo ""
echo "Starting training..."
echo ""

# 构建训练命令
TRAIN_CMD="python -m llava.train.train_mem \
    --model_name_or_path ${MODEL_PATH} \
    --vision_tower ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    --train_actor_only \
    --use_attention_actor \
    --actor_hidden_dim ${ACTOR_HIDDEN_DIM} \
    --actor_num_heads ${ACTOR_NUM_HEADS} \
    --actor_num_layers ${ACTOR_NUM_LAYERS} \
    --actor_dropout ${ACTOR_DROPOUT} \
    --actor_top_k ${ACTOR_TOP_K} \
    --actor_tau ${ACTOR_TAU} \
    --actor_ckpt ${ACTOR_PATH} \
    --save_only_actor \
    --freeze_backbone \
    --bf16 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_steps 0 \
    --max_steps -1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --num_train_epochs ${EPOCHS} \
    --bits 16 \
    --use_visualization \
    --visualization_output_dir ${OUTPUT_DIR} \
    --visualization_plots_save_steps ${VISUALIZATION_PLOTS_SAVE_STEPS} \
    --visualization_checkpoint_save_steps ${VISUALIZATION_CHECKPOINT_SAVE_STEPS} \
    --lambda_div ${LAMBDA_DIV} \
    --gamma_prune_rate ${GAMMA_PRUNE_RATE} \
    --report_to none"

# 添加 FSDP 参数（如果是多卡）
if [ "$USE_FSDP" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --fsdp ${FSDP_CONFIG} --fsdp_min_num_params ${FSDP_MIN_PARAMS} --fsdp_config ${FSDP_CONFIG_FILE}"
fi

# 设置 Python 路径，让 torchrun 能找到 llava 模块
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# RTX 3090 需要禁用 P2P 和 IB 通信
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 执行训练
if [ "$USE_FSDP" = true ]; then
    # FSDP 需要使用 torchrun 启动分布式训练
    # 提取训练参数（去掉 "python -m llava.train.train_mem " 前缀）
    TRAIN_ARGS="${TRAIN_CMD#python -m llava.train.train_mem }"
    torchrun --nproc_per_node=${NUM_GPUS} -m llava.train.train_mem ${TRAIN_ARGS}
else
    eval ${TRAIN_CMD}
fi

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "  - All outputs (checkpoints, visualizations) saved here"
echo "=========================================="
