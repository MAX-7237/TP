# Actor-only Training Configuration

**Date:** 2026-03-06 08:57:40

## Paths
- MODEL_PATH: `/data/users/airprofly/FastV/llava-v1.5-7b`
- DATA_PATH: `/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json`
- IMAGE_FOLDER: `/data/users/Actor/Attention_Actor/playground/data`
- OUTPUT_DIR: `/data/users/Actor/DynamicVLM/train_1`

## HF Train Args
- LR: `2e-4`
- BATCH_SIZE(per device): `2`
- EPOCHS: `1`
- MODEL_MAX_LENGTH: `2048`

## Actor Env
- SYS_PROMPT_LEN: `35`
- IMG_TOKEN_LEN: `576`
- PRUNE_LAYER_INDEX: `2`
- MODEL_DIM: `4096`
- NUM_HEADS: `8`
- DROPOUT: `0.1`
- TOP_K: `96`
- TAU: `1.0`
- USE_LAYERNORM: `1`
- HARD_MODE: `argmax`
- TARGET_PRUNE_RATIO: `0.8`
- LAMBDA_PRUNE: `5`

## Resume
- ACTOR_CKPT: `none`
