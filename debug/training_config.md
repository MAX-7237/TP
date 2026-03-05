# Attention Actor Training Configuration

**Training Date:** 2026-03-03 19:56:08

## Model Configuration
- **Model Path:** `/data/users/airprofly/FastV/llava-v1.5-7b`
- **Vision Tower:** `/data/users/airprofly/FastV/llava-v1.5-7b`

## Data Configuration
- **Dataset Path:** `/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json`
- **Image Folder:** `/data/users/Actor/Attention_Actor/playground/data`

## Training Configuration
- **Output Directory:** `/data/users/Actor/DynamicVLM/debug`
- **Learning Rate:** `2e-4`
- **Batch Size:** `1`
- **Epochs:** `1`
- **Num GPUs:** `1`

## Actor Configuration (与 dynamicvlm_actor.py 参数名一致)
- **Actor hidden_dim:** `1024`
- **Actor num_heads:** `8`
- **Actor num_layers:** `1`
- **Actor dropout:** `0.1`
- **Actor top_k:** `192`
- **Actor tau:** `1.0`
- **Actor checkpoint:** `None`

## Diversity Loss Configuration
- **Lambda Div (λ):** `0`

## Visualization Configuration
- **Plots Save Steps:** `100`
- **Checkpoints Save Steps:** `1000`

## Additional Configuration
- **FSDP:** `none`
- **FSDP Min Params:** `0`

