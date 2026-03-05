#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# ========== 全局 tokenizer 用于 debug 解码 ==========
_GLOBAL_TOKENIZER = None

def set_global_tokenizer(tokenizer):
    """在训练开始时设置全局 tokenizer"""
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = tokenizer

def get_global_tokenizer():
    """获取全局 tokenizer"""
    return _GLOBAL_TOKENIZER
# ========== 全局 tokenizer 结束 ==========


# ========== 自定义异常：用于跳过当前 batch ==========
class SkipBatchException(Exception):
    """用于在 compute_loss 中跳过当前 batch 的异常"""
    pass
# ========== 异常定义结束 ==========


from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

# ========== 从 visualizer.py 导入可视化工具和损失函数 ==========
from .visualizer import ActorVisualizer
from .compute_loss import compute_diversity_loss
from .compute_loss import compute_actor_loss_from_list_with_diversity


# ========== 计算图像token之间的余弦相似度矩阵 ==========
def compute_token_similarity(image_features):
    """
    计算图像token之间的余弦相似度矩阵。

    Args:
        image_features: 图像特征 [N, C] (2D) 或 [B, N, C] (3D) (不含CLS token)

    Returns:
        similarity_matrix: [N, N] 或 [B, N, N] 每两个token之间的余弦相似度矩阵
    """
    # 如果是2D [N, C]，添加batch维度变成 [1, N, C]
    is_2d = image_features.ndim == 2
    if is_2d:
        image_features = image_features.unsqueeze(0)
    
    # 归一化
    image_features_norm = F.normalize(image_features, p=2, dim=-1)
    # 计算余弦相似度矩阵
    similarity = torch.bmm(image_features_norm, image_features_norm.transpose(1, 2))
    
    # 如果原本是2D输入，返回2D输出
    if is_2d:
        similarity = similarity.squeeze(0)
    return similarity
# ============================================================

# ========== 导入结束 ==========

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

            # ----------Attention Actor (在 train.py 中初始化) ----------
            # 注意：Actor 的初始化移到 train.py 中进行，以确保使用命令行传入的参数
            self.attention_actor = None
            # ---------------------End-------------------------
            
            # ========== Initialize Actor Loss Attributes ==========
            # 这些属性用于存储上一次计算的 actor 相关损失，供 compute_loss 获取
            # 注意：这些属性实际存储在 self (即 LlavaMetaModel 实例) 上
            # 通过 self.get_model() 访问时，self.get_model() 返回的也是同一个对象
            self.actor_loss: Optional[torch.Tensor] = None
            self.prune_loss: Optional[torch.Tensor] = None
            self.weighted_prune_loss: Optional[torch.Tensor] = None
            self.p: Optional[torch.Tensor] = None
            self.keep_mask: Optional[np.ndarray] = None
            self.image_features: Optional[torch.Tensor] = None
            # ========== End of Initializing Actor Loss Attributes ==========

            # ========== Initialize Global Step Counter ==========
            # 用于可视化：每 N 步保存一次可视化图像
            self._global_step: int = 0
            self._visualization_save_steps: int = 100
            # ========== End of Global Step Counter ==========
            
            # ========== Initialize Visualization History Arrays ==========
            # 用于记录可视化数据的数组
            self._loss_history_steps: List[int] = []
            self._loss_history_total_loss: List[float] = []
            self._loss_history_lm_loss: List[float] = []
            # diversity_loss
            self._loss_history_diversity_loss: List[float] = []  # weighted diversity loss
            # 新增：记录 unweighted diversity loss, E(keep), diversity level
            self._loss_history_diversity_raw: List[float] = []
            self._loss_history_kept: List[float] = []
            self._loss_history_diversity_level: List[float] = []  # 1 - similarity
            # 剪枝率损失
            self._loss_history_prune_rate_loss: List[float] = []  # weighted pruning rate loss
            self._loss_history_actual_prune_rate: List[float] = []  # actual pruning rate
            # ========== End of Visualization History Arrays ==========
    #----------------------------------------------------------------

    def init_visualizer(self, output_dir: str = "actor_visualizations", 
                        plots_save_steps: int = 100,
                        checkpoint_save_steps: int = 1000):
        """
        Initialize the actor visualizer
        """
        from llava.model.visualizer import ActorVisualizer
        # Create visualizer with proper output directory
        vis_dir = os.path.join(output_dir, 'actor_visualizations')
        self.visualizer = ActorVisualizer(save_dir=vis_dir)
        self._visualization_save_steps = plots_save_steps
        self._checkpoint_save_steps = checkpoint_save_steps
        
        self.visualization_config = {
            'plots_save_steps': plots_save_steps,
            'checkpoint_save_steps': checkpoint_save_steps,
            'output_dir': output_dir,
        }
        
        print(f"[LlavaMetaModel] Initialized visualizer:")
        print(f"  - Plots saved every {plots_save_steps} steps")
        print(f"  - Output directory: {vis_dir}")
    #----------------------------------------------------------------

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            # vision_tower 已存在，检查是否是列表
            if isinstance(self.vision_tower, list):
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                else:
                    vision_tower = self.vision_tower[0] if len(self.vision_tower) > 0 else self.vision_tower
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass
    
    # #--- [actor训练: 添加compute_loss方法] ---
    # def compute_loss(self, model, inputs):
    #     """
    #     Custom compute_loss that includes prune loss.

    #     If actor is enabled, adds prune loss (λ * L_prune) to the total loss.
    #     L_total = LLM_loss + λ * (mean(keep_mask) - τ)²
    #     """
    #     # 使用默认的 compute_loss
    #     outputs = model(**inputs)

    #     # 获取 actor loss（如果存在）
    #     actor_loss = self.get_model()._last_actor_loss

    #     # 安全检查：确保 actor_loss 存在且有梯度
    #     lm_loss = outputs.loss

    #     print(f"\n[DEBUG Loss]")
    #     print(f"  lm_loss={lm_loss.item():.4f}, requires_grad={lm_loss.requires_grad}")

    #     # ========== [显存保护：跳过无梯度的 batch] ==========
    #     # 当 lm_loss 没有梯度时（显存优化导致计算图被释放），
    #     # 抛出 SkipBatchException，Trainer 会捕获并完全跳过这个 batch
    #     # if not lm_loss.requires_grad:
    #     #     print(f"  [WARNING] lm_loss 没有梯度（显存不足），跳过当前 batch")

    #     #     # 记录跳过信息
    #     #     model = self.get_model()
    #     #     model._skipped_batches = getattr(model, '_skipped_batches', 0) + 1
    #     #     print(f"  [INFO] 已跳过 batch {model._global_step}，累计跳过: {model._skipped_batches}")

    #     #     # 抛出异常，Trainer 会捕获并跳过这个 batch
    #     #     raise SkipBatchException()
    #     # ========== [显存保护结束] ==========
        
    #     print(f"  actor_loss={actor_loss.item():.4f}, requires_grad={actor_loss.requires_grad}")
    #     total_loss = lm_loss + actor_loss
    #     print(f"  total_loss=lm_loss + actor_loss={total_loss.item():.4f}")

    #     # 存储损失值用于可视化
    #     model = self.get_model()
    #     model.lm_loss = lm_loss
    #     model.total_loss = total_loss

    #     # 每次 step 记录损失到历史数组
    #     model._loss_history_steps.append(model._global_step)
    #     model._loss_history_total_loss.append(total_loss.item())
    #     model._loss_history_lm_loss.append(lm_loss.item())
        
    #     return total_loss

    # #--- [actor训练: 方法结束] ---
    #--- [actor训练: 添加compute_loss方法] ---
    #--- [actor训练: 添加compute_loss方法] ---
    #--- [actor训练: 添加compute_loss方法] ---
    def compute_loss(self, model, inputs):
        """
        Custom compute_loss that includes prune loss.

        If actor is enabled, adds prune loss (λ * L_prune) to the total loss.
        L_total = LLM_loss + λ * (mean(keep_mask) - τ)²

        所有的 step 计数、debug 打印、监控记录、可视化保存都在这里进行，
        因为 compute_loss 每个 batch 只调用一次，不受 gradient checkpointing 影响。
        """
        # ===== Forward =====
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # ===== 从 inputs 中获取 image_files 并保存到模型 =====
        if 'image_files' in inputs:
            model._sample_ids = inputs['image_files']
        else:
            model._sample_ids = [None] * inputs.get('input_ids', [[]]).__len__()

        # ===== Step 自增（只在这里，每个 batch 只执行一次） =====
        mdl = self.get_model()
        mdl._global_step += 1
        current_step = mdl._global_step
        
        # ===== 初始化损失历史列表 =====
        if not hasattr(mdl, '_loss_history_bce_loss'):
            mdl._loss_history_bce_loss = []
        
        # ===== 保存 input_ids 用于可视化 =====
        if 'input_ids' in inputs:
            input_ids_list = inputs['input_ids']
            # 只保存第一个样本的 input_ids（用于可视化）
            if isinstance(input_ids_list, torch.Tensor):
                mdl.input_ids = input_ids_list[0].cpu().tolist() if input_ids_list.shape[0] > 0 else []
            else:
                mdl.input_ids = input_ids_list[0] if input_ids_list else []

        # ===== 从 token_feats_list 和 keep_probs_list 计算 diversity loss =====
        token_feats_list = getattr(mdl, 'token_feats_list', [])
        mask_soft_list = getattr(mdl, 'keep_probs_list', [])  # keep_probs (soft)，可导
        mask_hard_list = getattr(mdl, 'hard_mask_list', [])
        
        # 获取 top_k 配置（用于可视化）
        top_k = getattr(mdl.config, 'actor_top_k', 64)
        
        # 检查是否使用 BCE mask 监督模式
        use_mask = getattr(mdl.config, 'use_mask', False)
        mask_dict = getattr(mdl, 'mask_dict', {})
        
        # Debug: 打印 mask 模式状态
        print(f"[DEBUG BCE] use_mask={use_mask}, mask_dict_len={len(mask_dict)}")
        
        if len(token_feats_list) > 0 and len(mask_soft_list) > 0:
            lambda_div = getattr(mdl.config, 'lambda_div', 1.0)

            # ===== 根据模式选择损失函数 =====
            if use_mask and len(mask_dict) > 0:
                # BCE Loss 模式
                # 使用 keep_probs (保留概率) 与 label_mask 计算 BCE loss
                keep_probs_list = getattr(mdl, 'keep_probs_list', [])
                label_mask_list = getattr(mdl, 'label_mask_list', [])
                
                # Debug: 打印列表状态
                print(f"[DEBUG BCE] keep_probs_list_len={len(keep_probs_list)}, label_mask_list_len={len(label_mask_list)}")
                
                if len(keep_probs_list) > 0 and len(label_mask_list) > 0:
                    keep_probs = keep_probs_list[-1]  # [N] 预测的保留概率 P(keep)
                    label_mask = label_mask_list[-1]  # [N] ground truth mask
                    
                    # 如果 label_mask 为 None（没有对应的 mask），跳过 BCE loss
                    if label_mask is None:
                        print("[DEBUG] No mask found for this sample, skipping BCE loss")
                        bce_loss_raw = torch.tensor(0.0, device=keep_probs.device, requires_grad=keep_probs.requires_grad)
                        weighted_bce_loss = torch.tensor(0.0, device=keep_probs.device, requires_grad=keep_probs.requires_grad)
                        mdl.bce_loss_raw = bce_loss_raw.detach()
                        mdl.weighted_bce_loss = weighted_bce_loss.detach()
                        diversity_loss_raw = bce_loss_raw
                        weighted_diversity_loss = weighted_bce_loss
                        prune_loss = weighted_bce_loss
                        weighted_prune_loss = weighted_bce_loss
                    else:
                        # Debug: 打印 label_mask 的信息
                        print(f"[DEBUG] label_mask: min={label_mask.min().item():.4f}, max={label_mask.max().item():.4f}, mean={label_mask.mean().item():.4f}, sum={label_mask.sum().item():.0f}")
                        print(f"[DEBUG] keep_probs: min={keep_probs.min().item():.4f}, max={keep_probs.max().item():.4f}, mean={keep_probs.mean().item():.4f}")
                        
                        # 确保在同一设备上
                        if label_mask.device != keep_probs.device:
                            label_mask = label_mask.to(keep_probs.device)
                        
                        # 计算真正的 BCE loss: -[label * log(pred) + (1-label) * log(1-pred)]
                        # 使用 clamp 防止 log(0)
                        keep_probs_clamped = torch.clamp(keep_probs, min=1e-7, max=1-1e-7)
                        bce_loss_raw = F.binary_cross_entropy(keep_probs_clamped, label_mask)
                        
                        # Mask 模式下，直接使用 BCE loss（不使用 lambda_div，因为 lambda_div=0）
                        weighted_bce_loss = bce_loss_raw
                        
                        # 保存用于日志
                        mdl.bce_loss_raw = bce_loss_raw.detach()
                        mdl.weighted_bce_loss = weighted_bce_loss.detach()
                        
                        # 使用 BCE loss 作为 prune_loss
                        diversity_loss_raw = bce_loss_raw
                        weighted_diversity_loss = weighted_bce_loss
                        prune_loss = weighted_bce_loss
                        weighted_prune_loss = weighted_bce_loss
                        
                        print(f"[DEBUG] BCE Loss: raw={bce_loss_raw.item():.10f}, weighted={weighted_bce_loss.item():.10f}")
                else:
                    # 如果没有 label_mask，使用 diversity loss 作为后备
                    diversity_loss_raw, weighted_diversity_loss = compute_actor_loss_from_list_with_diversity(
                        token_feats_list=token_feats_list,
                        mask_hard_list=mask_hard_list,
                        lambda_div=lambda_div,
                    )
                    prune_loss = weighted_diversity_loss
                    weighted_prune_loss = weighted_diversity_loss
                    bce_loss_raw = torch.tensor(0.0, device=token_feats_list[0].device if token_feats_list else torch.device('cpu'))
                    weighted_bce_loss = torch.tensor(0.0, device=token_feats_list[0].device if token_feats_list else torch.device('cpu'))
                    mdl.bce_loss_raw = bce_loss_raw.detach()
                    mdl.weighted_bce_loss = weighted_bce_loss.detach()
            else:
                # 原始 diversity loss 模式
                diversity_loss_raw, weighted_diversity_loss = compute_actor_loss_from_list_with_diversity(
                    token_feats_list=token_feats_list,
                    mask_hard_list=mask_hard_list,  # soft mask (keep_probs) 用于多样性损失，可导
                    lambda_div=lambda_div,
                )
                prune_loss = weighted_diversity_loss
                weighted_prune_loss = weighted_diversity_loss

            # 计算 E(keep)（用于监控），使用 STE_mask (STE 近似)
            STE_mask_list = getattr(mdl, 'STE_mask_list', [])
            if len(STE_mask_list) > 0:
                keep_mask = STE_mask_list[-1]  # [N] 使用 STE 近似的 keep_mask
                # E(keep) = sum(keep_mask)
                kept = keep_mask.sum().item()
            else:
                kept = 0.0

            # 保存用于日志（detach，不影响计算图）
            mdl.diversity_loss_raw = diversity_loss_raw.detach()
            mdl.weighted_diversity_loss = weighted_diversity_loss.detach()
            mdl.kept = kept
            mdl.prune_loss = prune_loss.detach()
            mdl.weighted_prune_loss = weighted_prune_loss.detach()

            # Debug 打印每个 mask 的统计信息 (keep_probs)
            print(f"[DEBUG] Step {current_step}: kept={kept:.4f}, top_k={top_k}")
            keep_probs_list = getattr(mdl, 'keep_probs_list', [])  # keep_probs
            for idx, m in enumerate(keep_probs_list):
                p_min, p_max = m.min().item(), m.max().item()
                p_mean = m.mean().item()
                print(f"[DEBUG] idx={idx}: p_min={p_min:.4f}, p_max={p_max:.4f}, p_mean={p_mean:.4f}")

            # 监控数据
            mdl.p = STE_mask_list[-1].detach()
            mdl.keep_mask = STE_mask_list[-1].detach().float().cpu().numpy()

            current_keep_ratio = STE_mask_list[-1].mean().item()
            
            # ===== 计算剪枝率损失 =====
            gamma_prune_rate = getattr(mdl.config, 'gamma_prune_rate', 0.0)
            if gamma_prune_rate > 0:
                # 目标保留率 = top_k / total_tokens
                total_tokens = STE_mask_list[-1].numel()
                target_keep_ratio = top_k / total_tokens
                # 实际保留率
                actual_keep_ratio = STE_mask_list[-1].mean()
                # 剪枝率损失 = (target - actual)^2
                prune_rate_loss_raw = (actual_keep_ratio - target_keep_ratio) ** 2
                weighted_prune_rate_loss = gamma_prune_rate * prune_rate_loss_raw
                
                # 保存用于日志
                mdl.prune_rate_loss = weighted_prune_rate_loss.detach()
                mdl.actual_prune_rate = (1.0 - actual_keep_ratio.item())
                
                print(f"  prune_rate: target={target_keep_ratio:.4f}, actual={actual_keep_ratio.item():.4f}, loss={prune_rate_loss_raw.item():.6f}")
            else:
                prune_rate_loss_raw = torch.tensor(0.0, device=lm_loss.device, requires_grad=True)
                weighted_prune_rate_loss = torch.tensor(0.0, device=lm_loss.device, requires_grad=True)
                mdl.prune_rate_loss = weighted_prune_rate_loss.detach()
                mdl.actual_prune_rate = 0.0
            
            mdl._loss_history_steps.append(current_step)
            
            # 记录监控数据
            mdl._loss_history_diversity_raw.append(diversity_loss_raw.item())
            mdl._loss_history_diversity_loss.append(weighted_diversity_loss.item())
            # 移除 entropy 记录
            # mdl._loss_history_entropy.append(entropy)
            mdl._loss_history_kept.append(kept)
            # diversity_level = 1 - similarity (diversity_loss_raw 就是相似度)
            diversity_level = 1.0 - diversity_loss_raw.item()
            mdl._loss_history_diversity_level.append(diversity_level)
            
        else:
            weighted_diversity_loss = torch.tensor(0.0, device=lm_loss.device, requires_grad=True)
            mdl.weighted_diversity_loss = weighted_diversity_loss
            # 没有 actor 时，也需要添加占位，保持列表长度一致
            mdl._loss_history_diversity_raw.append(0.0)
            mdl._loss_history_diversity_loss.append(0.0)
            mdl._loss_history_kept.append(0.0)
            mdl._loss_history_diversity_level.append(0.0)
            # 没有 actor 时，剪枝率损失为 0
            weighted_prune_rate_loss = torch.tensor(0.0, device=lm_loss.device, requires_grad=True)
            mdl.prune_rate_loss = weighted_prune_rate_loss.detach()
            mdl.actual_prune_rate = 0.0

        # ===== 打印损失信息 =====
        print(f"\n[DEBUG Loss]")
        print(f"  lm_loss={lm_loss.item():.4f}, requires_grad={lm_loss.requires_grad}")
        print(f"  diversity_loss={weighted_diversity_loss.item():.6f}, requires_grad={weighted_diversity_loss.requires_grad}")
        print(f"  prune_rate_loss={weighted_prune_rate_loss.item():.6f}, requires_grad={weighted_prune_rate_loss.requires_grad}")

        # ===== 安全合并损失 =====
        # 在 mask 模式下，只使用 BCE loss (weighted_diversity_loss)
        if use_mask and len(mask_dict) > 0:
            # Mask 模式：只用 BCE loss
            safe_bce = weighted_diversity_loss if weighted_diversity_loss.requires_grad else weighted_diversity_loss.detach()
            total_loss = safe_bce
            print(f"  [Mask Mode] Using only BCE loss: {total_loss.item():.10f}")
        else:
            # 原始模式：lm_loss + diversity_loss + prune_rate_loss
            safe_lm = lm_loss if lm_loss.requires_grad else lm_loss.detach()
            safe_div = weighted_diversity_loss if weighted_diversity_loss.requires_grad else weighted_diversity_loss.detach()
            safe_prune_rate = weighted_prune_rate_loss if weighted_prune_rate_loss.requires_grad else weighted_prune_rate_loss.detach()
            total_loss = safe_lm + safe_div + safe_prune_rate

        # 极端情况：两边都没梯度，挂一个 dummy 防止 Trainer backward 报错
        if not total_loss.requires_grad:
            print(f"  [WARNING] total_loss 无梯度，挂 dummy")
            for p in mdl.parameters():
                if p.requires_grad:
                    total_loss = total_loss + 0.0 * p.sum()
                    break

        print(f"  total_loss={total_loss.item():.4f}, requires_grad={total_loss.requires_grad}")

        # ===== 存储损失值 =====
        mdl.lm_loss = lm_loss.detach()
        mdl.total_loss = total_loss.detach()

        mdl._loss_history_steps.append(current_step)
        mdl._loss_history_total_loss.append(total_loss.item())
        mdl._loss_history_lm_loss.append(lm_loss.item())
        
        # 只有在有 actor 时才记录 diversity_loss
        has_actor = len(token_feats_list) > 0
        
        if has_actor:
            mdl._loss_history_diversity_loss.append(weighted_diversity_loss.item())
            # 记录剪枝率损失
            mdl._loss_history_prune_rate_loss.append(mdl.prune_rate_loss.item())
            mdl._loss_history_actual_prune_rate.append(mdl.actual_prune_rate)
            # 记录 BCE loss（如果存在）
            if hasattr(mdl, 'bce_loss_raw'):
                mdl._loss_history_bce_loss.append(mdl.bce_loss_raw.item())
            else:
                mdl._loss_history_bce_loss.append(0.0)
        else:
            # 没有 actor 时，追加 0 作为占位，保持列表长度一致
            mdl._loss_history_diversity_loss.append(0.0)
            mdl._loss_history_prune_rate_loss.append(0.0)
            mdl._loss_history_actual_prune_rate.append(0.0)
            mdl._loss_history_bce_loss.append(0.0)
        # ===== 可视化 =====
        if current_step % mdl._visualization_save_steps == 0:
            print(f"[DEBUG] Step {current_step}: Saving visualization")
            # 没有 actor 时，传入 None 而不是全0的列表
            loss_diversity_for_viz = mdl._loss_history_diversity_raw if has_actor else None
            diversity_raw_for_viz = mdl._loss_history_diversity_raw if has_actor else None
            # entropy_for_viz = mdl._loss_history_entropy if has_actor else None  # 移除
            kept_for_viz = mdl._loss_history_kept if has_actor else None
            diversity_level_for_viz = mdl._loss_history_diversity_level if has_actor else None
            prune_rate_loss_for_viz = mdl._loss_history_prune_rate_loss if has_actor else None
            actual_prune_rate_for_viz = mdl._loss_history_actual_prune_rate if has_actor else None
            bce_loss_for_viz = mdl._loss_history_bce_loss if has_actor else None
            self.run_visualization(
                loss_steps=mdl._loss_history_steps,
                loss_total=mdl._loss_history_total_loss,
                loss_lm=mdl._loss_history_lm_loss,
                loss_diversity=loss_diversity_for_viz,
                diversity_raw=diversity_raw_for_viz,
                # entropy=entropy_for_viz,  # 移除
                kept=kept_for_viz,
                diversity_level=diversity_level_for_viz,
                prune_rate_loss=prune_rate_loss_for_viz,
                actual_prune_rate=actual_prune_rate_for_viz,
                bce_loss=bce_loss_for_viz,
                top_k=top_k
            )

        # ===== 清空 token_feats_list 和 keep_probs_list，防止下次误用旧数据 =====
        mdl.token_feats_list = []
        mdl.keep_probs_list = []

        return total_loss
    #--- [actor训练: 方法结束] ---    #--- [actor训练: 方法结束] --- 

    #============= Visualization Callbacks =================
    def run_visualization(
        self,
        loss_steps: List[int],
        loss_total: List[float],
        loss_lm: List[float],
        loss_diversity: Optional[List[float]] = None,
        diversity_raw: Optional[List[float]] = None,
        # entropy: Optional[List[float]] = None,  # 移除
        kept: Optional[List[float]] = None,
        diversity_level: Optional[List[float]] = None,
        prune_rate_loss: Optional[List[float]] = None,
        actual_prune_rate: Optional[List[float]] = None,
        bce_loss: Optional[List[float]] = None,
        top_k: int = 64
    ):
        """
        Run visualization at the end of each training step (每100步调用)
        
        调用 visualizer.save_visualization() 绘制曲线图
        每1000步额外调用 visualizer.save_checkpoint() 保存 checkpoint
        """
        model = self.get_model()
        current_step = model._global_step
        
        print(f"[DEBUG run_visualization] step={current_step}")
        
        # 调用 visualizer 保存可视化图像
        if model.visualizer is not None:
            has_actor = len(model._loss_history_diversity_raw) > 0
            diversity_raw = model._loss_history_diversity_raw if has_actor else None
            # entropy = model._loss_history_entropy if has_actor else None  # 移除
            kept = model._loss_history_kept if has_actor else None
            diversity_level = model._loss_history_diversity_level if has_actor else None
            prune_rate_loss = model._loss_history_prune_rate_loss if has_actor else None
            actual_prune_rate = model._loss_history_actual_prune_rate if has_actor else None
            bce_loss = model._loss_history_bce_loss if has_actor else None
            # alpha_history = model._loss_history_alpha if has_actor else None  # 移除
            model.visualizer.save_visualization(
                loss_steps=loss_steps,
                loss_total=loss_total,
                loss_lm=loss_lm,
                loss_diversity=loss_diversity,
                diversity_raw=diversity_raw,
                # entropy=entropy,  # 移除
                kept=kept,
                diversity_level=diversity_level,
                prune_rate_loss=prune_rate_loss,
                actual_prune_rate=actual_prune_rate,
                bce_loss=bce_loss,
                # alpha_history=alpha_history,  # 移除
                # r_current=r_current,  # 移除
                top_k=top_k,
                step=current_step
            )
        
        # ========== 绘制图像可视化 (mask overlay 和 probs heatmap) ==========
        # 只在 visualization 触发时绘制一次
        images_list = getattr(model, 'images_list', [])
        probs_list = getattr(model, 'probs_list', [])
        mask_hard_list = getattr(model, 'mask_hard_list', [])
        
        if model.visualizer is not None and len(images_list) > 0 and len(probs_list) > 0:
            # 只取第一个样本进行可视化
            img = images_list[0]
            probs = probs_list[0]
            mask_hard = mask_hard_list[0]
            
            # 获取 input_ids 用于解码文本
            input_ids = getattr(model, 'input_ids', None)
            
            # 绘制 mask overlay
            model.visualizer.plot_mask_overlay(
                image=img,
                mask_hard=mask_hard,
                step=current_step,
                save=True,
                input_ids=input_ids
            )
            
            # 绘制 probs 分布分析
            model.visualizer.plot_probs_analysis(
                probs=probs,
                mask_hard=mask_hard,
                input_ids=input_ids,
                step=current_step,
                save=True
            )
            
            # 清空 input_ids
            model.input_ids = None
        
        # 每1000步保存 checkpoint（包含 actor 权重）
        if current_step % model._checkpoint_save_steps == 0 and model.visualizer is not None:
            # 获取 actor 权重
            actor_state_dict = model.attention_actor.state_dict() if model.attention_actor is not None else None
            
            model.visualizer.save_checkpoint(
                actor_state_dict=actor_state_dict,
                loss_steps=loss_steps,
                loss_total=loss_total,
                loss_lm=loss_lm,
                loss_diversity=loss_diversity,
                top_k=top_k,
                step=current_step
            )
    
    def save_final_visualization(self):
        """Save final visualizations at the end of training"""
        model = self.get_model()
        current_step = model._global_step
        
        # 获取 top_k 配置
        top_k = getattr(model.config, 'actor_top_k', 64)
        
        # 绘制并保存最终图像
        if model.visualizer is not None:
            has_actor = len(model._loss_history_diversity_raw) > 0
            model.visualizer.save_visualization(
                loss_steps=model._loss_history_steps,
                loss_total=model._loss_history_total_loss,
                loss_lm=model._loss_history_lm_loss,
                loss_diversity=model._loss_history_diversity_raw if has_actor else None,
                diversity_raw=model._loss_history_diversity_raw if has_actor else None,
                # entropy=model._loss_history_entropy if has_actor else None,  # 移除
                kept=model._loss_history_kept if has_actor else None,
                diversity_level=model._loss_history_diversity_level if has_actor else None,
                prune_steps=model._loss_history_steps,
                prunkept=model._loss_history_diversity_loss,
                bce_loss=model._loss_history_bce_loss if has_actor else None,
                top_k=top_k,
                step=current_step
            )
        
        # ========== 绘制图像可视化 (mask overlay 和 probs heatmap) ==========
        images_list = getattr(model, 'images_list', [])
        probs_list = getattr(model, 'probs_list', [])
        mask_hard_list = getattr(model, 'mask_hard_list', [])
        
        if model.visualizer is not None and len(images_list) > 0 and len(probs_list) > 0:
            img = images_list[0]
            probs = probs_list[0]
            mask_hard = mask_hard_list[0]
            
            # 获取 input_ids 用于解码文本
            input_ids = getattr(model, 'input_ids', None)
            
            model.visualizer.plot_mask_overlay(
                image=img,
                mask_hard=mask_hard,
                step=current_step,
                save=True,
                input_ids=input_ids
            )
            
            model.visualizer.plot_probs_analysis(
                probs=probs,
                mask_hard=mask_hard,
                input_ids=input_ids,
                step=current_step,
                save=True
            )
            
            # 清空 input_ids
            model.input_ids = None
        
        print(f"[ActorVisualizer] Final visualizations saved at step {current_step}")
    #============= End of Visualization Callbacks =================
    #--------------Actor Loss End----------------

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, return_attention=False):
        image_features, attention_scores = self.get_model().get_vision_tower()(images, return_attention=return_attention)
        image_features = self.get_model().mm_projector(image_features)
        if return_attention:
            return image_features, attention_scores
        return image_features

    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #     images, image_sizes=None, sample_id=None
    # ):
    #     # ========== Increment Global Step ==========
    #     # 每次处理一个 batch 后，global_step 加 1
    #     self.get_model()._global_step += 1
    #     current_step = self.get_model()._global_step
    #     logger.debug(f"Step {current_step}")
    #     # ========== End of Increment Global Step ==========

    #     logger.debug(f"prepare_inputs_labels_for_multimodal called! images type={type(images)}, images shape={getattr(images, 'shape', None)}")
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         if type(images) is list:
    #             images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
    #         image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
    #         if mm_patch_merge_type == 'flat':
    #             image_features = [x.flatten(0, 1) for x in image_features]
    #         elif mm_patch_merge_type.startswith('spatial'):
    #             new_image_features = []
    #             for image_idx, image_feature in enumerate(image_features):
    #                 if image_feature.shape[0] > 1:
    #                     base_image_feature = image_feature[0]
    #                     image_feature = image_feature[1:]
    #                     height = width = self.get_vision_tower().num_patches_per_side
    #                     assert height * width == base_image_feature.shape[0]
    #                     if image_aspect_ratio == 'anyres':
    #                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
    #                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #                     else:
    #                         raise NotImplementedError
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    #                         ), dim=-1)
    #                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #                     else:
    #                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
    #                         image_feature = image_feature.flatten(0, 3)
    #                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #                 else:
    #                     image_feature = image_feature[0]
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[None].to(image_feature.device)
    #                         ), dim=0)
    #                 new_image_features.append(image_feature)
    #             image_features = new_image_features
    #         else:
    #             raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    #     else:
    #         image_features = self.encode_images(images)

    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #         raise NotImplementedError

    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- FIXME
    #     _input_ids = input_ids
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0
        
    #     #----------用于收集 actor loss 的数据-------------
    #     actor_keep_keep_probs_list: List[torch.Tensor] = []  # 收集 STE keep_mask (可导)
    #     #------------------------------------------------
        
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         cur_new_input_embeds = []
    #         cur_new_labels = []

    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_image_idx += 1

    #                 # Debug: 检查原始 image features
    #                 if self.get_model().training and torch.isnan(cur_image_features).any():
    #                     # 使用真实的训练 step
    #                     step_info = current_step
    #                     print(f"[DEBUG] Step {step_info}: NaN in cur_image_features BEFORE mask!")

    #                 # ----------Apply Attention-based image feature pruning----------
    #                 attention_actor = self.get_model().attention_actor
    #                 logger.debug(f"ARCH TEST attention_actor = {type(attention_actor)}")
    #                 if attention_actor is not None:
    #                     # Get text embeddings for this segment (already encoded)
    #                     text_embeds_seg = cur_input_embeds_no_im[i]  # [T_text, D]
                        
    #                     # 使用 AttentionPruningActor 的新接口
    #                     # x_img: [T_img, D], x_txt: [T_text, D] (支持无 batch 维度)
    #                     x_img = cur_image_features  # [T_img, D]
    #                     x_txt = text_embeds_seg     # [T_text, D]
                        
    #                     # 创建文本 mask（True keeps）#######小伏笔#######
    #                     txt_mask = torch.ones(x_txt.shape[0], dtype=torch.bool, device=x_txt.device)
                        
    #                     # 调用 AttentionPruningActor（支持无 batch 维度输入）
    #                     actor_output = self.get_model().attention_actor(
    #                         x_img=x_img,
    #                         x_txt=x_txt,
    #                         txt_mask=txt_mask,
    #                         tau=self.config.T if hasattr(self.config, 'T') else 1.0
    #                     )
                        
    #                     logits = actor_output.logits  # [T_img]
    #                     probs = actor_output.probs   # [T_img]
                        
    #                     p = probs  # [T_img] 保留概率 (AttentionPruningActor 已经内部做了 sigmoid)
                        
    #                     # Create mask: keep tokens where p > 0.5 (prune if p <= 0.5)
    #                     # Use Straight-Through Estimator (STE) for differentiable pruning
    #                     # Forward: use hard mask (0 or 1)
    #                     # Backward: gradient flows through p (bypass the threshold)
    #                     p_hard = ((p > 0.5).to(p.dtype))  # [T_img], 0.0 or 1.0 (保持与 p 相同的 dtype)
    #                     keep_mask = p_hard.detach() + (p - p.detach())  # STE: [T_img]
                        
    #                     # Debug: 检查 p 和 mask 的分布
    #                     if self.get_model().training:
    #                         # 使用真实的训练 step
    #                         step_info = current_step
    #                         p_min, p_max = p.min().item(), p.max().item()
    #                         p_mean = p.mean().item()
    #                         kept_tokens = (keep_mask > 0.5).sum().item()
    #                         total_tokens = p.shape[0]
    #                         print(f"[DEBUG] Step {step_info}: p_min={p_min:.4f}, p_max={p_max:.4f}, p_mean={p_mean:.4f}, kept={kept_tokens}/{total_tokens}")
                        
    #                     cur_image_features = cur_image_features * keep_mask.unsqueeze(-1)  # [T_img, D]
                        
    #                     # 收集 keep_mask 用于计算损失
    #                     if self.get_model().training:
    #                         actor_keep_keep_probs_list.append(keep_mask)
    #                 # ---------------------Pruning End-------------------------
                    
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

    #     #-------------计算 actor loss（基于 prune_ratio_loss）--------------
    #     # 总损失 = LLM loss + λ * (mean(keep_mask) - τ)²
        
    #     if len(actor_keep_keep_probs_list) > 0:
    #         # 从 config 获取 target_ratio 和 lambda_prune
    #         target_ratio = getattr(self.config, 'target_keep_ratio', 1.0)
    #         lambda_prune = getattr(self.config, 'lambda_prune', 1.0)
            
    #         # compute_actor_loss_from_list 返回 (prune_loss, weighted_prune_loss)
    #         # 使用 STE 的 keep_mask 计算损失（可导）
    #         prune_loss, weighted_prune_loss = compute_actor_loss_from_list(
    #             actor_keep_keep_probs_list,
    #             target_ratio=target_ratio,
    #             lambda_prune=lambda_prune
    #         )
            
    #         # actor_loss = λ * prune_loss
    #         actor_loss = weighted_prune_loss
            
    #         # 存储用于日志记录
    #         model=self.get_model()
    #         model.actor_loss = actor_loss
    #         model.prune_loss = prune_loss
    #         model.weighted_prune_loss = weighted_prune_loss
            
    #         # 存储最后一个 p 值用于监控
    #         model.p = actor_keep_keep_probs_list[-1].detach()
    #         # 同时计算并保存 keep mask（转换为 float32 以支持 numpy）
    #         model.keep_mask = actor_keep_keep_probs_list[-1].detach().float().cpu().numpy()
            
    #         # 计算当前 keep ratio 并记录到历史数组
    #         current_keep_ratio = actor_keep_keep_probs_list[-1].mean().item()
    #         model._loss_history_steps.append(model._global_step)
    #         model._loss_history_actor_loss.append(current_keep_ratio)
            
    #         # 每 100 步保存可视化
    #         if model._global_step % model._visualization_save_steps == 0:
    #             print(f"[DEBUG][DEBUG] Step {model._global_step}: Saving visualization")
    #             self.run_visualization(
    #                 loss_steps=model._loss_history_steps,
    #                 loss_total=model._loss_history_total_loss,
    #                 loss_lm=model._loss_history_lm_loss,
    #                 prune_steps=model._loss_history_steps,
    #                 prunkept=model._loss_history_actor_loss
    #             )
    #     #---------------------actor loss end------------------------

    #     # Truncate sequences to max length as image embeddings can make the sequence longer
    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # Combine them
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 attention_mask[i, -cur_len:] = True
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 attention_mask[i, :cur_len] = True
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    # def prepare_inputs_labels_for_multimodal(
    #         self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #         images, image_sizes=None, sample_id=None
    #     ):
    #     # ========== Increment Global Step ==========
    #     # 每次处理一个 batch 后，global_step 加 1
    #     if torch.is_grad_enabled():

    #         self.get_model()._global_step += 1
    #     current_step = self.get_model()._global_step
    #     logger.debug(f"Step {current_step}")
    #     # ========== End of Increment Global Step ==========

    #     logger.debug(f"prepare_inputs_labels_for_multimodal called! images type={type(images)}, images shape={getattr(images, 'shape', None)}")
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         if type(images) is list:
    #             images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
    #         image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
    #         if mm_patch_merge_type == 'flat':
    #             image_features = [x.flatten(0, 1) for x in image_features]
    #         elif mm_patch_merge_type.startswith('spatial'):
    #             new_image_features = []
    #             for image_idx, image_feature in enumerate(image_features):
    #                 if image_feature.shape[0] > 1:
    #                     base_image_feature = image_feature[0]
    #                     image_feature = image_feature[1:]
    #                     height = width = self.get_vision_tower().num_patches_per_side
    #                     assert height * width == base_image_feature.shape[0]
    #                     if image_aspect_ratio == 'anyres':
    #                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
    #                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #                     else:
    #                         raise NotImplementedError
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    #                         ), dim=-1)
    #                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #                     else:
    #                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
    #                         image_feature = image_feature.flatten(0, 3)
    #                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #                 else:
    #                     image_feature = image_feature[0]
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[None].to(image_feature.device)
    #                         ), dim=0)
    #                 new_image_features.append(image_feature)
    #             image_features = new_image_features
    #         else:
    #             raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    #     else:
    #         image_features = self.encode_images(images)

    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #         raise NotImplementedError

    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     _input_ids = input_ids
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0
        
    #     #----------用于收集 actor loss 的数据-------------
    #     actor_keep_keep_probs_list: List[torch.Tensor] = []  # 收集 STE keep_mask (可导)
    #     #------------------------------------------------
        
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         cur_new_input_embeds = []
    #         cur_new_labels = []

    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_image_idx += 1

    #                 if self.get_model().training and torch.isnan(cur_image_features).any():
    #                     step_info = current_step
    #                     print(f"[DEBUG] Step {step_info}: NaN in cur_image_features BEFORE mask!")

    #                 # ----------Apply Attention-based image feature pruning----------
    #                 attention_actor = self.get_model().attention_actor
    #                 logger.debug(f"ARCH TEST attention_actor = {type(attention_actor)}")
    #                 if attention_actor is not None:
    #                     text_embeds_seg = cur_input_embeds_no_im[i]
                        
    #                     x_img = cur_image_features
    #                     x_txt = text_embeds_seg
                        
    #                     txt_mask = torch.ones(x_txt.shape[0], dtype=torch.bool, device=x_txt.device)
                        
    #                     actor_output = self.get_model().attention_actor(
    #                         x_img=x_img,
    #                         x_txt=x_txt,
    #                         txt_mask=txt_mask,
    #                         tau=self.config.T if hasattr(self.config, 'T') else 1.0
    #                     )
                        
    #                     logits = actor_output.logits
    #                     probs = actor_output.probs
                        
    #                     p = probs
                        
    #                     p_hard = ((p > 0.5).to(p.dtype))
    #                     keep_mask = p_hard.detach() + (p - p.detach())  # STE
                        
    #                     if self.get_model().training:
    #                         if self.get_model().training and torch.is_grad_enabled():

    #                             step_info = current_step
    #                             p_min, p_max = p.min().item(), p.max().item()
    #                             p_mean = p.mean().item()
    #                             kept_tokens = (keep_mask > 0.5).sum().item()
    #                             total_tokens = p.shape[0]
    #                             print(f"[DEBUG] Step {step_info}: p_min={p_min:.4f}, p_max={p_max:.4f}, p_mean={p_mean:.4f}, kept={kept_tokens}/{total_tokens}")
                        
    #                     cur_image_features = cur_image_features * keep_mask.unsqueeze(-1)
                        
    #                     if self.get_model().training:
    #                         actor_keep_keep_probs_list.append(keep_mask)
    #                 # ---------------------Pruning End-------------------------
                    
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

    #     #-------------[改动] 保存 keep_keep_probs_list 而不是计算 actor_loss--------------
    #     # 关键改动：不在 forward 中计算 actor_loss 标量，
    #     # 而是保存原始的 keep_keep_probs_list，在 compute_loss 中重新计算。
    #     # 这样即使 gradient checkpointing 导致 forward 被调用两次，
    #     # _last_keep_keep_probs_list 始终指向最新一次 forward 的计算图。
    #     model = self.get_model()
        
    #     if len(actor_keep_keep_probs_list) > 0:
    #         # 保存 keep_keep_probs_list（带梯度），compute_loss 中会用它计算 actor_loss
    #         model.keep_keep_probs_list = actor_keep_keep_probs_list
            
    #         # 以下仅用于监控/可视化（detach，不影响计算图）
    #         model.p = actor_keep_keep_probs_list[-1].detach()
    #         model.keep_mask = actor_keep_keep_probs_list[-1].detach().float().cpu().numpy()
            
    #         current_keep_ratio = actor_keep_keep_probs_list[-1].mean().item()
    #         model._loss_history_steps.append(model._global_step)
    #         model._loss_history_actor_loss.append(current_keep_ratio)
            
    #         if model._global_step % model._visualization_save_steps == 0:
    #             print(f"[DEBUG][DEBUG] Step {model._global_step}: Saving visualization")
    #             self.run_visualization(
    #                 loss_steps=model._loss_history_steps,
    #                 loss_total=model._loss_history_total_loss,
    #                 loss_lm=model._loss_history_lm_loss,
    #                 prune_steps=model._loss_history_steps,
    #                 prunkept=model._loss_history_actor_loss
    #             )
    #     else:
    #         model.keep_keep_probs_list = []
    #     #---------------------[改动结束]------------------------

    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 attention_mask[i, -cur_len:] = True
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 attention_mask[i, :cur_len] = True
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None, sample_id=None, image_files=None
        ):
            # ========== Global Step（仅读取，不自增） ==========
            # 自增在 compute_loss 中进行，避免 gradient checkpointing 重复计数
            current_step = self.get_model()._global_step
            logger.debug(f"Step {current_step}")
            # ========== End ==========

            logger.debug(f"prepare_inputs_labels_for_multimodal called! images type={type(images)}, images shape={getattr(images, 'shape', None)}")
            vision_tower = self.get_vision_tower()
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels

            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                image_features, attention_scores_tensor = self.encode_images(concat_images, return_attention=True)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                # attention_scores_tensor 形状是 [batch, seq_len-1]，直接在 batch 维度上拆分
                # 每张图片对应 attention_scores_tensor[i]
                num_images = len(images)
                attention_scores_list = [attention_scores_tensor[i] for i in range(num_images)]
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                if mm_patch_merge_type == 'flat':
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features, attention_scores_tensor = self.encode_images(images, return_attention=True)
                # images 可能是 [batch, C, H, W] 格式，需要按 batch 维度拆分
                # attention_scores_tensor 形状: [batch, seq_len-1]
                batch_size = attention_scores_tensor.shape[0]

            # TODO: image start / end is not implemented here to support pretraining.
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                raise NotImplementedError

            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            _input_ids = input_ids
            input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

            new_input_embeds = []
            new_labels = []
            cur_image_idx = 0

            # 新增：用于保存每个图像的 keep_mask（用于注意力掩码）
            STE_mask_list = []
            
            # 新增：BCE loss 模式下的 label_mask 列表
            use_mask = getattr(self.config, 'use_mask', False)
            
            # 懒加载 mask 数据
            mask_dict = getattr(self.get_model(), 'mask_dict', {})
            if use_mask and len(mask_dict) == 0:
                # 尝试从 config 获取 mask_path 并加载
                mask_path = getattr(self.config, 'mask_path', None)
                if mask_path:
                    import pickle
                    print(f"[INFO] Lazily loading mask data from {mask_path}")
                    with open(mask_path, 'rb') as f:
                        mask_data = pickle.load(f)
                    
                    # 构建 sample_id 到 top_indices 的映射
                    if isinstance(mask_data, dict) and 'masks' in mask_data:
                        for mask_entry in mask_data['masks']:
                            sample_id = mask_entry.get('sample_id', '')
                            top_indices = mask_entry.get('top_indices', [])
                            if sample_id and len(top_indices) > 0:
                                mask_dict[sample_id] = top_indices
                        
                        # 保存到 get_model() 以便后续使用
                        self.get_model().mask_dict = mask_dict
                        print(f"[INFO] Loaded {len(mask_dict)} masks")
            
            label_mask_list = []  # 用于 BCE loss
            
            # 获取 image_files 列表（用于 BCE loss 的 label_mask 查找）
            # 优先使用传入的 image_files 参数，否则从 batch 中获取
            if image_files is not None:
                sample_ids = image_files
            else:
                sample_ids = getattr(self, '_sample_ids', [None] * len(input_ids))
            
            #----------用于收集 actor diversity loss 的数据-------------
            actor_token_feats_list: List[torch.Tensor] = []  # x_fused
            # 新增：用于可视化的数据（images, probs, mask_hard）
            actor_images_list: List[torch.Tensor] = []        # 原始图像
            actor_probs_list: List[torch.Tensor] = []        # soft mask (probs)
            actor_mask_hard_list: List[torch.Tensor] = []    # hard mask (top-k indices)
            # 用于追踪图像 token 在最终序列中的位置：(batch_idx, start_pos, keep_mask)
            STE_mask_positions: List[Tuple[int, int, torch.Tensor]] = []
            #------------------------------------------------

            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    cur_image_idx += 1
                    continue

                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    tokenizer = get_global_tokenizer()
                    if tokenizer is not None:
                        print(f"[DEBUG TEXT] tokenizer exists={tokenizer is not None}, tokenizer={type(tokenizer)}")
                    if i==num_images:
                        print(f"[DEBUG] i={i}, num_images={num_images}")
                        print(f"[DEBUG] cur_new_input_embeds={cur_input_ids_noim[i].shape}")
                        decoded_text = tokenizer.decode(cur_input_ids_noim[i], skip_special_tokens=True)
                        print(f"[DEBUG] decoded_text={decoded_text}")
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1

                        # ----------Apply Attention-based image feature pruning----------
                        attention_actor = self.get_model().attention_actor
                        logger.debug(f"ARCH TEST attention_actor = {type(attention_actor)}")
                        print(f"[DEBUG TEXT] attention_actor={attention_actor is not None}, i={i}")
                        if attention_actor is not None:
                            # Debug: 打印当前 segment 的 token IDs 和对应的文本内容
                            cur_text_token_ids = cur_input_ids_noim[i]
                            print(f"[DEBUG TEXT] Segment {i}: token_ids exists={cur_text_token_ids is not None}")
                            if cur_text_token_ids is not None:
                                decoded_text = tokenizer.decode(cur_text_token_ids, skip_special_tokens=True)
                                print(f"[DEBUG TEXT] Segment {i}: decoded='{decoded_text[:200]}...'")
                            else:
                                print(f"[DEBUG TEXT] Segment {i}: values={cur_text_token_ids[:20].tolist()}")
                            ###这里有BUG，这个是image之前的文本token，不包含实际的user_prompt，所以不能拿来作为text

                            x_img = cur_image_features
                            x_txt = cur_input_embeds_no_im[i]

                            # 调用新的 dynamicvlm_actor
                            actor_output = self.get_model().attention_actor(
                                x_img=x_img,
                                x_txt=x_txt,
                                tau=getattr(self.config, 'actor_tau', 1.0),
                                top_k=getattr(self.config, 'actor_top_k', 64)
                            )

                            prune_probs = actor_output.prune_probs  # P(prune)
                            keep_probs = actor_output.keep_probs   # P(keep)
                            mask_hard = actor_output.mask_hard     # hard mask: 0.0 or 1.0

                            keep_mask = mask_hard.detach() + (keep_probs - keep_probs.detach())  # STE

                            cur_image_features = cur_image_features * keep_mask.unsqueeze(-1)
                            
                            # 保存 keep_mask 用于后续注意力掩码和损失计算
                            # 注意：不要 detach，否则损失无法反向传播
                            STE_mask_list.append(keep_mask)
                            
                            # 追踪图像 token 在最终序列中的位置 (start_pos, keep_mask)
                            # 当前 cur_new_input_embeds 的长度就是图像的起始位置
                            img_start_pos = sum(x.shape[0] for x in cur_new_input_embeds)
                            STE_mask_positions.append((batch_idx, img_start_pos, keep_mask))
                            
                            # ===== BCE Loss: 查找并保存 label_mask =====
                            if use_mask and len(mask_dict) > 0:
                                # 获取当前图像的 sample_id (需要从完整路径中提取图片 ID)
                                full_sample_id = sample_ids[batch_idx] if batch_idx < len(sample_ids) else None
                                
                                # 从完整路径中提取图片 ID (如 "coco/train2017/000000570452.jpg" -> "000000570452")
                                if full_sample_id:
                                    # 提取文件名（去掉路径和扩展名）
                                    sample_id = full_sample_id.split('/')[-1].rsplit('.', 1)[0]
                                else:
                                    sample_id = None
                                
                                # Debug: 打印 sample_id 和 mask_dict 状态
                                if batch_idx == 0:
                                    print(f"[DEBUG] full_sample_id={full_sample_id}, extracted_id={sample_id}, mask_dict keys (first 3)={list(mask_dict.keys())[:3] if mask_dict else 'empty'}")
                                    if sample_id:
                                        print(f"[DEBUG] sample_id in mask_dict: {sample_id in mask_dict}")
                                if sample_id and sample_id in mask_dict:
                                    # mask_dict[sample_id] 现在直接就是 top_indices 列表
                                    top_indices = mask_dict[sample_id]
                                    
                                    # Debug
                                    if batch_idx == 0:
                                        print(f"[DEBUG] Found mask: top_indices count = {len(top_indices)}")
                                    
                                    num_total_tokens = keep_mask.shape[0]  # 与 keep_mask 长度对齐
                                    
                                    # 创建 ground truth mask (0/1)
                                    gt_mask = torch.zeros(num_total_tokens, dtype=keep_mask.dtype, device=keep_mask.device)
                                    for idx in top_indices:
                                        if idx < num_total_tokens:
                                            gt_mask[idx] = 1.0
                                    
                                    label_mask_list.append(gt_mask)
                                else:
                                    # 如果没有找到对应的 mask，使用 None 占位，后续会跳过
                                    label_mask_list.append(None)
                            # ===== BCE Loss 结束 =====
                            
                            # Ensure cur_image_features has the same dtype as the model
                            # (actor converts to fp16, but we need to ensure consistency)
                            model_dtype = next(self.get_model().parameters()).dtype
                            cur_image_features = cur_image_features.to(dtype=model_dtype)

                            # 收集 x_img 和 keep_probs（用于可视化）
                            # 训练模式：带梯度；推理模式：detach
                            if self.get_model().training:
                                actor_token_feats_list.append(x_img)  # 原始图像特征
                            else:
                                # 推理模式：记录用于可视化
                                actor_token_feats_list.append(x_img.detach())
                            
                            # 收集 images, keep_probs, mask_hard 用于可视化
                            # images 是列表，取当前处理的图像索引
                            if images is not None and cur_image_idx - 1 < len(images):
                                img = images[cur_image_idx - 1]  # cur_image_idx 已经+1了，所以-1
                                actor_images_list.append(img.detach().cpu() if isinstance(img, torch.Tensor) else img)
                                # 训练模式：保留梯度用于 actor loss 反向传播；推理模式：detach
                                if self.get_model().training:
                                    actor_probs_list.append(keep_probs)  # P(keep) 保留梯度
                                    actor_mask_hard_list.append(mask_hard)  # 保留梯度
                                else:
                                    actor_probs_list.append(keep_probs.detach().cpu())
                                    actor_mask_hard_list.append(mask_hard.detach().cpu())
                            
                            # 收集 DualHeadPruningActor 的 alpha 用于可视化  # 移除
                        # if alpha is not None:
                        #     if self.get_model().training:
                        #         actor_alpha_list.append(alpha)  # 保留梯度
                        #     else:
                        #         actor_alpha_list.append(alpha.detach().cpu() if isinstance(alpha, torch.Tensor) else alpha)
                        else:
                            # 没有 Actor 时，创建一个全1的 mask（不剪枝）
                            STE_mask_list.append(torch.ones(cur_image_features.shape[0], device=cur_image_features.device, dtype=cur_image_features.dtype))
                            # 追踪位置（全1 mask 不需要剪枝，但需要记录位置）
                            img_start_pos = sum(x.shape[0] for x in cur_new_input_embeds)
                            STE_mask_positions.append((batch_idx, img_start_pos, torch.ones(cur_image_features.shape[0], device=cur_image_features.device, dtype=cur_image_features.dtype)))
                        # ---------------------Pruning End-------------------------

                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

            #------------- 保存 token_feats_list 和 keep_probs_list（用于 compute_loss 计算 diversity loss） --------------
            # compute_loss 中会用 _last_token_feats_list 和 _last_keep_probs_list 计算 diversity loss
            model = self.get_model()
            if len(actor_token_feats_list) > 0:
                model.images_list = actor_images_list
                model.token_feats_list = actor_token_feats_list
                model.keep_probs_list = actor_probs_list  # 使用 probs 作为 mask
                model.hard_mask_list = actor_mask_hard_list
                model.STE_mask_list = STE_mask_list
                model.STE_mask_positions = STE_mask_positions  # 图像 token 位置
                model.label_mask_list = label_mask_list  # BCE loss 需要的 label_mask
            else:
                model.token_feats_list = []
                model.keep_probs_list = []
                model.images_list = []
                model.hard_mask_list = []
                model.STE_mask_list = []
                model.STE_mask_positions = []
                model.label_mask_list = []
            #---------------------结束------------------------

            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            # ===== 应用剪枝注意力掩码 =====
            # 获取保存的图像 token 位置信息
            model = self.get_model()
            STE_mask_positions = getattr(model, 'STE_mask_positions', [])
            
            if len(STE_mask_positions) > 0 and attention_mask is not None:
                padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')
                
                # Debug: 打印原始 attention_mask 信息
                original_valid_count = attention_mask.sum().item()
                
                for batch_idx, img_start_pos, keep_mask in STE_mask_positions:
                    num_img_tokens = keep_mask.shape[0]
                    # 检查是否在有效范围内
                    if padding_side == "left":
                        # left padding: 有效位置在 [-cur_len:] 范围
                        # 需要根据实际序列长度计算
                        cur_len = new_input_embeds[batch_idx].shape[0]
                        img_start_padded = max_len - cur_len + img_start_pos
                    else:
                        img_start_padded = img_start_pos
                    
                    # Debug: 打印位置信息
                    pruned_count = 0
                    kept_count = 0
                    # 对每个被剪枝的 token，设置 attention_mask 为 0
                    for token_idx in range(num_img_tokens):
                        if keep_mask[token_idx].item() == 0:  # 被剪枝
                            pruned_count += 1
                            key_pos = img_start_padded + token_idx
                            if key_pos < max_len:
                                attention_mask[batch_idx, key_pos] = False
                        else:
                            kept_count += 1
                    
                    # Debug: 打印剪枝信息
                    if batch_idx == 0:  # 只打印第一个 batch 的 debug 信息
                        print(f"[DEBUG Attention] batch_idx={batch_idx}, img_start_pos={img_start_pos}, img_start_padded={img_start_padded}, num_img_tokens={num_img_tokens}, pruned={pruned_count}, kept={kept_count}")
                
                # Debug: 打印 attention_mask 变化
                final_valid_count = attention_mask.sum().item()
                print(f"[DEBUG Attention] Original valid: {original_valid_count}, After pruning: {final_valid_count}, Masked out: {original_valid_count - final_valid_count}")

            if _labels is None:
                new_labels = None
            else:
                new_labels = new_labels_padded

            if _attention_mask is None:
                attention_mask = None
            else:
                attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

            if _position_ids is None:
                position_ids = None

            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
