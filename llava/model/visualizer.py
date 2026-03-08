"""
Dual Head Pruning Actor 可视化工具和损失函数
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def smooth_curve(data: List[float], alpha: float = 0.9) -> List[float]:
    """
    对曲线进行指数移动平均平滑
    
    Args:
        data: 原始数据列表
        alpha: 平滑因子，值越大越平滑 (0 < alpha < 1)
    
    Returns:
        平滑后的数据列表
    """
    if len(data) == 0:
        return []
    
    smoothed = []
    current = data[0]
    for value in data:
        current = alpha * current + (1 - alpha) * value
        smoothed.append(current)
    return smoothed


class ActorVisualizer:
    """
    Dual Head Pruning Actor 可视化工具
    
    用于记录和可视化:
    - 损失曲线 (total_loss, lm_loss)
    - 保留比例 (keep_ratio)
    """
    
    def __init__(
        self,
        save_dir: str = "visualizations",
        smooth_alpha: float = 0.9
    ):
        self.save_dir = save_dir
        self.smooth_alpha = smooth_alpha  # 平滑因子
        
        # 创建子目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'loss_curves'), exist_ok=True)
        
        self.current_step = 0

    def update_step(self, step: int):
        """Update the current step in visualizer"""
        self.current_step = step

    # ============== Plotting Methods (传入数组参数) ==============
    
    def plot_loss_curves(
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
        top_k: int = 64,
        step: int = 0,
        save: bool = True
    ):
        """绘制并保存损失曲线图（分别绘制 total_loss、lm_loss、diversity_loss、E(keep) 和 diversity_level）"""
        if len(loss_steps) == 0:
            return
        
        # ========== 处理长度不一致问题 ==========
        # 确保 loss_steps, loss_total, loss_lm 长度一致
        min_len = min(len(loss_steps), len(loss_total), len(loss_lm))
        loss_steps_trimmed = loss_steps[:min_len]
        loss_total_trimmed = loss_total[:min_len]
        loss_lm_trimmed = loss_lm[:min_len]
        
        # 计算平滑曲线
        loss_total_smooth = smooth_curve(loss_total_trimmed, self.smooth_alpha)
        loss_lm_smooth = smooth_curve(loss_lm_trimmed, self.smooth_alpha)
        loss_diversity_smooth = smooth_curve(loss_diversity, self.smooth_alpha) if loss_diversity else None
        diversity_raw_smooth = smooth_curve(diversity_raw, self.smooth_alpha) if diversity_raw else None
        # entropy_smooth = smooth_curve(entropy, self.smooth_alpha) if entropy else None  # 移除
        kept_smooth = smooth_curve(kept, self.smooth_alpha) if kept else None
        diversity_level_smooth = smooth_curve(diversity_level, self.smooth_alpha) if diversity_level else None
        # 剪枝率损失平滑曲线
        prune_rate_loss_smooth = smooth_curve(prune_rate_loss, self.smooth_alpha) if prune_rate_loss else None
        actual_prune_rate_smooth = smooth_curve(actual_prune_rate, self.smooth_alpha) if actual_prune_rate else None
        # BCE loss 平滑曲线
        bce_loss_smooth = smooth_curve(bce_loss, self.smooth_alpha) if bce_loss else None
        
        # ========== 图1: Total Loss ==========
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # 原始曲线
        ax1.plot(loss_steps_trimmed, loss_total_trimmed, 'b-', linewidth=1.5, alpha=0.5, label='Total Loss (Raw)')
        # 平滑曲线
        ax1.plot(loss_steps_trimmed, loss_total_smooth, 'b-', linewidth=2.5, label=f'Total Loss (Smoothed, α={self.smooth_alpha})')
        
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Total Loss Over Training (top_k={top_k})', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        final_loss = loss_total_trimmed[-1] if loss_total_trimmed else 0
        min_loss = min(loss_total_trimmed) if loss_total_trimmed else 0
        avg_loss = np.mean(loss_total_trimmed) if loss_total_trimmed else 0
        stats_text = f'Final: {final_loss:.4f}\nMin: {min_loss:.4f}\nAvg: {avg_loss:.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            loss_path = os.path.join(self.save_dir, 'loss_curves', f'total_loss_step_{step}.png')
            os.makedirs(os.path.dirname(loss_path), exist_ok=True)
            plt.savefig(loss_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[ActorVisualizer] Saved total loss curve: {loss_path}")
        else:
            plt.show()
        
        # ========== 图2: LLM Loss ==========
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        # 原始曲线
        ax2.plot(loss_steps_trimmed, loss_lm_trimmed, 'g-', linewidth=1.5, alpha=0.5, label='LLM Loss (Raw)')
        # 平滑曲线
        ax2.plot(loss_steps_trimmed, loss_lm_smooth, 'g-', linewidth=2.5, label=f'LLM Loss (Smoothed, α={self.smooth_alpha})')
        
        ax2.set_xlabel('Steps', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'LLM Loss Over Training (top_k={top_k})', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        final_loss = loss_lm_trimmed[-1] if loss_lm_trimmed else 0
        min_loss = min(loss_lm_trimmed) if loss_lm_trimmed else 0
        avg_loss = np.mean(loss_lm_trimmed) if loss_lm_trimmed else 0
        stats_text = f'Final: {final_loss:.4f}\nMin: {min_loss:.4f}\nAvg: {avg_loss:.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            loss_path = os.path.join(self.save_dir, 'loss_curves', f'llm_loss_step_{step}.png')
            os.makedirs(os.path.dirname(loss_path), exist_ok=True)
            plt.savefig(loss_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[ActorVisualizer] Saved LLM loss curve: {loss_path}")
        else:
            plt.show()
        
        # ========== 图3: Diversity Loss ==========
        if loss_diversity and len(loss_diversity) > 0:
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
            
            # 确保 loss_steps 和 loss_diversity 长度一致
            min_len = min(len(loss_steps), len(loss_diversity))
            loss_steps_div = loss_steps[:min_len]
            loss_diversity_trimmed = loss_diversity[:min_len]
            
            # 原始曲线
            ax3.plot(loss_steps_div, loss_diversity_trimmed, 'r-', linewidth=1.5, alpha=0.5, label='Diversity Loss (Raw)')
            # 平滑曲线
            loss_diversity_smooth_trimmed = loss_diversity_smooth[:min_len] if loss_diversity_smooth else None
            if loss_diversity_smooth_trimmed:
                ax3.plot(loss_steps_div, loss_diversity_smooth_trimmed, 'r-', linewidth=2.5, 
                        label=f'Diversity Loss (Smoothed, α={self.smooth_alpha})')
            
            ax3.set_xlabel('Steps', fontsize=12)
            ax3.set_ylabel('Loss', fontsize=12)
            ax3.set_title(f'Diversity Loss Over Training (top_k={top_k})', fontsize=14)
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息
            final_loss = loss_diversity_trimmed[-1] if loss_diversity_trimmed else 0
            min_loss = min(loss_diversity_trimmed) if loss_diversity_trimmed else 0
            avg_loss = np.mean(loss_diversity_trimmed) if loss_diversity_trimmed else 0
            stats_text = f'Final: {final_loss:.4f}\nMin: {min_loss:.4f}\nAvg: {avg_loss:.4f}'
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                loss_path = os.path.join(self.save_dir, 'loss_curves', f'diversity_loss_step_{step}.png')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved Diversity loss curve: {loss_path}")
            else:
                plt.show()
        
        # ========== 图6: E(keep) ==========
        if kept and len(kept) > 0:
            fig6, ax6 = plt.subplots(1, 1, figsize=(10, 6))
            
            # 确保 loss_steps 和 kept 长度一致
            min_len = min(len(loss_steps), len(kept))
            loss_steps_ek = loss_steps[:min_len]
            kept_trimmed = kept[:min_len]
            
            # 原始曲线
            ax6.plot(loss_steps_ek, kept_trimmed, 'orange', linewidth=1.5, alpha=0.5, label='E(keep) (Raw)')
            # 平滑曲线
            kept_smooth_trimmed = kept_smooth[:min_len] if kept_smooth else None
            if kept_smooth_trimmed:
                ax6.plot(loss_steps_ek, kept_smooth_trimmed, 'orange', linewidth=2.5, 
                        label=f'E(keep) (Smoothed, α={self.smooth_alpha})')
            
            ax6.set_xlabel('Steps', fontsize=12)
            ax6.set_ylabel('E(keep) = sum(probs)', fontsize=12)
            ax6.set_title(f'Expected Keep Tokens Over Training (top_k={top_k})', fontsize=14)
            ax6.legend(loc='upper right', fontsize=10)
            ax6.grid(True, alpha=0.3)
            
            # 添加统计信息
            final_ek = kept_trimmed[-1] if kept_trimmed else 0
            min_ek = min(kept_trimmed) if kept_trimmed else 0
            avg_ek = np.mean(kept_trimmed) if kept_trimmed else 0
            stats_text = f'Final: {final_ek:.4f}\nMin: {min_ek:.4f}\nAvg: {avg_ek:.4f}'
            ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                loss_path = os.path.join(self.save_dir, 'loss_curves', f'kept_step_{step}.png')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved E(keep) curve: {loss_path}")
            else:
                plt.show()
        
        # ========== 图7: Diversity Level (1 - Similarity) ==========
        if diversity_level and len(diversity_level) > 0:
            fig7, ax7 = plt.subplots(1, 1, figsize=(10, 6))
            
            # 确保 loss_steps 和 diversity_level 长度一致
            min_len = min(len(loss_steps), len(diversity_level))
            loss_steps_div = loss_steps[:min_len]
            diversity_level_trimmed = diversity_level[:min_len]
            
            # 原始曲线
            ax7.plot(loss_steps_div, diversity_level_trimmed, 'purple', linewidth=1.5, alpha=0.5, label='Diversity Level (Raw)')
            # 平滑曲线
            diversity_level_smooth_trimmed = diversity_level_smooth[:min_len] if diversity_level_smooth else None
            if diversity_level_smooth_trimmed:
                ax7.plot(loss_steps_div, diversity_level_smooth_trimmed, 'purple', linewidth=2.5, 
                        label=f'Diversity Level (Smoothed, α={self.smooth_alpha})')
            
            ax7.set_xlabel('Steps', fontsize=12)
            ax7.set_ylabel('Diversity Level (1 - Similarity)', fontsize=12)
            ax7.set_title(f'Diversity Level Over Training (top_k={top_k})', fontsize=14)
            ax7.legend(loc='upper right', fontsize=10)
            ax7.grid(True, alpha=0.3)
            
            # 添加统计信息
            final_div = diversity_level_trimmed[-1] if diversity_level_trimmed else 0
            min_div = min(diversity_level_trimmed) if diversity_level_trimmed else 0
            avg_div = np.mean(diversity_level_trimmed) if diversity_level_trimmed else 0
            stats_text = f'Final: {final_div:.4f}\nMin: {min_div:.4f}\nAvg: {avg_div:.4f}'
            ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                loss_path = os.path.join(self.save_dir, 'loss_curves', f'diversity_level_step_{step}.png')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved Diversity Level curve: {loss_path}")
            else:
                plt.show()
        
        # ========== 图8: Pruning Rate Loss ==========
        if prune_rate_loss and len(prune_rate_loss) > 0:
            fig8, ax8 = plt.subplots(1, 1, figsize=(10, 6))
            
            # 确保长度一致
            min_len = min(len(loss_steps), len(prune_rate_loss))
            loss_steps_pr = loss_steps[:min_len]
            prune_rate_loss_trimmed = prune_rate_loss[:min_len]
            
            # 原始曲线
            ax8.plot(loss_steps_pr, prune_rate_loss_trimmed, 'orange', linewidth=1.5, alpha=0.5, label='Pruning Rate Loss (Raw)')
            # 平滑曲线
            prune_rate_loss_smooth_trimmed = prune_rate_loss_smooth[:min_len] if prune_rate_loss_smooth else None
            if prune_rate_loss_smooth_trimmed:
                ax8.plot(loss_steps_pr, prune_rate_loss_smooth_trimmed, 'orange', linewidth=2.5, 
                        label=f'Pruning Rate Loss (Smoothed, α={self.smooth_alpha})')
            
            ax8.set_xlabel('Steps', fontsize=12)
            ax8.set_ylabel('Loss', fontsize=12)
            ax8.set_title(f'Pruning Rate Loss Over Training (top_k={top_k})', fontsize=14)
            ax8.legend(loc='upper right', fontsize=10)
            ax8.grid(True, alpha=0.3)
            
            # 添加统计信息
            final_pr = prune_rate_loss_trimmed[-1] if prune_rate_loss_trimmed else 0
            min_pr = min(prune_rate_loss_trimmed) if prune_rate_loss_trimmed else 0
            avg_pr = np.mean(prune_rate_loss_trimmed) if prune_rate_loss_trimmed else 0
            stats_text = f'Final: {final_pr:.6f}\nMin: {min_pr:.6f}\nAvg: {avg_pr:.6f}'
            ax8.text(0.02, 0.98, stats_text, transform=ax8.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                loss_path = os.path.join(self.save_dir, 'loss_curves', f'prune_rate_loss_step_{step}.png')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved Pruning Rate Loss curve: {loss_path}")
            else:
                plt.show()
        
        # ========== 图8b: BCE Loss ==========
        if bce_loss and len(bce_loss) > 0:
            fig8b, ax8b = plt.subplots(1, 1, figsize=(10, 6))
            
            # 确保长度一致
            min_len = min(len(loss_steps), len(bce_loss))
            loss_steps_bce = loss_steps[:min_len]
            bce_loss_trimmed = bce_loss[:min_len]
            
            # 原始曲线
            ax8b.plot(loss_steps_bce, bce_loss_trimmed, 'purple', linewidth=1.5, alpha=0.5, label='BCE Loss (Raw)')
            # 平滑曲线
            bce_loss_smooth_trimmed = bce_loss_smooth[:min_len] if bce_loss_smooth else None
            if bce_loss_smooth_trimmed:
                ax8b.plot(loss_steps_bce, bce_loss_smooth_trimmed, 'purple', linewidth=2.5, 
                        label=f'BCE Loss (Smoothed, α={self.smooth_alpha})')
            
            ax8b.set_xlabel('Steps', fontsize=12)
            ax8b.set_ylabel('Loss', fontsize=12)
            ax8b.set_title(f'BCE Loss Over Training (top_k={top_k})', fontsize=14)
            ax8b.legend(loc='upper right', fontsize=10)
            ax8b.grid(True, alpha=0.3)
            
            # 添加统计信息
            final_bce = bce_loss_trimmed[-1] if bce_loss_trimmed else 0
            min_bce = min(bce_loss_trimmed) if bce_loss_trimmed else 0
            avg_bce = np.mean(bce_loss_trimmed) if bce_loss_trimmed else 0
            stats_text = f'Final: {final_bce:.6f}\nMin: {min_bce:.6f}\nAvg: {avg_bce:.6f}'
            ax8b.text(0.02, 0.98, stats_text, transform=ax8b.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                loss_path = os.path.join(self.save_dir, 'loss_curves', f'bce_loss_step_{step}.png')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved BCE Loss curve: {loss_path}")
            else:
                plt.show()
        
        # ========== 图9: Actual Prune Rate ==========
        if actual_prune_rate and len(actual_prune_rate) > 0:
            fig9, ax9 = plt.subplots(1, 1, figsize=(10, 6))
            
            # 确保长度一致
            min_len = min(len(loss_steps), len(actual_prune_rate))
            loss_steps_apr = loss_steps[:min_len]
            actual_prune_rate_trimmed = actual_prune_rate[:min_len]
            
            # 目标剪枝率
            target_prune_rate = 1.0 - (top_k / 576) if top_k < 576 else 0.0  # 假设最多 576 tokens
            
            # 原始曲线
            ax9.plot(loss_steps_apr, actual_prune_rate_trimmed, 'green', linewidth=1.5, alpha=0.5, label='Actual Prune Rate (Raw)')
            # 平滑曲线
            actual_prune_rate_smooth_trimmed = actual_prune_rate_smooth[:min_len] if actual_prune_rate_smooth else None
            if actual_prune_rate_smooth_trimmed:
                ax9.plot(loss_steps_apr, actual_prune_rate_smooth_trimmed, 'green', linewidth=2.5, 
                        label=f'Actual Prune Rate (Smoothed, α={self.smooth_alpha})')
            # 目标剪枝率
            ax9.axhline(y=target_prune_rate, color='red', linestyle='--', linewidth=2, label=f'Target Prune Rate ({target_prune_rate:.4f})')
            
            ax9.set_xlabel('Steps', fontsize=12)
            ax9.set_ylabel('Prune Rate', fontsize=12)
            ax9.set_title(f'Actual vs Target Pruning Rate (top_k={top_k})', fontsize=14)
            ax9.legend(loc='upper right', fontsize=10)
            ax9.grid(True, alpha=0.3)
            
            # 添加统计信息
            final_apr = actual_prune_rate_trimmed[-1] if actual_prune_rate_trimmed else 0
            avg_apr = np.mean(actual_prune_rate_trimmed) if actual_prune_rate_trimmed else 0
            stats_text = f'Final: {final_apr:.4f}\nAvg: {avg_apr:.4f}\nTarget: {target_prune_rate:.4f}'
            ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                loss_path = os.path.join(self.save_dir, 'loss_curves', f'actual_prune_rate_step_{step}.png')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved Actual Prune Rate curve: {loss_path}")
            else:
                plt.show()
    
    def plot_mask_overlay(
        self,
        image: torch.Tensor,
        mask_hard: torch.Tensor,
        step: int = 0,
        save: bool = True,
        input_ids: Optional[List[int]] = None
    ):
        """
        绘制 mask_hard 叠加在原始图像上的可视化
        
        Args:
            image: 原始图像 tensor [C, H, W] 或 [H, W, C]
            mask_hard: hard mask (bool tensor), 1D
            step: 当前步数
            save: 是否保存
            input_ids: 对应的 input_ids，用于解码文本
        """
        # 尝试解码文本
        text = ""
        if input_ids is not None:
            try:
                from transformers import AutoTokenizer
                # 尝试获取 tokenizer（从环境变量或默认路径）
                tokenizer = None
                # 常见模型路径
                model_paths = [
                    "/data/users/airprofly/FastV/llava-v1.5-7b",  # 实际使用的模型路径
                    "/data/users/Actor/liuhao/llava-v1.5-7b",
                    "/data/users/Actor/liuhao/llava-v1.5-13b",
                    "liuhao/llava-v1.5-7b",
                    "llava-v1.5-7b"
                ]
                for path in model_paths:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                        break
                    except Exception as e:
                        continue
                
                if tokenizer is not None:
                    text = tokenizer.decode(input_ids, skip_special_tokens=True)
                else:
                    # 如果找不到 tokenizer，直接显示 token IDs 的统计信息
                    text = f"input_ids length: {len(input_ids)}"
            except Exception as e:
                text = f"[Cannot decode: {e}]"
        
        try:
            # 处理 image 格式
            if isinstance(image, torch.Tensor):
                img = image.detach().float().cpu().numpy()  # 转换为 float32
            else:
                img = np.array(image).astype(np.float32)
            
            # 转换为 [H, W, C] 格式
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            
            # 反归一化（如果图像已经经过 ImageNet 归一化）
            # ImageNet mean 和 std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # 检测是否经过归一化：范围在 [-3, 3] 左右（std 倍数）
            img_min, img_max = img.min(), img.max()
            if img_min < -0.5 or img_max > 2.5:
                # 图像经过归一化，需要反归一化
                img = img * std + mean
            
            # 归一化到 [0, 1]
            img = np.clip(img, 0, 1)  # 裁剪到有效范围
            
            # 处理 mask_hard
            if isinstance(mask_hard, torch.Tensor):
                mask = mask_hard.detach().cpu().numpy()
            else:
                mask = np.array(mask_hard)
            
            # 获取图像尺寸
            h, w = img.shape[:2]
            n_tokens = len(mask)
            n_selected = int(mask.sum())
            
            # 计算 grid 尺寸
            grid_size = int(np.ceil(np.sqrt(n_tokens)))
            
            # 创建 mask grid
            mask_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
            mask_grid_flat = mask.flatten()[:grid_size * grid_size]
            mask_grid_flat = mask_grid_flat.astype(np.float32)
            mask_grid = mask_grid_flat.reshape(grid_size, grid_size)
            
            # 创建更大的显示图（每个格子用更大的像素块表示）
            # 计算每个格子显示为多大
            pixel_per_cell = max(8, int(800 / grid_size))  # 动态调整，保证清晰度
            display_size = grid_size * pixel_per_cell
            
            # 调整图像大小
            from PIL import Image
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img_resized = pil_img.resize((display_size, display_size), Image.BILINEAR)
            img_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
            
            # 调整 mask grid 大小并添加边框效果
            mask_resized = Image.fromarray((mask_grid * 255).astype(np.uint8), mode='L')
            mask_resized = mask_resized.resize((display_size, display_size), Image.NEAREST)
            mask_final = np.array(mask_resized).astype(np.float32) / 255.0
            
            # 创建带边框的 mask grid 可视化
            # 用插值方式放大，然后手动添加网格线
            mask_display = np.kron(mask_grid, np.ones((pixel_per_cell, pixel_per_cell), dtype=np.float32))
            # 添加网格线（黑色边框）
            for i in range(0, grid_size * pixel_per_cell, pixel_per_cell):
                mask_display[i:i+1, :] = 0.3  # 水平线
                mask_display[:, i:i+1] = 0.3  # 垂直线
            
            # 创建叠加图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原图
            axes[0].imshow(img)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')
            
            # Mask grid（带网格线）
            axes[1].imshow(mask_display, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f'Mask Grid ({grid_size}x{grid_size}, selected={n_selected}/{n_tokens})', fontsize=12)
            axes[1].axis('off')
            
            # 叠加图
            overlay = img_resized.copy()
            mask_3ch = np.stack([mask_final] * 3, axis=-1)
            # 红色表示被选中的区域
            overlay = np.where(mask_3ch > 0.5, 
                             overlay * 0.5 + np.array([1, 0, 0]) * 0.5, 
                             overlay * 0.7)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay (Red=Selected)', fontsize=12)
            axes[2].axis('off')
            
            # 添加文本标题（始终显示，即使解码失败）
            if text:
                # 只显示前200个字符
                text_short = text[:200] + "..." if len(text) > 200 else text
            else:
                text_short = "[No text available]"
            
            # 在图的顶部添加文本（避免被 tight_layout 裁掉）
            fig.text(0.5, 0.99, f"Text: {text_short}", ha='center', va='top', fontsize=9, wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.suptitle(f'Step {step}: Mask Hard Visualization', fontsize=14, y=0.95)
            plt.tight_layout()
            
            if save:
                save_path = os.path.join(self.save_dir, 'mask_overlay', f'mask_overlay_step_{step}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved mask overlay: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"[ActorVisualizer] Error in plot_mask_overlay: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_probs_analysis(
        self,
        probs: torch.Tensor,
        mask_hard: torch.Tensor,
        input_ids: Optional[List[int]] = None,
        step: int = 0,
        save: bool = True
    ):
        """
        绘制 probs 的分布可视化（适合大量 token）
        
        Args:
            probs: soft mask probabilities, 1D tensor
            mask_hard: hard mask (0.0 or 1.0), 1D tensor
            input_ids: 对应的 input_ids，用于解码文本
            step: 当前步数
            save: 是否保存
        """
        # 尝试解码文本
        text = ""
        if input_ids is not None:
            try:
                from transformers import AutoTokenizer
                tokenizer = None
                model_paths = [
                    "/data/users/airprofly/FastV/llava-v1.5-7b",  # 实际使用的模型路径
                    "/data/users/Actor/liuhao/llava-v1.5-7b",
                    "/data/users/Actor/liuhao/llava-v1.5-13b",
                    "liuhao/llava-v1.5-7b",
                    "llava-v1.5-7b"
                ]
                for path in model_paths:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                        break
                    except Exception as e:
                        continue
                
                if tokenizer is not None:
                    text = tokenizer.decode(input_ids, skip_special_tokens=True)
                else:
                    text = f"input_ids length: {len(input_ids)}"
            except Exception as e:
                text = f"[Cannot decode: {str(e)}]"
        else:
            text = "[No input_ids provided]"
        
        print(f"[DEBUG plot_mask_overlay] input_ids={'provided' if input_ids is not None else 'None'}, text_len={len(text)}")
        
        try:
            # 处理数据
            if isinstance(probs, torch.Tensor):
                p = probs.detach().float().cpu().numpy()  # 转换为 float32
            else:
                p = np.array(probs)
            
            if isinstance(mask_hard, torch.Tensor):
                m = mask_hard.detach().cpu().numpy()
            else:
                m = np.array(mask_hard)
            
            n_tokens = len(p)
            n_selected = int(m.sum())
            
            # 创建 2x2 子图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # ===== 图1: 概率分布直方图 =====
            ax1 = axes[0, 0]
            ax1.hist(p, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(p.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={p.mean():.4f}')
            ax1.axvline(np.median(p), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(p):.4f}')
            ax1.set_xlabel('Probability')
            ax1.set_ylabel('Count')
            ax1.set_title(f'Probability Distribution (n={n_tokens})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ===== 图2: Top-K token 的概率（条形图）=====
            ax2 = axes[0, 1]
            # 找出被选中的 token 索引
            selected_idx = np.where(m > 0.5)[0]
            if len(selected_idx) > 0:
                selected_probs = p[selected_idx]
                # 只显示前 50 个（如果太多）
                display_idx = min(50, len(selected_idx))
                ax2.bar(range(display_idx), selected_probs[:display_idx], color='coral', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Selected Token Index (top {})'.format(display_idx))
                ax2.set_ylabel('Probability')
                ax2.set_title(f'Top {display_idx} Selected Tokens (total selected={n_selected})')
                ax2.grid(True, alpha=0.3, axis='y')
            
            # ===== 图3: 按概率排序的曲线 =====
            ax3 = axes[1, 0]
            sorted_probs = np.sort(p)[::-1]  # 降序排列
            ax3.plot(sorted_probs, 'b-', linewidth=1.5, alpha=0.7)
            ax3.axhline(p.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={p.mean():.4f}')
            ax3.fill_between(range(len(sorted_probs)), sorted_probs, alpha=0.3)
            ax3.set_xlabel('Token Rank (sorted by prob)')
            ax3.set_ylabel('Probability')
            ax3.set_title('Sorted Probabilities (descending)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # ===== 图4: 统计信息 =====
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # 如果有文本，在图4中显示
            if text:
                # 截断过长的文本
                display_text = text[:500] + "..." if len(text) > 500 else text
            else:
                display_text = "[No text available]"
            ax4.text(0.5, 0.95, f"Input Text:", transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='center', fontweight='bold')
            ax4.text(0.5, 0.85, display_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            # 计算统计信息
            stats_text = f"""
            ================= Statistics =================
            
            Total Tokens:     {n_tokens}
            Selected (top-k): {n_selected} ({100*n_selected/n_tokens:.1f}%)
            
            ----- Probability Stats -----
            Mean:             {p.mean():.6f}
            Std:              {p.std():.6f}
            Min:              {p.min():.6f}
            Max:              {p.max():.6f}
            Median:           {np.median(p):.6f}
            
            ----- Percentiles -----
            25th:             {np.percentile(p, 25):.6f}
            75th:             {np.percentile(p, 75):.6f}
            90th:             {np.percentile(p, 90):.6f}
            95th:             {np.percentile(p, 95):.6f}
            99th:             {np.percentile(p, 99):.6f}
            
            ----- Selected Tokens -----
            Selected Mean:    {p[m > 0.5].mean():.6f} (if any selected)
            """
            ax4.text(0.1, 0.4 if text else 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'Step {step}: Token Probability Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                save_path = os.path.join(self.save_dir, 'probs_analysis', f'probs_analysis_step_{step}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved probs analysis: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"[ActorVisualizer] Error in plot_probs_distribution: {e}")
            import traceback
            traceback.print_exc()

    def plot_rd_distribution(
        self,
        r_values: torch.Tensor,
        d_values: torch.Tensor,
        step: int = 0,
        save: bool = True
    ):
        """绘制并保存 r（文本相关性得分）和 d（图像冗余得分）在当前 step 下的分布图"""
        try:
            # 处理 r values
            if isinstance(r_values, torch.Tensor):
                r = r_values.detach().float().cpu().numpy()
            else:
                r = np.array(r_values)
            
            # 处理 d values
            if isinstance(d_values, torch.Tensor):
                d = d_values.detach().float().cpu().numpy()
            else:
                d = np.array(d_values)
            
            n_tokens = len(r)
            
            # 创建 2x2 子图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # ===== 图1: r 分布直方图 =====
            ax1 = axes[0, 0]
            ax1.hist(r, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(r.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={r.mean():.4f}')
            ax1.axvline(np.median(r), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(r):.4f}')
            ax1.set_xlabel('r Value (Text Relevance Score)')
            ax1.set_ylabel('Count')
            ax1.set_title(f'r Distribution (Text Relevance, n={n_tokens})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ===== 图2: d 分布直方图 =====
            ax2 = axes[0, 1]
            ax2.hist(d, bins=50, alpha=0.7, color='coral', edgecolor='black')
            ax2.axvline(d.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={d.mean():.4f}')
            ax2.axvline(np.median(d), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(d):.4f}')
            ax2.set_xlabel('d Value (Image Importance Score)')
            ax2.set_ylabel('Count')
            ax2.set_title(f'd Distribution (Image Importance, n={n_tokens})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # ===== 图3: r 和 d 散点图 =====
            ax3 = axes[1, 0]
            ax3.scatter(r, d, alpha=0.5, s=10, c='purple')
            ax3.set_xlabel('r (Text Relevance)')
            ax3.set_ylabel('d (Image Importance)')
            ax3.set_title('r vs d Scatter Plot')
            ax3.grid(True, alpha=0.3)
            
            # 添加相关系数
            correlation = np.corrcoef(r, d)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax3.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # ===== 图4: r 和 d 的 index 分布曲线 =====
            ax4 = axes[1, 1]
            indices = np.arange(n_tokens)
            ax4.plot(indices, r, 'b-', linewidth=1.5, alpha=0.7, label='r (Text Relevance)')
            ax4.plot(indices, d, 'r-', linewidth=1.5, alpha=0.7, label='d (Image Importance)')
            ax4.set_xlabel('Token Index')
            ax4.set_ylabel('Score Value')
            ax4.set_title('r and d Values by Token Index')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Step {step}: r and d Distribution Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                save_path = os.path.join(self.save_dir, 'loss_curves', f'rd_distribution_step_{step}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved r/d distribution: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"[ActorVisualizer] Error in plot_rd_distribution: {e}")
            import traceback
            traceback.print_exc()

    def plot_rds_over_index(
        self,
        r_values: torch.Tensor,
        d_values: torch.Tensor,
        scores_values: Optional[torch.Tensor] = None,
        step: int = 0,
        save: bool = True
    ):
        """绘制并保存 r（文本相关性）、d（密度）、scores（最终得分）随 token index 变化的分布图"""
        try:
            # 处理 r values
            if isinstance(r_values, torch.Tensor):
                r = r_values.detach().float().cpu().numpy()
            else:
                r = np.array(r_values)
            
            # 处理 d values
            if isinstance(d_values, torch.Tensor):
                d = d_values.detach().float().cpu().numpy()
            else:
                d = np.array(d_values)
            
            # 处理 scores values
            if scores_values is not None:
                if isinstance(scores_values, torch.Tensor):
                    scores = scores_values.detach().float().cpu().numpy()
                else:
                    scores = np.array(scores_values)
            else:
                # 如果没有提供 scores，计算 scores = alpha * r + (1 - alpha) * d
                # 这里假设 alpha = 0.5 作为默认值
                scores = 0.5 * r + 0.5 * d
            
            n_tokens = len(r)
            indices = np.arange(n_tokens)
            
            # 创建 2x2 子图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # ===== 图1: r 随 index 变化 =====
            ax1 = axes[0, 0]
            ax1.plot(indices, r, 'b-', linewidth=1.0, alpha=0.7)
            ax1.axhline(y=r.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={r.mean():.4f}')
            ax1.fill_between(indices, r, alpha=0.3)
            ax1.set_xlabel('Token Index', fontsize=11)
            ax1.set_ylabel('r (Text Relevance)', fontsize=11)
            ax1.set_title('r (Text Relevance) over Token Index', fontsize=12)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # ===== 图2: d 随 index 变化 =====
            ax2 = axes[0, 1]
            ax2.plot(indices, d, 'g-', linewidth=1.0, alpha=0.7)
            ax2.axhline(y=d.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={d.mean():.4f}')
            ax2.fill_between(indices, d, alpha=0.3, color='green')
            ax2.set_xlabel('Token Index', fontsize=11)
            ax2.set_ylabel('d (Token Density)', fontsize=11)
            ax2.set_title('d (Token Density) over Token Index', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # ===== 图3: scores 随 index 变化 =====
            ax3 = axes[1, 0]
            ax3.plot(indices, scores, 'purple', linewidth=1.0, alpha=0.7)
            ax3.axhline(y=scores.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={scores.mean():.4f}')
            ax3.fill_between(indices, scores, alpha=0.3, color='purple')
            ax3.set_xlabel('Token Index', fontsize=11)
            ax3.set_ylabel('scores (Final)', fontsize=11)
            ax3.set_title('scores (Final) over Token Index', fontsize=12)
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            
            # ===== 图4: r, d, scores 叠加对比 =====
            ax4 = axes[1, 1]
            ax4.plot(indices, r, 'b-', linewidth=1.0, alpha=0.7, label='r (Text Relevance)')
            ax4.plot(indices, d, 'g-', linewidth=1.0, alpha=0.7, label='d (Token Density)')
            ax4.plot(indices, scores, 'r-', linewidth=1.5, alpha=0.9, label='scores (Final)')
            ax4.set_xlabel('Token Index', fontsize=11)
            ax4.set_ylabel('Value', fontsize=11)
            ax4.set_title('Comparison: r, d, scores over Token Index', fontsize=12)
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Step {step}: r, d, scores Distribution over Token Index (n_tokens={n_tokens})', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                save_path = os.path.join(self.save_dir, 'loss_curves', f'rds_over_index_step_{step}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[ActorVisualizer] Saved r/d/scores over index: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"[ActorVisualizer] Error in plot_rds_over_index: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_probs_distribution(
        self,
        scores: Optional[List[float]] = None,
        step: int = 0,
        save: bool = True
    ):
        """绘制并保存 scores 分布条形图（按 scores 从大到小排序）"""
        if scores is None or len(scores) == 0:
            return
        
        # 按 scores 从大到小排序，同时保留原始 index
        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        # 限制显示的 token 数量（太多会看不清）
        max_display = min(512, len(sorted_scores))
        sorted_scores_display = sorted_scores[:max_display]
        
        # 提取排序后的 scores 和对应的原始 index
        sorted_indices = [x[0] for x in sorted_scores_display]
        sorted_values = [x[1] for x in sorted_scores_display]
        x_positions = list(range(len(sorted_scores_display)))
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # 绘制条形图
        bars = ax.bar(x_positions, sorted_values, color='steelblue', alpha=0.7, width=0.8)
        
        # 在每个条形上标注原始 index（前50个太多了，只标注前20个）
        for i, (bar, orig_idx) in enumerate(zip(bars[:20], sorted_indices[:20])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{orig_idx}',
                    ha='center', va='bottom', fontsize=6, rotation=0)
        
        ax.set_xlabel('Sorted Index (from high to low score)', fontsize=12)
        ax.set_ylabel('Score (raw, not normalized)', fontsize=12)
        ax.set_title(f'Scores Distribution (sorted, top_k={self._get_top_k_from_context()})', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加统计信息
        avg_score = np.mean(sorted_values)
        max_score = sorted_values[0] if sorted_values else 0
        min_score = sorted_values[-1] if sorted_values else 0
        top_k = self._get_top_k_from_context()
        # 找到 top_k 位置的 score 值
        top_k_score = sorted_values[top_k-1] if top_k <= len(sorted_values) else 0
        stats_text = f'Max: {max_score:.4f}\nAvg: {avg_score:.4f}\nMin: {min_score:.4f}\nTop-{top_k} score: {top_k_score:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            score_path = os.path.join(self.save_dir, 'loss_curves', f'scores_dist_step_{step}.png')
            os.makedirs(os.path.dirname(score_path), exist_ok=True)
            plt.savefig(score_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[ActorVisualizer] Saved scores distribution: {score_path}")
        else:
            plt.show()
    
    def _get_top_k_from_context(self) -> int:
        """从上下文获取 top_k 值（用于标题显示）"""
        # 尝试从保存的配置中读取，或使用默认值
        return getattr(self, '_cached_top_k', 64)
    
    def set_top_k(self, top_k: int):
        """设置 top_k 值用于可视化标题"""
        self._cached_top_k = top_k
    
    def save_visualization(
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
        scores: Optional[List[float]] = None,
        # alpha_history: Optional[List[float]] = None,  # 移除
        # r_current: Optional[torch.Tensor] = None,  # 移除
        # d_current: Optional[torch.Tensor] = None,  # 移除
        top_k: int = 64,
        step: int = 0
    ):
        """
        保存可视化图像（每100步调用）
        
        Args:
            loss_steps: 损失记录的步数数组
            loss_total: 总损失数组
            loss_lm: LLM损失数组
            loss_diversity: 加权后的 Diversity 损失数组
            diversity_raw: 未加权的 Diversity 损失数组（相似度）
            # entropy: 熵数组  # 移除
            kept: 期望保留 token 数量数组
            diversity_level: 多样性水平数组 (1 - similarity)
            scores: 原始 scores 分布数据（未归一化）
            # alpha_history: alpha 参数历史（用于绘制随 step 变化曲线）  # 移除
            # r_current: 当前 step 的 r 值（文本相关性得分）  # 移除
            # d_current: 当前 step 的 d 值（图像冗余得分）  # 移除
            top_k: 保留的 token 数量
            step: 当前步数
        """
        # 设置 top_k 用于标题显示
        self.set_top_k(top_k)
        
        # 绘制损失曲线图
        self.plot_loss_curves(
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
            top_k=top_k,
            step=step,
            save=True
        )
        
        # 绘制 scores 分布条形图
        if scores is not None and len(scores) > 0:
            self.plot_probs_distribution(
                scores=scores,
                step=step,
                save=True
            )
        
        # 绘制 alpha 随 step 变化的曲线  # 移除
        # if alpha_history is not None and len(alpha_history) > 0:
        #     self.plot_alpha_over_steps(
        #         steps=loss_steps,
        #         alpha_history=alpha_history,
        #         step=step,
        #         save=True
        #     )
        
        # 绘制当前 step 的 r 和 d 分布  # 移除
        # if r_current is not None and d_current is not None:
        #     self.plot_rd_distribution(
        #         r_values=r_current,
        #         d_values=d_current,
        #         step=step,
        #         save=True
        #     )
    
    def save_checkpoint(
        self,
        actor_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        loss_steps: Optional[List[int]] = None,
        loss_total: Optional[List[float]] = None,
        loss_lm: Optional[List[float]] = None,
        loss_diversity: Optional[List[float]] = None,
        top_k: int = 64,
        step: int = 0
    ):
        """
        保存可视化checkpoint（每1000步调用）
        
        Args:
            actor_state_dict: Actor模型权重
            loss_steps: 损失记录的步数数组
            loss_total: 总损失数组
            loss_lm: LLM损失数组
            loss_diversity: Diversity损失数组
            prune_steps: 剪枝率记录的步数数组
            prunkept: 剪枝率数组
            top_k: 保留的 token 数量
            step: 当前步数
        """
        checkpoint_data = {
            'step': step,
            'top_k': top_k,
            'loss_history': {
                'steps': loss_steps[-1000:] if loss_steps else [],
                'total_loss': loss_total[-1000:] if loss_total else [],
                'lm_loss': loss_lm[-1000:] if loss_lm else [],
                'diversity_loss': loss_diversity[-1000:] if loss_diversity else [],
            },
            'top_k': top_k,
        }
        
        # 保存Actor模型权重（如果有）
        if actor_state_dict is not None:
            actor_weights_path = os.path.join(self.save_dir, f'actor_weights_step_{step}.pt')
            torch.save(actor_state_dict, actor_weights_path)
            checkpoint_data['actor_weights_path'] = actor_weights_path
            print(f"[ActorVisualizer] Saved actor weights: {actor_weights_path}")
        
        checkpoint_file = os.path.join(self.save_dir, f'visualization_checkpoint_step_{step}.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"[ActorVisualizer] Saved visualization checkpoint: {checkpoint_file}")
