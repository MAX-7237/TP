"""
Attention Actor Training Monitor
==========================

This module provides real-time monitoring and visualization for Attention Actor training:
1. Track LLM loss, Actor loss, Total loss over steps
2. Track pruning ratio (keep ratio) over steps
3. Visualize mask application on images

Usage:
    # As a module (integrate with training script)
    from monitor import TrainingMonitor
    
    monitor = TrainingMonitor(
        output_dir="./actor_checkpoints/mmbench",
        num_save_plots=100,
        num_save_samples=5
    )
    
    # In training loop
    monitor.step(step, llm_loss, actor_loss, total_loss, keep_ratio, image, mask)
    
    # At the end
    monitor.close()
    
    # Or run standalone to plot from logs
    python monitor.py --log-file ./actor_checkpoints/mmbench/training.log
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from PIL import Image

# Try to import for standalone mode
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class TrainingMonitor:
    """
    Monitor Attention Actor training with real-time visualization.
    """
    
    def __init__(
        self,
        output_dir: str,
        num_save_plots: int = 100,
        num_save_samples: int = 5,
        device: str = "cuda"
    ):
        """
        Args:
            output_dir: Directory to save plots and samples
            num_save_plots: Save plot every N steps
            num_save_samples: Number of sample images to save per save interval
            device: Device for tensor operations
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.samples_dir = self.output_dir / "samples"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_save_plots = num_save_plots
        self.num_save_samples = num_save_samples
        self.device = device
        
        # Metrics history
        self.steps = []
        self.llm_losses = []
        self.actor_losses = []
        self.total_losses = []
        self.pruning_ratios = []  # keep ratio
        
        # Sample data
        self.sample_data = []  # List of (step, image, mask, keep_ratio)
        
        # Current step
        self.current_step = 0
        
        print(f"[Monitor] Initialized")
        print(f"[Monitor] Plots directory: {self.plots_dir}")
        print(f"[Monitor] Samples directory: {self.samples_dir}")
    
    def step(
        self,
        step: int,
        llm_loss: float,
        actor_loss: float,
        keep_ratio: float,
        images: Optional[List[torch.Tensor]] = None,
        masks: Optional[List[torch.Tensor]] = None,
        force_save: bool = False
    ):
        """
        Record a training step.
        
        Args:
            step: Current step number
            llm_loss: LLM loss value
            actor_loss: Actor loss value
            keep_ratio: Pruning keep ratio (0~1)
            images: List of input images [C, H, W]
            masks: List of mask tensors [T_img] (bool)
            force_save: Force save regardless of step interval
        """
        self.current_step = step
        self.steps.append(step)
        self.llm_losses.append(llm_loss)
        self.actor_losses.append(actor_loss)
        self.total_losses.append(llm_loss + 0.1 * actor_loss if actor_loss else llm_loss)
        self.pruning_ratios.append(keep_ratio)
        
        # Collect sample data
        if images is not None and masks is not None:
            for img, mask in zip(images, masks):
                if len(self.sample_data) < self.num_save_samples * 10:  # Limit total samples
                    self.sample_data.append({
                        'step': step,
                        'image': img.detach().cpu().clone(),
                        'mask': mask.detach().cpu().clone() if isinstance(mask, torch.Tensor) else mask
                    })
        
        # Save plots at intervals
        if step % self.num_save_plots == 0 or force_save:
            self._save_plots(step)
            self._save_samples(step)
            self._save_metrics_json(step)
    
    def _save_plots(self, step: int):
        """Save loss and pruning ratio plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Attention Actor Training Progress (Step {step})', fontsize=14, fontweight='bold')
        
        # Color scheme
        colors = {
            'llm': '#2E86AB',
            'actor': '#A23B72',
            'total': '#F18F01',
            'pruning': '#C73E1D'
        }
        
        # ----- Plot 1: All Losses -----
        ax1 = axes[0, 0]
        if len(self.steps) > 1:
            ax1.plot(self.steps, self.llm_losses, label='LLM Loss', color=colors['llm'], linewidth=2, alpha=0.8)
            if any(l > 0 for l in self.actor_losses):
                ax1.plot(self.steps, self.actor_losses, label='Actor Loss', color=colors['actor'], linewidth=2, alpha=0.8)
            ax1.plot(self.steps, self.total_losses, label='Total Loss', color=colors['total'], linewidth=2, alpha=0.8, linestyle='--')
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # ----- Plot 2: Log Scale Losses -----
        ax2 = axes[0, 1]
        if len(self.steps) > 1:
            ax2.semilogy(self.steps, self.llm_losses, label='LLM Loss', color=colors['llm'], linewidth=2, alpha=0.8)
            if any(l > 0 for l in self.actor_losses):
                ax2.semilogy(self.steps, self.actor_losses, label='Actor Loss', color=colors['actor'], linewidth=2, alpha=0.8)
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Loss (log)', fontsize=11)
        ax2.set_title('Loss Curves (Log Scale)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.set_xlim(left=0)
        
        # ----- Plot 3: Pruning Ratio -----
        ax3 = axes[1, 0]
        if len(self.steps) > 1:
            ax3.plot(self.steps, self.pruning_ratios, label='Keep Ratio', color=colors['pruning'], linewidth=2)
            ax3.fill_between(self.steps, 0, self.pruning_ratios, alpha=0.3, color=colors['pruning'])
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Keep Ratio', fontsize=11)
        ax3.set_title('Image Token Keep Ratio (Pruning Rate)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        ax3.set_ylim(0, 1.05)
        
        # ----- Plot 4: Statistics Summary -----
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Training Statistics
        ═══════════════════════════════
        
        Current Step: {step:,}
        Total Steps: {len(self.steps):,}
        
        ┌─────────────────────────────────────┐
        │ Loss Summary                        │
        ├─────────────────────────────────────┤
        │ LLM Loss:     {self.llm_losses[-1]:.6f}  │
        │ Actor Loss:   {self.actor_losses[-1] if any(l > 0 for l in self.actor_losses) else 0:.6f}  │
        │ Total Loss:   {self.total_losses[-1]:.6f}  │
        └─────────────────────────────────────┘
        
        Keep Ratio: {self.pruning_ratios[-1]:.2%} ({int(self.pruning_ratios[-1] * 100)}% tokens kept)
        
        Average LLM Loss:    {np.mean(self.llm_losses):.6f}
        Average Actor Loss:  {np.mean(self.actor_losses):.6f}
        Average Keep Ratio:  {np.mean(self.pruning_ratios):.2%}
        
        Min Keep Ratio: {min(self.pruning_ratios):.2%}
        Max Keep Ratio: {max(self.pruning_ratios):.2%}
        """
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"training_progress_{step:06d}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[Monitor] Saved plot: {plot_path}")
        
        # Also save as latest.png for easy viewing
        latest_path = self.plots_dir / "latest.png"
        plt.savefig(latest_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _save_samples(self, step: int):
        """Save sample images with mask visualization."""
        if len(self.sample_data) == 0:
            return
        
        # Select samples from current step or nearby
        current_step_samples = [s for s in self.sample_data if s['step'] == step]
        
        if len(current_step_samples) == 0:
            # Find closest step samples
            if self.sample_data:
                current_step_samples = [self.sample_data[-1]]
        
        if len(current_step_samples) == 0:
            return
        
        # Limit samples per save
        samples_to_plot = current_step_samples[:self.num_save_samples]
        
        n_samples = len(samples_to_plot)
        if n_samples == 0:
            return
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Mask Visualization (Step {step})', fontsize=14, fontweight='bold')
        
        for idx, sample in enumerate(samples_to_plot):
            image = sample['image']
            mask = sample['mask']
            
            # Convert tensor to numpy
            if isinstance(image, torch.Tensor):
                # Handle different tensor shapes
                if image.dim() == 4:
                    # [B, C, H, W] or [B, C, H, W] - take first image
                    image = image[0]
                if image.dim() == 3:
                    image_np = image.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
                else:
                    # Handle unexpected shape
                    image_np = image.detach().cpu().numpy()  # [C, H, W]
                # Denormalize if needed
                if image_np.max() > 1.1:  # Probably normalized
                    mean = np.array([0.48145466, 0.4578275, 0.40821073])
                    std = np.array([0.26862954, 0.26130296, 0.27577711])
                    image_np = image_np * std + mean
                    image_np = np.clip(image_np, 0, 1)
            else:
                image_np = image
            
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            
            # Handle different mask shapes
            if mask_np.ndim == 1:
                # 1D mask [T_img], reshape to 2D grid
                T = len(mask_np)
                # Try common grid shapes
                grid_h, grid_w = int(np.sqrt(T)), int(np.sqrt(T))
                if grid_h * grid_w < T:
                    # Try other aspect ratios
                    for h in range(1, int(np.sqrt(T)) + 1):
                        if T % h == 0:
                            grid_h, grid_w = h, T // h
                            break
                mask_2d = mask_np.reshape(grid_h, grid_w)
            else:
                mask_2d = mask_np
            
            # ----- Original Image -----
            axes[idx, 0].imshow(image_np)
            axes[idx, 0].set_title('Original Image', fontsize=10)
            axes[idx, 0].axis('off')
            
            # ----- Mask (2D grid) -----
            im1 = axes[idx, 1].imshow(mask_2d, cmap='RdYlGn', vmin=0, vmax=1)
            axes[idx, 1].set_title('Keep Mask (Green=Keep, Red=Prune)', fontsize=10)
            axes[idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)
            
            # ----- Masked Image -----
            # Apply mask to image by zeroing out pruned regions
            masked_image = image_np.copy()
            # Resize mask to image size for visualization
            h, w = masked_image.shape[:2]
            mask_resized = np.zeros((h, w), dtype=np.float32)
            
            # Create a grid visualization
            grid_h, grid_w = mask_2d.shape
            cell_h, cell_w = h // grid_h, w // grid_w
            
            for i in range(grid_h):
                for j in range(grid_w):
                    if mask_2d[i, j]:  # Keep
                        mask_resized[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 1.0
            
            # Apply mask (darken pruned regions)
            for c in range(3):
                masked_image[:, :, c] = masked_image[:, :, c] * mask_resized
            
            axes[idx, 2].imshow(masked_image)
            axes[idx, 2].set_title(f'Masked Image (Keep: {mask_np.mean():.1%})', fontsize=10)
            axes[idx, 2].axis('off')
            
            # ----- Mask as 1D bar -----
            # 使用阈值 0.5 判断保留/剪枝
            keep_threshold = 0.5
            # 确保 mask 是 1D 的（2D mask 需要 flatten）
            mask_1d = mask_np.flatten()
            bar_colors = ['green' if m > keep_threshold else 'red' for m in mask_1d]
            axes[idx, 3].bar(range(len(mask_1d)), mask_1d, 
                            color=bar_colors, alpha=0.7)
            axes[idx, 3].set_xlabel('Token Position', fontsize=9)
            axes[idx, 3].set_ylabel('Keep (1) / Prune (0)', fontsize=9)
            axes[idx, 3].set_title('1D Mask', fontsize=10)
            axes[idx, 3].set_ylim(-0.1, 1.1)
            axes[idx, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        sample_path = self.samples_dir / f"mask_samples_{step:06d}.png"
        plt.savefig(sample_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[Monitor] Saved samples: {sample_path}")
    
    def _save_metrics_json(self, step: int):
        """Save metrics to JSON file."""
        metrics = {
            'step': step,
            'total_steps': len(self.steps),
            'timestamp': datetime.now().isoformat(),
            'current': {
                'llm_loss': self.llm_losses[-1],
                'actor_loss': self.actor_losses[-1] if any(l > 0 for l in self.actor_losses) else 0.0,
                'total_loss': self.total_losses[-1],
                'keep_ratio': self.pruning_ratios[-1]
            },
            'average': {
                'llm_loss': float(np.mean(self.llm_losses)),
                'actor_loss': float(np.mean(self.actor_losses)),
                'total_loss': float(np.mean(self.total_losses)),
                'keep_ratio': float(np.mean(self.pruning_ratios))
            },
            'min': {
                'keep_ratio': float(min(self.pruning_ratios))
            },
            'max': {
                'keep_ratio': float(max(self.pruning_ratios))
            }
        }
        
        json_path = self.plots_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def close(self):
        """Finalize monitoring and save final plots."""
        if len(self.steps) > 0:
            self._save_plots(self.steps[-1])
            self._save_metrics_json(self.steps[-1])
            self._save_samples(self.steps[-1])
        
        print(f"[Monitor] Training completed!")
        print(f"[Monitor] Final metrics:")
        print(f"  - Steps: {len(self.steps)}")
        print(f"  - Avg LLM Loss: {np.mean(self.llm_losses):.6f}")
        print(f"  - Avg Actor Loss: {np.mean(self.actor_losses):.6f}")
        print(f"  - Avg Keep Ratio: {np.mean(self.pruning_ratios):.2%}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'total_steps': len(self.steps),
            'llm_loss': {'mean': np.mean(self.llm_losses), 'std': np.std(self.llm_losses)},
            'actor_loss': {'mean': np.mean(self.actor_losses), 'std': np.std(self.actor_losses)},
            'keep_ratio': {'mean': np.mean(self.pruning_ratios), 'min': min(self.pruning_ratios), 'max': max(self.pruning_ratios)}
        }


def plot_from_log(log_file: str, output_dir: Optional[str] = None):
    """
    Plot training curves from a log file.
    
    Args:
        log_file: Path to training log file
        output_dir: Output directory (default: same as log file directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(log_file)
    
    # Parse log file
    steps = []
    llm_losses = []
    actor_losses = []
    total_losses = []
    pruning_ratios = []
    
    print(f"Parsing log file: {log_file}")
    with open(log_file, 'r') as f:
        for line in f:
            if 'Step' in line and 'LLM_loss' in line:
                try:
                    # Extract values using regex or string parsing
                    parts = line.split()
                    # This is a simplified parser - adjust based on your log format
                    for part in parts:
                        if 'LLM_loss' in part:
                            llm_losses.append(float(part.split('=')[1].strip(',')))
                        elif 'Actor_loss' in part:
                            actor_losses.append(float(part.split('=')[1].strip(',')))
                except:
                    pass
    
    # Create a simple monitor and save plots
    monitor = TrainingMonitor(output_dir=output_dir, num_save_plots=len(steps))
    monitor.steps = steps or list(range(len(llm_losses)))
    monitor.llm_losses = llm_losses
    monitor.actor_losses = actor_losses
    monitor.total_losses = [l + 0.1 * a for l, a in zip(llm_losses, actor_losses)]
    monitor.pruning_ratios = [0.5] * len(steps)  # Placeholder
    
    monitor._save_plots(steps[-1] if steps else 0)
    print(f"Plots saved to: {output_dir}/plots/")


def main():
    parser = argparse.ArgumentParser(description="Attention Actor Training Monitor")
    parser.add_argument("--log-file", type=str, help="Path to training log file")
    parser.add_argument("--output-dir", type=str, default="./attention_actor_checkpoints/mmbench",
                        help="Output directory")
    parser.add_argument("--num-save-plots", type=int, default=100,
                        help="Save plot every N steps")
    parser.add_argument("--num-save-samples", type=int, default=5,
                        help="Number of samples to save per interval")
    
    args = parser.parse_args()
    
    if args.log_file:
        plot_from_log(args.log_file, args.output_dir)
    else:
        # Interactive mode - just initialize monitor
        monitor = TrainingMonitor(
            output_dir=args.output_dir,
            num_save_plots=args.num_save_plots,
            num_save_samples=args.num_save_samples
        )
        print("[Monitor] Ready! Integrate with your training script:")
        print("""
    from monitor import TrainingMonitor
    
    monitor = TrainingMonitor(output_dir="./attention_actor_checkpoints/mmbench")
    
    for step in range(total_steps):
        # ... training step ...
        monitor.step(
            step=step,
            llm_loss=llm_loss.item(),
            actor_loss=actor_loss.item() if actor_loss else 0,
            keep_ratio=keep_ratio,
            images=[image_tensor],
            masks=[mask]
        )
    
    monitor.close()
        """)


if __name__ == "__main__":
    main()
