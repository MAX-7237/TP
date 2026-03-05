import torch
from typing import Tuple, List, Optional
import torch.nn.functional as F
import math


# ============================================================
# BCE/MSE Loss: (pred - label)^2 的平均数
# ============================================================
def compute_bce_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE/MSE 损失：预测值与标签的差值平方的平均数
    
    Args:
        pred: [N, T] 预测的保留概率 (0-1)
        label: [N, T] ground truth mask (0-1)
        valid: [N, T] 有效位置 mask (可选)
        
    Returns:
        损失值
    """
    # MSE: (pred - label)^2
    mse = (pred - label) ** 2
    
    if valid is not None:
        mse = mse.masked_fill(~valid, 0.0)
        return mse.sum() / valid.sum()
    else:
        return mse.mean()


# ============================================================
# Diversity loss (weighted mean pairwise cosine similarity)
#   - Uses mask (soft weights or hard 0/1)
#   - Computes OFF-DIAGONAL weighted AVERAGE similarity
#   - Minimize => encourage diversity among selected tokens
# ============================================================

def compute_diversity_loss(
    token_feats: torch.Tensor,            # [N, D]
    mask: Optional[torch.Tensor] = None,  # [N] weights
    eps: float = 1e-6,
) -> torch.Tensor:
    assert token_feats.ndim == 2, "token_feats should be [N, D]"
    N, D = token_feats.shape

    if N <= 1:
        return token_feats.new_zeros((), requires_grad=True)

    x = F.normalize(token_feats, dim=-1)  # [N, D]

    if mask is None:
        w = torch.full((N,), 1.0 / N, device=x.device, dtype=x.dtype)
    else:
        w = mask.to(dtype=x.dtype)
        w = w / (w.sum() + eps)  # sum=1

    # cosine similarity matrix (INCLUDING diagonal)
    sim = x @ x.t()  # [N, N]

    # weight matrix (INCLUDING diagonal)
    ww = w[:, None] * w[None, :]

    denom = ww.sum().clamp_min(eps)  # when sum(w)=1, denom = 1
    div_loss = (sim * ww).sum() / denom
    return div_loss #相当于是平均值


# ============================================================
# Total loss = lm_loss + lambda_div * diversity_loss
# ============================================================

def compute_actor_loss_from_list_with_diversity(
    token_feats_list: List[torch.Tensor],
    mask_hard_list: Optional[List[Optional[torch.Tensor]]] = None,
    lambda_div: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从多个 keep_mask 计算平均多样性损失

    Args:
        token_feats_list: 特征列表
        mask_hard_list: hard mask 列表（用于多样性损失，只计算保留的 token）
        lambda_div: 多样性损失的权重

    Returns:
        div_loss: 平均 div_loss (原始多样性损失)
        weighted_div: lambda_div * div_loss
    """
    if len(token_feats_list) == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy, dummy

    if mask_hard_list is None:
        mask_hard_list = [None] * len(token_feats_list)

    div_losses = []
    
    for idx, (feats, m_hard) in enumerate(zip(token_feats_list, mask_hard_list)):
        # 确保 mask 和 feats 在同一设备
        if m_hard is not None and m_hard.device != feats.device:
            m_hard = m_hard.to(feats.device)
        
        # 多样性损失：使用 hard mask（只计算保留的 token）
        div_losses.append(compute_diversity_loss(feats, mask=m_hard))
    
    if len(div_losses) == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy, dummy
        
    avg_div_loss = torch.stack(div_losses).mean()
    weighted_div = lambda_div * avg_div_loss
    
    # 返回：原始 diversity loss、加权 diversity loss
    return avg_div_loss, weighted_div