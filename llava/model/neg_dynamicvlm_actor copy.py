import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def _make_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    scores: [N]
    return: [N] float mask 0/1
    """
    N = scores.size(0)
    kk = min(max(int(k), 1), N)
    top_idx = torch.topk(scores, kk, dim=0).indices
    mask = torch.zeros(N, device=scores.device, dtype=torch.float32)
    mask[top_idx] = 1.0
    return mask 

# --------------------------- 输出 ---------------------------
@dataclass
class ActorOutput:
    prune_probs: torch.Tensor     # [N]  (we use keep-logit as score)
    keep_probs: torch.Tensor      # [N]  (P(keep))
    mask_hard: torch.Tensor  # [N] float 0/1 (argmax over 2-class softmax)


# --------------------------- 文本 Self-Attn Block（单样本，返回attn） ---------------------------
class TextSelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # batch=1
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [M, D]
        return:
          x: [M, D]
          attn: [H, M, M]
        """
        x = x.unsqueeze(0)  # [1, M, D]

        attn_out, attn_w = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,  # [1, H, M, M]
        )

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        x = x.squeeze(0)           # [M, D]
        attn_w = attn_w.squeeze(0) # [H, M, M]
        return x, attn_w


# --------------------------- Actor（单样本） ---------------------------
class dynamicvlm_actor(nn.Module):
    """
    2-class soft mask + argmax hard mask

    - fusion_mlp outputs logits2: [N, 2] (prune, keep)
    - M_soft = softmax(logits2 / tau)
    - probs  = M_soft[:, keep_id]
    - M_hard = argmax(M_soft) -> {0,1} then convert to keep-mask float
    """

    def __init__(
        self,
        text_dim: int = 4096,
        image_dim: int = 4096,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        top_k: int = 192,   # 不再用于 hard mask（你要 argmax），保留参数是为了兼容
        tau: float = 1.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(dropout)
        self.top_k = top_k
        self.tau = tau

        # 文本：投影 + self-attn blocks
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.text_blocks = nn.ModuleList(
            [TextSelfAttnBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # 图像：先过 MLP
        self.image_mlp = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 融合 MLP：输出两类 logits（prune/keep）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # <-- 改成 2
        )

    def forward(
        self,
        x_img: torch.Tensor,  # [N, C_img]
        x_txt: torch.Tensor,  # [M, C_txt]
        tau: float = None,
        top_k: int = None,    # 保留兼容，但 argmax hard 不用它
        **kwargs,
    ) -> ActorOutput:

        if tau is None:
            tau = self.tau

        N = x_img.shape[0]
        M = x_txt.shape[0]
        device = x_img.device
        top_k=top_k if top_k is not None else self.top_k

        # --------- 文本：proj + blocks，拿最后一层 attn ---------
        t = self.text_proj(x_txt)  # [M, D]
        for blk in self.text_blocks:
            t, attn_w = blk(t)     # t:[M,D], attn_w:[H,M,M]

        attn = attn_w.mean(dim=0)  # [M, M]

        # received_i = sum_q attn[q,i]（排除 self）
        eye = torch.eye(M, device=device, dtype=torch.bool)
        attn = attn.masked_fill(eye, 0.0)
        received = attn.sum(dim=0)           # [M]
        w = F.softmax(received, dim=-1)      # [M]
        txt_guided_vec = (w.unsqueeze(-1) * t).sum(dim=0)  # [D]

        # --------- 图像：MLP + global mean ---------
        img_feat = self.image_mlp(x_img)     # [N, D]
        img_global = img_feat.mean(dim=0)    # [D]

        # --------- 融合：broadcast 拼接 ---------
        img_global_b = img_global.unsqueeze(0).expand(N, -1)        # [N, D]
        txt_guided_b = txt_guided_vec.unsqueeze(0).expand(N, -1)    # [N, D]
        fused = torch.cat([img_feat, img_global_b, txt_guided_b], dim=-1)  # [N, 3D]

        # --------- 输出两类 logits ---------
        logits2 = self.fusion_mlp(self.drop(fused))  # [N, 2]

        # soft mask: [N, 2]
        m_soft = F.softmax(logits2 / tau, dim=-1)

        # probs: P(keep) => [N]
        prune_probs = m_soft[:,0] # prune
        keep_probs = m_soft[:, 1] # keep

        # hard mask: argmax over 2-class => [N] in {0,1}, then convert to keep-mask float
        m_hard_idx = torch.argmax(m_soft, dim=-1)  # [N]
        mask_hard = (m_hard_idx == 1).to(torch.float32)  # [N]每一个token单独决定自己是否保留
        mask_hard = _make_topk_mask(-keep_probs, top_k) # [N]取负数，直接选择前面topk个概率最大的token作为保留

        return ActorOutput(prune_probs=prune_probs, keep_probs=keep_probs, mask_hard=mask_hard)
    