import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def make_topk_keep_mask(keep_scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    keep_scores: [B, N]  (higher => more likely keep)
    return:      [B, N]  float mask 0/1, exactly k kept per sample (clamped to [1,N])
    """
    B, N = keep_scores.shape
    kk = min(max(int(k), 1), N)
    top_idx = torch.topk(keep_scores, kk, dim=-1).indices  # [B, kk]
    mask = torch.zeros((B, N), device=keep_scores.device, dtype=torch.float32)
    mask.scatter_(dim=-1, index=top_idx, value=1.0)
    return mask

@dataclass
class ActorOutput:
    logits: torch.Tensor      # [B, N, 2]  (0=prune, 1=keep)
    prune_probs: torch.Tensor  # [B, N]
    keep_probs: torch.Tensor   # [B, N]
    mask_soft: torch.Tensor    # [B, N]  (keep prob or keep gate)
    mask_hard: torch.Tensor    # [B, N]  float 0/1 (top-k keep by default)


class dynamicvlm_actor(nn.Module):
    """
    Batch actor:
      - project img/txt to d_proj
      - cross-attn: Q = img, K/V = txt
      - residual: img_proj + attn_out
      - MLP -> logits2 (prune/keep)

    logits2[...,0] = prune, logits2[...,1] = keep
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        d_proj: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        top_k: int = 192,
        tau: float = 1.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        assert d_proj % num_heads == 0, "d_proj must be divisible by num_heads"

        self.d_proj = d_proj
        self.num_heads = num_heads
        self.top_k = top_k
        self.tau = tau

        # projections
        self.image_proj = nn.Linear(image_dim, d_proj)
        self.text_proj = nn.Linear(text_dim, d_proj)

        # cross-attn (batch_first)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_proj,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.ln_image = nn.LayerNorm(d_proj) if use_layernorm else nn.Identity()
        self.ln_text = nn.LayerNorm(d_proj) if use_layernorm else nn.Identity()
        self.ln_fused = nn.LayerNorm(d_proj) if use_layernorm else nn.Identity()

        # token-wise classifier
        self.mlp = nn.Sequential(
            nn.Linear(d_proj, d_proj//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj//4, d_proj//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj//16, 2),  # [prune, keep]
        )

    def forward(
        self,
        image: torch.Tensor,              # [B, N, img_dim]
        text: torch.Tensor,              # [B, M, txt_dim]
        text_mask: Optional[torch.Tensor], # [B, M]  True=valid or 1=valid? (see below)
        hard_mode: str = "argmax",           # "topk" or "argmax"
    ) -> ActorOutput:
        """
        txt_mask convention:
          - If txt_mask is bool: True means VALID token
          - If txt_mask is 0/1: 1 means VALID token
        MHA expects key_padding_mask: True means PAD (to be ignored).
        """
        tau = self.tau
        top_k = self.top_k
        B, N, _ = image.shape
        _, M, _ = text.shape
        device = image.device
        
        #-----------------------------------------------------------------
        #添加dummy，防止torch run崩溃
        # 1) text length is 0 -> dummy 1 token
        if M == 0:
            # create 1 dummy token on same device/dtype
            text = text.new_zeros((B, 1, text.shape[-1]))
            M = 1
            # no valid tokens => don't mask anything (or mask none)
            text_mask = None

        # 2) image length is 0 -> return safe empty outputs
        if N == 0:
            empty_logits = image.new_zeros((B, 0, 2))
            empty = image.new_zeros((B, 0))
            return ActorOutput(
                logits=empty_logits,
                prune_probs=empty,
                keep_probs=empty,
                mask_soft=empty,
                mask_hard=empty,
            )
        #-----------------------------------------------------------------

        # Project + LayerNorm
        image=self.image_proj(image)
        image = self.ln_image(image)  # [B, N, d]
        text = self.text_proj(text)   # [B, M, d]
        text = self.ln_text(text)     # [B, M, d]

        # Build key_padding_mask for MHA: True => ignore
        key_padding_mask = None
        if text_mask is not None:
            if text_mask.dtype != torch.bool:
                text_mask_bool = (text_mask != 0)
            else:
                text_mask_bool = text_mask
            key_padding_mask = ~text_mask_bool  # True means PAD

        # Cross-attention: Q=image, K/V=text
        attn_out, _ = self.cross_attn(
            query=image,
            key=text,
            value=text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # [B, N, d]

        fused = self.ln_fused(image + self.drop(attn_out))  # [B, N, d]

        # Classify per image token
        logits = self.mlp(self.drop(fused))  # [B, N, 2]

        # Soft probabilities
        probs = F.softmax(logits / tau, dim=-1)  # [B, N, 2]
        prune_probs = probs[..., 0]
        keep_probs = probs[..., 1]

        # mask_soft: use keep prob (continuous gate)
        mask_soft = keep_probs  # [B, N], in (0,1)

        # mask_hard
        if hard_mode == "argmax":
            hard_idx = torch.argmax(probs, dim=-1)      # [B, N]
            mask_hard = (hard_idx == 1).float()          # keep=1
        elif hard_mode == "topk":
            mask_hard = make_topk_keep_mask(keep_probs, top_k)  # [B, N]
        else:
            raise ValueError(f"Unknown hard_mode={hard_mode}, use 'topk' or 'argmax'.")

        return ActorOutput(
            logits=logits,
            prune_probs=prune_probs,
            keep_probs=keep_probs,
            mask_soft=mask_soft,
            mask_hard=mask_hard,
        )
        
    