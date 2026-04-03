"""
cross_attention.py — Bidirectional Cross-Attention Fusion block.

Implements the core fusion mechanism:
    1. Cross-attention A→B:  mammography queries ultrasound
    2. Cross-attention B→A:  ultrasound queries mammography
    3. Residual connections + LayerNorm
    4. Concatenation along token dimension

All computations handle variable sequence lengths (N_m ≠ N_u),
which occurs when using ViT (196 tokens) vs CNN (49 tokens) backbones.
"""

import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional Cross-Attention block for multimodal token fusion.

    Given mammography tokens ``(B, N_m, D)`` and ultrasound tokens
    ``(B, N_u, D)``, computes two-way cross-attention and returns
    a concatenated fused representation ``(B, N_m + N_u, D)``.

    Attention weights are saved as instance variables for
    explainability visualisation:
        - ``self.attn_weights_m2u``: shape ``(B, N_m, N_u)``
        - ``self.attn_weights_u2m``: shape ``(B, N_u, N_m)``

    Args:
        embed_dim:      Token embedding dimension (default: 512).
        num_heads:      Number of attention heads (default: 8).
        attn_dropout:   Dropout on attention weights (default: 0.1).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # Cross-attention A→B: mammo queries ultrasound
        self.cross_attn_m2u = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Cross-attention B→A: ultrasound queries mammography
        self.cross_attn_u2m = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Layer norms for residual connections
        self.norm_mammo = nn.LayerNorm(embed_dim)
        self.norm_us = nn.LayerNorm(embed_dim)

        # Attention weights saved for interpretability
        self.attn_weights_m2u: torch.Tensor = torch.empty(0)
        self.attn_weights_u2m: torch.Tensor = torch.empty(0)

    def forward(
        self,
        mammo_tokens: torch.Tensor,
        us_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bidirectional cross-attention and fuse modalities.

        Args:
            mammo_tokens: Mammography features ``(B, N_m, D)``.
            us_tokens:    Ultrasound features ``(B, N_u, D)``.

        Returns:
            Fused token sequence ``(B, N_m + N_u, D)``.
        """
        # ── Cross-attention A→B: mammo queries ultrasound ──────
        # Q = mammo, K/V = ultrasound
        attended_mammo, attn_m2u = self.cross_attn_m2u(
            query=mammo_tokens,
            key=us_tokens,
            value=us_tokens,
            need_weights=True,
            average_attn_weights=True,  # Average over heads → (B, N_m, N_u)
        )
        # attended_mammo: (B, N_m, D)
        # attn_m2u:       (B, N_m, N_u)

        # ── Cross-attention B→A: ultrasound queries mammography ─
        # Q = ultrasound, K/V = mammo
        attended_us, attn_u2m = self.cross_attn_u2m(
            query=us_tokens,
            key=mammo_tokens,
            value=mammo_tokens,
            need_weights=True,
            average_attn_weights=True,  # Average over heads → (B, N_u, N_m)
        )
        # attended_us: (B, N_u, D)
        # attn_u2m:    (B, N_u, N_m)

        # ── Save attention weights for explainability ──────────
        self.attn_weights_m2u = attn_m2u.detach()
        self.attn_weights_u2m = attn_u2m.detach()

        # ── Residual connections + LayerNorm ───────────────────
        attended_mammo = self.norm_mammo(attended_mammo + mammo_tokens)
        attended_us = self.norm_us(attended_us + us_tokens)

        # ── Concatenate along token dimension ──────────────────
        fused = torch.cat([attended_mammo, attended_us], dim=1)
        # fused: (B, N_m + N_u, D)

        return fused
