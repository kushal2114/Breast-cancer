"""
fusion_model.py — Full Two-Stream Cross-Attention Fusion Model.

Architecture:
    Stream A (Mammography Encoder) ──┐
                                     ├→ BidirectionalCrossAttention
    Stream B (Ultrasound Encoder) ───┘        │
                                              ▼
                                    TransformerEncoder (2 layers)
                                              │
                                    Global Average Pooling → (B, 512)
                                              │
                                    ClassificationHead → (B, 2)

Modality dropout (training only):
    With p=0.2, randomly zero out one modality's token sequence.
    Never drops both simultaneously.
"""

import random

import torch
import torch.nn as nn

from src.models.cross_attention import BidirectionalCrossAttention
from src.models.encoders import build_encoder


class CrossAttentionFusionModel(nn.Module):
    """Two-stream CNN/ViT + Bidirectional Cross-Attention Transformer
    fusion model for multimodal breast cancer classification.

    Args:
        encoder_backbone:    Name of the encoder backbone
                             (``"efficientnet_b0"``, ``"densenet121"``,
                              ``"vit_b_16"``).
        token_dim:           Dimensionality of token embeddings.
        num_heads:           Number of attention heads in cross-attention
                             and transformer encoder.
        attn_dropout:        Dropout on attention weights.
        transformer_layers:  Number of TransformerEncoder layers.
        transformer_ffn_dim: Feed-forward hidden dimension in transformer.
        classifier_hidden:   Hidden dimension in the classification MLP.
        classifier_dropout:  Dropout in the classification MLP.
        modality_dropout_p:  Probability of dropping a modality stream
                             during training.
    """

    def __init__(
        self,
        encoder_backbone: str = "efficientnet_b0",
        token_dim: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_ffn_dim: int = 1024,
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.3,
        modality_dropout_p: float = 0.2,
    ):
        super().__init__()
        self.modality_dropout_p = modality_dropout_p

        # ── Two-stream encoders ────────────────────────────────
        self.mammo_encoder = build_encoder(
            encoder_backbone, token_dim=token_dim)
        self.us_encoder = build_encoder(
            encoder_backbone, token_dim=token_dim)

        # ── Bidirectional Cross-Attention ──────────────────────
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

        # ── Transformer Encoder ────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # ── Classification Head ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 2),
        )

    def forward(
        self,
        mammo: torch.Tensor,
        us: torch.Tensor,
        modality_dropout: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the full fusion pipeline.

        Args:
            mammo:             Mammography images ``(B, 3, 224, 224)``.
            us:                Ultrasound images ``(B, 3, 224, 224)``.
            modality_dropout:  If True, randomly zero out one stream
                               during training (never both).

        Returns:
            Classification logits ``(B, 2)``.
        """
        # ── Encode both modalities ─────────────────────────────
        mammo_tokens = self.mammo_encoder(mammo)  # (B, N_m, D)
        us_tokens = self.us_encoder(us)            # (B, N_u, D)

        # ── Modality dropout (training only) ───────────────────
        if modality_dropout and self.training:
            r = random.random()
            if r < self.modality_dropout_p:
                # Zero out mammography stream
                mammo_tokens = torch.zeros_like(mammo_tokens)
            elif r < 2 * self.modality_dropout_p:
                # Zero out ultrasound stream
                us_tokens = torch.zeros_like(us_tokens)
            # Otherwise keep both → never drop both simultaneously

        # ── Cross-attention fusion ─────────────────────────────
        fused = self.cross_attention(mammo_tokens, us_tokens)
        # fused: (B, N_m + N_u, D)

        # ── Transformer encoder ────────────────────────────────
        fused = self.transformer_encoder(fused)
        # fused: (B, N_m + N_u, D)

        # ── Global average pooling ─────────────────────────────
        pooled = fused.mean(dim=1)  # (B, D)

        # ── Classification ─────────────────────────────────────
        logits = self.classifier(pooled)  # (B, 2)
        return logits

    @classmethod
    def from_config(cls, config: dict) -> "CrossAttentionFusionModel":
        """Construct model from config.yaml dict.

        Args:
            config: Parsed config.yaml as a nested dict.

        Returns:
            Initialised CrossAttentionFusionModel.
        """
        m = config["model"]
        return cls(
            encoder_backbone=m["encoder_backbone"],
            token_dim=m["token_dim"],
            num_heads=m["num_attention_heads"],
            attn_dropout=m["attention_dropout"],
            transformer_layers=m["transformer_layers"],
            transformer_ffn_dim=m["transformer_ffn_dim"],
            classifier_hidden=m["classifier_hidden_dim"],
            classifier_dropout=m["classifier_dropout"],
            modality_dropout_p=m["modality_dropout_prob"],
        )
