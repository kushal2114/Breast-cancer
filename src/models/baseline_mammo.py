"""
baseline_mammo.py — Mammography-only baseline model.

Architecture:
    Encoder (EfficientNet-B0 / DenseNet-121 / ViT-B/16)
    → (B, N, 512) tokens
    → Global Average Pooling → (B, 512)
    → Linear(512, 256) → GELU → Dropout(0.3) → Linear(256, 2)
"""

import torch
import torch.nn as nn

from src.models.encoders import build_encoder


class MammographyOnlyModel(nn.Module):
    """Single-stream mammography baseline classifier.

    Uses the same configurable encoder backbone as the fusion model,
    followed by global average pooling and a two-layer MLP classifier.

    Args:
        encoder_backbone:   Name of the encoder backbone.
        token_dim:          Token embedding dimension from encoder.
        classifier_hidden:  Hidden dimension of the classification MLP.
        classifier_dropout: Dropout probability in the MLP.
    """

    def __init__(
        self,
        encoder_backbone: str = "efficientnet_b0",
        token_dim: int = 512,
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = build_encoder(
            encoder_backbone, token_dim=token_dim)

        self.classifier = nn.Sequential(
            nn.Linear(token_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Mammography images ``(B, 3, 224, 224)``.

        Returns:
            Classification logits ``(B, 2)``.
        """
        tokens = self.encoder(x)           # (B, N, D)
        pooled = tokens.mean(dim=1)        # (B, D) — global avg pool
        logits = self.classifier(pooled)   # (B, 2)
        return logits

    @classmethod
    def from_config(cls, config: dict) -> "MammographyOnlyModel":
        """Construct model from config.yaml dict."""
        m = config["model"]
        return cls(
            encoder_backbone=m["encoder_backbone"],
            token_dim=m["token_dim"],
            classifier_hidden=m["classifier_hidden_dim"],
            classifier_dropout=m["classifier_dropout"],
        )
