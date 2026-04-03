"""
baseline_concat.py — Simple concatenation fusion baseline (no attention).

Architecture:
    Mammography Encoder → (B, N_m, 512) → Global Avg Pool → (B, 512)
    Ultrasound  Encoder → (B, N_u, 512) → Global Avg Pool → (B, 512)
    Concatenate → (B, 1024)
    Linear(1024, 512) → GELU → Dropout(0.3) → Linear(512, 2)

No attention mechanism of any kind — serves as a fusion baseline
to demonstrate the value of cross-attention.
"""

import torch
import torch.nn as nn

from src.models.encoders import build_encoder


class ConcatFusionModel(nn.Module):
    """Two-stream concatenation fusion baseline.

    Two independent encoders extract features from each modality.
    Features are globally averaged and concatenated into a single
    vector, then classified by a two-layer MLP.

    Args:
        encoder_backbone:   Name of the encoder backbone.
        token_dim:          Token embedding dimension from each encoder.
        classifier_hidden:  Hidden dimension of the classification MLP.
        classifier_dropout: Dropout probability in the MLP.
    """

    def __init__(
        self,
        encoder_backbone: str = "efficientnet_b0",
        token_dim: int = 512,
        classifier_hidden: int = 512,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()

        self.mammo_encoder = build_encoder(
            encoder_backbone, token_dim=token_dim)
        self.us_encoder = build_encoder(
            encoder_backbone, token_dim=token_dim)

        self.classifier = nn.Sequential(
            nn.Linear(token_dim * 2, classifier_hidden),  # 1024 → 512
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 2),
        )

    def forward(
        self,
        mammo: torch.Tensor,
        us: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through both streams with concatenation fusion.

        Args:
            mammo: Mammography images ``(B, 3, 224, 224)``.
            us:    Ultrasound images  ``(B, 3, 224, 224)``.

        Returns:
            Classification logits ``(B, 2)``.
        """
        # Encode both modalities
        mammo_tokens = self.mammo_encoder(mammo)    # (B, N_m, D)
        us_tokens = self.us_encoder(us)              # (B, N_u, D)

        # Global average pooling
        mammo_pooled = mammo_tokens.mean(dim=1)     # (B, D)
        us_pooled = us_tokens.mean(dim=1)            # (B, D)

        # Concatenate
        combined = torch.cat([mammo_pooled, us_pooled], dim=1)  # (B, 2D)

        # Classify
        logits = self.classifier(combined)  # (B, 2)
        return logits

    @classmethod
    def from_config(cls, config: dict) -> "ConcatFusionModel":
        """Construct model from config.yaml dict."""
        m = config["model"]
        return cls(
            encoder_backbone=m["encoder_backbone"],
            token_dim=m["token_dim"],
            classifier_dropout=m["classifier_dropout"],
        )
