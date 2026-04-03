"""
encoders.py — Pluggable encoder backbones for multimodal breast cancer classification.

Supported backbones (all pretrained on ImageNet, classification head removed):
    1. EfficientNet-B0  → (B, 49, 512) tokens  [1280→512 projection]
    2. DenseNet-121     → (B, 49, 512) tokens  [1024→512 projection]
    3. ViT-B/16         → (B, 196, 512) tokens [768→512 projection]

All encoders output shape ``(B, N, token_dim)`` where ``N`` varies by backbone
(49 for CNN backbones, 196 for ViT) and ``token_dim = 512``.

Usage:
    encoder = build_encoder("efficientnet_b0", token_dim=512)
    tokens = encoder(batch_of_images)  # (B, N, 512)
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder producing spatial feature tokens.

    Architecture:
        EfficientNet-B0 features (without classifier) → (B, 1280, 7, 7)
        Flatten spatial dims → (B, 49, 1280)
        Linear projection   → (B, 49, 512)

    Args:
        token_dim: Dimensionality of the output token embeddings.
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, token_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Remove the classification head — keep only feature extractor
        self.features = backbone.features  # Output: (B, 1280, 7, 7)
        self.projection = nn.Linear(1280, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project spatial feature tokens.

        Args:
            x: Input images of shape ``(B, 3, 224, 224)``.

        Returns:
            Token embeddings of shape ``(B, 49, token_dim)``.
        """
        feat = self.features(x)            # (B, 1280, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C) = (B, 49, 1280)
        tokens = self.projection(feat)     # (B, 49, token_dim)
        return tokens


class DenseNetEncoder(nn.Module):
    """DenseNet-121 encoder producing spatial feature tokens.

    Architecture:
        DenseNet-121 features (before global avg pool) → (B, 1024, 7, 7)
        Flatten spatial dims → (B, 49, 1024)
        Linear projection   → (B, 49, 512)

    The projection normalises DenseNet's 1024-dim features to the same
    512-dim token space as EfficientNet for cross-attention compatibility.

    Args:
        token_dim: Dimensionality of the output token embeddings.
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, token_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        # DenseNet features block (before classifier)
        self.features = backbone.features     # Output: (B, 1024, 7, 7)
        self.final_relu = nn.ReLU(inplace=True)
        self.projection = nn.Linear(1024, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project spatial feature tokens.

        Args:
            x: Input images of shape ``(B, 3, 224, 224)``.

        Returns:
            Token embeddings of shape ``(B, 49, token_dim)``.
        """
        feat = self.features(x)            # (B, 1024, 7, 7)
        feat = self.final_relu(feat)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C) = (B, 49, 1024)
        tokens = self.projection(feat)     # (B, 49, token_dim)
        return tokens


class ViTEncoder(nn.Module):
    """ViT-B/16 encoder producing patch tokens (CLS token discarded).

    Architecture:
        ViT-B/16 encoder → (B, 197, 768)  [196 patch + 1 CLS]
        Discard CLS token → (B, 196, 768)
        Linear projection → (B, 196, 512)

    NOTE: ViT produces 196 tokens vs 49 for CNN backbones.
    The cross-attention module handles variable token lengths.

    Args:
        token_dim: Dimensionality of the output token embeddings.
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, token_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        backbone = models.vit_b_16(weights=weights)

        # Keep only the encoder layers (conv_proj + encoder)
        self.conv_proj = backbone.conv_proj          # Patch embedding
        self.encoder = backbone.encoder              # Transformer encoder
        self.class_token = backbone.class_token      # (1, 1, 768)
        self.pos_embedding = backbone.encoder.pos_embedding  # (1, 197, 768)

        self.projection = nn.Linear(768, token_dim)
        self.hidden_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from ViT (discarding CLS token).

        Args:
            x: Input images of shape ``(B, 3, 224, 224)``.

        Returns:
            Token embeddings of shape ``(B, 196, token_dim)``.
        """
        B = x.shape[0]

        # Patch embedding
        x = self.conv_proj(x)                   # (B, 768, 14, 14)
        x = x.flatten(2).permute(0, 2, 1)       # (B, 196, 768)

        # Prepend CLS token
        cls_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)   # (B, 197, 768)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.encoder.ln(self.encoder.layers(x))  # (B, 197, 768)

        # Discard CLS token → keep only patch tokens
        patch_tokens = x[:, 1:, :]               # (B, 196, 768)

        # Project
        tokens = self.projection(patch_tokens)   # (B, 196, token_dim)
        return tokens


# ════════════════════════════════════════════════════════════════
#  Factory Function
# ════════════════════════════════════════════════════════════════

_ENCODER_REGISTRY = {
    "efficientnet_b0": EfficientNetEncoder,
    "densenet121": DenseNetEncoder,
    "vit_b_16": ViTEncoder,
}


def build_encoder(backbone_name: str,
                  token_dim: int = 512,
                  pretrained: bool = True) -> nn.Module:
    """Factory function to build a feature encoder by name.

    Args:
        backbone_name: One of ``"efficientnet_b0"``, ``"densenet121"``,
                       ``"vit_b_16"``.
        token_dim: Dimensionality of the output token embeddings.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        An ``nn.Module`` encoder that maps ``(B, 3, 224, 224)`` images
        to ``(B, N, token_dim)`` token sequences.

    Raises:
        ValueError: If backbone_name is not recognised.
    """
    if backbone_name not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from: {list(_ENCODER_REGISTRY.keys())}"
        )
    return _ENCODER_REGISTRY[backbone_name](
        token_dim=token_dim, pretrained=pretrained,
    )
