"""
explainer.py — Grad-CAM++ generation for the BreastAI Classifier API.

Generates Grad-CAM++ heatmaps overlaid on the original uploaded image.
Uses the same MammoStreamWrapper / USStreamWrapper pattern as
src/explain.py, targeting DenseNet-121's last dense block
(features.denseblock4) as the convolutional target layer.

The result is returned as a base64-encoded PNG string for embedding
directly in the JSON API response.
"""

import base64
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.fusion_model import CrossAttentionFusionModel

logger = logging.getLogger("breastai.explainer")


class MammoStreamWrapper(nn.Module):
    """Wrapper that runs the fusion model with a fixed dummy US input.

    Allows pytorch-grad-cam to compute gradients with respect to
    the mammography stream only.
    """

    def __init__(
        self, fusion_model: CrossAttentionFusionModel, dummy_us: torch.Tensor
    ):
        super().__init__()
        self.fusion_model = fusion_model
        self.dummy_us = dummy_us

    def forward(self, mammo: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fusion model with frozen US stream."""
        return self.fusion_model(
            mammo,
            self.dummy_us.expand(mammo.size(0), -1, -1, -1),
            modality_dropout=False,
        )


class USStreamWrapper(nn.Module):
    """Wrapper that runs the fusion model with a fixed dummy mammo input.

    Allows pytorch-grad-cam to compute gradients with respect to
    the ultrasound stream only.
    """

    def __init__(
        self,
        fusion_model: CrossAttentionFusionModel,
        dummy_mammo: torch.Tensor,
    ):
        super().__init__()
        self.fusion_model = fusion_model
        self.dummy_mammo = dummy_mammo

    def forward(self, us: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fusion model with frozen mammo stream."""
        return self.fusion_model(
            self.dummy_mammo.expand(us.size(0), -1, -1, -1),
            us,
            modality_dropout=False,
        )


class GradCAMExplainer:
    """Generates Grad-CAM++ heatmaps for the active modality stream.

    The target layer is DenseNet-121's last dense block
    (encoder.features.denseblock4) inside the correct stream
    encoder (mammography or ultrasound).

    Args:
        model: The loaded CrossAttentionFusionModel instance.
    """

    def __init__(self, model: CrossAttentionFusionModel) -> None:
        self.model = model

    def generate(
        self,
        mammo_tensor: torch.Tensor,
        us_tensor: torch.Tensor,
        modality: str,
        original_image_bytes: bytes,
    ) -> str:
        """Generate a Grad-CAM++ heatmap overlaid on the original image.

        Args:
            mammo_tensor: Preprocessed mammography tensor (1, 3, 224, 224).
            us_tensor: Preprocessed ultrasound tensor (1, 3, 224, 224).
            modality: "mammography" or "ultrasound".
            original_image_bytes: Raw bytes of the uploaded image for overlay.

        Returns:
            Base64-encoded PNG string of the heatmap overlay.
        """
        # Build the single-stream wrapper and identify target layer
        if modality == "mammography":
            wrapper = MammoStreamWrapper(self.model, us_tensor.detach())
            target_layer = self.model.mammo_encoder.features.denseblock4
            input_tensor = mammo_tensor
        else:
            wrapper = USStreamWrapper(self.model, mammo_tensor.detach())
            target_layer = self.model.us_encoder.features.denseblock4
            input_tensor = us_tensor

        # Run Grad-CAM++
        cam = GradCAMPlusPlus(model=wrapper, target_layers=[target_layer])

        # Use the predicted class as target
        with torch.no_grad():
            logits = self.model(
                mammo_tensor, us_tensor, modality_dropout=False
            )
            pred_class = logits.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]  # (H, W) float32 in [0, 1]

        # Load original image for overlay
        original_pil = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        original_np = np.array(original_pil)  # (H, W, 3) uint8 RGB

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(
            heatmap,
            (original_np.shape[1], original_np.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        # Apply colormap and overlay
        heatmap_coloured = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

        overlaid = cv2.addWeighted(
            original_np, 0.5, heatmap_coloured, 0.5, 0
        )

        # Encode as base64 PNG
        overlay_pil = Image.fromarray(overlaid)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        buffer.seek(0)
        base64_string = base64.b64encode(buffer.read()).decode("utf-8")

        logger.info(
            "Grad-CAM++ generated for %s stream (predicted class %d)",
            modality,
            pred_class,
        )
        return base64_string
