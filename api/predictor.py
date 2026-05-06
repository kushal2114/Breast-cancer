"""
predictor.py — Model loading and inference logic for the BreastAI Classifier API.

Loads the fusion_densenet121 model once at startup and provides
preprocessing and prediction methods for single-image inference.
The missing modality stream is filled with a zero tensor, which
works because the model was trained with modality dropout.
"""

import io
import logging
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.models.fusion_model import CrossAttentionFusionModel
from src.preprocessing import get_mammography_transforms, get_ultrasound_transforms

logger = logging.getLogger("breastai.predictor")

# Class index → label mapping
CLASS_LABELS = {0: "benign", 1: "malignant"}


class BreastCancerPredictor:
    """Handles model loading, image preprocessing, and inference.

    The model is loaded once during initialisation and kept in eval()
    mode. Each prediction preprocesses the uploaded image, pairs it
    with a zero tensor for the missing modality, and runs a single
    forward pass through the fusion model.

    Args:
        config_path: Path to configs/config.yaml.
        checkpoint_path: Path to fusion_densenet121_best.pth.
    """

    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        # ── Load configuration ─────────────────────────────────
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # ── Auto-detect device ─────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", self.device)

        # ── Build model with DenseNet-121 backbone ─────────────
        # Override config backbone to always use densenet121
        self.config["model"]["encoder_backbone"] = "densenet121"
        self.model = CrossAttentionFusionModel.from_config(self.config)

        # ── Load checkpoint weights ────────────────────────────
        logger.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Model loaded successfully — fusion_densenet121 on %s", self.device
        )

        # ── Build val/test transforms (no augmentation) ────────
        img_size = self.config["data"]["image_size"]  # 224
        self.mammo_transform = get_mammography_transforms(
            image_size=img_size, is_training=False
        )
        self.us_transform = get_ultrasound_transforms(
            image_size=img_size, is_training=False
        )

    def preprocess(
        self, image_bytes: bytes, modality: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess an uploaded image into the two-stream tensor pair.

        The uploaded image is transformed using the appropriate
        modality-specific pipeline. The other stream receives a
        zero tensor of the same shape.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            modality: Either "mammography" or "ultrasound".

        Returns:
            Tuple of (mammo_tensor, us_tensor), each of shape
            (1, 3, 224, 224), moved to the active device.
        """
        # Decode bytes → PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply the correct modality transform
        if modality == "mammography":
            transformed = self.mammo_transform(pil_image)  # (3, 224, 224)
            mammo_tensor = transformed.unsqueeze(0)         # (1, 3, 224, 224)
            us_tensor = torch.zeros(1, 3, 224, 224)
        elif modality == "ultrasound":
            transformed = self.us_transform(pil_image)      # (3, 224, 224)
            us_tensor = transformed.unsqueeze(0)            # (1, 3, 224, 224)
            mammo_tensor = torch.zeros(1, 3, 224, 224)
        else:
            raise ValueError(
                f"Invalid modality '{modality}'. "
                f"Must be 'mammography' or 'ultrasound'."
            )

        # Move to device
        mammo_tensor = mammo_tensor.to(self.device)
        us_tensor = us_tensor.to(self.device)

        return mammo_tensor, us_tensor

    def predict(
        self, image_bytes: bytes, modality: str
    ) -> dict:
        """Run full inference on an uploaded image.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            modality: Either "mammography" or "ultrasound".

        Returns:
            Dictionary containing:
                - prediction: "benign" or "malignant"
                - confidence: float (max probability)
                - probabilities: {"benign": float, "malignant": float}
                - inference_time_ms: float
        """
        mammo_tensor, us_tensor = self.preprocess(image_bytes, modality)

        # Timed inference
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(
                mammo_tensor, us_tensor, modality_dropout=False
            )  # (1, 2)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Softmax → probabilities
        probs = F.softmax(logits, dim=1).squeeze(0)  # (2,)
        confidence, pred_idx = probs.max(dim=0)

        prediction = CLASS_LABELS[pred_idx.item()]
        probabilities = {
            "benign": round(probs[0].item(), 4),
            "malignant": round(probs[1].item(), 4),
        }

        return {
            "prediction": prediction,
            "confidence": round(confidence.item(), 4),
            "probabilities": probabilities,
            "inference_time_ms": round(elapsed_ms, 1),
            "mammo_tensor": mammo_tensor,
            "us_tensor": us_tensor,
        }

    def predict_fusion(
        self, mammo_bytes: bytes, us_bytes: bytes
    ) -> dict:
        """Run full fusion inference with both modality images.

        Both mammography and ultrasound images are preprocessed with
        their respective pipelines and fed as real tensors to the
        fusion model — no zero-filling.

        Args:
            mammo_bytes: Raw bytes of the mammography image file.
            us_bytes: Raw bytes of the ultrasound image file.

        Returns:
            Dictionary containing:
                - prediction: "benign" or "malignant"
                - confidence: float (max probability)
                - probabilities: {"benign": float, "malignant": float}
                - inference_time_ms: float
                - mammo_tensor, us_tensor: preprocessed tensors
        """
        # Decode and transform mammography image
        mammo_pil = Image.open(io.BytesIO(mammo_bytes)).convert("RGB")
        mammo_tensor = self.mammo_transform(mammo_pil).unsqueeze(0).to(self.device)

        # Decode and transform ultrasound image
        us_pil = Image.open(io.BytesIO(us_bytes)).convert("RGB")
        us_tensor = self.us_transform(us_pil).unsqueeze(0).to(self.device)

        # Timed inference
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(
                mammo_tensor, us_tensor, modality_dropout=False
            )  # (1, 2)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Softmax → probabilities
        probs = F.softmax(logits, dim=1).squeeze(0)  # (2,)
        confidence, pred_idx = probs.max(dim=0)

        prediction = CLASS_LABELS[pred_idx.item()]
        probabilities = {
            "benign": round(probs[0].item(), 4),
            "malignant": round(probs[1].item(), 4),
        }

        return {
            "prediction": prediction,
            "confidence": round(confidence.item(), 4),
            "probabilities": probabilities,
            "inference_time_ms": round(elapsed_ms, 1),
            "mammo_tensor": mammo_tensor,
            "us_tensor": us_tensor,
        }
