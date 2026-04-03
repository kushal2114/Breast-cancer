"""
explain.py — Explainability visualisations: Grad-CAM++ and cross-attention maps.

Generates three types of figures:
    1. Grad-CAM++ on mammography stream (CNN backbones only)
    2. Grad-CAM++ on ultrasound stream (CNN backbones only)
    3. Cross-attention weight maps (all backbones)

For ViT backbone, Grad-CAM++ is skipped and only attention maps are produced.

Usage:
    python src/explain.py --checkpoint outputs/checkpoints/fusion_efficientnet_b0_best.pth
    python src/explain.py --checkpoint outputs/checkpoints/fusion_vit_b_16_best.pth
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataloaders, load_cbis, load_busi
from src.preprocessing import get_mammography_transforms, get_ultrasound_transforms
from src.models.fusion_model import CrossAttentionFusionModel

# Plotting defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
})


# ════════════════════════════════════════════════════════════════
#  Grad-CAM++ for CNN backbones
# ════════════════════════════════════════════════════════════════

def get_target_layer(encoder: nn.Module, backbone_name: str) -> nn.Module:
    """Identify the last convolutional layer for Grad-CAM++.

    Args:
        encoder:       Encoder module (mammo_encoder or us_encoder).
        backbone_name: One of efficientnet_b0, densenet121.

    Returns:
        The target nn.Module (last conv block).

    Raises:
        ValueError: If backbone is ViT (not applicable).
    """
    if backbone_name == "efficientnet_b0":
        # Last block in EfficientNet features
        return encoder.features[-1]
    elif backbone_name == "densenet121":
        # Last DenseBlock in DenseNet features
        return encoder.features.denseblock4
    else:
        raise ValueError(
            f"Grad-CAM++ is not defined for backbone '{backbone_name}'. "
            f"Use attention rollout for ViT instead."
        )


def generate_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Generate a Grad-CAM++ heatmap for a given input and target class.

    Uses the pytorch-grad-cam library for Grad-CAM++ computation.

    Args:
        model:        Full fusion model or encoder wrapper.
        target_layer: The convolutional layer to hook.
        input_tensor: Preprocessed image tensor (1, 3, H, W).
        target_class: Class index for gradient computation.
        device:       Compute device.

    Returns:
        Heatmap as (H, W) float32 numpy array in [0, 1].
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    targets = [ClassifierOutputTarget(target_class)]

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]  # (H, W)


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an original image.

    Args:
        original_image: (H, W, 3) uint8 BGR or RGB image.
        heatmap:        (H, W) float32 in [0, 1].
        alpha:          Blending factor.

    Returns:
        (H, W, 3) uint8 overlaid image.
    """
    heatmap_coloured = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)
    if original_image.shape[:2] != heatmap_coloured.shape[:2]:
        heatmap_coloured = cv2.resize(
            heatmap_coloured, (original_image.shape[1], original_image.shape[0]))
    overlaid = cv2.addWeighted(original_image, 1 - alpha,
                                heatmap_coloured, alpha, 0)
    return overlaid


# ════════════════════════════════════════════════════════════════
#  Single-stream wrapper for Grad-CAM++
# ════════════════════════════════════════════════════════════════

class MammoStreamWrapper(nn.Module):
    """Wrapper that runs the fusion model with a fixed dummy US input.

    This lets pytorch-grad-cam compute gradients w.r.t. the mammo
    stream only.
    """

    def __init__(self, fusion_model: CrossAttentionFusionModel,
                 dummy_us: torch.Tensor):
        super().__init__()
        self.fusion_model = fusion_model
        self.dummy_us = dummy_us

    def forward(self, mammo: torch.Tensor) -> torch.Tensor:
        return self.fusion_model(mammo, self.dummy_us.expand(
            mammo.size(0), -1, -1, -1), modality_dropout=False)


class USStreamWrapper(nn.Module):
    """Wrapper that runs the fusion model with a fixed dummy mammo input.

    This lets pytorch-grad-cam compute gradients w.r.t. the US
    stream only.
    """

    def __init__(self, fusion_model: CrossAttentionFusionModel,
                 dummy_mammo: torch.Tensor):
        super().__init__()
        self.fusion_model = fusion_model
        self.dummy_mammo = dummy_mammo

    def forward(self, us: torch.Tensor) -> torch.Tensor:
        return self.fusion_model(self.dummy_mammo.expand(
            us.size(0), -1, -1, -1), us, modality_dropout=False)


# ════════════════════════════════════════════════════════════════
#  Cross-attention map visualisation
# ════════════════════════════════════════════════════════════════

def visualise_cross_attention(
    model: CrossAttentionFusionModel,
    mammo_tensor: torch.Tensor,
    us_tensor: torch.Tensor,
    backbone_name: str,
    original_mammo: np.ndarray,
    original_us: np.ndarray,
    save_path: str,
    mammo_gradcam: Optional[np.ndarray] = None,
    us_gradcam: Optional[np.ndarray] = None,
) -> None:
    """Visualise cross-attention maps and optionally Grad-CAM++ overlays.

    Generates the KEY figure for the paper: a side-by-side figure with
    original images, Grad-CAM overlays, and cross-attention maps.

    For CNN (N=49): attention reshaped to (7, 7) spatial map.
    For ViT (N=196): attention reshaped to (14, 14) grid.

    Args:
        model:         Fusion model (already ran forward pass).
        mammo_tensor:  (1, 3, 224, 224) mammography tensor.
        us_tensor:     (1, 3, 224, 224) ultrasound tensor.
        backbone_name: Backbone name for spatial reshape.
        original_mammo: (H, W, 3) uint8 original mammography image.
        original_us:    (H, W, 3) uint8 original ultrasound image.
        save_path:      Output file path.
        mammo_gradcam:  Optional Grad-CAM++ heatmap for mammo stream.
        us_gradcam:     Optional Grad-CAM++ heatmap for US stream.
    """
    # Get attention weights from the model
    attn_m2u = model.cross_attention.attn_weights_m2u[0].cpu().numpy()
    attn_u2m = model.cross_attention.attn_weights_u2m[0].cpu().numpy()

    # Spatial grid size based on backbone
    if backbone_name == "vit_b_16":
        grid_size = 14  # sqrt(196)
    else:
        grid_size = 7   # sqrt(49)

    # Average attention across target tokens → per-query spatial map
    # attn_m2u: (N_m, N_u) → average over N_u → (N_m,) → reshape
    mammo_attn = attn_m2u.mean(axis=1)  # (N_m,)
    us_attn = attn_u2m.mean(axis=1)     # (N_u,)

    mammo_attn_map = mammo_attn.reshape(grid_size, grid_size)
    us_attn_map = us_attn.reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    mammo_attn_map = (mammo_attn_map - mammo_attn_map.min()) / \
        (mammo_attn_map.max() - mammo_attn_map.min() + 1e-8)
    us_attn_map = (us_attn_map - us_attn_map.min()) / \
        (us_attn_map.max() - us_attn_map.min() + 1e-8)

    # Resize attention maps to original image size
    h, w = original_mammo.shape[:2]
    mammo_attn_resized = cv2.resize(mammo_attn_map, (w, h),
                                     interpolation=cv2.INTER_CUBIC)
    us_attn_resized = cv2.resize(us_attn_map,
                                  (original_us.shape[1], original_us.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

    # Build figure
    has_gradcam = mammo_gradcam is not None and us_gradcam is not None
    n_cols = 4 if has_gradcam else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Column 0: Original mammo
    axes[0].imshow(original_mammo)
    axes[0].set_title("Original Mammogram")
    axes[0].axis("off")

    col = 1
    if has_gradcam:
        # Column 1: Mammo Grad-CAM
        mammo_overlay = overlay_heatmap(original_mammo, mammo_gradcam)
        axes[col].imshow(mammo_overlay)
        axes[col].set_title("Mammo Grad-CAM++")
        axes[col].axis("off")
        col += 1

    # US Grad-CAM or Original US
    if has_gradcam:
        us_overlay = overlay_heatmap(original_us, us_gradcam)
        axes[col].imshow(us_overlay)
        axes[col].set_title("US Grad-CAM++")
        axes[col].axis("off")
        col += 1
    else:
        axes[col].imshow(original_us)
        axes[col].set_title("Original Ultrasound")
        axes[col].axis("off")
        col += 1

    # Cross-attention map
    axes[col].imshow(original_mammo)
    axes[col].imshow(mammo_attn_resized, cmap="hot", alpha=0.6)
    axes[col].set_title("Cross-Attention Map")
    axes[col].axis("off")

    fig.suptitle("Multimodal Explainability", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Attention visualisation → {save_path}")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    """Generate explainability figures for a trained fusion model."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM++ and cross-attention visualisations.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a fusion model checkpoint.")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml.")
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of correct samples per class to visualise.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    # ── Load model ─────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = ckpt["backbone"]
    config["model"]["encoder_backbone"] = backbone

    model = CrossAttentionFusionModel.from_config(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    is_vit = backbone == "vit_b_16"
    if is_vit:
        print("[INFO] ViT backbone detected → skipping Grad-CAM++, "
              "using attention rollout from cross-attention weights instead.")

    # ── Output directories ─────────────────────────────────────
    fig_dir = config["output"]["figure_dir"]
    gradcam_mammo_dir = os.path.join(fig_dir, "gradcam_mammo")
    gradcam_us_dir = os.path.join(fig_dir, "gradcam_us")
    attn_dir = os.path.join(fig_dir, "attention_maps")
    for d in [gradcam_mammo_dir, gradcam_us_dir, attn_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Build test data ────────────────────────────────────────
    data = build_dataloaders(config, model_type="fusion")
    test_loader = data["test"]

    data_cfg = config["data"]
    img_size = data_cfg["image_size"]
    mammo_eval_tf = get_mammography_transforms(img_size, is_training=False)
    us_eval_tf = get_ultrasound_transforms(img_size, is_training=False)

    # ── Collect correct predictions ────────────────────────────
    correct_benign = []   # (mammo_tensor, us_tensor, mammo_orig, us_orig)
    correct_malignant = []

    print("\n[STEP 1] Collecting correctly classified samples...")
    for batch in tqdm(test_loader, desc="  Scanning"):
        mammo, us, labels = batch
        mammo = mammo.to(device)
        us = us.to(device)

        with torch.no_grad():
            logits = model(mammo, us, modality_dropout=False)
            preds = logits.argmax(dim=1)

        for i in range(len(labels)):
            if preds[i].item() == labels[i].item():
                entry = (
                    mammo[i:i+1].cpu(),
                    us[i:i+1].cpu(),
                    labels[i].item(),
                )
                if labels[i].item() == 0 and len(correct_benign) < args.num_samples:
                    correct_benign.append(entry)
                elif labels[i].item() == 1 and len(correct_malignant) < args.num_samples:
                    correct_malignant.append(entry)

        if (len(correct_benign) >= args.num_samples and
                len(correct_malignant) >= args.num_samples):
            break

    print(f"  Collected {len(correct_benign)} benign, "
          f"{len(correct_malignant)} malignant samples")

    all_samples = (
        [(s, "benign") for s in correct_benign] +
        [(s, "malignant") for s in correct_malignant]
    )

    # ── Generate visualisations ────────────────────────────────
    print("\n[STEP 2] Generating explainability figures...")

    for idx, ((mammo_t, us_t, label), class_name) in enumerate(
            tqdm(all_samples, desc="  Generating")):

        mammo_t = mammo_t.to(device)
        us_t = us_t.to(device)

        # Forward pass to populate attention weights
        with torch.no_grad():
            _ = model(mammo_t, us_t, modality_dropout=False)

        # Denormalise for visualisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        mammo_vis = (mammo_t.cpu() * std + mean).clamp(0, 1)
        us_vis = (us_t.cpu() * std + mean).clamp(0, 1)
        mammo_np = (mammo_vis[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        us_np = (us_vis[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        mammo_gradcam_map = None
        us_gradcam_map = None

        if not is_vit:
            # ── Grad-CAM++ for mammography stream ──────────────
            try:
                dummy_us = us_t.detach()
                mammo_wrapper = MammoStreamWrapper(model, dummy_us)
                mammo_target = get_target_layer(
                    model.mammo_encoder, backbone)
                mammo_gradcam_map = generate_gradcam(
                    mammo_wrapper, mammo_target, mammo_t,
                    target_class=label, device=device)

                mammo_overlay = overlay_heatmap(mammo_np, mammo_gradcam_map)
                save_name = f"{class_name}_{idx:02d}.png"
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(mammo_overlay)
                ax.set_title(f"Mammo Grad-CAM++ ({class_name})")
                ax.axis("off")
                fig.savefig(os.path.join(gradcam_mammo_dir, save_name),
                            dpi=200, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  [WARN] Mammo Grad-CAM++ failed for sample {idx}: {e}")

            # ── Grad-CAM++ for ultrasound stream ───────────────
            try:
                dummy_mammo = mammo_t.detach()
                us_wrapper = USStreamWrapper(model, dummy_mammo)
                us_target = get_target_layer(model.us_encoder, backbone)
                us_gradcam_map = generate_gradcam(
                    us_wrapper, us_target, us_t,
                    target_class=label, device=device)

                us_overlay = overlay_heatmap(us_np, us_gradcam_map)
                save_name = f"{class_name}_{idx:02d}.png"
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(us_overlay)
                ax.set_title(f"US Grad-CAM++ ({class_name})")
                ax.axis("off")
                fig.savefig(os.path.join(gradcam_us_dir, save_name),
                            dpi=200, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  [WARN] US Grad-CAM++ failed for sample {idx}: {e}")

        # ── Cross-attention map ────────────────────────────────
        save_name = f"{class_name}_{idx:02d}_combined.png"
        visualise_cross_attention(
            model=model,
            mammo_tensor=mammo_t,
            us_tensor=us_t,
            backbone_name=backbone,
            original_mammo=mammo_np,
            original_us=us_np,
            save_path=os.path.join(attn_dir, save_name),
            mammo_gradcam=mammo_gradcam_map,
            us_gradcam=us_gradcam_map,
        )

    print(f"\n[DONE] Explainability figures saved to {fig_dir}/")
    print(f"  Grad-CAM++ (mammo): {gradcam_mammo_dir}/")
    print(f"  Grad-CAM++ (US):    {gradcam_us_dir}/")
    print(f"  Attention maps:     {attn_dir}/")


if __name__ == "__main__":
    main()
