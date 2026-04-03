"""
evaluate.py — Full evaluation script for all model configurations.

Evaluates all 6 ablation models on the held-out test set and produces:
    - Side-by-side 6-model comparison table (console + CSV)
    - outputs/results/metrics_comparison.csv
    - outputs/figures/roc_comparison.png  (all 6 ROC curves)
    - outputs/figures/confusion_matrices.png

Additional tests for the fusion model:
    - Missing-modality robustness (zero out one stream)
    - OOD confidence distribution using BUSI normal/ images

Usage:
    python src/evaluate.py --config configs/config.yaml
    python src/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/fusion_efficientnet_b0_best.pth --model fusion --backbone efficientnet_b0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    build_dataloaders,
    load_busi,
    load_cbis,
    get_mammography_transforms,
    get_ultrasound_transforms,
)
from src.preprocessing import get_mammography_transforms, get_ultrasound_transforms
from src.models.fusion_model import CrossAttentionFusionModel
from src.models.baseline_mammo import MammographyOnlyModel
from src.models.baseline_us import UltrasoundOnlyModel
from src.models.baseline_concat import ConcatFusionModel

# Plotting style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ════════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════════

def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint.
        device:          Compute device.

    Returns:
        (model, checkpoint_metadata) tuple.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = ckpt["model_type"]
    backbone = ckpt["backbone"]
    config = ckpt["config"]
    config["model"]["encoder_backbone"] = backbone

    if model_type == "fusion":
        model = CrossAttentionFusionModel.from_config(config)
    elif model_type == "mammo":
        model = MammographyOnlyModel.from_config(config)
    elif model_type == "us":
        model = UltrasoundOnlyModel.from_config(config)
    elif model_type == "concat":
        model = ConcatFusionModel.from_config(config)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, ckpt


# ════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    model_type: str,
    mammo_zeroed: bool = False,
    us_zeroed: bool = False,
) -> Dict:
    """Run full evaluation on a data loader.

    Args:
        model:        Trained model in eval mode.
        loader:       Test DataLoader.
        device:       Compute device.
        model_type:   One of fusion/mammo/us/concat.
        mammo_zeroed: If True, zero out mammography input (robustness).
        us_zeroed:    If True, zero out ultrasound input (robustness).

    Returns:
        Dict with metrics: accuracy, sensitivity, specificity,
        precision, f1, auc, all_labels, all_probs.
    """
    model.eval()
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        if model_type in ("fusion", "concat"):
            mammo, us, labels = batch
            mammo = mammo.to(device, non_blocking=True)
            us = us.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mammo_zeroed:
                mammo = torch.zeros_like(mammo)
            if us_zeroed:
                us = torch.zeros_like(us)

            if model_type == "fusion":
                logits = model(mammo, us, modality_dropout=False)
            else:
                logits = model(mammo, us)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(
        all_labels, preds, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "sensitivity": recall_score(all_labels, preds, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs)
            if len(np.unique(all_labels)) > 1 else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "all_labels": all_labels,
        "all_probs": all_probs,
    }
    return metrics


# ════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════

def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: str,
) -> None:
    """Plot ROC curves for all models on a single figure.

    Args:
        results: Dict mapping model names to their evaluation results.
        save_path: Output file path for the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    for (name, metrics), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(metrics["all_labels"], metrics["all_probs"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] ROC curves → {save_path}")


def plot_confusion_matrices(
    results: Dict[str, Dict],
    save_path: str,
) -> None:
    """Plot confusion matrices for all models in a grid.

    Args:
        results: Dict mapping model names to their evaluation results.
        save_path: Output file path for the figure.
    """
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, metrics) in enumerate(results.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=axes[idx],
        )
        axes[idx].set_title(name, fontsize=11)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Confusion matrices → {save_path}")


def plot_ood_confidence(
    confidences: np.ndarray,
    save_path: str,
) -> None:
    """Plot OOD confidence distribution for normal BUSI images.

    Args:
        confidences: Array of malignant-class probabilities for OOD inputs.
        save_path:   Output file path for the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(confidences, bins=30, color="coral", edgecolor="black",
            alpha=0.75)
    ax.axvline(x=0.5, color="red", linestyle="--", lw=2,
               label="Decision boundary (0.5)")
    ax.set_xlabel("Malignancy Confidence (P(malignant))")
    ax.set_ylabel("Count")
    ax.set_title("OOD Confidence Distribution — BUSI Normal Images")
    ax.legend()
    ax.grid(alpha=0.3)
    mean_conf = np.mean(confidences)
    ax.annotate(f"Mean confidence: {mean_conf:.3f}",
                xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] OOD confidence distribution → {save_path}")


# ════════════════════════════════════════════════════════════════
#  OOD Evaluation
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ood(
    model: nn.Module,
    config: dict,
    device: torch.device,
) -> np.ndarray:
    """Evaluate fusion model on OOD normal ultrasound images.

    Pairs each normal US image with a random benign mammogram.
    Returns the malignant-class confidence array.

    Args:
        model:  Fusion model in eval mode.
        config: Parsed config dict.
        device: Compute device.

    Returns:
        1-D array of P(malignant) for each OOD sample.
    """
    from src.dataset import load_cbis
    from PIL import Image
    import random as rng

    data_cfg = config["data"]
    img_size = data_cfg["image_size"]

    # Load normal US images
    ood_dir = Path(data_cfg["us_ood_dir"])
    ood_files = sorted([
        f for f in ood_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        and "_mask" not in f.stem.lower()
    ])

    # Load benign mammograms for pairing
    mammo_df = load_cbis(data_cfg["mammo_csv_train"],
                         data_cfg["mammo_jpeg_root"])
    benign_mammo = mammo_df[mammo_df["label"] == 0]["image_path"].tolist()

    mammo_tf = get_mammography_transforms(img_size, is_training=False)
    us_tf = get_ultrasound_transforms(img_size, is_training=False)

    model.eval()
    confidences = []

    for ood_path in tqdm(ood_files, desc="  OOD eval"):
        # Load OOD ultrasound image
        us_img = Image.open(str(ood_path)).convert("RGB")
        us_tensor = us_tf(us_img).unsqueeze(0).to(device)

        # Pair with random benign mammogram
        mammo_path = rng.choice(benign_mammo)
        mammo_img = Image.open(mammo_path).convert("RGB")
        mammo_tensor = mammo_tf(mammo_img).unsqueeze(0).to(device)

        logits = model(mammo_tensor, us_tensor, modality_dropout=False)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().item()
        confidences.append(prob)

    return np.array(confidences)


# ════════════════════════════════════════════════════════════════
#  Main evaluation
# ════════════════════════════════════════════════════════════════

def evaluate_single(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> Tuple[str, Dict]:
    """Load and evaluate a single checkpoint.

    Returns:
        (display_name, metrics) tuple.
    """
    model, ckpt = load_trained_model(checkpoint_path, device)
    model_type = ckpt["model_type"]
    backbone = ckpt["backbone"]
    name = f"{model_type}_{backbone}"

    # Build test loader
    data = build_dataloaders(config, model_type=model_type)
    test_loader = data["test"]

    print(f"\n[EVAL] {name}")
    metrics = evaluate_model(model, test_loader, device, model_type)

    return name, metrics, model, model_type, data


def main():
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate breast cancer classification models.")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Evaluate a single checkpoint (optional).")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["fusion", "mammo", "us", "concat"],
        help="Model type (used with --checkpoint).")
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=["efficientnet_b0", "densenet121", "vit_b_16"],
        help="Backbone (used with --checkpoint).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    out_cfg = config["output"]
    os.makedirs(out_cfg["figure_dir"], exist_ok=True)
    os.makedirs(out_cfg["results_dir"], exist_ok=True)

    # ── Discover all checkpoints ───────────────────────────────
    if args.checkpoint:
        # Single checkpoint mode
        if args.backbone:
            config["model"]["encoder_backbone"] = args.backbone
        name, metrics, model, model_type, data = evaluate_single(
            args.checkpoint, config, device)
        all_results = {name: metrics}

        # Run robustness + OOD if fusion
        if model_type == "fusion":
            test_loader = data["test"]
            print("\n[ROBUSTNESS] Mammo-only inference (US zeroed):")
            m_only = evaluate_model(
                model, test_loader, device, "fusion",
                us_zeroed=True)
            print(f"  AUC = {m_only['auc']:.4f}")

            print("[ROBUSTNESS] US-only inference (Mammo zeroed):")
            u_only = evaluate_model(
                model, test_loader, device, "fusion",
                mammo_zeroed=True)
            print(f"  AUC = {u_only['auc']:.4f}")

            all_results[f"{name}_mammo_only"] = m_only
            all_results[f"{name}_us_only"] = u_only

            # OOD test
            print("\n[OOD] Evaluating on BUSI normal/ images...")
            ood_conf = evaluate_ood(model, config, device)
            plot_ood_confidence(
                ood_conf,
                os.path.join(out_cfg["figure_dir"],
                             "ood_confidence_distribution.png"))
    else:
        # Auto-discover all checkpoints in the checkpoint directory
        ckpt_dir = Path(out_cfg["checkpoint_dir"])
        ckpt_files = sorted(ckpt_dir.glob("*_best.pth"))

        if not ckpt_files:
            print(f"[ERROR] No checkpoints found in {ckpt_dir}")
            return

        all_results = {}
        fusion_model = None
        fusion_data = None
        fusion_name = None

        for ckpt_file in ckpt_files:
            name, metrics, model, model_type, data = evaluate_single(
                str(ckpt_file), config, device)
            all_results[name] = metrics

            # Track the first fusion model for robustness/OOD tests
            if model_type == "fusion" and fusion_model is None:
                fusion_model = model
                fusion_data = data
                fusion_name = name

        # Run robustness & OOD on fusion model
        if fusion_model is not None:
            test_loader = fusion_data["test"]
            print(f"\n[ROBUSTNESS] {fusion_name} — Mammo-only (US zeroed):")
            m_only = evaluate_model(
                fusion_model, test_loader, device, "fusion",
                us_zeroed=True)
            print(f"  AUC = {m_only['auc']:.4f}")

            print(f"[ROBUSTNESS] {fusion_name} — US-only (Mammo zeroed):")
            u_only = evaluate_model(
                fusion_model, test_loader, device, "fusion",
                mammo_zeroed=True)
            print(f"  AUC = {u_only['auc']:.4f}")

            print("\n[OOD] Evaluating on BUSI normal/ images...")
            ood_conf = evaluate_ood(fusion_model, config, device)
            plot_ood_confidence(
                ood_conf,
                os.path.join(out_cfg["figure_dir"],
                             "ood_confidence_distribution.png"))

    # ── Comparison table ───────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("  MODEL COMPARISON — TEST SET METRICS")
    print(f"{'═' * 90}")
    header = (f"{'Model':<35} {'Acc':>7} {'Sens':>7} {'Spec':>7} "
              f"{'Prec':>7} {'F1':>7} {'AUC':>7}")
    print(header)
    print("─" * 90)

    rows_for_csv = []
    for name, metrics in all_results.items():
        row_str = (
            f"{name:<35} "
            f"{metrics['accuracy']:>7.4f} "
            f"{metrics['sensitivity']:>7.4f} "
            f"{metrics['specificity']:>7.4f} "
            f"{metrics['precision']:>7.4f} "
            f"{metrics['f1']:>7.4f} "
            f"{metrics['auc']:>7.4f}"
        )
        print(row_str)
        rows_for_csv.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        })
    print(f"{'═' * 90}\n")

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = os.path.join(out_cfg["results_dir"],
                            "metrics_comparison.csv")
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)
    print(f"[SAVED] Metrics CSV → {csv_path}")

    # ── Save JSON ──────────────────────────────────────────────
    json_results = {}
    for name, metrics in all_results.items():
        json_results[name] = {
            k: v for k, v in metrics.items()
            if k not in ("all_labels", "all_probs")
        }
    json_path = os.path.join(out_cfg["results_dir"],
                             "metrics_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"[SAVED] Metrics JSON → {json_path}")

    # ── Plots ──────────────────────────────────────────────────
    plot_roc_curves(
        all_results,
        os.path.join(out_cfg["figure_dir"], "roc_comparison.png"),
    )
    plot_confusion_matrices(
        all_results,
        os.path.join(out_cfg["figure_dir"], "confusion_matrices.png"),
    )

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
