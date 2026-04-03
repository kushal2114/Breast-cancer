"""
train.py — Unified training script for all model configurations.

Supports four model types and three backbone options via CLI:
    python src/train.py --model fusion  --backbone efficientnet_b0
    python src/train.py --model mammo   --backbone efficientnet_b0
    python src/train.py --model us      --backbone efficientnet_b0
    python src/train.py --model concat  --backbone efficientnet_b0
    python src/train.py --model fusion  --backbone densenet121
    python src/train.py --model fusion  --backbone vit_b_16
    python src/train.py --model fusion  --backbone efficientnet_b0 --resume outputs/checkpoints/fusion_efficientnet_b0_best.pth

Features:
    - AdamW optimiser with CosineAnnealingLR scheduler
    - Weighted CrossEntropy loss for class imbalance
    - Early stopping on validation AUC (patience=10)
    - Mixed-precision training (torch.amp)
    - CSV and TensorBoard logging
    - Fully resumable from checkpoint
    - Sanity checks at startup (paths, shapes, forward pass)
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataloaders
from src.models.fusion_model import CrossAttentionFusionModel
from src.models.baseline_mammo import MammographyOnlyModel
from src.models.baseline_us import UltrasoundOnlyModel
from src.models.baseline_concat import ConcatFusionModel


# ════════════════════════════════════════════════════════════════
#  Reproducibility
# ════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════
#  Model Factory
# ════════════════════════════════════════════════════════════════

def build_model(model_type: str, config: dict,
                backbone: str) -> nn.Module:
    """Build model by type with the specified backbone.

    Args:
        model_type: One of ``"fusion"``, ``"mammo"``, ``"us"``, ``"concat"``.
        config:     Parsed config.yaml dict.
        backbone:   Backbone name to override config.

    Returns:
        Initialised model (nn.Module).
    """
    # Override backbone in config
    config["model"]["encoder_backbone"] = backbone

    if model_type == "fusion":
        return CrossAttentionFusionModel.from_config(config)
    elif model_type == "mammo":
        return MammographyOnlyModel.from_config(config)
    elif model_type == "us":
        return UltrasoundOnlyModel.from_config(config)
    elif model_type == "concat":
        return ConcatFusionModel.from_config(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ════════════════════════════════════════════════════════════════
#  Sanity Checks
# ════════════════════════════════════════════════════════════════

def sanity_check(model: nn.Module, model_type: str,
                 device: torch.device, config: dict) -> None:
    """Run pre-training sanity checks.

    Verifies:
        1. All dataset paths exist.
        2. A dummy forward pass succeeds with correct output shape.
        3. Prints model parameter count and tensor shapes.
    """
    data_cfg = config["data"]
    print("\n" + "═" * 60)
    print("  SANITY CHECKS")
    print("═" * 60)

    # 1. Verify dataset paths
    paths_to_check = [
        data_cfg["mammo_csv_train"],
        data_cfg["mammo_csv_test"],
        data_cfg["mammo_jpeg_root"],
        data_cfg["us_dir"],
    ]
    for p in paths_to_check:
        exists = os.path.exists(p)
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}] {p}")
        if not exists:
            raise FileNotFoundError(f"Dataset path not found: {p}")

    # 2. Dummy forward pass
    img_size = data_cfg["image_size"]
    dummy_batch = torch.randn(2, 3, img_size, img_size, device=device)

    model.eval()
    with torch.no_grad():
        if model_type in ("fusion", "concat"):
            output = model(dummy_batch, dummy_batch)
        else:
            output = model(dummy_batch)
    print(f"\n  Dummy forward pass: input (2, 3, {img_size}, {img_size})"
          f" → output {tuple(output.shape)}")
    assert output.shape == (2, 2), (
        f"Expected output shape (2, 2), got {output.shape}")
    print("  [✓] Output shape correct: (B, 2)")

    # 3. Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    model_type: str,
    use_amp: bool = True,
) -> float:
    """Train for one epoch, returning the average loss.

    Args:
        model:      The model to train.
        loader:     Training DataLoader.
        criterion:  Loss function.
        optimizer:  Optimiser.
        scaler:     GradScaler for mixed precision.
        device:     Compute device.
        model_type: Model type string for dispatch.
        use_amp:    Whether to use automatic mixed precision.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if model_type in ("fusion", "concat"):
            mammo, us, labels = batch
            mammo = mammo.to(device, non_blocking=True)
            us = us.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if model_type == "fusion":
                logits = model(mammo, us, modality_dropout=True)
            elif model_type == "concat":
                logits = model(mammo, us)
            else:
                logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    use_amp: bool = True,
) -> dict:
    """Validate on a data split, returning loss, accuracy, and AUC.

    Returns:
        Dict with keys ``"loss"``, ``"accuracy"``, ``"auc"``.
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        if model_type in ("fusion", "concat"):
            mammo, us, labels = batch
            mammo = mammo.to(device, non_blocking=True)
            us = us.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if model_type == "fusion":
                logits = model(mammo, us, modality_dropout=False)
            elif model_type == "concat":
                logits = model(mammo, us)
            else:
                logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    avg_loss = running_loss / max(n_batches, 1)
    preds = (all_probs >= 0.5).astype(int)
    accuracy = (preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # single-class edge case

    return {"loss": avg_loss, "accuracy": accuracy, "auc": auc}


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    """Entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train multimodal breast cancer classification models.")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["fusion", "mammo", "us", "concat"],
        help="Model type to train.")
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=["efficientnet_b0", "densenet121", "vit_b_16"],
        help="Encoder backbone (overrides config.yaml).")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml.")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override backbone from CLI if provided
    backbone = args.backbone or config["model"]["encoder_backbone"]
    config["model"]["encoder_backbone"] = backbone

    train_cfg = config["training"]
    out_cfg = config["output"]

    # ── Seed & device ──────────────────────────────────────────
    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")
    if device.type == "cuda":
        print(f"         GPU: {torch.cuda.get_device_name(0)}")

    # ── Create output directories ──────────────────────────────
    for d in [out_cfg["checkpoint_dir"], out_cfg["log_dir"],
              out_cfg["figure_dir"], out_cfg["results_dir"]]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(out_cfg["log_dir"], "tensorboard"),
                exist_ok=True)

    # ── Build model ────────────────────────────────────────────
    model = build_model(args.model, config, backbone)
    model = model.to(device)

    # ── Sanity checks ──────────────────────────────────────────
    sanity_check(model, args.model, device, config)

    # ── Build dataloaders ──────────────────────────────────────
    data = build_dataloaders(config, model_type=args.model)
    train_loader = data["train"]
    val_loader = data["val"]
    class_weights = data["class_weights"].to(device)

    # ── Loss, optimiser, scheduler ─────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=train_cfg["lr_min"],
    )
    scaler = torch.amp.GradScaler(enabled=train_cfg["mixed_precision"])

    # ── Resume from checkpoint ─────────────────────────────────
    start_epoch = 0
    best_val_auc = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\n[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        print(f"         Resuming from epoch {start_epoch}, "
              f"best val AUC = {best_val_auc:.4f}")

    # ── Logging setup ──────────────────────────────────────────
    run_name = f"{args.model}_{backbone}"
    log_csv_path = os.path.join(out_cfg["log_dir"],
                                f"{run_name}_log.csv")
    tb_dir = os.path.join(out_cfg["log_dir"], "tensorboard", run_name)
    writer = SummaryWriter(log_dir=tb_dir)

    # Initialise or append to CSV log
    write_header = (not os.path.exists(log_csv_path)) or (start_epoch == 0)
    log_file = open(log_csv_path, "a" if start_epoch > 0 else "w",
                    newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "epoch", "train_loss", "val_loss", "val_accuracy",
            "val_auc", "lr", "time_sec",
        ])

    # ── Checkpoint path ────────────────────────────────────────
    ckpt_path = os.path.join(out_cfg["checkpoint_dir"],
                             f"{run_name}_best.pth")

    # ── Training loop ──────────────────────────────────────────
    patience_counter = 0
    use_amp = train_cfg["mixed_precision"] and device.type == "cuda"

    print(f"\n{'═' * 60}")
    print(f"  TRAINING: {run_name}")
    print(f"  Epochs: {train_cfg['epochs']}, Batch size: {train_cfg['batch_size']}")
    print(f"  LR: {train_cfg['learning_rate']}, AMP: {use_amp}")
    print(f"{'═' * 60}\n")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.model, use_amp,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args.model, use_amp,
        )

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - epoch_start

        # ── Logging ────────────────────────────────────────────
        print(
            f"Epoch [{epoch + 1:3d}/{train_cfg['epochs']}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        log_writer.writerow([
            epoch + 1, f"{train_loss:.6f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['accuracy']:.6f}",
            f"{val_metrics['auc']:.6f}",
            f"{current_lr:.8f}",
            f"{elapsed:.2f}",
        ])
        log_file.flush()

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch + 1)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch + 1)
        writer.add_scalar("AUC/val", val_metrics["auc"], epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)

        # ── Checkpoint (best AUC) ──────────────────────────────
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "model_type": args.model,
                "backbone": backbone,
                "config": config,
            }, ckpt_path)
            print(f"         ★ New best AUC: {best_val_auc:.4f} → saved {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"\n[EARLY STOP] No improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs. "
                      f"Best AUC: {best_val_auc:.4f}")
                break

    # ── Cleanup ────────────────────────────────────────────────
    log_file.close()
    writer.close()

    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE: {run_name}")
    print(f"  Best validation AUC: {best_val_auc:.4f}")
    print(f"  Checkpoint saved to: {ckpt_path}")
    print(f"  Log saved to: {log_csv_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
