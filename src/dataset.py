"""
dataset.py — Dataset classes and data-loading utilities.

Implements:
    - load_cbis():  Load CBIS-DDSM mass mammography data via CSV → JPEG bridge.
    - load_busi():  Load BUSI ultrasound data (benign/malignant only).
    - MultiModalBreastDataset:  Class-level pseudo-paired multimodal dataset.
    - MammographyDataset:       Single-modality mammography dataset.
    - UltrasoundDataset:        Single-modality ultrasound dataset.
    - build_dataloaders():      One-stop builder for all splits and modalities.
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.preprocessing import get_mammography_transforms, get_ultrasound_transforms


# ================================================================
#  CBIS-DDSM Loader
# ================================================================

def load_cbis(csv_path: str, jpeg_root: str) -> pd.DataFrame:
    """Load CBIS-DDSM mass mammography dataset via CSV → JPEG bridge.

    The CSV ``image file path`` column has the format:
        ``Mass-Training_P_00001_LEFT_CC / UID_study / UID_series / 000000.dcm``

    Split by ``/`` → ``parts[2]`` is the SeriesInstanceUID folder that
    actually exists inside ``jpeg_root``.  ``parts[0]`` is a human-readable
    prefix and ``parts[1]`` is a StudyInstanceUID — neither exists on disk.

    Inside the SeriesInstanceUID folder there is typically a single ``.jpg``
    file (the full mammogram).  When a folder contains more than one file
    we select the **largest** file (the full mammogram) and skip smaller
    files (ROI masks).

    Args:
        csv_path:  Absolute path to the mass case description CSV.
        jpeg_root: Absolute path to the ``jpeg/`` directory.

    Returns:
        DataFrame with columns ``[image_path, label, patient_id]``.
    """
    df = pd.read_csv(csv_path)
    records: List[Dict] = []

    for _, row in df.iterrows():
        filepath = str(row["image file path"])
        parts = filepath.split("/")

        # parts[2] = SeriesInstanceUID → real folder inside jpeg_root
        uid_folder = parts[2]
        top_dir = Path(jpeg_root) / uid_folder

        if not top_dir.exists():
            continue

        # Find all .jpg files in the UID folder (flat structure)
        jpg_files = list(top_dir.glob("*.jpg"))
        if not jpg_files:
            continue

        # Select the largest file (full mammogram)
        largest_file = max(jpg_files, key=lambda f: f.stat().st_size)

        # Map pathology label
        pathology = str(row["pathology"]).upper()
        if "MALIGNANT" in pathology:
            label = 1
        elif "BENIGN" in pathology:
            label = 0
        else:
            continue  # skip unknown

        patient_id = str(row["patient_id"])

        records.append({
            "image_path": str(largest_file),
            "label": label,
            "patient_id": patient_id,
        })

    result = pd.DataFrame(records)
    # De-duplicate by image_path (same image may appear from multiple rows)
    result = result.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return result


# ================================================================
#  BUSI Loader
# ================================================================

def load_busi(busi_root: str,
              include_normal: bool = False) -> pd.DataFrame:
    """Load BUSI ultrasound dataset (benign / malignant).

    By default the ``normal/`` class is **excluded** from training
    and reserved exclusively for OOD robustness evaluation.

    Args:
        busi_root:      Path to ``data/busi/``.
        include_normal: If True, also load the normal class (label=-1).

    Returns:
        DataFrame with columns ``[image_path, label]``.
    """
    records: List[Dict] = []
    class_dirs = {"benign": 0, "malignant": 1}
    if include_normal:
        class_dirs["normal"] = -1  # sentinel for OOD

    for class_name, label in class_dirs.items():
        class_dir = Path(busi_root) / class_name
        if not class_dir.exists():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                # Skip mask files (files with '_mask' in name)
                if "_mask" in img_file.stem.lower():
                    continue
                records.append({
                    "image_path": str(img_file),
                    "label": label,
                })

    return pd.DataFrame(records)


# ================================================================
#  Splitting utilities
# ================================================================

def patient_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split CBIS DataFrame using patient-level stratified grouping.

    Ensures that all images from the same patient stay in the same split,
    while approximately preserving the class balance across splits.

    Args:
        df:          DataFrame with ``patient_id`` and ``label`` columns.
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation.
        seed:        Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) DataFrames.
    """
    # First split: train vs (val + test)
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=1.0 - train_ratio,
        random_state=seed,
    )
    groups = df["patient_id"].values
    labels = df["label"].values
    train_idx, temp_idx = next(gss1.split(df, labels, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: val vs test (from the remaining data)
    relative_val = val_ratio / (1.0 - train_ratio)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=1.0 - relative_val,
        random_state=seed,
    )
    temp_groups = temp_df["patient_id"].values
    temp_labels = temp_df["label"].values
    val_idx, test_idx = next(gss2.split(temp_df, temp_labels, temp_groups))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def image_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame using image-level stratification on ``label``.

    Args:
        df:          DataFrame with ``label`` column.
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation.
        seed:        Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) DataFrames.
    """
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - train_ratio,
        random_state=seed,
    )
    labels = df["label"].values
    train_idx, temp_idx = next(sss1.split(df, labels))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    relative_val = val_ratio / (1.0 - train_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - relative_val,
        random_state=seed,
    )
    temp_labels = temp_df["label"].values
    val_idx, test_idx = next(sss2.split(temp_df, temp_labels))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Args:
        labels: 1-D array of integer labels.

    Returns:
        Float tensor of shape ``(num_classes,)`` with class weights.
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    weight_tensor = torch.zeros(max(classes) + 1, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        weight_tensor[cls] = w
    return weight_tensor


# ================================================================
#  Dataset Classes
# ================================================================

class MammographyDataset(Dataset):
    """Single-modality mammography dataset.

    Args:
        dataframe: DataFrame with ``image_path`` and ``label`` columns.
        transform: Preprocessing callable (PIL Image → Tensor).
    """

    def __init__(self, dataframe: pd.DataFrame,
                 transform: Optional[Callable] = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class UltrasoundDataset(Dataset):
    """Single-modality ultrasound dataset.

    Args:
        dataframe: DataFrame with ``image_path`` and ``label`` columns.
        transform: Preprocessing callable (PIL Image → Tensor).
    """

    def __init__(self, dataframe: pd.DataFrame,
                 transform: Optional[Callable] = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MultiModalBreastDataset(Dataset):
    """Class-level pseudo-paired multimodal dataset.

    For each ``__getitem__`` call, returns one mammography image and
    one **randomly sampled** ultrasound image of the **same class**.
    This implements the standard class-level pairing strategy for
    non-patient-matched multimodal datasets.

    The dataset length equals the mammography set length.  The
    ultrasound partner is re-sampled randomly on every access,
    providing natural augmentation.

    Args:
        mammo_df:       Mammography DataFrame (image_path, label).
        us_df:          Ultrasound DataFrame (image_path, label).
        mammo_transform: Transform for mammography images.
        us_transform:    Transform for ultrasound images.
    """

    def __init__(
        self,
        mammo_df: pd.DataFrame,
        us_df: pd.DataFrame,
        mammo_transform: Optional[Callable] = None,
        us_transform: Optional[Callable] = None,
    ):
        self.mammo_df = mammo_df.reset_index(drop=True)
        self.mammo_transform = mammo_transform
        self.us_transform = us_transform

        # Build per-class index lists for ultrasound random pairing
        self.us_by_class: Dict[int, List[int]] = {}
        for idx, row in us_df.iterrows():
            label = int(row["label"])
            self.us_by_class.setdefault(label, []).append(idx)
        self.us_df = us_df.reset_index(drop=True)

        # Re-index after reset
        self.us_by_class = {}
        for idx in range(len(self.us_df)):
            label = int(self.us_df.iloc[idx]["label"])
            self.us_by_class.setdefault(label, []).append(idx)

    def __len__(self) -> int:
        return len(self.mammo_df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (mammo_tensor, us_tensor, label).

        The ultrasound image is randomly sampled from all ultrasound
        images sharing the same class label as the mammography image.
        """
        mammo_row = self.mammo_df.iloc[idx]
        label = int(mammo_row["label"])

        # Load mammography image
        mammo_img = Image.open(mammo_row["image_path"]).convert("RGB")
        if self.mammo_transform is not None:
            mammo_img = self.mammo_transform(mammo_img)

        # Randomly sample an ultrasound image of the same class
        us_candidates = self.us_by_class[label]
        us_idx = random.choice(us_candidates)
        us_row = self.us_df.iloc[us_idx]
        us_img = Image.open(us_row["image_path"]).convert("RGB")
        if self.us_transform is not None:
            us_img = self.us_transform(us_img)

        return mammo_img, us_img, label


# ================================================================
#  Dataloader Builder
# ================================================================

def build_dataloaders(
    config: dict,
    model_type: str = "fusion",
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for the specified model type.

    Args:
        config:     Parsed config.yaml as a nested dict.
        model_type: One of ``"fusion"``, ``"mammo"``, ``"us"``, ``"concat"``.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        DataLoaders, plus ``"class_weights"`` (Tensor) and
        ``"mammo_test_df"`` / ``"us_test_df"`` for evaluation.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    seed = train_cfg["seed"]
    img_size = data_cfg["image_size"]
    batch_size = train_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]

    # ── Load raw data ──────────────────────────────────────────
    mammo_train_df = load_cbis(
        data_cfg["mammo_csv_train"],
        data_cfg["mammo_jpeg_root"],
    )
    mammo_test_df = load_cbis(
        data_cfg["mammo_csv_test"],
        data_cfg["mammo_jpeg_root"],
    )

    us_df = load_busi(data_cfg["us_dir"], include_normal=False)

    print(f"[DATA] CBIS-DDSM train CSV entries -> {len(mammo_train_df)} images resolved")
    print(f"[DATA] CBIS-DDSM test  CSV entries -> {len(mammo_test_df)} images resolved")
    print(f"[DATA] BUSI (benign+malignant)     -> {len(us_df)} images")

    # ── Split mammography (patient-level) ──────────────────────
    mammo_tr, mammo_val, _ = patient_stratified_split(
        mammo_train_df,
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
        seed=seed,
    )
    # The provided test CSV is the held-out test set
    mammo_te = mammo_test_df

    print(f"[SPLIT] Mammo  - train: {len(mammo_tr)}, val: {len(mammo_val)}, "
          f"test: {len(mammo_te)}")

    # ── Split ultrasound (image-level stratified) ──────────────
    us_tr, us_val, us_te = image_stratified_split(
        us_df,
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
        seed=seed,
    )

    print(f"[SPLIT] US     - train: {len(us_tr)}, val: {len(us_val)}, "
          f"test: {len(us_te)}")

    # ── Transforms ─────────────────────────────────────────────
    mammo_train_tf = get_mammography_transforms(img_size, is_training=True)
    mammo_eval_tf = get_mammography_transforms(img_size, is_training=False)
    us_train_tf = get_ultrasound_transforms(img_size, is_training=True)
    us_eval_tf = get_ultrasound_transforms(img_size, is_training=False)

    # ── Class weights (from training labels) ───────────────────
    if model_type in ("fusion", "concat"):
        all_train_labels = np.concatenate([
            mammo_tr["label"].values, us_tr["label"].values
        ])
    elif model_type == "mammo":
        all_train_labels = mammo_tr["label"].values
    else:  # "us"
        all_train_labels = us_tr["label"].values
    class_weights = compute_class_weights(all_train_labels)

    # ── Build datasets ─────────────────────────────────────────
    if model_type in ("fusion", "concat"):
        train_ds = MultiModalBreastDataset(
            mammo_tr, us_tr, mammo_train_tf, us_train_tf)
        val_ds = MultiModalBreastDataset(
            mammo_val, us_val, mammo_eval_tf, us_eval_tf)
        test_ds = MultiModalBreastDataset(
            mammo_te, us_te, mammo_eval_tf, us_eval_tf)
    elif model_type == "mammo":
        train_ds = MammographyDataset(mammo_tr, mammo_train_tf)
        val_ds = MammographyDataset(mammo_val, mammo_eval_tf)
        test_ds = MammographyDataset(mammo_te, mammo_eval_tf)
    else:  # "us"
        train_ds = UltrasoundDataset(us_tr, us_train_tf)
        val_ds = UltrasoundDataset(us_val, us_eval_tf)
        test_ds = UltrasoundDataset(us_te, us_eval_tf)

    # ── DataLoaders ────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "mammo_test_df": mammo_te,
        "us_test_df": us_te,
    }
