# Attention-Based Multimodal Breast Cancer Classification

**Bidirectional Cross-Attention Fusion of Mammography and Ultrasound Images**

A two-stream CNN/ViT + Bidirectional Cross-Attention Transformer fusion model
for classifying breast lesions as **Benign** or **Malignant** using mammography
(CBIS-DDSM) and ultrasound (BUSI) imaging modalities simultaneously.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION ARCHITECTURE                       │
├────────────────────────────┬─────────────────────────────────────────────┤
│                            │                                             │
│   MAMMOGRAPHY STREAM       │       ULTRASOUND STREAM                     │
│                            │                                             │
│   ┌──────────────────┐     │     ┌──────────────────┐                    │
│   │  JPEG Image      │     │     │  PNG Image        │                   │
│   │  (CBIS-DDSM)     │     │     │  (BUSI)           │                   │
│   └────────┬─────────┘     │     └────────┬──────────┘                   │
│            ▼               │              ▼                              │
│   ┌──────────────────┐     │     ┌──────────────────┐                    │
│   │  CLAHE + Resize  │     │     │ Wavelet Denoise  │                    │
│   │  → 3ch × 224²    │     │     │ (db2) → 3ch×224² │                    │
│   └────────┬─────────┘     │     └────────┬──────────┘                   │
│            ▼               │              ▼                              │
│   ┌──────────────────┐     │     ┌──────────────────┐                    │
│   │  ENCODER          │     │     │  ENCODER          │                   │
│   │  ┌──────────────┐ │     │     │  ┌──────────────┐ │                   │
│   │  │EfficientNetB0│ │     │     │  │EfficientNetB0│ │                   │
│   │  │  DenseNet121 │ │     │     │  │  DenseNet121 │ │                   │
│   │  │   ViT-B/16   │ │     │     │  │   ViT-B/16   │ │                   │
│   │  └──────────────┘ │     │     │  └──────────────┘ │                   │
│   │  → (B, N, 512)    │     │     │  → (B, N, 512)    │                   │
│   └────────┬──────────┘     │     └────────┬──────────┘                   │
│            │               │              │                              │
├────────────┴───────────────┴──────────────┴──────────────────────────────┤
│                                                                          │
│              BIDIRECTIONAL CROSS-ATTENTION FUSION                         │
│                                                                          │
│    Mammo queries US:  Q=mammo, K/V=US  → attended_mammo (B, N_m, 512)   │
│    US queries Mammo:  Q=US, K/V=mammo  → attended_us    (B, N_u, 512)   │
│    Residual + LayerNorm                                                  │
│    Concatenate → (B, N_m + N_u, 512)                                    │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│              TRANSFORMER ENCODER (2 layers, 8 heads)                     │
│              → (B, N_m + N_u, 512)                                       │
│                                                                          │
│              GLOBAL AVERAGE POOLING → (B, 512)                           │
│                                                                          │
│              CLASSIFIER: Linear→GELU→Drop→Linear → (B, 2)              │
│                                                                          │
│              OUTPUT:  Benign (0) │ Malignant (1)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Token counts by backbone:**
| Backbone | N (tokens) | Feature dim → Projected dim |
|---|---|---|
| EfficientNet-B0 | 49 (7×7) | 1280 → 512 |
| DenseNet-121 | 49 (7×7) | 1024 → 512 |
| ViT-B/16 | 196 (14×14) | 768 → 512 |

---

## Setup

### 1. Create Environment

```bash
conda create -n breast-cancer python=3.10 -y
conda activate breast-cancer
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### CBIS-DDSM (Mammography)

1. Download the CBIS-DDSM dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).
2. Place the data in `data/cbis_ddsm/`:

```
data/cbis_ddsm/
├── csv/
│   ├── mass_case_description_train_set.csv
│   ├── mass_case_description_test_set.csv
│   └── dicom_info.csv  (optional, for reference)
└── jpeg/
    ├── 1.3.6.1.4.1.9590.100.1.2.XXXXXXXXX/   ← UID folder
    │   ├── 1-263.jpg                           ← full mammogram
    │   └── 2-241.jpg                           ← ROI mask (smaller, ignored)
    └── ...
```

> **Important:** The `jpeg/` folder contains UID-named folders (long numeric strings),
> NOT human-readable names like `Mass-Training_P_00001_LEFT_CC`.
> The CSV `image file path` column bridges human-readable names to UIDs:
> - Format: `HumanName/UID_study/UID_series/000000.dcm`
> - Split by `/` → `parts[2]` is the SeriesInstanceUID folder in `jpeg/`
> - The code automatically selects the largest `.jpg` file (full mammogram)
>   and ignores smaller files (ROI masks).

#### BUSI (Ultrasound)

1. Download the BUSI dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
2. Place in `data/busi/`:

```
data/busi/
├── benign/       (images used for training/evaluation)
├── malignant/    (images used for training/evaluation)
└── normal/       (EXCLUDED from training — used only for OOD evaluation)
```

---

## Training

All models are trained via a single unified script with CLI arguments:

```bash
# ── Proposed Model (3 backbone variants) ──────────────────
python src/train.py --model fusion  --backbone efficientnet_b0
python src/train.py --model fusion  --backbone densenet121
python src/train.py --model fusion  --backbone vit_b_16

# ── Baseline Models ───────────────────────────────────────
python src/train.py --model mammo   --backbone efficientnet_b0
python src/train.py --model us      --backbone efficientnet_b0
python src/train.py --model concat  --backbone efficientnet_b0

# ── Resume from Checkpoint ────────────────────────────────
python src/train.py --model fusion --backbone efficientnet_b0 \
    --resume outputs/checkpoints/fusion_efficientnet_b0_best.pth
```

**Key training settings** (configurable via `configs/config.yaml`):
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-6)
- Loss: CrossEntropyLoss with class weights
- Early stopping: patience=10 on validation AUC
- Mixed precision: enabled by default on GPU

---

## Evaluation

Evaluate all trained models on the held-out test set:

```bash
# Evaluate all checkpoints automatically
python src/evaluate.py

# Evaluate a single checkpoint
python src/evaluate.py --checkpoint outputs/checkpoints/fusion_efficientnet_b0_best.pth \
    --model fusion --backbone efficientnet_b0
```

**Outputs:**
- `outputs/results/metrics_comparison.csv` — Side-by-side comparison table
- `outputs/figures/roc_comparison.png` — All ROC curves overlaid
- `outputs/figures/confusion_matrices.png` — Confusion matrix grid
- `outputs/figures/ood_confidence_distribution.png` — OOD robustness plot

---

## Explainability

Generate Grad-CAM++ overlays and cross-attention visualisations:

```bash
python src/explain.py --checkpoint outputs/checkpoints/fusion_efficientnet_b0_best.pth
```

> **Note:** For ViT backbone, Grad-CAM++ is skipped automatically.
> Only cross-attention maps are generated.

**Outputs:**
- `outputs/figures/gradcam_mammo/` — Grad-CAM++ on mammography stream
- `outputs/figures/gradcam_us/` — Grad-CAM++ on ultrasound stream
- `outputs/figures/attention_maps/` — Cross-attention + combined figures

---

## Project Structure

```
├── data/                          ← Datasets (user-provided)
├── src/
│   ├── dataset.py                 ← Dataset and DataLoader classes
│   ├── preprocessing.py           ← Modality-specific transforms
│   ├── models/
│   │   ├── encoders.py            ← Pluggable encoder backbones
│   │   ├── cross_attention.py     ← Bidirectional Cross-Attention block
│   │   ├── fusion_model.py        ← Full two-stream fusion model
│   │   ├── baseline_mammo.py      ← Mammography-only baseline
│   │   ├── baseline_us.py         ← Ultrasound-only baseline
│   │   └── baseline_concat.py     ← Simple concatenation baseline
│   ├── train.py                   ← Unified training script
│   ├── evaluate.py                ← Full evaluation + metrics
│   └── explain.py                 ← Grad-CAM++ + attention maps
├── notebooks/
│   └── results_analysis.ipynb     ← Publication-ready figures
├── outputs/
│   ├── checkpoints/               ← Saved model weights
│   ├── logs/                      ← Training logs (CSV + TensorBoard)
│   ├── figures/                   ← Visualisations
│   └── results/                   ← Metrics JSON and CSV
├── configs/
│   └── config.yaml                ← All hyperparameters
├── requirements.txt
└── README.md
```

---

## Ablation Study Matrix

| # | Model | Backbone | Notes |
|---|---|---|---|
| 1 | Mammo-only | EfficientNet-B0 | Baseline 1 |
| 2 | US-only | EfficientNet-B0 | Baseline 2 |
| 3 | Concat Fusion | EfficientNet-B0 | Baseline 3 (no attention) |
| 4 | Cross-Attn Fusion | EfficientNet-B0 | **Proposed (variant A)** |
| 5 | Cross-Attn Fusion | DenseNet-121 | **Proposed (variant B)** |
| 6 | Cross-Attn Fusion | ViT-B/16 | **Proposed (variant C)** |

---

## Expected Results

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| Mammo-only (EffNet) | — | — | — | — | — |
| US-only (EffNet) | — | — | — | — | — |
| Concat (EffNet) | — | — | — | — | — |
| **Fusion (EffNet)** | — | — | — | — | — |
| **Fusion (DenseNet)** | — | — | — | — | — |
| **Fusion (ViT)** | — | — | — | — | — |

*(Fill in after training)*

---

## Citation

```bibtex
@article{yourname2025multimodal,
  title={Attention-Based Multimodal Breast Cancer Classification Using
         Bidirectional Cross-Attention Fusion of Mammography and
         Ultrasound Images},
  author={Your Name},
  journal={},
  year={2025}
}
```

---

## License

This project is for academic and research purposes.
