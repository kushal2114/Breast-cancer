# Attention-Based Multimodal Breast Cancer Classification

**Bidirectional Cross-Attention Fusion of Mammography and Ultrasound Images**

A two-stream CNN/ViT + Bidirectional Cross-Attention Transformer fusion model for classifying breast lesions as **Benign** or **Malignant** using mammography (CBIS-DDSM) and ultrasound (BUSI) imaging modalities simultaneously.

This repository contains the complete machine learning pipeline (preprocessing, training, evaluation, explainability) and a full-stack clinical decision support web application (FastAPI backend + React/Vite/Tailwind CSS frontend).

---

## 📖 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Project Structure](#-project-structure)
3. [Setup & Installation](#-setup--installation)
   - [Backend Environment](#1-backend-environment)
   - [Frontend Environment](#2-frontend-environment)
4. [Dataset Preparation](#-dataset-preparation)
   - [CBIS-DDSM (Mammography)](#1-cbis-ddsm-mammography)
   - [BUSI (Ultrasound)](#2-busi-ultrasound)
5. [Training the Models](#-training-the-models)
6. [Evaluation & Metrics](#-evaluation--metrics)
7. [Model Explainability (Grad-CAM++)](#-model-explainability-grad-cam)
8. [Running the Web Application](#-running-the-web-application)
   - [Start FastAPI Backend](#1-start-fastapi-backend)
   - [Start React Frontend](#2-start-react-frontend)
   - [Using the Application](#3-using-the-application)
9. [Ablation Study Matrix & Expected Results](#-ablation-study-matrix--expected-results)
10. [Troubleshooting](#-troubleshooting)
11. [Citation & License](#-citation--license)

---

## 🏗️ Architecture Overview

The system processes mammography and ultrasound images through two parallel streams, extracting spatial features using pre-trained convolutional (EfficientNet-B0, DenseNet-121) or transformer-based (ViT-B/16) backbones. The streams are fused using a **Bidirectional Cross-Attention block**, allowing the model to attend to features in one modality based on queries from the other, followed by a joint Transformer Encoder and classification head.

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
│              CLASSIFIER: Linear→GELU→Drop→Linear → (B, 2)                │
│                                                                          │
│              OUTPUT:  Benign (0) | Malignant (1)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Token counts by backbone:**
| Backbone | N (tokens) | Feature dim → Projected dim |
|---|---|---|
| EfficientNet-B0 | 49 (7×7) | 1280 → 512 |
| DenseNet-121 | 49 (7×7) | 1024 → 512 |
| ViT-B/16 | 196 (14×14) | 768 → 512 |

---

## 📁 Project Structure

```
project_root/
├── src/                          ← Deep learning source code
│   ├── __init__.py
│   ├── dataset.py                ← Dataset loading (CBIS-DDSM + BUSI)
│   ├── preprocessing.py          ← Image transforms (CLAHE, wavelet, augmentation)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py           ← EfficientNet-B0, DenseNet-121, ViT-B/16
│   │   ├── cross_attention.py    ← Bidirectional Cross-Attention block
│   │   ├── fusion_model.py       ← Full two-stream fusion model
│   │   ├── baseline_mammo.py     ← Mammography-only baseline
│   │   ├── baseline_us.py        ← Ultrasound-only baseline
│   │   └── baseline_concat.py    ← Early concatenation baseline
│   ├── train.py                  ← Training script (all model variants)
│   ├── evaluate.py               ← Evaluation + metrics comparison
│   └── explain.py                ← Grad-CAM++ and attention visualisations
│
├── api/                          ← FastAPI inference backend
│   ├── __init__.py
│   ├── main.py                   ← FastAPI app (endpoints + startup)
│   ├── predictor.py              ← Model loading + inference logic
│   ├── explainer.py              ← Grad-CAM++ generation for API
│   └── schemas.py                ← Pydantic request/response models
│
├── frontend/                     ← React + Vite + Tailwind CSS app
│   ├── src/
│   │   ├── main.jsx              ← React entry point
│   │   ├── App.jsx               ← Main application UI shell
│   │   ├── api.js                ← API client (health check + predict)
│   │   ├── index.css             ← Stylesheet with design system
│   │   └── components/
│   │       ├── Header.jsx        ← Navigation + health status indicator
│   │       ├── UploadCard.jsx    ← Image dropzone & modality selection
│   │       ├── ResultsSection.jsx ← Model prediction scores & clinical advice
│   │       ├── ExplainabilitySection.jsx ← Interactive Grad-CAM heatmap viewer
│   │       └── Footer.jsx        ← Copy and versioning
│   ├── .env.example              ← Frontend environment variable template
│   ├── index.html                ← App HTML layout
│   ├── vite.config.js            ← Vite config
│   └── package.json              ← Frontend NPM packages
│
├── configs/
│   └── config.yaml               ← Hyperparameters & training configuration
│
├── data/                         ← Datasets (git-ignored)
│   ├── cbis_ddsm/                ← CBIS-DDSM Mammography dataset
│   └── busi/                     ← BUSI Ultrasound dataset
│
├── outputs/                      ← Training outputs (git-ignored)
│   ├── checkpoints/              ← Trained weights (.pth files)
│   ├── logs/                     ← TensorBoard and training logs
│   ├── figures/                  ← ROC curves, Grad-CAM overlays, maps
│   └── results/                  ← Evaluation metrics CSVs
│
├── requirements.txt              ← Python training dependencies
├── api_requirements.txt          ← FastAPI dependencies
├── .gitignore                    ← File to exclude large data/caches from git
└── README.md                     ← This guide
```

---

## ⚙️ Setup & Installation

### 1. Backend Environment

Ensure you have **Python 3.9–3.11** installed. Isolation with Conda or `virtualenv` is recommended.

```bash
# Create and activate environment
conda create -n breastai python=3.9 -y
conda activate breastai

# Install main deep learning dependencies
pip install -r requirements.txt

# Install API dependencies
pip install -r api_requirements.txt
```

> **Note:** If a CUDA-compatible GPU is available, PyTorch will leverage it automatically. Otherwise, it will fallback to CPU (which will make training much slower).

### 2. Frontend Environment

Ensure you have **Node.js 18+** installed.

```bash
cd frontend

# Copy local environment configuration
# On Linux/macOS:
cp .env.example .env
# On Windows:
copy .env.example .env

# Install Node dependencies
npm install
```

---

## 📊 Dataset Preparation

Create a folder named `data/` in the root of the project to place the datasets.

### 1. CBIS-DDSM (Mammography)
1. Download the dataset from [CBIS-DDSM on Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).
2. Extract the files and structure them as follows:
   ```
   data/cbis_ddsm/
   ├── csv/
   │   ├── mass_case_description_train_set.csv
   │   └── mass_case_description_test_set.csv
   └── jpeg/
       ├── <UID-named-folders>/
       │   └── <image-files>.jpg
   ```
   *Note: The dataset loader maps the image names via the UID strings (e.g., `1.3.6.1.4...`) matching `parts[2]` (or `parts[1]` depending on paths) from the CSV paths.*

### 2. BUSI (Ultrasound)
1. Download the dataset from [BUSI on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
2. Extract and structure them as follows:
   ```
   data/busi/
   ├── benign/
   │   ├── benign (1).png
   │   └── ...
   ├── malignant/
   │   ├── malignant (1).png
   │   └── ...
   └── normal/          ← Excluded from training; used only for OOD evaluation
       ├── normal (1).png
       └── ...
   ```

---

## 🚂 Training the Models

Train any of the 6 configuration setups. The training scripts read configs from `configs/config.yaml`.

```bash
# Baseline: Mammography-only
python src/train.py --model mammo  --backbone efficientnet_b0

# Baseline: Ultrasound-only
python src/train.py --model us     --backbone efficientnet_b0

# Baseline: Early concatenation
python src/train.py --model concat --backbone efficientnet_b0

# Proposed Fusion (EfficientNet-B0)
python src/train.py --model fusion --backbone efficientnet_b0

# Proposed Fusion (DenseNet-121) — Used by default in the web app
python src/train.py --model fusion --backbone densenet121

# Proposed Fusion (ViT-B/16)
python src/train.py --model fusion --backbone vit_b_16
```

### Resume Training
To resume training a model from its latest checkpoint:
```bash
python src/train.py --model fusion --backbone densenet121 --resume outputs/checkpoints/fusion_densenet121_best.pth
```

---

## 📈 Evaluation & Metrics

Evaluate all trained models across test partitions and generate final metrics:

```bash
python src/evaluate.py
```

This generates:
- `outputs/results/metrics_comparison.csv` — Full accuracy/F1/AUC comparison table.
- `outputs/figures/roc_comparison.png` — Multi-model ROC curves.
- `outputs/figures/confusion_matrices.png` — Test-set confusion matrices.
- `outputs/figures/ood_confidence_distribution.png` — Out-of-distribution confidence chart.

---

## 🔍 Model Explainability (Grad-CAM++)

Extract heatmap overlays highlighting regions of interest (lesion areas) that the model focused on:

```bash
python src/explain.py --checkpoint outputs/checkpoints/fusion_densenet121_best.pth
```

Outputs are saved under:
- `outputs/figures/gradcam_mammo/` (Mammography heatmaps)
- `outputs/figures/gradcam_us/` (Ultrasound heatmaps)
- `outputs/figures/attention_maps/` (Cross-attention token distributions)

---

## 💻 Running the Web Application

The interactive web application provides a clean GUI for clinicians to upload images, select modalities, view prediction probabilities, and overlay Grad-CAM++ explanation heatmaps.

### 1. Start FastAPI Backend

```bash
# From the project root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

To confirm the API is active, navigate to `http://localhost:8000/health` in your browser. You should see a JSON response stating `"status": "ok"`.

### 2. Start React Frontend

```bash
# In a new terminal
cd frontend
npm run dev
```

Open your browser at **`http://localhost:5173`**.

### 3. Using the Application
1. **Connection Check**: Ensure the header indicator is a green dot (meaning connection to the backend is successful).
2. **Upload Images**: Drag-and-drop or select a breast imaging file (Mammography or Ultrasound).
3. **Select Modality**: Set the toggle/radio selection matching your image type.
4. **Analyze**: Click "Analyze Image". The result card will display the classification labels, confidence distributions, clinical advice, and a toggle to overlay the Grad-CAM++ heatmap.

---

## 🧪 Ablation Study Matrix & Expected Results

| # | Model Type | Backbone | Mode | Status |
|---|---|---|---|---|
| 1 | Mammo-only | EfficientNet-B0 | Single Stream | Baseline |
| 2 | Ultrasound-only | EfficientNet-B0 | Single Stream | Baseline |
| 3 | Early Concatenation | EfficientNet-B0 | Dual Stream | Baseline |
| 4 | Bidirectional Cross-Attn | EfficientNet-B0 | Dual Stream | Proposed |
| 5 | Bidirectional Cross-Attn | DenseNet-121 | Dual Stream | Proposed |
| 6 | Bidirectional Cross-Attn | ViT-B/16 | Dual Stream | Proposed |

### Experimental Results Table

| Model | Accuracy | Sensitivity | Specificity | F1-Score | AUC |
|---|---|---|---|---|---|
| Mammo-only (EffNet) | — | — | — | — | — |
| US-only (EffNet) | — | — | — | — | — |
| Concat (EffNet) | — | — | — | — | — |
| **Fusion (EffNet)** | — | — | — | — | — |
| **Fusion (DenseNet)** | — | — | — | — | — |
| **Fusion (ViT)** | — | — | — | — | — |

*(Fill in with metrics from `outputs/results/metrics_comparison.csv` post-training)*

---

## 🛠️ Troubleshooting

| Issue | Root Cause & Resolution |
|---|---|
| **Health Dot is Red** | FastAPI server is not running or running on a port other than `8000`. Run `uvicorn` and make sure it outputs `http://127.0.0.1:8000`. |
| **CORS Errors in Browser** | Ensure `uvicorn` is run with `--host 0.0.0.0` or verify that backend CORS middleware configurations match the port of the React dev server (default `5173`). |
| **CUDA Out of Memory** | PyTorch ran out of VRAM. Reduce `batch_size` in `configs/config.yaml` or train on a CPU (will be slower). |
| **Checkpoint file not found** | Verify that `outputs/checkpoints/fusion_densenet121_best.pth` exists. Make sure you trained the model first, or place pre-trained checkpoints in the folder. |
| **ModuleNotFoundError** | Ensure you run the python commands from the project root directory, and verify you are in the correct activated conda environment (`breastai`). |

---

## 📄 Citation & License

### Citation
```bibtex
@article{breastai2025multimodal,
  title={Attention-Based Multimodal Breast Cancer Classification Using Bidirectional Cross-Attention Fusion of Mammography and Ultrasound Images},
  author={Your Name},
  journal={Academic Research Repository},
  year={2025}
}
```

### License
This project is licensed for academic, educational, and research purposes.
