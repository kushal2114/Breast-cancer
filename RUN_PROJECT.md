# How to Run the Full BreastAI Classifier Project

A complete, beginner-friendly guide to setting up, training, evaluating, and deploying the **Bidirectional Cross-Attention Fusion** breast cancer classification system.

---

## 0. Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.9+ | 3.9–3.11 recommended |
| **Node.js** | 18+ | For the React frontend |
| **CUDA GPU** | Optional | RTX 3060+ recommended; CPU works but is much slower |
| **Conda or virtualenv** | Any | For Python environment isolation |
| **Git** | Any | For cloning the repository |

---

## 1. Project Structure Overview

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
│   │   ├── main.jsx              ← Entry point
│   │   ├── App.jsx               ← Main application shell
│   │   ├── api.js                ← API client (health check + predict)
│   │   ├── index.css             ← Design system + Tailwind config
│   │   └── components/
│   │       ├── Header.jsx        ← Branding + health indicator
│   │       ├── UploadCard.jsx    ← Drag-and-drop upload + modality selection
│   │       ├── ResultsSection.jsx ← Classification + clinical info cards
│   │       ├── ExplainabilitySection.jsx ← Collapsible Grad-CAM viewer
│   │       └── Footer.jsx        ← Attribution footer
│   ├── .env.example              ← Environment variable template
│   ├── index.html                ← HTML shell with Google Fonts
│   ├── vite.config.js            ← Vite + Tailwind plugin config
│   └── package.json
│
├── configs/
│   └── config.yaml               ← Model + training hyperparameters
│
├── data/                         ← Datasets (not in git)
│   ├── cbis_ddsm/                ← CBIS-DDSM mammography dataset
│   └── busi/                     ← BUSI ultrasound dataset
│
├── outputs/                      ← Training outputs (not in git)
│   ├── checkpoints/              ← Model weights (.pth files)
│   ├── logs/                     ← TensorBoard logs
│   ├── figures/                  ← Visualisation outputs
│   └── results/                  ← Metrics CSVs
│
├── requirements.txt              ← Main Python dependencies
├── api_requirements.txt          ← API-specific Python dependencies
└── RUN_PROJECT.md                ← This file
```

---

## 2. Set Up the Python Environment

```bash
# Create a fresh Conda environment
conda create -n breastai python=3.9 -y
conda activate breastai

# Install main dependencies (includes PyTorch with CUDA support)
pip install -r requirements.txt

# Install API-specific dependencies
pip install -r api_requirements.txt
```

> **Note:** If you don't have a CUDA GPU, PyTorch will automatically use CPU. Training will be significantly slower (12–24 hours per model vs 2–3 hours on GPU).

---

## 3. Prepare the Datasets

### 3.1 CBIS-DDSM (Mammography)

**Download:** [CBIS-DDSM on Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

Place the data so the directory structure looks like:

```
data/cbis_ddsm/
├── csv/
│   ├── mass_case_description_train_set.csv
│   └── mass_case_description_test_set.csv
└── jpeg/
    ├── <UID-named-folders>/
    │   └── <image-files>.jpg
    └── ...
```

> **Important:** The `jpeg/` directory contains UID-named folders (e.g., `1.3.6.1.4...`), not human-readable names. The CSV files bridge labels to images via the image file path column — the dataset loader uses `parts[1]` of that path.

### 3.2 BUSI (Ultrasound)

**Download:** [BUSI on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

Place the data so the directory structure looks like:

```
data/busi/
├── benign/
│   ├── benign (1).png
│   └── ...
├── malignant/
│   ├── malignant (1).png
│   └── ...
└── normal/          ← NOT used in training; OOD evaluation only
    ├── normal (1).png
    └── ...
```

> **Note:** The `normal/` class is excluded from training and used only for out-of-distribution (OOD) evaluation in `evaluate.py`.

---

## 4. Train the Models

Train all 6 ablation configurations for the full experimental comparison:

```bash
# Baseline: Mammography-only
python src/train.py --model mammo  --backbone efficientnet_b0

# Baseline: Ultrasound-only
python src/train.py --model us     --backbone efficientnet_b0

# Baseline: Early concatenation
python src/train.py --model concat --backbone efficientnet_b0

# Fusion: EfficientNet-B0
python src/train.py --model fusion --backbone efficientnet_b0

# Fusion: DenseNet-121 ← Used by the web app
python src/train.py --model fusion --backbone densenet121

# Fusion: ViT-B/16
python src/train.py --model fusion --backbone vit_b_16
```

> **💡 Tip:** The web app serves `fusion_densenet121_best.pth`. Train this one first if you want to test the app quickly:
> ```bash
> python src/train.py --model fusion --backbone densenet121
> ```

### Resume from Checkpoint

If training is interrupted, resume from the last checkpoint:

```bash
python src/train.py --model fusion --backbone densenet121 \
    --resume outputs/checkpoints/fusion_densenet121_best.pth
```

### Expected Training Times

| Hardware | Time per Model |
|---|---|
| GPU (RTX 3060+) | ~2–3 hours |
| CPU only | ~12–24 hours (not recommended) |

---

## 5. Evaluate All Models

Run the full evaluation suite across all trained models:

```bash
python src/evaluate.py
```

**Outputs:**

| File | Description |
|---|---|
| `outputs/results/metrics_comparison.csv` | Accuracy, AUC, F1 for all models |
| `outputs/figures/roc_comparison.png` | ROC curves comparison |
| `outputs/figures/confusion_matrices.png` | Confusion matrices grid |
| `outputs/figures/ood_confidence_distribution.png` | OOD detection confidence |

---

## 6. Generate Explainability Figures

```bash
python src/explain.py --checkpoint outputs/checkpoints/fusion_densenet121_best.pth
```

**Outputs:**

| Directory | Contents |
|---|---|
| `outputs/figures/gradcam_mammo/` | Grad-CAM++ overlays on mammography images |
| `outputs/figures/gradcam_us/` | Grad-CAM++ overlays on ultrasound images |
| `outputs/figures/attention_maps/` | Cross-attention weight visualisations |

> **Note:** For ViT backbone, Grad-CAM++ is skipped and only attention maps are produced. CNN backbones (EfficientNet-B0, DenseNet-121) produce both.

---

## 7. Start the FastAPI Backend

```bash
conda activate breastai
cd project_root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Verify It's Running

Open **http://localhost:8000/health** in your browser. You should see:

```json
{
    "status": "ok",
    "model": "fusion_densenet121",
    "backbone": "densenet121",
    "device": "cuda"
}
```

> The `device` field will show `"cpu"` if no CUDA GPU is available.

### Test with curl

```bash
curl -X POST http://localhost:8000/predict \
    -F "image=@sample.jpg" \
    -F "modality=mammography"
```

Expected response:

```json
{
    "prediction": "benign",
    "confidence": 0.873,
    "probabilities": {"benign": 0.873, "malignant": 0.127},
    "modality": "mammography",
    "gradcam_image": "<base64-encoded PNG string>",
    "inference_time_ms": 142.3
}
```

### API Documentation

FastAPI provides auto-generated interactive docs:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 8. Start the Frontend

```bash
cd frontend

# Copy environment template
cp .env.example .env

# Install Node.js dependencies
npm install

# Start the dev server
npm run dev
```

Open **http://localhost:5173** in your browser.

> **Note:** On Windows, use `copy .env.example .env` instead of `cp`.

---

## 9. Using the Application

1. **Open** http://localhost:5173 in your browser
2. **Check the health dot** in the header — a green dot confirms the backend is connected
3. **Upload an image** — drag and drop or click to browse for a mammography or ultrasound image (JPG or PNG)
4. **Select the modality** using the radio buttons (Mammography or Ultrasound)
5. **Click "Analyze Image"** to submit for classification
6. **Wait for results** — inference takes a few seconds on GPU, longer on CPU
7. **Read the prediction** — view the classification label, confidence score, and probability bars
8. **Review clinical information** — the right card shows what the prediction means with recommended next steps
9. **View the Grad-CAM++ heatmap** — click "Show Model Explanation" to see where the model focused
10. **Analyze another image** — click "Analyze Another Image" to reset and start over

---

## 10. Troubleshooting

| Problem | Solution |
|---|---|
| **CUDA out of memory** | Reduce `batch_size` in `configs/config.yaml` and retrain |
| **Checkpoint not found** | Verify `outputs/checkpoints/fusion_densenet121_best.pth` exists. Train the model first (Step 4). |
| **Backend CORS error in browser** | Confirm uvicorn is running on port 8000 with `--host 0.0.0.0` |
| **Frontend shows blank page** | Check `VITE_API_URL` in `frontend/.env` is set to `http://localhost:8000` |
| **Health dot is red** | Backend is not running. Complete Step 7 first. |
| **Slow inference** | Normal on CPU. GPU is strongly recommended for interactive use. |
| **`ModuleNotFoundError`** | Run uvicorn from the project root directory, not from inside `api/` |
| **Port already in use** | Kill the existing process or use a different port: `--port 8001` |

---

## 11. Architecture Overview

```
    ┌─────────────────────────────────┐
    │  Browser: localhost:5173        │
    │  (React + Vite + Tailwind CSS)  │
    └────────────┬────────────────────┘
                 │  HTTP POST /predict
                 │  (image + modality)
                 ▼
    ┌─────────────────────────────────┐
    │  FastAPI Backend: localhost:8000 │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  BreastCancerPredictor          │
    │    preprocess(image, modality)   │
    │    → mammo_tensor or us_tensor   │
    │    → zero tensor for inactive    │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  CrossAttentionFusionModel      │
    │    Backbone: DenseNet-121       │
    │    Bidirectional Cross-Attention │
    │    Transformer Fusion Head      │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  GradCAMExplainer               │
    │    Target: last DenseNet conv    │
    │    Returns: base64 heatmap PNG   │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  JSON Response to Frontend      │
    │    prediction, confidence,       │
    │    probabilities, gradcam_image  │
    └─────────────────────────────────┘
```

---

## 12. Quick Reference

### Start Everything (Two Terminals)

**Terminal 1 — Backend:**
```bash
conda activate breastai
cd project_root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd project_root/frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.
