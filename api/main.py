"""
main.py — FastAPI application for the BreastAI Classifier.

Serves predictions from the trained fusion_densenet121 model.
The model is loaded once at startup and shared across all requests.

Endpoints:
    POST /predict  — Classify a breast image as benign/malignant
    GET  /health   — Check API status and model info

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import BreastCancerPredictor
from api.explainer import GradCAMExplainer
from api.schemas import HealthResponse, PredictionResponse

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("breastai.api")

# ── Constants ──────────────────────────────────────────────────
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "config.yaml"
)
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "checkpoints", "fusion_densenet121_best.pth"
)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
VALID_MODALITIES = {"mammography", "ultrasound"}

# ── Global references (populated at startup) ───────────────────
predictor: BreastCancerPredictor = None  # type: ignore
explainer: GradCAMExplainer = None       # type: ignore


# ── Lifespan (startup / shutdown) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; clean up CUDA cache at shutdown."""
    global predictor, explainer

    logger.info("=" * 60)
    logger.info("  BreastAI Classifier API — Starting up")
    logger.info("=" * 60)

    # ── Startup ────────────────────────────────────────────────
    try:
        predictor = BreastCancerPredictor(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
        )
        explainer = GradCAMExplainer(model=predictor.model)
        logger.info("Startup complete — model ready for inference")
    except FileNotFoundError as e:
        logger.error("Checkpoint not found: %s", e)
        logger.error(
            "Train the model first: python src/train.py "
            "--model fusion --backbone densenet121"
        )
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    yield  # ← Application is running

    # ── Shutdown ───────────────────────────────────────────────
    logger.info("Shutting down — cleaning up resources")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="BreastAI Classifier API",
    description=(
        "Breast cancer classification using Bidirectional "
        "Cross-Attention Fusion with DenseNet-121 backbone."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow all origins for development) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════
#  Endpoints
# ════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health, loaded model, and compute device."""
    return HealthResponse(
        status="ok",
        model="fusion_densenet121",
        backbone="densenet121",
        device=str(predictor.device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Breast image (JPG or PNG, max 10 MB)"),
    modality: str = Form(..., description="'mammography' or 'ultrasound'"),
):
    """Classify a breast image as benign or malignant.

    Accepts a single mammography or ultrasound image. The missing
    modality stream is filled with a zero tensor (the model was
    trained with modality dropout for robustness to missing inputs).

    Always returns a Grad-CAM++ heatmap overlay because DenseNet-121
    is a CNN backbone that supports gradient-based explanations.
    """
    # ── Validate modality ──────────────────────────────────────
    if modality not in VALID_MODALITIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid modality '{modality}'. "
                f"Must be 'mammography' or 'ultrasound'."
            ),
        )

    # ── Validate content type ──────────────────────────────────
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{image.content_type}'. "
                f"Accepted types: JPEG, PNG."
            ),
        )

    # ── Read and validate file size ────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum allowed size is 10 MB."
            ),
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # ── Run inference ──────────────────────────────────────────
    try:
        result = predictor.predict(image_bytes, modality)
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again with a different image.",
        )

    # ── Generate Grad-CAM++ ────────────────────────────────────
    try:
        gradcam_b64 = explainer.generate(
            mammo_tensor=result["mammo_tensor"],
            us_tensor=result["us_tensor"],
            modality=modality,
            original_image_bytes=image_bytes,
        )
    except Exception as e:
        logger.error("Grad-CAM++ generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM++ generation failed. Please try again.",
        )

    # ── Build response ─────────────────────────────────────────
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        modality=modality,
        gradcam_image=gradcam_b64,
        inference_time_ms=result["inference_time_ms"],
    )


@app.post("/predict-fusion", response_model=PredictionResponse)
async def predict_fusion(
    mammo_image: UploadFile = File(
        ..., description="Mammography image (JPG or PNG, max 10 MB)"
    ),
    us_image: UploadFile = File(
        ..., description="Ultrasound image (JPG or PNG, max 10 MB)"
    ),
):
    """Classify using both mammography and ultrasound images (true fusion).

    Accepts one mammography image and one ultrasound image. Both are
    preprocessed with their respective pipelines and fed as real tensors
    to the cross-attention fusion model — no zero-filling is used.

    Returns Grad-CAM++ heatmap overlays for both streams.
    """
    # ── Validate content types ─────────────────────────────────
    for img, name in [(mammo_image, "mammography"), (us_image, "ultrasound")]:
        if img.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid file type for {name}: '{img.content_type}'. "
                    f"Accepted types: JPEG, PNG."
                ),
            )

    # ── Read and validate file sizes ───────────────────────────
    mammo_bytes = await mammo_image.read()
    us_bytes = await us_image.read()

    for data, name in [(mammo_bytes, "mammography"), (us_bytes, "ultrasound")]:
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name.capitalize()} file too large "
                    f"({len(data) / 1024 / 1024:.1f} MB). "
                    f"Maximum allowed size is 10 MB."
                ),
            )
        if len(data) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded {name} file is empty.",
            )

    # ── Run fusion inference ───────────────────────────────────
    try:
        result = predictor.predict_fusion(mammo_bytes, us_bytes)
    except Exception as e:
        logger.error("Fusion inference failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again with different images.",
        )

    # ── Generate Grad-CAM++ for both streams ───────────────────
    try:
        gradcam_mammo_b64 = explainer.generate(
            mammo_tensor=result["mammo_tensor"],
            us_tensor=result["us_tensor"],
            modality="mammography",
            original_image_bytes=mammo_bytes,
        )
        gradcam_us_b64 = explainer.generate(
            mammo_tensor=result["mammo_tensor"],
            us_tensor=result["us_tensor"],
            modality="ultrasound",
            original_image_bytes=us_bytes,
        )
    except Exception as e:
        logger.error("Grad-CAM++ generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM++ generation failed. Please try again.",
        )

    # ── Build response ─────────────────────────────────────────
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        modality="fusion",
        gradcam_mammo=gradcam_mammo_b64,
        gradcam_us=gradcam_us_b64,
        inference_time_ms=result["inference_time_ms"],
    )
