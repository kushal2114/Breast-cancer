"""
schemas.py — Pydantic request/response models for the BreastAI Classifier API.

Defines the JSON response schemas for the /predict, /predict-fusion,
and /health endpoints.
All fields are documented with descriptions for automatic OpenAPI documentation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for POST /predict.

    Contains the classification result, confidence scores,
    modality information, Grad-CAM++ heatmap, and timing data.
    """

    prediction: str = Field(
        ...,
        description="Classification label: 'benign' or 'malignant'.",
        examples=["malignant"],
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the predicted class (0.0–1.0).",
        ge=0.0,
        le=1.0,
        examples=[0.873],
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Class-wise probability distribution.",
        examples=[{"benign": 0.127, "malignant": 0.873}],
    )
    modality: str = Field(
        ...,
        description="Input modality used: 'mammography', 'ultrasound', or 'fusion'.",
        examples=["mammography"],
    )
    gradcam_image: Optional[str] = Field(
        None,
        description="Base64-encoded PNG of the Grad-CAM++ heatmap overlay. "
                    "Present for single-modality predictions.",
    )
    gradcam_mammo: Optional[str] = Field(
        None,
        description="Base64-encoded PNG of the Grad-CAM++ heatmap overlay "
                    "for the mammography stream. Present in fusion mode.",
    )
    gradcam_us: Optional[str] = Field(
        None,
        description="Base64-encoded PNG of the Grad-CAM++ heatmap overlay "
                    "for the ultrasound stream. Present in fusion mode.",
    )
    inference_time_ms: float = Field(
        ...,
        description="Model inference time in milliseconds.",
        examples=[142.3],
    )


class HealthResponse(BaseModel):
    """Response schema for GET /health.

    Reports the API status, loaded model, backbone architecture,
    and compute device.
    """

    status: str = Field(
        ...,
        description="API health status.",
        examples=["ok"],
    )
    model: str = Field(
        ...,
        description="Name of the loaded model.",
        examples=["fusion_densenet121"],
    )
    backbone: str = Field(
        ...,
        description="Encoder backbone architecture.",
        examples=["densenet121"],
    )
    device: str = Field(
        ...,
        description="Compute device: 'cuda' or 'cpu'.",
        examples=["cuda"],
    )
