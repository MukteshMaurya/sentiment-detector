"""HTTP routes for the sentiment API.

Exposes ``GET /`` (service info), ``GET /health``, and ``POST /api/predict``
on a shared ``APIRouter``. This backend is deployed separately (Render) from
the static frontend (Vercel), which calls ``/api/predict`` over HTTPS.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app import __version__, config
from app.models import HealthResponse, PredictRequest, PredictResponse
from app.services import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", include_in_schema=False)
def index() -> dict[str, str]:
    """Service information for the API root (the frontend lives elsewhere)."""
    return {
        "service": "AI Sentiment Detector API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /api/predict",
    }


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Report the service and model status."""
    analyzer = get_analyzer()
    return HealthResponse(
        status="ok",
        model=config.MODEL_NAME,
        labels=list(config.LABELS),
        loaded=analyzer.loaded,
    )


@router.post("/api/predict", response_model=PredictResponse, tags=["predict"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the sentiment of ``request.text``."""
    try:
        prediction = get_analyzer().predict(request.text)
    except Exception as exc:  # missing model, runtime/ONNX errors -> 503
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {exc}",
        ) from exc
    return PredictResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        scores=prediction.scores,
    )