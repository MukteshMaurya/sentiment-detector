"""Application configuration.

All configuration is read from environment variables so nothing
application-specific (paths, limits, origins) is hard-coded elsewhere
and no secrets are ever committed to the repository.

This project does not call any paid or proprietary API; the only
model runs locally, so there are no API keys to manage.
"""

from __future__ import annotations

import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}")


def _list_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment variable into a list."""
    raw = os.getenv(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


# Repository root (parent of the `app` package).
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory holding the exported ONNX model + tokenizer files.
MODEL_DIR = Path(os.getenv("SENTIMENT_MODEL_DIR", str(BASE_DIR / "app" / "model_assets")))

# Directory holding the served frontend (index.html, CSS, JS).
STATIC_DIR = Path(os.getenv("SENTIMENT_STATIC_DIR", str(BASE_DIR / "app" / "static")))

# Human-readable model identifier returned by the API.
MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)",
)

# Order of the model's output logits (matches config.json id2label).
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

# Maximum characters accepted by POST /api/predict (input validation).
MAX_INPUT_CHARS = _int_env("SENTIMENT_MAX_INPUT_CHARS", 2000)

# Maximum tokens fed to the model (RoBERTa supports 512; 256 is ample
# for typical inputs and bounds inference latency).
MAX_TOKENS = _int_env("SENTIMENT_MAX_TOKENS", 256)

# Allowed CORS origins, comma-separated (default: same-origin only + localhost
# dev + the production deployment so the hosted frontend may call the API
# cross-origin if it is ever served from a different host).
CORS_ORIGINS: list[str] = _list_env(
    "SENTIMENT_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,https://sentiment-detector-tau.vercel.app",
)

# Log level for the application logger.
LOG_LEVEL = os.getenv("SENTIMENT_LOG_LEVEL", "INFO").upper()
