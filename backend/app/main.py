"""FastAPI application assembly.

Run with ``uvicorn app.main:app``; the ``app`` module-level instance is the
single source of truth used by the dev server, deployment entrypoints, and
tests.

Serves only the API: the sentiment frontend is a separate static deployment
(Vercel) that calls this backend over HTTPS.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__, config
from app.api.routes import router


def create_app() -> FastAPI:
    """Build and configure the application."""
    application = FastAPI(
        title="AI Sentiment Detector",
        description="Local sentiment classification via an int8-quantized "
        "RoBERTa ONNX model.",
        version=__version__,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()