"""Tests for the deployment configuration after the architecture migration.

The Vercel project deploys only the static ``frontend/`` directory (no Python
function, no model), and the Render backend runs from ``backend/`` via `uvicorn
app.main:app`. These tests keep the deployment config consistent with the
repository: no Vercel Python function, the FastAPI entrypoint intact, the ONNX
model still Git-LFS-tracked at its new path, and runtime assets present.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frontend_vercel_json_is_valid() -> None:
    config = _json(FRONTEND / "vercel.json")
    assert isinstance(config, dict)


def test_frontend_has_no_vercel_python_function() -> None:
    config = _json(FRONTEND / "vercel.json")
    assert "functions" not in config, "Vercel must not define Python functions"
    assert "rwfs" not in config


def test_frontend_uses_static_build_with_url_injection() -> None:
    assert (FRONTEND / "index.html").is_file()
    assert (FRONTEND / "app.js").is_file()
    assert (FRONTEND / "build.js").is_file()
    build_js = (FRONTEND / "build.js").read_text(encoding="utf-8")
    assert "VITE_API_URL" in build_js


def test_old_root_vercel_python_config_removed() -> None:
    assert not (ROOT / "vercel.json").exists()


def test_backend_entrypoint_exports_fastapi_app() -> None:
    import sys

    backend_root = str(BACKEND)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    import app.main  # noqa: F401

    from app.main import app

    assert isinstance(app, FastAPI)
    assert app.title == "AI Sentiment Detector"


def test_backend_layout_is_clean() -> None:
    assert (BACKEND / "requirements.txt").is_file()
    assert (BACKEND / "app" / "main.py").is_file()
    assert (BACKEND / "render.yaml").is_file()
    assert not (BACKEND / "app" / "static").exists()


def test_model_is_lfs_tracked_at_backend_path() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "backend/app/model_assets/model.onnx" in attributes
    assert "filter=lfs" in attributes


def test_model_file_present_on_disk() -> None:
    model = BACKEND / "app" / "model_assets" / "model.onnx"
    assert model.is_file()
    assert model.stat().st_size > 100_000_000  # real int8 binary, not a pointer


def test_tokenizer_config_present_on_disk() -> None:
    assets = BACKEND / "app" / "model_assets"
    for name in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        assert (assets / name).is_file()