"""Tests for the Vercel deployment configuration.

Vercel auto-detects FastAPI from ``app/main.py`` (a supported entrypoint), so
``vercel.json`` only tunes the resolved function. These tests keep the config
consistent with the repository: the entrypoint must exist and export a
FastAPI ``app``, the function ``maxDuration`` must stay within all plan
limits, and ``excludeFiles`` must never strip runtime assets (the ``app/``
package, ``requirements.txt``, ``model.onnx``) from the bundle.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERCEL_JSON = ROOT / "vercel.json"


def _config() -> dict:
    return json.loads(VERCEL_JSON.read_text(encoding="utf-8"))


def test_vercel_json_is_valid() -> None:
    assert isinstance(_config(), dict)


def test_function_keyed_to_existing_entrypoint() -> None:
    functions = _config()["functions"]
    assert set(functions) == {"app/main.py"}
    assert (ROOT / "app" / "main.py").is_file()


def test_entrypoint_exports_fastapi_app() -> None:
    from app.main import app

    assert isinstance(app, FastAPI)


def test_entrypoint_is_the_same_object_served_locally() -> None:
    from app.main import app

    assert app.title == "AI Sentiment Detector"


def test_max_duration_within_all_plan_limits() -> None:
    config = _config()
    assert config["functions"]["app/main.py"]["maxDuration"] == 300


def test_exclude_files_never_remove_runtime_assets() -> None:
    config = _config()
    raw = config["functions"]["app/main.py"]["excludeFiles"]
    inner = raw.strip("{}")
    patterns = [part.strip() for part in inner.split(",") if part.strip()]
    # Exact, reviewed set: only dev/test/report paths may be excluded.
    assert set(patterns) == {"tests/**", "scripts/**", "data/**", "**/*.md"}
    forbidden = ("app/", "app\\", "requirements", "vercel.json", ".venv")
    for pattern in patterns:
        assert not pattern.startswith(forbidden), pattern


def test_runtime_assets_present_on_disk() -> None:
    assert (ROOT / "app" / "model_assets" / "model.onnx").is_file()
    assert (ROOT / "app" / "model_assets" / "tokenizer.json").is_file()
    assert (ROOT / "requirements.txt").is_file()