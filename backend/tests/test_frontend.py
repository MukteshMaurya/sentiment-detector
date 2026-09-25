"""Tests for the API service info and the frontend/backend integration boundary.

After the architecture migration, the frontend is a separate static deployment
(Vercel) — the backend no longer serves HTML. These tests cover the API root
(``GET /`` -> JSON service info), the docs surface, the health endpoint, and
the CORS configuration that lets the Vercel frontend call the Render backend.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

ALLOWED_FRONTEND_ORIGIN = "https://sentiment-detector-tau.vercel.app"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "origin",
    [
        ALLOWED_FRONTEND_ORIGIN,
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)
def test_cors_allows_frontend_origin(client: TestClient, origin: str) -> None:
    assert origin in config.CORS_ORIGINS
    response = client.post(
        "/api/predict",
        json={"text": "nice"},
        headers={"Origin": origin},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


def test_frontend_url_is_prepended_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRONTEND_URL", "https://my-frontend.example")
    reloaded = importlib.reload(config)
    assert "https://my-frontend.example" in reloaded.CORS_ORIGINS
    assert ALLOWED_FRONTEND_ORIGIN in reloaded.CORS_ORIGINS


def test_root_returns_service_info(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "AI Sentiment Detector API"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
    assert "predict" in body


def test_root_content_is_json_not_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.headers["content-type"].startswith("application/json")


def test_docs_still_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "redoc" in response.text.lower()


def test_openapi_still_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/predict" in response.text


def test_health_still_available(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"