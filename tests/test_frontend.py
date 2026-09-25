"""Tests for the served frontend: GET / and the /static assets.

Verifies the production root route serves the sentiment detector UI (not the
FastAPI 404 JSON), the static assets resolve, and the API/docs surface stays
intact alongside the frontend.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_frontend(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "AI Sentiment Detector" in html
    assert "sentiment-form" in html
    assert "text-input" in html
    assert "analyze-btn" in html


def test_root_frontend_references_static_assets(client: TestClient) -> None:
    html = client.get("/").text
    for asset in ("/static/styles.css", "/static/app.js", "/static/favicon.svg"):
        assert asset in html, f"index.html must reference {asset}"


def test_static_css_served(client: TestClient) -> None:
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_static_js_served(client: TestClient) -> None:
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_static_favicon_served(client: TestClient) -> None:
    response = client.get("/static/favicon.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg")


def test_static_dir_config_points_at_existing_frontend() -> None:
    assert (config.STATIC_DIR / "index.html").is_file()
    assert (config.STATIC_DIR / "styles.css").is_file()
    assert (config.STATIC_DIR / "app.js").is_file()


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