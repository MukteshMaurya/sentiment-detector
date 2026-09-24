"""API tests for the FastAPI application (served through TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config
from app.api import routes as routes_mod
from app.main import app
from app.services import SentimentAnalyzer

ALLOWED_ORIGIN = "http://localhost:8000"
DISALLOWED_ORIGIN = "https://evil.example"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_unloaded_state(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """/health must not force-load the model; a fresh analyzer is unloaded."""
    fresh = SentimentAnalyzer()
    monkeypatch.setattr(routes_mod, "get_analyzer", lambda: fresh)

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == config.MODEL_NAME
    assert body["labels"] == list(config.LABELS)
    assert body["loaded"] is False


def test_predict_happy_path(client: TestClient) -> None:
    response = client.post(
        "/api/predict", json={"text": "I absolutely love this product!"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == config.MODEL_NAME
    assert body["label"] in config.LABELS
    assert body["confidence"] == pytest.approx(
        body["scores"][body["label"]], abs=1e-9
    )
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-4)
    assert set(body["scores"]) == set(config.LABELS)


def test_predict_strips_whitespace(client: TestClient) -> None:
    response = client.post(
        "/api/predict", json={"text": "   Absolutely fantastic!   "}
    )
    assert response.status_code == 200
    assert response.json()["label"] in config.LABELS


def test_health_reports_loaded_after_predict(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["loaded"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "x" * (config.MAX_INPUT_CHARS + 1)},
        {"text": 123},
        {"texty": "hi"},
        {"text": "hi", "extra": 1},
    ],
)
def test_predict_invalid_payloads_return_422(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_inference_failure_returns_503(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    class FailingAnalyzer:
        loaded = False

        def predict(self, text: str) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(routes_mod, "get_analyzer", lambda: FailingAnalyzer())

    response = client.post("/api/predict", json={"text": "hello"})
    assert response.status_code == 503
    assert "boom" in response.json()["detail"]


def test_cors_allowed_origin_is_echoed(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"text": "nice"},
        headers={"Origin": ALLOWED_ORIGIN},
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    )


def test_cors_disallowed_origin_not_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})
    assert response.headers.get("access-control-allow-origin") is None


def test_cors_preflight(client: TestClient) -> None:
    response = client.options(
        "/api/predict",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in (200, 204)
    assert (
        response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    )


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404


def test_get_on_predict_returns_405(client: TestClient) -> None:
    assert client.get("/api/predict").status_code == 405