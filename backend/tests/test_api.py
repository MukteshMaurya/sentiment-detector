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


# --------------------------------------------------------------------------
# Hindi / Hinglish translation fields (additive, must not break the contract)
# --------------------------------------------------------------------------


def test_predict_english_reports_not_needed(client: TestClient) -> None:
    body = client.post(
        "/api/predict", json={"text": "I absolutely love this product!"}
    ).json()
    assert body["translation_status"] == "not_needed"
    assert body["translated_text"] is None
    assert body["original_text"] == "I absolutely love this product!"


def test_predict_devanagari_is_translated(client: TestClient) -> None:
    response = client.post("/api/predict", json={"text": "यह उत्पाद बहुत खराब है।"})
    assert response.status_code == 200
    body = response.json()
    assert body["translation_status"] == "translated"
    assert body["translated_text"]
    assert body["original_text"] == "यह उत्पाद बहुत खराब है।"
    assert body["label"] in config.LABELS


def test_predict_hinglish_is_translated(client: TestClient) -> None:
    response = client.post(
        "/api/predict", json={"text": "Mujhe ye product bahut achha laga"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["translation_status"] == "translated"
    assert "product" in body["translated_text"].lower()
    assert body["original_text"] == "Mujhe ye product bahut achha laga"


def test_predict_english_with_a_person_name_is_not_translated(
    client: TestClient,
) -> None:
    """Regression: a person named "Mukesh" must not trigger translation.

    "Mukesh" and "name" used to be treated as weak Hinglish markers, so this
    ordinary English sentence was sent to the MT model. It must stay on the
    normal English sentiment path: no translation, original text untouched.
    """
    text = "My friend Mukesh told me the name of the restaurant"
    response = client.post("/api/predict", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["translation_status"] == "not_needed"
    assert body["translated_text"] is None
    assert body["original_text"] == text
    assert body["model"] == config.MODEL_NAME
    assert body["label"] in config.LABELS


def test_predict_translation_failure_still_returns_200(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A translation failure degrades to analyzing the original text."""
    import app.services.translation as translation_mod

    class Broken:
        loaded = False

        def translate(self, text: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(translation_mod, "get_translator", lambda: Broken())
    response = client.post("/api/predict", json={"text": "Ye product bahut kharab hai"})
    assert response.status_code == 200
    body = response.json()
    assert body["translation_status"] == "failed"
    assert body["translated_text"] is None
    assert body["label"] in config.LABELS


def test_existing_response_fields_are_unchanged(client: TestClient) -> None:
    """The original contract must keep its exact field set and validation."""
    body = client.post("/api/predict", json={"text": "nice"}).json()
    for field in ("model", "label", "confidence", "scores"):
        assert field in body
    assert body["model"] == config.MODEL_NAME
    assert set(body["scores"]) == set(config.LABELS)
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-4)