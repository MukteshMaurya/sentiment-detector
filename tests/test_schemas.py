"""Unit tests for the Pydantic request/response schemas.

These run without loading the model, so they are fast and safe in any
environment.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app import config
from app.models import HealthResponse, PredictRequest, PredictResponse


def test_request_strips_whitespace() -> None:
    assert PredictRequest(text="  hi there  ").text == "hi there"


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_request_rejects_blank(text: str) -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text=text)


def test_request_rejects_too_long() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="x" * (config.MAX_INPUT_CHARS + 1))


def test_request_accepts_limit_length() -> None:
    request = PredictRequest(text="y" * config.MAX_INPUT_CHARS)
    assert len(request.text) == config.MAX_INPUT_CHARS


def test_request_rejects_non_string() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text=123)


def test_request_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="hi", extra=1)


def test_response_roundtrip() -> None:
    response = PredictResponse(
        label="positive",
        confidence=0.9,
        scores={"negative": 0.05, "neutral": 0.05, "positive": 0.9},
    )
    dumped = response.model_dump()
    assert dumped == json.loads(response.model_dump_json())
    assert dumped["model"] == config.MODEL_NAME
    assert dumped["label"] == "positive"
    assert dumped["confidence"] == 0.9


def test_response_rejects_bad_label() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(
            label="angry",
            confidence=1.0,
            scores={"negative": 0, "neutral": 0, "positive": 1},
        )


def test_response_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(
            label="negative",
            confidence=1.5,
            scores={"negative": 1, "neutral": 0, "positive": 0},
        )


def test_response_rejects_incomplete_scores() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(
            label="negative",
            confidence=1.0,
            scores={"negative": 1.0, "neutral": 0.0},
        )


def test_response_rejects_unknown_score_label() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(
            label="negative",
            confidence=1.0,
            scores={
                "negative": 1.0,
                "neutral": 0.0,
                "positive": 0.0,
                "extra": 0.0,
            },
        )


def test_health_response_roundtrip() -> None:
    health = HealthResponse(
        status="ok",
        model="m",
        labels=list(config.LABELS),
        loaded=True,
    )
    assert health.model_dump() == {
        "status": "ok",
        "model": "m",
        "labels": list(config.LABELS),
        "loaded": True,
    }