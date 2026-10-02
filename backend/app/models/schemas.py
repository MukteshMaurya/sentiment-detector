"""Pydantic request/response schemas for the sentiment API.

These mirror the service-layer ``Prediction`` result
(``app/services/sentiment.py``) and enforce input limits
(``SENTIMENT_MAX_INPUT_CHARS``) and probability bounds at the API boundary,
so validation failures surface as HTTP 422 without touching the model.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from app import config

# The three sentiment labels, in model logit order (matches config.json id2label).
Label = Literal[*config.LABELS]

# Outcome of Hindi/Hinglish preprocessing (see app/services/translation.py).
TranslationStatus = Literal["not_needed", "translated", "failed"]

# Non-empty (after stripping whitespace) and bounded by MAX_INPUT_CHARS.
TextInput = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=config.MAX_INPUT_CHARS,
    ),
]

# Probabilities produced by softmax: always within [0, 1].
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class PredictRequest(BaseModel):
    """Input for POST /api/predict."""

    model_config = ConfigDict(extra="forbid")

    text: TextInput


class PredictResponse(BaseModel):
    """Output for POST /api/predict."""

    model_config = ConfigDict(extra="forbid")

    # Human-readable model identifier (in case the deployed artifact changes).
    model: str = Field(default_factory=lambda: config.MODEL_NAME)
    label: Label
    confidence: Probability
    scores: dict[str, Probability]

    # --- Optional Hindi/Hinglish metadata -------------------------------
    # These are additive: existing clients reading model/label/confidence/
    # scores keep working, and every field below may legitimately be null
    # for English input or when translation is unavailable.
    #: Exactly the text that was submitted, after request validation.
    original_text: str | None = None
    #: English rendering of the input, or null when none was produced.
    translated_text: str | None = None
    #: "not_needed" for English, "translated", or "failed" (original analyzed).
    translation_status: TranslationStatus | None = None

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, scores: dict[str, float]) -> dict[str, float]:
        """Require exactly the model's labels, no more and no less."""
        missing = set(config.LABELS) - set(scores)
        if missing:
            raise ValueError(f"scores missing labels: {sorted(missing)}")
        unexpected = set(scores) - set(config.LABELS)
        if unexpected:
            raise ValueError(f"scores has unknown labels: {sorted(unexpected)}")
        return scores


class HealthResponse(BaseModel):
    """Output for GET /health."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    model: str
    labels: list[Label]

    # True once the service has instantiated the ONNX session for this process.
    loaded: bool