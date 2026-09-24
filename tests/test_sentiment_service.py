"""Smoke tests for the sentiment inference service.

These load the real ONNX model + tokenizer from ``app/model_assets/`` once
per module (via a module-scoped fixture) and run a handful of inferences.
"""

from __future__ import annotations

import pytest

from app import config
from app.services import Prediction, SentimentAnalyzer, get_analyzer


@pytest.fixture(scope="module")
def analyzer() -> SentimentAnalyzer:
    """One shared analyzer for the module: the model is loaded on first use."""
    return SentimentAnalyzer()


def test_predict_returns_structured_prediction(analyzer: SentimentAnalyzer) -> None:
    prediction = analyzer.predict(
        "This is absolutely fantastic, I loved every second of it!"
    )
    assert isinstance(prediction, Prediction)
    assert prediction.label in config.LABELS
    assert set(prediction.scores) == set(config.LABELS)
    assert 0.0 <= prediction.confidence <= 1.0
    assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert prediction.scores[prediction.label] == pytest.approx(
        prediction.confidence, abs=1e-9
    )
    # as_dict is JSON-safe and matches the dataclass fields
    dumped = prediction.as_dict()
    assert dumped["label"] == prediction.label
    assert dumped["confidence"] == prediction.confidence
    assert dumped["scores"] == prediction.scores


def test_sentiment_directions(analyzer: SentimentAnalyzer) -> None:
    positive = analyzer.predict("I love this, it is wonderful and amazing!")
    negative = analyzer.predict("This is awful, I hate it with a passion.")
    assert positive.label == "positive"
    assert positive.confidence > 0.5
    assert negative.label == "negative"
    assert negative.confidence > 0.5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I love this!", "positive"),
        ("I hate this!", "negative"),
        ("The package arrived on Tuesday.", "neutral"),
    ],
)
def test_labels_for_known_texts(
    analyzer: SentimentAnalyzer, text: str, expected: str
) -> None:
    assert analyzer.predict(text).label == expected


def test_lazy_load_and_idempotent_load() -> None:
    fresh = SentimentAnalyzer()
    assert fresh.loaded is False
    assert fresh.load() is fresh
    assert fresh.load() is fresh
    assert fresh.loaded is True


def test_non_string_rejected(analyzer: SentimentAnalyzer) -> None:
    with pytest.raises(TypeError):
        analyzer.predict(1234)  # type: ignore[arg-type]


def test_max_tokens_override_and_validation(analyzer: SentimentAnalyzer) -> None:
    short = analyzer.predict("I love this!", max_tokens=16)
    assert short.label == "positive"
    with pytest.raises(ValueError):
        analyzer.predict("test", max_tokens=0)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_rejected(analyzer: SentimentAnalyzer, blank: str) -> None:
    with pytest.raises(ValueError):
        analyzer.predict(blank)


def test_get_analyzer_returns_singleton() -> None:
    assert get_analyzer() is get_analyzer()
    assert isinstance(get_analyzer(), SentimentAnalyzer)