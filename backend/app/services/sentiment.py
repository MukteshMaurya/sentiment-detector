"""Sentiment analysis inference service.

Loads the int8-quantized ONNX model and the RoBERTa tokenizer from the
runtime asset directory once per process (lazily, on first use) and runs
inference on ONNX Runtime. PyTorch is never imported at runtime.

Public API:
    - ``Prediction``        structured result of a single prediction
    - ``SentimentAnalyzer`` loads the model and predicts sentiment
    - ``get_analyzer()``    returns the process-wide analyzer singleton
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app import config

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_ANALYZER_LOCK = threading.Lock()


@dataclass(frozen=True)
class Prediction:
    """Result of a sentiment prediction for a single piece of text."""

    label: str
    confidence: float
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """Flat dict representation, safe for JSON serialization."""
        return asdict(self)


class SentimentAnalyzer:
    """Predicts sentiment with RoBERTa exported to an int8-quantized ONNX graph.

    The model and tokenizer are loaded once on first use (see ``load``); the
    heavy imports are deferred to that point so ``import app.services`` stays
    cheap for serverless cold starts.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir is not None else Path(config.MODEL_DIR)
        self._labels: tuple[str, ...] = tuple(config.LABELS)
        self._session: Any = None
        self._tokenizer: Any = None

    @property
    def loaded(self) -> bool:
        """True once the ONNX session has been created."""
        return self._session is not None

    def load(self) -> "SentimentAnalyzer":
        """Load and hold the ONNX session + tokenizer (idempotent, thread-safe)."""
        if self._session is not None:
            return self
        with _INIT_LOCK:
            if self._session is not None:
                return self
            import onnxruntime
            from transformers import AutoTokenizer

            model_path = self._model_dir / "model.onnx"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Run scripts/prepare_model.py to export it first."
                )
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            logger.info("SentimentAnalyzer loaded model from %s", self._model_dir)
        return self

    def predict(self, text: str, *, max_tokens: int | None = None) -> Prediction:
        """Run sentiment inference on ``text`` and return a Prediction.

        ``max_tokens`` overrides ``config.MAX_TOKENS`` for the tokenizer's
        padding/truncation length (used by evaluation tooling to bound
        latency on large corpora). ``None`` (the default) keeps the runtime
        behavior.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        text = text.strip()
        if not text:
            raise ValueError("text must be a non-empty string.")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}.")
        self.load()

        encoding = self._tokenizer(
            text,
            max_length=config.MAX_TOKENS if max_tokens is None else max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        feed = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        (logits,) = self._session.run(None, feed)

        probs = _softmax(np.asarray(logits[0], dtype=np.float32))
        scores = {label: float(p) for label, p in zip(self._labels, probs)}
        index = int(np.argmax(probs))
        return Prediction(
            label=self._labels[index],
            confidence=float(probs[index]),
            scores=scores,
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis of a 1-D input."""
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


_analyzer: SentimentAnalyzer | None = None


def get_analyzer() -> SentimentAnalyzer:
    """Return the process-wide SentimentAnalyzer; created once, on first use.

    The returned analyzer is not forced to load; the load happens lazily on
    the first ``predict`` call.
    """
    global _analyzer
    if _analyzer is None:
        with _ANALYZER_LOCK:
            if _analyzer is None:
                _analyzer = SentimentAnalyzer()
    return _analyzer