"""Unit tests for scripts/evaluate_model.py.

These cover the pure helpers only (pair loading, metric computation, report
and confusion-matrix writers) so they run fast, offline, and without loading
the ONNX model. Full end-to-end runs against the real TweetEval data are done
manually via the script.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_model import (
    _build_report,
    _compute_metrics,
    _load_pairs,
    _parse_args,
    _write_confusion_csv,
)

LABELS = ("negative", "neutral", "positive")


@pytest.fixture
def dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    text_path = tmp_path / "val_text.txt"
    label_path = tmp_path / "val_labels.txt"
    text_path.write_text(
        "Great stuff!\nok fine\nterrible\n\n one more \n",
        encoding="utf-8",
    )
    label_path.write_text("2\n1\n0\n\n1\n", encoding="utf-8")
    return text_path, label_path


def test_load_pairs_aligns_and_drops_blanks(dataset_files: tuple[Path, Path]) -> None:
    text_path, label_path = dataset_files
    pairs = _load_pairs(text_path, label_path, labels=LABELS)
    assert pairs == [("Great stuff!", 2), ("ok fine", 1), ("terrible", 0), ("one more", 1)]


def test_load_pairs_rejects_non_integer_label(
    tmp_path: Path, dataset_files: tuple[Path, Path]
) -> None:
    _, label_path = dataset_files
    text_path = tmp_path / "t.txt"
    text_path.write_text("hello\n", encoding="utf-8")
    label_path.write_text("spam\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid label"):
        _load_pairs(text_path, label_path, labels=LABELS)


def test_load_pairs_rejects_out_of_range_label(
    tmp_path: Path, dataset_files: tuple[Path, Path]
) -> None:
    _, label_path = dataset_files
    text_path = tmp_path / "t.txt"
    text_path.write_text("hello\n", encoding="utf-8")
    label_path.write_text("-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of range"):
        _load_pairs(text_path, label_path, labels=LABELS)


def test_compute_metrics_known_values() -> None:
    y_true = ["negative", "negative", "neutral", "positive"]
    y_pred = ["negative", "negative", "neutral", "neutral"]
    metrics = _compute_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)
    assert metrics["weighted_f1"] == pytest.approx((2 * 1.0 + 2 / 3) / 4)
    assert metrics["micro_f1"] == pytest.approx(0.75)
    assert metrics["support"] == [2, 1, 1]

    cm = np.asarray(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm.tolist() == [[2, 0, 0], [0, 1, 0], [0, 1, 0]]

    # zero-division class yields precision 0, not NaN
    assert metrics["precision"] == [1.0, pytest.approx(0.5), 0.0]


def test_build_report_contains_key_sections() -> None:
    y_true = ["negative", "negative", "neutral", "positive"]
    y_pred = ["negative", "negative", "neutral", "neutral"]
    metrics = _compute_metrics(y_true, y_pred, LABELS)
    text = _build_report(
        metrics,
        LABELS,
        model_name="test model (ONNX int8)",
        split="val",
        n_samples=4,
        elapsed_seconds=1.26,
    )
    assert "TweetEval Sentiment Model Evaluation Report" in text
    assert "test model (ONNX int8)" in text
    assert "Split        : val" in text
    assert "Samples      : 4" in text
    assert "Eval time    : 1.3s" in text
    assert "Accuracy      : 0.7500" in text
    for name in LABELS:
        assert f"{name:<12}" in text


def test_write_confusion_csv_round_trip(tmp_path: Path) -> None:
    cm = np.array([[2, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=int)
    dest = tmp_path / "confusion_matrix.csv"
    _write_confusion_csv(cm, LABELS, dest)

    with dest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["true_label", *LABELS]
    assert rows[1:] == [
        ["negative", "2", "0", "0"],
        ["neutral", "0", "1", "0"],
        ["positive", "0", "1", "0"],
    ]


def test_parse_args_defaults_and_overrides() -> None:
    defaults = _parse_args([])
    assert defaults.split == "test"
    assert defaults.max_samples == 0

    overridden = _parse_args(["--split", "val", "--max-samples", "50"])
    assert overridden.split == "val"
    assert overridden.max_samples == 50


def test_predict_all_passes_max_tokens_through() -> None:
    from scripts.evaluate_model import _predict_all

    class DummyAnalyzer:
        def __init__(self) -> None:
            self.calls: list[int | None] = []

        def predict(self, text: str, *, max_tokens: int | None = None) -> object:
            self.calls.append(max_tokens)
            return SimpleNamespace(label="positive")

    analyzer = DummyAnalyzer()
    pairs = [("one", 2), ("two", 2)]
    y_true, y_pred = _predict_all(analyzer, pairs, max_tokens=64)
    assert y_true == ["positive", "positive"]
    assert y_pred == ["positive", "positive"]
    assert analyzer.calls == [64, 64]