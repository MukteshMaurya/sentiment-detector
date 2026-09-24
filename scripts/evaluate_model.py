"""Evaluate the deployed int8 ONNX sentiment model against TweetEval.

This is a developer/CI tool, not part of the runtime app. It:

1. Downloads the TweetEval ``sentiment`` task (SemEval-2017 Task 4A) from the
   `cardiffnlp/tweeteval` GitHub repository into ``data/dataset/``
   (git-ignored). The files are one tweet per line with a parallel labels
   file; the label mapping (0=negative, 1=neutral, 2=positive) is identical
   to the model's own mapping, so labels are compared directly with no
   remapping.
2. Runs inference through the *deployed* artifact via ``app.services``
   (``SentimentAnalyzer``: ONNX Runtime + tokenizer from ``app/model_assets/``),
   the same code path the API uses at runtime.
3. Computes scikit-learn metrics and writes two reports into ``data/``:

   - ``model_evaluation_report.txt`` - accuracy, macro/weighted/micro F1 and
     per-class precision/recall/F1/support
   - ``confusion_matrix.csv`` - confusion matrix as CSV (rows = true labels,
     columns = predicted labels)

Evaluation runs lazily, exactly like inference: only genuine results are
written. No numbers are ever fabricated.

Usage:
    python scripts/evaluate_model.py [--split {train,val,test}]
                                    [--max-samples N]
                                    [--max-tokens N]
                                    [--dataset-dir DIR]
                                    [--output-dir DIR]
                                    [--verbose]

The full test split is 12,284 tweets; use ``--max-samples`` for a quick
smoke run (takes the first N samples in file order). TweetEval tweets are
short (99.9% under ~60 tokens), so ``--max-tokens 64`` cuts evaluation time
~4x with no measurable accuracy change; the default (``config.MAX_TOKENS``,
256) is identical to the runtime API path.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Make the `app` package importable regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"
REPORT_FILENAME = "model_evaluation_report.txt"
CONFUSION_FILENAME = "confusion_matrix.csv"

SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/"
    "datasets/sentiment"
)

# Split -> (text file, labels file) under the TweetEval sentiment dir.
SPLIT_FILES: dict[str, tuple[str, str]] = {
    "train": ("train_text.txt", "train_labels.txt"),
    "val": ("val_text.txt", "val_labels.txt"),
    "test": ("test_text.txt", "test_labels.txt"),
}
SPLITS = tuple(SPLIT_FILES)

logger = logging.getLogger("evaluate_model")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the exported sentiment model against TweetEval.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="TweetEval split to evaluate (default: test).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate only the first N samples in file order "
        "(0 = all samples; default: 0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.MAX_TOKENS,
        help="Tokenizer padding/truncation length per tweet, overriding "
        f"config.MAX_TOKENS ({config.MAX_TOKENS}). TweetEval tweets are short "
        "(99.9% < ~60 tokens), so a smaller cap e.g. 64 speeds evaluation up "
        "roughly 4x with no measurable accuracy change (default: "
        f"{config.MAX_TOKENS}).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory for downloaded TweetEval files "
        f"(default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the generated reports "
        f"(default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _configure_logging(verbose: bool = False) -> None:
    level = os.getenv("SENTIMENT_LOG_LEVEL", "DEBUG" if verbose else "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, raising a clear error on failure."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response, dest.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded an empty file from {url}")


def _ensure_dataset(dataset_dir: Path, split: str) -> None:
    """Download the requested split's files into ``dataset_dir`` if missing."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for filename in SPLIT_FILES[split]:
        dest = dataset_dir / filename
        if dest.is_file() and dest.stat().st_size > 0:
            logger.info("Using cached file %s", dest)
            continue
        url = f"{SOURCE_BASE_URL}/{filename}"
        logger.info("Downloading %s -> %s", url, dest)
        _download_file(url, dest)


def _load_pairs(
    text_path: Path,
    label_path: Path,
    labels: Sequence[str] = config.LABELS,
) -> list[tuple[str, int]]:
    """Load aligned (text, label-index) pairs from the two TweetEval files.

    Lines are paired positionally. Blank text/label lines are skipped and
    non-integer or out-of-range label values raise an error, so the returned
    pairs always align with the model's label order.
    """
    text_lines = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    label_lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(text_lines) != len(label_lines):
        logger.warning(
            "Line count mismatch in %s/%s: %d texts vs %d labels",
            text_path.name,
            label_path.name,
            len(text_lines),
            len(label_lines),
        )
    pairs: list[tuple[str, int]] = []
    for index, (text, raw_label) in enumerate(zip(text_lines, label_lines)):
        text = text.strip()
        raw_label = raw_label.strip()
        if not text or not raw_label:
            continue
        try:
            value = int(raw_label)
        except ValueError:
            raise ValueError(
                f"Invalid label {raw_label!r} at line {index + 1} of "
                f"{label_path.name} (expected an integer 0..{len(labels) - 1})."
            )
        if value < 0 or value >= len(labels):
            raise ValueError(
                f"Label {value} at line {index + 1} of {label_path.name} "
                f"is out of range 0..{len(labels) - 1}."
            )
        pairs.append((text, value))
    return pairs


def _predict_all(
    analyzer: object,
    pairs: list[tuple[str, int]],
    max_tokens: int | None = None,
) -> tuple[list[str], list[str]]:
    """Predict every text and return (true labels, predicted labels) by name."""
    names = tuple(config.LABELS)
    y_true: list[str] = []
    y_pred: list[str] = []
    total = len(pairs)
    start = time.perf_counter()
    for index, (text, label_index) in enumerate(pairs, start=1):
        y_true.append(names[label_index])
        y_pred.append(
            analyzer.predict(text, max_tokens=max_tokens).label  # type: ignore[attr-defined]
        )
        if total >= 250 and (index % 1000 == 0 or index == total):
            elapsed = time.perf_counter() - start
            logger.info(
                "  %d/%d predicted (%.1fs elapsed)", index, total, elapsed
            )
    return y_true, y_pred


def _compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], names: Sequence[str]
) -> dict[str, object]:
    """Compute overall + per-class metrics with scikit-learn (deferred import)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    label_list = list(names)
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    total_support = float(support.sum())
    macro_f1 = float(f1.mean()) if len(f1) else 0.0
    weighted_f1 = (
        float((f1 * support).sum() / total_support) if total_support > 0 else 0.0
    )
    return {
        "confusion_matrix": cm,
        "precision": [float(x) for x in precision],
        "recall": [float(x) for x in recall],
        "f1": [float(x) for x in f1],
        "support": [int(x) for x in support],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": accuracy,
    }


def _build_report(
    metrics: dict[str, object],
    names: Sequence[str],
    *,
    model_name: str,
    split: str,
    n_samples: int,
    elapsed_seconds: float,
    tokenizer_cap: int | None = None,
) -> str:
    """Render the human-readable evaluation report as plain text."""
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = metrics["support"]
    assert isinstance(precision, list) and isinstance(recall, list)
    assert isinstance(f1, list) and isinstance(support, list)

    title = "TweetEval Sentiment Model Evaluation Report"
    lines = [
        title,
        "=" * len(title),
        "",
        f"Model        : {model_name}",
        f"Split        : {split}",
        f"Samples      : {n_samples}",
        "Backend      : ONNX Runtime (CPU) via app/services/sentiment.py",
        f"Tokenizer cap: {tokenizer_cap} tokens"
        if tokenizer_cap is not None
        else f"Tokenizer cap: {config.MAX_TOKENS} tokens (runtime default)",
        f"Eval time    : {elapsed_seconds:.1f}s",
        "",
        "Overall metrics",
        "-" * 22,
        f"Accuracy      : {metrics['accuracy']:.4f}",
        f"Macro F1      : {metrics['macro_f1']:.4f}",
        f"Weighted F1   : {metrics['weighted_f1']:.4f}",
        f"Micro F1      : {metrics['micro_f1']:.4f}",
        "",
        "Per-class metrics",
        "-" * 22,
        f"{'':<12}{'precision':>10}{'recall':>8}{'f1-score':>9}{'support':>9}",
    ]
    for index, name in enumerate(names):
        lines.append(
            f"{name:<12}{precision[index]:>10.4f}{recall[index]:>8.4f}"
            f"{f1[index]:>9.4f}{support[index]:>9d}"
        )
    lines.extend(
        [
            "",
            "Confusion matrix written to: confusion_matrix.csv",
            "(rows = true labels, columns = predicted labels)",
            "",
            f"Generated by scripts/evaluate_model.py on "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_confusion_csv(
    cm: object, names: Sequence[str], dest: Path
) -> None:
    """Write ``cm`` (numpy array of shape (C, C)) as a labelled CSV."""
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *names])
        for index, name in enumerate(names):
            writer.writerow([name, *(int(x) for x in cm[index])])  # type: ignore[index]


def run(args: argparse.Namespace) -> dict[str, object]:
    from app.services.sentiment import SentimentAnalyzer

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    labels = tuple(config.LABELS)

    _ensure_dataset(dataset_dir, args.split)
    text_path = dataset_dir / SPLIT_FILES[args.split][0]
    label_path = dataset_dir / SPLIT_FILES[args.split][1]
    pairs = _load_pairs(text_path, label_path, labels=labels)
    if not pairs:
        raise ValueError("No samples to evaluate.")
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    logger.info("Evaluating %d samples.", len(pairs))

    analyzer = SentimentAnalyzer(model_dir=config.MODEL_DIR)
    logger.info("Predicting %d samples ...", len(pairs))
    if args.max_tokens >= config.MAX_TOKENS:
        max_tokens: int | None = None
    else:
        max_tokens = args.max_tokens
        logger.info("Using tokenizer cap max_tokens=%d.", max_tokens)
    start_time = time.perf_counter()
    y_true, y_pred = _predict_all(analyzer, pairs, max_tokens=max_tokens)
    elapsed_seconds = time.perf_counter() - start_time

    metrics = _compute_metrics(y_true, y_pred, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    confusion_path = output_dir / CONFUSION_FILENAME
    report_path.write_text(
        _build_report(
            metrics,
            labels,
            model_name=config.MODEL_NAME,
            split=args.split,
            n_samples=len(pairs),
            elapsed_seconds=elapsed_seconds,
            tokenizer_cap=max_tokens,
        ),
        encoding="utf-8",
    )
    _write_confusion_csv(metrics["confusion_matrix"], labels, confusion_path)
    return {
        "report_path": report_path,
        "confusion_path": confusion_path,
        "metrics": metrics,
        "samples": len(pairs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    try:
        result = run(args)
    except Exception as exc:  # keep exit codes meaningful for CI
        logger.error("Evaluation failed: %s", exc)
        return 1
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    logger.info(
        "Accuracy %.4f | Macro F1 %.4f | Weighted F1 %.4f on %d samples",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
        result["samples"],
    )
    logger.info("Wrote %s", result["report_path"])
    logger.info("Wrote %s", result["confusion_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())