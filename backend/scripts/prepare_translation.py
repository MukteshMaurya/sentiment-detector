"""One-time export of the Hindi -> English translator to fp16 ONNX.

This is a developer/CI tool, not part of the runtime app. It:

1. Downloads ``Helsinki-NLP/opus-mt-hi-en`` from the Hugging Face Hub. All
   downloaded sources are cached under ``scripts/.model_cache/`` (git-ignored).
2. Exports the MarianMT model to a decoder-less FP32 ONNX graph with Optimum's
   OnnxExport (``text2text-generation-with-past``), producing
   ``encoder_model.onnx`` and ``decoder_model.onnx``.
3. Converts both graphs to FP16 with ONNX Runtime's own converter, keeping the
   graph I/O in FP32 so the runtime feeds and reads plain float32 tensors.
4. Writes the runtime artifacts to ``app/translation_assets/``:

   - ``encoder_model.onnx`` / ``decoder_model.onnx`` - FP16 graphs (Git LFS)
   - ``config.json``, ``generation_config.json`` - model configuration
   - ``source.spm`` / ``target.spm`` and the tokenizer files for
     ``transformers.AutoTokenizer``

The runtime service (``app/services/translation.py``) loads the two graphs with
ONNX Runtime and decodes greedily, so PyTorch is never needed in production.

Why FP16: FP32 halves the shipped size (528 MiB -> 264 MiB) with no measurable
quality loss on this model, and CPU-only FP16 inference stays well under a
second per short sentence. int8 *dynamic* quantization was rejected because it
produced repeated/garbled output; see README for the full comparison.

Usage:
    python scripts/prepare_translation.py [--model-id MODEL_ID]
                                         [--output-dir DIR]
                                         [--cache-dir DIR]
                                         [--opset N]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# The project root is the parent of the `scripts` directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "app" / "translation_assets"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "scripts" / ".model_cache"

# Only these two graphs are shipped: the runtime re-evaluates the decoder over
# the growing prefix, so the merged and with-past variants are not needed.
SHIPPED_GRAPHS = ("encoder_model.onnx", "decoder_model.onnx")

logger = logging.getLogger("prepare_translation")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Hindi->English translator to fp16 ONNX "
        "in app/translation_assets/.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Runtime asset directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for downloaded model sources "
        f"(default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version for the exported graph (default: 18).",
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    level = os.getenv("SENTIMENT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _verify_graphs(output_dir: Path) -> None:
    """Load both FP16 graphs and check their I/O signatures.

    Confirms the artifacts are structurally valid and that the graph inputs
    still match what ``app/services/translation.py`` feeds them.
    """
    import onnxruntime

    expected_inputs = {
        "encoder_model.onnx": {"input_ids", "attention_mask"},
        "decoder_model.onnx": {
            "input_ids",
            "encoder_hidden_states",
            "encoder_attention_mask",
        },
    }
    for name in SHIPPED_GRAPHS:
        path = output_dir / name
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        names = {inp.name for inp in session.get_inputs()}
        logger.info("Verifying %s", path)
        logger.info("  inputs      : %s", sorted(names))
        if names != expected_inputs[name]:
            raise RuntimeError(
                f"{name} expects {sorted(names)}, "
                f"expected {sorted(expected_inputs[name])}."
            )
        for inp in session.get_inputs():
            if inp.type != "tensor(int64)":
                raise RuntimeError(
                    f"{name} input {inp.name} is {inp.type}, expected tensor(int64)."
                )
        logger.info("  outputs     : %s", [o.name for o in session.get_outputs()])
        logger.info("  signature   : OK")


def _translate_sample(output_dir: Path) -> None:
    """Run one real sentence through the runtime code path as a smoke test."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.services.translation import HindiTranslator

    translator = HindiTranslator(model_dir=output_dir)
    sample = "मुझे यह उत्पाद बहुत अच्छा लगा।"
    result = translator.translate(sample)
    logger.info("Smoke test: %r -> %r", sample, result)
    if not result:
        raise RuntimeError("Smoke test produced no translation.")


def run(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Direct the Hugging Face cache into scripts/.model_cache/ BEFORE any
    # huggingface/transformers import so nothing is written to the user's
    # global cache. `optimum` imports `transformers` at import time, so the
    # heavy imports below must happen after this env var is set.
    if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["HF_HOME"] = str(cache_dir)

    import onnx
    from optimum.exporters.onnx import main_export
    from onnxruntime.transformers import float16
    from transformers import AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model id     : %s", args.model_id)
    logger.info("Cache dir    : %s", cache_dir)
    logger.info("Output dir   : %s", output_dir)
    logger.info("ONNX opset   : %s", args.opset)

    logger.info("Downloading model + tokenizer (cached in %s) ...", cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=str(cache_dir))

    logger.info("Exporting FP32 ONNX graphs ...")
    # Optimum writes encoder_model.onnx / decoder_model.onnx (plus merged and
    # with-past variants) into this directory.
    export_dir = cache_dir / "hi_en_fp32"
    export_dir.mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path=args.model_id,
        output=export_dir,
        task="text2text-generation-with-past",
        opset=args.opset,
        device="cpu",
        do_validation=False,
        cache_dir=str(cache_dir),
    )

    logger.info("Converting graphs to FP16 (graph I/O stays FP32) ...")
    for name in SHIPPED_GRAPHS:
        source = export_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Optimum did not produce {source}")
        before = source.stat().st_size
        converted = float16.convert_float_to_float16(
            onnx.load(str(source)), keep_io_types=True
        )
        target = output_dir / name
        onnx.save(converted, str(target))
        after = target.stat().st_size
        logger.info("  %s: %.1f MiB -> %.1f MiB", name, before / 1048576, after / 1048576)

    # Remove any stale graphs the runtime does not use.
    for stale in export_dir.glob("*.onnx"):
        if stale.name not in SHIPPED_GRAPHS:
            stale.unlink(missing_ok=True)

    logger.info("Saving config + tokenizer to %s", output_dir)
    tokenizer.save_pretrained(output_dir)
    # Marian needs both SentencePiece models alongside the fast tokenizer.
    for extra in ("source.spm", "target.spm"):
        cached = cache_dir / f"models--{args.model_id.replace('/', '--')}"
        for candidate in cached.rglob(extra):
            (output_dir / extra).write_bytes(candidate.read_bytes())
            break

    _verify_graphs(output_dir)
    _translate_sample(output_dir)

    logger.info("Done. Runtime assets in %s:", output_dir)
    for artifact in sorted(p.name for p in output_dir.iterdir()):
        size_kb = (output_dir / artifact).stat().st_size / 1024
        logger.info("  - %s (%.1f KiB)", artifact, size_kb)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # keep exit codes meaningful for CI
        logger.error("Export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
