"""One-time export of the sentiment classifier to int8-quantized ONNX.

This is a developer/CI tool, not part of the runtime app. It:

1. Downloads ``cardiffnlp/twitter-roberta-base-sentiment-latest`` from the
   Hugging Face Hub. All downloaded sources are cached under
   ``scripts/.model_cache/`` (git-ignored).
2. Exports the model to an FP32 ONNX graph with Optimum's OnnxExport.
3. Applies int8 *dynamic* (weights-only) quantization with ONNX Runtime.
4. Writes the runtime artifacts to ``app/model_assets/``:

   - ``model.onnx``  - int8-quantized ONNX graph (tracked via Git LFS by
                       ``.gitattributes``)
   - ``config.json`` - model configuration (id2label label map, architecture)
   - tokenizer files - ``vocab.json``, ``merges.txt``, ``tokenizer.json``,
                       ``special_tokens_map.json``, ``tokenizer_config.json``

The runtime service (``app/services``) loads ``model.onnx`` with ONNX Runtime
and the tokenizer with ``transformers.AutoTokenizer``, so PyTorch/ONNX are
never needed in production.

The FP32 source graph is kept only under ``scripts/.model_cache/`` and removed
immediately after quantization.

Usage:
    python scripts/prepare_model.py [--model-id MODEL_ID]
                                    [--output-dir DIR]
                                    [--cache-dir DIR]
                                    [--opset N]
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# The project root is the parent of the `scripts` directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "app" / "model_assets"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "scripts" / ".model_cache"

logger = logging.getLogger("prepare_model")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the sentiment model to int8-quantized ONNX "
        "in app/model_assets/.",
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
        default=14,
        help="ONNX opset version for the exported graph (default: 14).",
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    level = os.getenv("SENTIMENT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _verify_model(model_path: Path, expected_outputs: int) -> None:
    """Load the ONNX graph with ONNX Runtime and run one inference pass.

    This confirms the artifact is structurally valid and produces the expected
    number of logits without needing the full application or any dataset.
    """
    import numpy as np
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    logger.info("Verifying %s", model_path)
    logger.info("  inputs : %s", input_names)
    logger.info("  outputs: %s", output_names)

    feed = {}
    for inp in session.get_inputs():
        if inp.name == "attention_mask":
            feed[inp.name] = np.ones((1, 16), dtype=np.int64)
        else:
            feed[inp.name] = np.random.randint(2, 50264, size=(1, 16), dtype=np.int64)

    results = session.run(output_names, feed)
    logits = np.asarray(results[0])
    if logits.shape[-1] != expected_outputs:
        raise RuntimeError(
            f"Model output has {logits.shape[-1]} logits; "
            f"expected {expected_outputs} label scores."
        )
    logger.info("  logits shape: %s (OK)", logits.shape)


def run(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Direct the Hugging Face cache into scripts/.model_cache/ BEFORE any
    # huggingface/transformers import so nothing is written to the user's
    # global cache. `optimum` imports `transformers` at import time, so the
    # heavy imports below must happen after this env var is set.
    if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["HF_HOME"] = str(cache_dir)

    import onnxruntime
    from onnxruntime.quantization import (
        QuantType,
        quantize_dynamic,
    )
    from optimum.exporters.onnx import export
    from optimum.exporters.onnx.model_configs import RobertaOnnxConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model id     : %s", args.model_id)
    logger.info("Cache dir    : %s", cache_dir)
    logger.info("Output dir   : %s", output_dir)
    logger.info("ONNX opset   : %s", args.opset)

    logger.info("Downloading model + tokenizer (cached in %s) ...", cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=str(cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, cache_dir=str(cache_dir)
    )
    model.eval()

    n_labels = len(model.config.id2label) if model.config.id2label else 3
    logger.info("Loaded model with %s labels.", n_labels)

    # 1) FP32 ONNX export to the git-ignored cache dir.
    fp32_path = cache_dir / "model_fp32.onnx"
    onnx_config = RobertaOnnxConfig(
        model.config, task="text-classification"
    )
    logger.info("Exporting FP32 ONNX graph ...")
    input_names, output_names = export(
        model,
        onnx_config,
        fp32_path,
        opset=args.opset,
        device="cpu",
    )
    logger.info("FP32 export done. inputs=%s outputs=%s", input_names, output_names)

    # 2) int8 dynamic (weights-only) quantization, no calibration data needed.
    quantized_path = output_dir / "model.onnx"
    logger.info("Applying int8 dynamic quantization -> %s", quantized_path)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
    )

    # 3) Runtime support files: model config + tokenizer.
    logger.info("Saving config.json and tokenizer to %s", output_dir)
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 4) Verify the final artifact on ONNX Runtime.
    _verify_model(quantized_path, expected_outputs=n_labels)

    # 5) Drop the temporary FP32 graph (sources stay cached for re-runs).
    fp32_path.unlink(missing_ok=True)
    logger.info("Removed temporary FP32 graph %s", fp32_path.name)

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