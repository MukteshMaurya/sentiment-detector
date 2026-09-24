# Project

AI Sentiment Detector

## Recovery Status

This file was reconstructed after the previous OpenCode session ended
unexpectedly (context/token limit). It reflects the **actual** state of the
files on disk as inspected during recovery. Nothing was assumed from memory;
no features are claimed unless present in the repository.

## Current Phase

**Phase 1 — Project Scaffolding (mostly complete).**

The package skeleton, configuration layer, dev dependencies, model-export
script, exported ONNX int8 model, the **inference service**, the **Pydantic
schemas**, the **FastAPI routes + app assembly**, the **automated test
suite**, the **model evaluation** (`scripts/evaluate_model.py`, TweetEval
val + test results), and the **Vercel deployment config** (`vercel.json` +
root `README.md`) are all in place. Still not implemented: a frontend.
Next action is the initial git commit (NEXT STEP #10).

## Completed Work

Only what is actually implemented:

- Python package skeleton:
  - `app/__init__.py` — package with `__version__ = "1.0.0"`
  - `app/api/__init__.py` — declared (empty) API layer
  - `app/models/__init__.py` — re-exports the schemas from
    `app/models/schemas.py` (`PredictRequest`, `PredictResponse`)
  - `app/services/__init__.py` — re-exports the public service API
    (`Prediction`, `SentimentAnalyzer`, `get_analyzer`) from
    `app/services/sentiment.py`
  - `tests/` — **automated test suite (new this session)**:
  - `tests/test_schemas.py` (14 tests) — unit tests for `PredictRequest`,
    `PredictResponse`, `HealthResponse`: whitespace stripping, blank / too-
    long / non-string / extra-field rejection, `MAX_INPUT_CHARS` boundary,
    label/confidence/scores-key validation, JSON round-trips. No model load.
  - `tests/test_sentiment_service.py` (12 tests) — smoke tests on the real
    ONNX model (module-scoped `SentimentAnalyzer` fixture): structured
    `Prediction` invariants, sentiment direction (positive/negative/neutral
    labels), lazy + idempotent `load()`, input-validation errors, `max_tokens`
    override/validation, singleton `get_analyzer()`.
  - `tests/test_api.py` (17 tests) — `TestClient` API tests: `/health`
    (unloaded before predict / loaded after), predict happy path + whitespace
    stripping, 7 invalid payloads → 422, inference failure → 503 (stubbed
    analyzer), CORS echo / disallowed origin / preflight, 404 unknown route,
    405 GET on `POST /api/predict`.
  - `tests/test_evaluate_script.py` (8 tests, new this session) — offline unit
    tests for `scripts/evaluate_model.py` helpers: aligned TweetEval pair
    loading (blank-line skipping, non-int / out-of-range label rejection),
    known-value metric computation (accuracy / macro / weighted / micro F1,
    per-class precision/recall), report text contents, confusion-matrix CSV
    round-trip, CLI arg defaults/overrides, and `max_tokens` plumbing through
    `_predict_all`.
  - `tests/test_vercel_config.py` (7 tests, new this session) — checks the
    Vercel deployment config stays in sync with the repo: `vercel.json` is
    valid JSON, the function is keyed to the real entrypoint `app/main.py`
    which exports a FastAPI `app`, `maxDuration` (300) is within all plan
    limits, and the `excludeFiles` globs are the reviewed set and never match
    runtime assets (`app/`, `requirements.txt`, `model.onnx`).
  - **`pytest tests -q` → 58 passed** (51 previous + 7 vercel-config tests).
    Files also pass in isolation. One harmless `StarletteDeprecationWarning`
    from `fastapi.testclient`.

- `vercel.json` + `README.md` — **Vercel deployment config (new this
  session)**. Vercel zero-config-detects FastAPI: it looks for a `FastAPI`
  instance named `app` in `app/main.py` (a supported entrypoint), so the
  serverless entrypoint is the existing `app.main:app` — no duplicate
  `api/index.py` was added to avoid entrypoint ambiguity. `vercel.json` keys
  the resolved function (`"app/main.py"`) with `maxDuration: 300` (the Hobby
  default *and* maximum, so it works on every plan and comfortably covers the
  ~15-30 s lazy model cold start) and `excludeFiles:
  "{tests/**,scripts/**,data/**,**/*.md}"` to slim the bundle. Grounded in
  Vercel's current FastAPI/Python docs (fetched this session): 500 MB
  uncompressed Python bundle limit (standard) — the ~231 MB model + deps fits;
  5 GB Large Functions Beta via `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`; Hobby 2 GB
  memory; 4.5 MB request-body limit. Root `README.md` documents setup, local
  run, tests, model export, TweetEval evaluation (with the real numbers), the
  env-var table, and Vercel deploy steps (git-import + CLI `vercel dev` /
  `vercel --prod`). The README/`vercel.json` state explicitly that deployment
  was **not executed** from this machine — the config follows documented
  guidance and must be confirmed with `vercel dev` / a real deploy.
- `tests/test_vercel_config.py` — see `tests/` entry above.

- `scripts/prepare_model.py` — one-time model export tool (**already run —
  see fixes in "Files Modified"**). Downloads the model from HF (cached under
  `scripts/.model_cache/`), exports FP32 ONNX via Optimum, applies int8
  dynamic (weights-only) quantization via
  `onnxruntime.quantization.quantize_dynamic` (QUInt8), writes
  `app/model_assets/model.onnx` + `config.json` + tokenizer files
  (`vocab.json`, `merges.txt`, `tokenizer.json`, `special_tokens_map.json`,
  `tokenizer_config.json`), then verifies the quantized graph with an ONNX
  Runtime inference pass checking the logits count (3) and removes the
  temporary FP32 graph. Heavy imports (torch/transformers/optimum) are
  deferred inside `run()` and `HF_HOME` is pointed at the cache dir first.
- **`scripts/evaluate_model.py` — model evaluation tool (new this session,
  RUN)**. Downloads TweetEval sentiment (SemEval-2017 Task 4A) raw files from
  `cardiffnlp/tweeteval` GitHub into `data/dataset/` (git-ignored) with stdlib
  `urllib` (no `datasets` dependency needed), loads aligned (text, label)
  pairs, runs inference through the deployed `app.services.SentimentAnalyzer`
  (same int8 ONNX artifact the API serves), and writes scikit-learn metrics to
  `data/model_evaluation_report.txt` (overall + per-class) and
  `data/confusion_matrix.csv` (rows = true labels, columns = predicted).
  CLI: `--split {train,val,test}` (default `test`), `--max-samples N`,
  `--max-tokens N` (overrides `config.MAX_TOKENS`; default 256), dataset/output
  dirs, `--verbose`. **Results are real (see Model Evaluation section):**
  test split 12,284 samples → accuracy **0.7208**, macro F1 **0.7201**,
  weighted F1 **0.7208** (run with `--max-tokens 64`; TweetEval tweets are
  short — max 59 tokens on val — so the 64 cap truncates effectively nothing
  and matches the model's published ~0.718 within noise); val split 2,000 →
  accuracy **0.7645**. `SentimentAnalyzer.predict()` gained an optional
  `max_tokens=` kwarg (default `None` = runtime `config.MAX_TOKENS`) so
  evaluation latency can be bounded without changing the runtime path.
- `app/config.py` — environment-driven application configuration:
  - `BASE_DIR`, `MODEL_DIR` (default `app/model_assets`), `MODEL_NAME`
    (default `cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)`),
  - `LABELS = ("negative", "neutral", "positive")`
  - `MAX_INPUT_CHARS` (default 2000), `MAX_TOKENS` (default 256)
  - `CORS_ORIGINS` (default localhost:8000 / 127.0.0.1:8000)
  - `LOG_LEVEL`
  - Verified importable and functional in the current venv.
- `requirements.txt` — runtime deps: fastapi, uvicorn, pydantic, onnxruntime,
  transformers (tokenizer only), numpy.
- `requirements-dev.txt` — dev deps: pytest, httpx, scikit-learn (evaluation),
  onnx + optimum for one-time model export; notes `pip install torch --index-url .../whl/cpu`.
- `.python-version` — 3.12
- `.gitignore` — ignores `.venv`, `.env`, `scripts/.model_cache/`, `data/dataset/`,
  `data/*.log`, `.vercel/`, model temp artifacts, caches.
- `.gitattributes` — Git LFS for `app/model_assets/*.onnx`
- `LICENSE` — MIT, plus third-party notices for cardiffnlp model (CC BY 4.0)
  and TweetEval (CC BY 3.0).
- `data/README.md` — documents the evaluation dataset: TweetEval
  `sentiment` (SemEval-2017 Task 4A), classes negative/neutral/positive,
  splits train 45,615 / val 2,000 / test 12,284, label mapping identical to
  the model's own (0=neg, 1=neu, 2=pos); generated outputs
  `model_evaluation_report.txt` and `confusion_matrix.csv` now exist in
  `data/`.
- `.venv` with current runtime dependencies installed (fastapi 0.141.1,
  uvicorn 0.53.0, pydantic 2.13.5, onnxruntime 1.30.0, transformers 4.57.6,
  numpy 2.5.3, huggingface_hub 0.36.2, tokenizers 0.22.2).
- **`app/model_assets/` — EXPORTED and verified.** Contains `model.onnx`
  (int8-quantized, ~231 MiB / 242,491,205 bytes), `config.json`, and the
  tokenizer files listed above. Verification inside `prepare_model.py` passed
  (logits shape `(1, 3)`); an additional end-to-end ONNX Runtime inference
  sanity check produced correct predictions (see Testing Status).
- `app/services/sentiment.py` — **inference service (new this session)**:
  - `Prediction` — frozen dataclass `{label, confidence, scores}` with
    `as_dict()`; `scores` is a `{"negative": .., "neutral": .., "positive": ..}`
    probability map.
  - `SentimentAnalyzer` — loads the int8 ONNX model via
    `onnxruntime.InferenceSession` (CPUExecutionProvider) + RoBERTa tokenizer
    via `transformers.AutoTokenizer` from `config.MODEL_DIR`, both lazily on
    first use under a thread lock (`load()` is idempotent); heavy imports
    (onnxruntime/transformers) deferred so importing the module is cheap for
    cold starts.
  - `predict(text)` — validates input (non-string → `TypeError`; blank →
    `ValueError`), tokenizes with
    `max_length=config.MAX_TOKENS, padding="max_length", truncation=True`,
    runs the session, applies stable softmax, and returns a `Prediction`.
  - `get_analyzer()` — process-wide singleton (double-checked locking).
- `app/models/schemas.py` — **Pydantic schemas (new this session)**:
  - `PredictRequest` — `text` with `StringConstraints(strip_whitespace=True,
    min_length=1, max_length=config.MAX_INPUT_CHARS)` (2000 default);
    `extra="forbid"`. Blank/whitespace-only/long/non-string inputs rejected.
  - `PredictResponse` — `model` (defaults to `config.MODEL_NAME`), `label`
    (`Literal[*config.LABELS]`), `confidence` and `scores` values constrained
    to `[0, 1]`; a field validator requires `scores` keys to be exactly the
    three model labels. `extra="forbid"`.
  - Re-exported from `app.models` as the public schema API.
- `app/api/routes.py` — **FastAPI routes (new this session)**:
  - `GET /health` — returns `HealthResponse{status:"ok", model, labels,
    loaded}`; `loaded` reflects whether the ONNX session has been created yet
    (lazy, so false until the first prediction).
  - `POST /api/predict` — accepts `PredictRequest`, calls the analyzer
    singleton, returns `PredictResponse`; unexpected inference failures are
    logged and surfaced as `503 Model inference failed: ...`.
  - Shared `APIRouter`, re-exported from `app.api`.
- `app/main.py` — **FastAPI application assembly (new this session)**:
  - `create_app()` builds the app (title/description/version from
    `app.__version__`), wires CORS middleware from `config.CORS_ORIGINS`
    (`allow_credentials=True`, all methods/headers), and includes the router.
  - Module-level `app = create_app()` for `uvicorn app.main:app`.
- `app/models/schemas.py` — added `HealthResponse` (see routes entry).

## Project Structure

```
nlp model/
├── .gitattributes            Git LFS tracking for model_assets/*.onnx
├── .gitignore
├── .python-version           3.12
├── LICENSE                   MIT + third-party notices
├── OPENCODE_PROGRESS.md      This recovery file
├── README.md                 Root docs: setup/run/test/eval/deploy (NEW)
├── vercel.json               Vercel function config: entrypoint/maxDuration/excludes (NEW)
├── requirements.txt          Runtime deps
├── requirements-dev.txt      Dev/test/eval/export deps
├── app/
│   ├── __init__.py           Package, __version__ = 1.0.0
│   ├── config.py             Environment-driven configuration (functional)
│   ├── main.py               FastAPI assembly: CORS + router (functional)
│   ├── api/                  HTTP routes (functional)
│   │   ├── __init__.py       Re-exports the routes router
│   │   └── routes.py         GET /health, POST /api/predict
│   ├── models/               Pydantic schemas (functional)
│   │   ├── __init__.py       Re-exports PredictRequest/Response + HealthResponse
│   │   └── schemas.py        Request/response schemas + input validation
│   ├── services/             Inference service (functional)
│   │   ├── __init__.py      Re-exports Prediction/SentimentAnalyzer/get_analyzer
│   │   └── sentiment.py     ONNX Runtime + tokenizer wrapper, softmax, singleton
│   └── model_assets/         model.onnx + config + tokenizer (EXPORTED)
├── data/
│   ├── README.md             Dataset + evaluation output documentation
│   ├── model_evaluation_report.txt   TweetEval test result report (GENERATED)
│   ├── confusion_matrix.csv   Test-split confusion matrix (GENERATED)
│   └── dataset/              Downloaded TweetEval files (git-ignored)
├── tests/
│   ├── __init__.py             Test suite package
│   ├── test_schemas.py         14 unit tests (schemas, no model load)
│   ├── test_sentiment_service.py  12 smoke tests (real ONNX model)
│   ├── test_api.py             17 API tests via TestClient
│   ├── test_evaluate_script.py  8 offline unit tests for the eval tool
│   └── test_vercel_config.py    7 Vercel-config/entrypoint tests (NEW)
├── scripts/
│   ├── prepare_model.py        One-time ONNX int8 export tool (run)
│   ├── evaluate_model.py       TweetEval evaluation tool (run; see Model Evaluation)
│   └── .model_cache/         (created at first run; git-ignored)
└── .venv/                    Virtualenv with runtime + dev deps
```

## Files Created

| File | Purpose |
|---|---|
| `app/config.py` | Central, env-based configuration |
| `app/services/sentiment.py` | Inference service: lazy ONNX + tokenizer loading, `predict`, singleton |
| `app/models/schemas.py` | Pydantic request/response schemas with input validation (+ `HealthResponse`) |
| `app/api/routes.py` | FastAPI routes: `GET /health`, `POST /api/predict` |
| `app/main.py` | FastAPI app assembly (`create_app` + `app`), CORS from config |
| `tests/test_schemas.py` | 14 schema unit tests |
| `tests/test_sentiment_service.py` | 12 service smoke tests (real model) |
| `tests/test_api.py` | 17 API tests via TestClient |
| `tests/test_evaluate_script.py` | 8 offline unit tests for the evaluation tool |
| `tests/test_vercel_config.py` | 7 Vercel-config/entrypoint tests (NEW) |
| `vercel.json` | Vercel function config: `"app/main.py"` entrypoint, `maxDuration 300`, safe `excludeFiles` (NEW) |
| `README.md` | Root README: setup/run/test/eval + Vercel deploy instructions (NEW) |
| `app/__init__.py` | Package skeleton placeholder |
| `scripts/prepare_model.py` | One-time ONNX int8 export tool (run; two bugs fixed) |
| `scripts/evaluate_model.py` | TweetEval evaluation tool (run; outputs in `data/`) |
| `data/model_evaluation_report.txt` | TweetEval **test-split** report (accuracy 0.7208) |
| `data/confusion_matrix.csv` | Test-split confusion matrix (rows = true labels) |
| `app/model_assets/` | `model.onnx` (int8, ~231 MiB) + `config.json` + tokenizer files — **exported this session** |
| `requirements.txt` / `requirements-dev.txt` | Dependency manifests |
| `.python-version` | Python 3.12 pin |
| `.gitignore` / `.gitattributes` | Ignore rules + Git LFS config |
| `LICENSE` | MIT + third-party notices |
| `data/README.md` | Dataset & evaluation documentation |
| `OPENCODE_PROGRESS.md` | THIS recovery document |

## Files Modified

- `scripts/prepare_model.py` — two bugs fixed while running the export
  (both required for the export to succeed with the installed versions):
  1. `RobertaOnnxConfig(..., task="sequence-classification")` →
     `task="text-classification"` (Optimum 2.1.0 registers the task as
     `text-classification`; `sequence-classification` raised
     `Export failed: 'sequence-classification'`).
  2. `onnxruntime.quantization` was referenced without an explicit import
     (onnxruntime 1.30.0 does not auto-expose the submodule). Added
     `from onnxruntime.quantization import QuantType, quantize_dynamic` and
     changed the call site to use those names directly.

- `app/services/__init__.py` — replaced the empty placeholder with re-exports
  of the new public service API (`Prediction`, `SentimentAnalyzer`,
  `get_analyzer`).
- `app/services/sentiment.py` — **`predict()` gained an optional
  `max_tokens: int | None = None` keyword** (default `None` keeps the runtime
  `config.MAX_TOKENS` behavior; values < 1 are rejected). Used by the
  evaluation tool to bound per-sample latency on large corpora. Default API
  behavior is unchanged.
- `app/models/__init__.py` — replaced the empty placeholder with re-exports
  of the new public schema API (`PredictRequest`, `PredictResponse`).
- `app/models/schemas.py` — added `HealthResponse` (status/model/labels/
  loaded) for the new `/health` route.
- `app/models/__init__.py` and `app/api/__init__.py` — re-export the new
  `HealthResponse` schema and the routes `router`, respectively.

Notably still absent: only a frontend (not planned as a numbered NEXT STEP).

## Architecture

**Actual** (implemented and tested):

```
Frontend (not yet created)
   └─ HTTP → FastAPI API (app/api/routes.py + app/main.py)
                └─ Pydantic schemas (app/models/schemas.py)
                     └─ Service layer (app/services/sentiment.py)
                          └─ ONNX Runtime inference
                               └─ cardiffnlp/twitter-roberta-base-sentiment-latest
                                  (int8-quantized ONNX, tokenizer via transformers)
```

- Inference intended to run on **ONNX Runtime** (no PyTorch at runtime;
  PyTorch only for one-time export via `scripts/prepare_model.py`).
- Evaluation implemented via `scripts/evaluate_model.py` against TweetEval
  (real results in the Model Evaluation section).
- **Actual implemented architecture:** `app/config.py` configuration layer,
  `app/services/sentiment.py` inference service, `app/models/` Pydantic
  schemas, `app/api/routes.py` routes, `app/main.py` FastAPI assembly
  (CORS + router), `scripts/evaluate_model.py` evaluation, and the Vercel
  deployment config (`vercel.json`, `README.md`, entrypoint `app/main.py`) —
  all functional. The only remaining planned layer is a frontend; deployment
  config is now added.

## NLP Model

**EXPORTED AND VERIFIED.**

- Chosen model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  (referenced in `app/config.py` default `MODEL_NAME`, license notices, and
  dependency comments).
- Format: **ONNX, int8-quantized** (weights-only dynamic quant, QUInt8),
  deployed in `app/model_assets/` as `model.onnx` (~231 MiB).
- Export flow (executed this session): HF download (cached in
  `scripts/.model_cache/`) → Optimum FP32 ONNX export (opset 14, inputs
  `input_ids` + `attention_mask`, outputs `logits`) → `quantize_dynamic`
  int8 → tokenizer + config files saved to `app/model_assets/` → ONNX
  Runtime verification (logits shape `(1, 3)` OK) → temp FP32 graph removed.
- Model downloads used `scripts/.model_cache/` (HF_HOME); the FP32 source
  graph was deleted after quantization as designed.
- Verified with an end-to-end inference sanity check outside the script:
  `AutoTokenizer` + `onnxruntime.InferenceSession` on three sample texts
  ("This is absolutely amazing! ..." → positive 98.0%; "I hate this, it is
  terrible." → negative 93.8%; "It is okay, nothing special." → neutral
  52.8%).

## Dataset

**DOWNLOADED INTO `data/dataset/` (git-ignored) AND VERIFIED.**

- TweetEval `sentiment` (SemEval-2017 Task 4A), 3 classes, label mapping
  0=negative / 1=neutral / 2=positive (identical to the model's, so no
  remapping during evaluation), license CC BY 3.0.
- Raw files fetched from `cardiffnlp/tweeteval` GitHub (raw URLs) by
  `scripts/evaluate_model.py`: `train_text/labels.txt` (4.74 MiB), `val_text/
  labels.txt` (0.21 MiB), `test_text/labels.txt` (1.1 MiB). All cached under
  `data/dataset/`; subsequent runs skip re-downloading.
- **Verified line counts (loaded via `_load_pairs`):** train **45,615**,
  val **2,000**, test **12,284** — all match `data/README.md`. Label
  distributions confirmed sane (e.g. train: 0→7,093 / 1→20,673 / 2→17,849).
- None of the dataset files are committed (`.gitignore` ignores
  `data/dataset/`).

## Testing Status

- **Automated test suite exists** (`tests/`): **58 tests, all passing**
  (`pytest tests -q`). Breakdown: 14 schema unit tests, 12 service smoke
  tests, 17 API tests, 8 evaluation-script offline tests, 7 Vercel-config
  tests; each file also passes in isolation.
  Run command: `python -m pytest tests -q` (or `.venv\Scripts\python.exe -m pytest tests -q`).
- Checks actually run in the current session (export run):
  - `python scripts/prepare_model.py` → **passed**. Full pipeline ran:
    download (cached) → FP32 ONNX export → int8 dynamic quantization →
    tokenizer/config saved → ONNX Runtime verification (`logits shape
    (1, 3)` OK) → FP32 graph removed.
  - End-to-end inference sanity check (tokenizer + ONNX Runtime, 3 texts) →
    **passed** with correct predictions (see NLP Model section).
- Checks actually run in the current session (service):
  - `python -m compileall -q app` → **passed**.
  - Direct verification of `app.services` (see Completed Work): 4 texts
    predicted via `get_analyzer().predict(...)` → all returned correct labels
    and confidence; `scores` keys exactly `{negative, neutral, positive}`;
    scores sum ≈ 1.0; `scores[label] == confidence`; confidence in [0,1].
    `predict("   ")` → `ValueError`, `predict(123)` → `TypeError` — both
    OK. Model + tokenizer load lazily on first predict (deferred imports).
  - Direct verification of `app.models` schemas (see Completed Work):
    valid request accepted (whitespace stripped); whitespace-only / empty /
    > `MAX_INPUT_CHARS` / non-string / extra fields → rejected; response with
    bad label, out-of-range confidence, or missing/extra `scores` labels →
    rejected; `model_dump_json()` round-trips. Full checks run in a throwaway
    script in the temp dir (not committed).
  - Integration check: `PredictResponse` built from a real
    `get_analyzer().predict(...)` result validated and serialized to the
    expected JSON (`model`, `label`, `confidence`, `scores`).
  - API checks via `fastapi.testclient.TestClient` on `app.main:app` (see
    Completed Work): `/health` → 200 with `status=ok`, correct model/labels,
    `loaded=false` before prediction / `true` after; `/api/predict` happy path
    (200, scores sum ≈ 1, `confidence` in-range, whitespace-stripped input);
    blank/empty/too-long/non-string/missing/extra-field payloads → 422; CORS:
    `Origin: http://localhost:8000` echoed in `access-control-allow-origin`,
    disallowed origin not echoed, preflight OPTIONS OK; stubbed analyzer
    raising → 503 with detail; unknown route → 404; `GET /api/predict` → 405.
  - Live smoke test: `uvicorn app.main:app` on port 8765 (subprocess) served
    `/health`, `/api/predict` (positive → 200), and 422 for empty text —
    **passed**.
  - `python -m compileall -q app scripts tests` → **passed**; `pytest tests -q`
    → **51 passed** (43 schema/service/API + 8 evaluation-script tests);
    plus 7 new Vercel-config tests this session → **58 passed**.
  - Evaluation script end-to-end runs (see Model Evaluation):
    `scripts/evaluate_model.py --split val` (2,000 samples, ~10.6 min) and
    `--split test` (12,284 samples, ~68 min) both **passed** and wrote the
    expected `model_evaluation_report.txt` + `confusion_matrix.csv`.

## Model Evaluation

**DONE — real numbers from TweetEval, no fabrication.**

`scripts/evaluate_model.py` was implemented and run against the TweetEval
`sentiment` test split (12,284 samples) and validation split (2,000 samples).
Inference uses the deployed `app/services/sentiment.py` ONNX Runtime path
with the exact int8 artifact from `app/model_assets/`. Runs used
`--max-tokens 64` (Tweets are short — max observed 59 tokens on val — so the
cap truncates effectively nothing while cutting wall-clock >4x).

Committed artifacts (via `python scripts/evaluate_model.py --split test
--max-tokens 64`, default output dir):

- `data/model_evaluation_report.txt`
- `data/confusion_matrix.csv` (rows = true labels, columns = predicted)

### Test split (12,284 samples, SemEval-2017 Task 4A test)

| Metric | Value |
|---|---|
| Accuracy | **0.7208** |
| Macro F1 | **0.7201** |
| Weighted F1 | **0.7208** |
| Micro F1 | 0.7208 |

Per class: precision / recall / F1 (support 3,972 / 5,937 / 2,375):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.7196 | 0.7341 | 0.7268 | 3,972 |
| neutral | 0.7241 | 0.7145 | 0.7193 | 5,937 |
| positive | 0.7144 | 0.7141 | 0.7143 | 2,375 |

Confusion matrix (rows=true, cols=predicted): [[2916,995,61],[1078,4242,617],
[58,621,1696]] — row sums equal the per-class supports; accuracy ≈ 0.7208
checks out. The result is consistent with the model's published TweetEval
accuracy (~0.718, cardiffnlp model card), a sanity signal the eval is honest.

### Validation split (2,000 samples, sanity run, not committed to `data/`)

Accuracy **0.7645**, Macro F1 0.7532, Weighted F1 0.7661.

Note: these numbers are for the deployed **int8-quantized** ONNX artifact;
small (<0.005) deviations from the FP32 model card figures are expected.

## Current Errors

- **No runtime errors found.** `compileall` passed and `app.config` imports
  cleanly in the venv.

## Current Blockers

**None.** The model artifact at `app/model_assets/` exists and is verified,
the model is evaluated (TweetEval test split -> accuracy 0.7208), and the
Vercel deployment config (`vercel.json` + `README.md`) is in place.
Remaining work (frontend, initial git commit) is unblocked.

## Vercel Status

**CONFIGURED (files in place, deployment not yet executed).**

- `vercel.json` keys the resolved FastAPI function (`"app/main.py"`) with
  `maxDuration: 300` (valid on every plan; Hobby default and maximum) and
  `excludeFiles: "{tests/**,scripts/**,data/**,**/*.md}"` to slim the bundle.
- Entrypoint: Vercel zero-config-detects FastAPI from `app.main`:app (`app`
  in `app/main.py`, a supported entrypoint). No `api/index.py` was added to
  avoid duplicate-entrypoint ambiguity. `.gitignore` ignores `.vercel/`.
- Grounded in Vercel's current FastAPI/Python docs (fetched this session):
  500 MB uncompressed Python bundle limit (standard) — the ~231 MB int8 model
  + deps fit; Large Functions beta up to 5 GB via
  `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`; Hobby 2 GB RAM / 1 vCPU; 4.5 MB
  request-body limit; lazy ~15-30 s model cold start covered by 300 s.
- Root `README.md` includes full deploy instructions (git-import +
  `vercel dev` / `vercel --prod`) and the env-var table
  (esp. `SENTIMENT_CORS_ORIGINS` for a production frontend origin).
- Blocking-caveat: deployment has **not** been executed from this machine;
  the config follows documented Vercel guidance and must be confirmed with
  `vercel dev` then a real deploy.

## Dependencies

Installed in `.venv` (runtime):
`fastapi 0.141.1`, `uvicorn 0.53.0`, `pydantic 2.13.5`, `onnxruntime 1.30.0`,
`transformers 4.57.6`, `numpy 2.5.3`, `huggingface_hub 0.36.2`,
`tokenizers 0.22.2`, plus transitive deps (starlette 1.7.0, anyio, etc.).

Installed in `.venv` (dev/eval/export) in this session:
`pytest 9.1.1`, `httpx 0.28.1`, `scikit-learn 1.9.1`, `onnx 1.23.0`,
`torch 2.14.0+cpu` (CUDA **not** bundled — verified `+cpu` tag), `optimum 2.1.0`,
`optimum-onnx 0.1.0`, plus new transitive deps (scipy, joblib, iniconfig,
pluggy, pygments, httpcore, jinja2, etc.).

Note: torch was pulled in automatically as an `optimum` dependency and is the
CPU build (`2.14.0+cpu`), so the separate
`pip install torch --index-url https://download.pytorch.org/whl/cpu` step is
**not required** on this machine.

## Environment

- OS: Windows (win32), Python 3.12 (`.python-version`), venv at `.venv/`.
- Git repo initialized on branch `master` but **no commits yet** (all files untracked).
- Git LFS configured in `.gitattributes` but not verified installed.
- No `.env` file present; no secrets exist in the repo.
- Active local dev server origins: `http://localhost:8000`, `http://127.0.0.1:8000`.

## NEXT STEPS

Based on the actual state, in dependency order:

1. **DONE — Install dev dependencies**: `pip install -r requirements-dev.txt`.
   (`pytest`, `httpx`, `scikit-learn`, `onnx`, `optimum`, `optimum-onnx`
   installed; `torch 2.14.0+cpu` pulled implicitly — CPU-only, CUDA not
   bundled, so no separate torch install needed.)
2. **DONE — Write `scripts/prepare_model.py`** — exports
   `cardiffnlp/twitter-roberta-base-sentiment-latest` to int8-quantized ONNX +
   tokenizer into `app/model_assets/`.
3. **DONE — Run the export** — `python scripts/prepare_model.py` produced
   `app/model_assets/` (`model.onnx` ~231 MiB, `config.json`, tokenizer files);
   inline verification passed (logits `(1, 3)`) plus an end-to-end inference
   sanity check. Required two fixes to the script (see "Files Modified"):
   Optimum task name `sequence-classification` → `text-classification`, and
   explicit `from onnxruntime.quantization import QuantType, quantize_dynamic`.
4. **DONE — Implement `app/services/`** inference service
   (`app/services/sentiment.py`). `SentimentAnalyzer` lazily loads the ONNX
   session (CPUExecutionProvider) + `AutoTokenizer` from `config.MODEL_DIR`,
   `predict(text)` returns `Prediction{label, confidence,
   scores:{negative,neutral,positive}}`; `get_analyzer()` provides a
   singleton. Verified with direct inference on 4 texts + input validation
   checks (see Testing Status).
5. **DONE — Implement `app/models/`** Pydantic schemas
   (`app/models/schemas.py`, re-exported from `app.models`).
   `PredictRequest.text` = non-blank string ≤ `MAX_INPUT_CHARS` (2000),
   whitespace-stripped; `PredictResponse` = `{model, label (Literal over
   config.LABELS), confidence ∈ [0,1], scores {negative,neutral,positive}}`
   with `scores` keys validated and `extra="forbid"`. Verified with 12+ schema
   checks plus an integration round-trip from a real service prediction (see
   Testing Status).
6. **DONE — Implement `app/api/`** FastAPI routes and application assembly.
   `app/api/routes.py` exposes `GET /health` (status/model/labels/loaded) and
   `POST /api/predict` (schema-validated, 503 on inference failure);
   `app/main.py` builds the FastAPI app with CORS from `config.CORS_ORIGINS`
   and the router. Verified via `TestClient` (happy path, 422s, CORS echo +
   disallowed origin + preflight, 503 stub, 404/405) and a live `uvicorn`
   smoke test (see Testing Status).
7. **DONE — Write tests** under `tests/` (service smoke test + API tests with
   httpx), then run `pytest`. Suite = `tests/test_schemas.py` (14) +
   `tests/test_sentiment_service.py` (12, incl. `max_tokens`) +
   `tests/test_api.py` (17) + `tests/test_evaluate_script.py` (8) =
   **51 tests. `pytest tests -q` → 51 passed** (also passes per-file).
8. **DONE — Write `scripts/evaluate_model.py`** — download TweetEval into
   `data/dataset/`, run evaluation, emit `data/model_evaluation_report.txt`
   and `data/confusion_matrix.csv`. Ran on the full test split (12,284
   samples): Accuracy **0.7208**, Macro F1 **0.7201**, Weighted F1 **0.7208**
   (see Model Evaluation). `SentimentAnalyzer.predict()` gained an optional
   `max_tokens` kwarg used by the tool; dataset files downloaded + verified
   (train 45,615 / val 2,000 / test 12,284).
9. **DONE — Add Vercel config** (`vercel.json` + serverless entrypoint) and a
   root `README.md` with setup/deploy instructions. Entrypoint = the existing
   `app/main.py` (Vercel auto-detects FastAPI from it; zero-config), so no
   duplicate `api/index.py` was created. `vercel.json` tunes the resolved
   function: `maxDuration` 300 (all plans) + safe `excludeFiles`; `README.md`
   documents setup/run/test/eval and Vercel deploy steps (see Vercel Status).
   `tests/test_vercel_config.py` guards the config (7 tests). Deployment
   itself was **not** run from this machine.
10. **Commit initial state** via git (with Git LFS for the ONNX model) when
    ready.

## RESUME INSTRUCTION

Read OPENCODE_PROGRESS.md first.
Inspect the actual project files.
Do not repeat completed work.
Continue from NEXT STEPS.
Run tests after changes.
Update OPENCODE_PROGRESS.md after every major milestone.