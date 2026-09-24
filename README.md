# AI Sentiment Detector

Local sentiment classification for English text. It serves a
`cardiffnlp/twitter-roberta-base-sentiment-latest` sentiment model exported to
an **int8-quantized ONNX** graph (~231 MB) and runs inference on **ONNX
Runtime** (no PyTorch at request time), wrapped in a minimal FastAPI service.

The same FastAPI instance is used by the local dev server, the automated
tests, and the (zero-config) [Vercel serverless
deployment](#deploy-to-vercel).

## What's here

```
├── app/
│   ├── main.py              FastAPI assembly: CORS + router (the Vercel entrypoint)
│   ├── config.py            Environment-driven configuration
│   ├── api/routes.py        GET /health, POST /api/predict
│   ├── models/schemas.py    Pydantic request/response schemas + validation
│   ├── services/sentiment.py  ONNX Runtime + tokenizer service, softmax, singleton
│   └── model_assets/        model.onnx (int8) + config.json + tokenizer files
├── scripts/
│   ├── prepare_model.py     One-time exporter to int8-quantized ONNX (dev tool)
│   └── evaluate_model.py    TweetEval evaluation (dev tool, see below)
├── data/                    Generated evaluation reports + downloaded dataset
└── tests/                    pytest suite (schema / service / API / eval / vercel)
```

## Prerequisites

- Python **3.12** (pin in `.python-version`; Vercel's Python runtime uses 3.12
  by default)
- `git` and (if deploying) the Vercel CLI `>= 48.1.8`

## Local setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional, for tests, model export, and evaluation:
pip install -r requirements-dev.txt
```

## Run the API locally

```bash
uvicorn app.main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health check:

```
GET /health
```

```json
{
  "status": "ok",
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)",
  "labels": ["negative", "neutral", "positive"],
  "loaded": false
}
```

`loaded` flips to `true` after the first prediction (the ~231 MB model is
loaded lazily on first use).

- Predict sentiment:

```bash
curl -i http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is absolutely amazing!"}'
```

```json
{
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)",
  "label": "positive",
  "confidence": 0.9801,
  "scores": { "negative": 0.0015, "neutral": 0.0184, "positive": 0.9801 }
}
```

`text` is required, whitespace-stripped, and limited to `MAX_INPUT_CHARS`
(2000) characters. Invalid payloads return **422**; a failed prediction
(model missing, ONNX/runtime error) returns **503**.

## Tests

```bash
pytest tests -q        # 58 tests: schemas, real-model service, API, eval, vercel config
```

## Model export (one-time, dev only)

```bash
python scripts/prepare_model.py
```

Downloads the model from Hugging Face, exports FP32 ONNX via Optimum, applies
int8 dynamic (weights-only) quantization, and writes
`app/model_assets/model.onnx` + `config.json` + tokenizer files. Verification
(inference sanity + logits shape) runs inside the script. PyTorch is only
needed for this step (`requirements-dev.txt`).

## Evaluation (dev only)

```bash
python scripts/evaluate_model.py --split test --max-tokens 64
```

Downloads TweetEval `sentiment` (SemEval-2017 Task 4A) into `data/dataset/`,
runs the deployed int8 model through `app/services`, and writes
`data/model_evaluation_report.txt` + `data/confusion_matrix.csv`.

Real results (already generated in this repo):

| Dataset | Samples | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| TweetEval test | 12,284 | **0.7208** | 0.7201 | 0.7208 |
| TweetEval val | 2,000 | 0.7645 | 0.7532 | 0.7661 |

(Token cap 64; TweetEval tweets are short, so the cap truncates ~nothing.)

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `SENTIMENT_MODEL_DIR` | `app/model_assets` (repo-relative) | Directory with `model.onnx` + tokenizer files |
| `SENTIMENT_MODEL_NAME` | `cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)` | Label shown in responses / health |
| `SENTIMENT_MAX_INPUT_CHARS` | `2000` | Max request `text` length |
| `SENTIMENT_MAX_TOKENS` | `256` | Tokenizer padding/truncation length |
| `SENTIMENT_CORS_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Comma-separated allowed origins (see CORS note) |
| `SENTIMENT_LOG_LEVEL` | `INFO` | Logging level |

No API keys are used; the model runs locally.

## Deploy to Vercel

Vercel detects FastAPI automatically: it looks for a `FastAPI` instance named
`app` in `app/main.py` (a supported entrypoint) and serves the whole app as a
single Vercel Function. `vercel.json` only tunes that function and slims the
bundle.

```json
// vercel.json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "functions": {
    "app/main.py": {
      "maxDuration": 300,
      "excludeFiles": "{tests/**,scripts/**,data/**,**/*.md}"
    }
  }
}
```

- **Entrypoint:** `app/main.py` exports `app` — the same object used by
  `uvicorn app.main:app` and the tests, so nothing Vercel-specific is shipped
  in code.
- **maxDuration:** 300 s. The 231 MB model loads lazily on each cold function
  instance; 300 s is the Hobby default *and* maximum and comfortably covers
  cold-start + inference. (Pro/Enterprise allow up to 800 s.)
- **Bundle size:** the Python function bundle limit is **500 MB uncompressed**
  (standard). Model (~231 MB) + `onnxruntime`/`transformers`/FastAPI stay
  well under that on the standard path. If you ever exceed it, set
  `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` (Large Functions beta, up to 5 GB, Fluid
  compute).
- **CORS:** the server allows only `SENTIMENT_CORS_ORIGINS`. You must set
  that env var to the real frontend origin in production (it defaults to
  localhost only). Do **not** allow credentials with `*`.
- **Memory:** Hobby gives 2 GB RAM / 1 vCPU, which is enough to hold the model
  plus dependencies.

### From a Git repository (recommended)

1. Push this repo to GitHub/GitLab/Bitbucket.
2. Import it at https://vercel.com/new — Vercel detects the FastAPI framework.
3. Set project environment variables (dashboard → Settings → Environment
   Variables), e.g. `SENTIMENT_CORS_ORIGINS` for your frontend origin.
4. Deploy. Get the deployment URL and call:

```bash
curl -i -X POST "<project>.vercel.app/api/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text": "I hate this."}'
```

### From the Vercel CLI

```bash
npm i -g vercel          # requires CLI >= 48.1.8
vercel dev               # local, mimics the serverless runtime (http://localhost:3000)
vercel --prod            # deploy
vercel env add SENTIMENT_CORS_ORIGINS production
```

> Note: deployment has **not** been executed from this machine; the config
> follows Vercel's documented FastAPI guidance. Confirm with `vercel dev`
> locally, then deploy.

## License and attribution

- Project code: MIT (`LICENSE`).
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` — license `CC BY
  4.0` (see `LICENSE` for the model notice).
- Evaluation data: TweetEval `sentiment` (SemEval-2017 Task 4A), `CC BY 3.0`.