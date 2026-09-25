# AI Sentiment Detector

Sentiment classification for English text (Positive / Neutral / Negative)
backed by a `cardiffnlp/twitter-roberta-base-sentiment-latest` model exported
to an **int8-quantized ONNX** graph (~231 MB) that runs on **ONNX Runtime**
(no PyTorch at request time).

## Architecture

```
                        USER
                          │
                          ▼
               ┌────────────────────┐
               │      VERCEL        │  static HTML/CSS/JS (no Python, no model)
               │  frontend/         │
               └─────────┬──────────┘
                         │ HTTPS POST /api/predict
                         ▼
               ┌────────────────────┐
               │      RENDER        │  FastAPI Web Service from backend/
               │  backend/app       │  uvicorn app.main:app
               │  ONNX Runtime      │
               │  tokenizer         │
               │  model.onnx        │
               └─────────┬──────────┘
                         │
                         ▼
                   Positive / Neutral / Negative
                         │
                         ▼
                        VERCEL UI
```

- **Vercel** — frontend only. Deploys the static files in `frontend/`. It never
  packages FastAPI, ONNX Runtime, or `model.onnx`.
- **Render** — the FastAPI backend from `backend/` (module `app.main:app`),
  including ONNX Runtime, the tokenizer/config, and the existing ONNX model.
- **GitHub** — source of truth for both; `model.onnx` is stored via Git LFS.
- The frontend calls the Render backend at `POST /api/predict` with
  `{"text": "..."}` and receives `{model, label, confidence, scores}`.

> The previous single-run Vercel deployment (FastAPI + 231 MB model as one
> Vercel Python function) was abandoned because the bundle exceeded Vercel's
> function size limit (291 MB > 225 MB) and the Git-LFS model arrived as an
> unresolved pointer (HTTP 503).

## Repository layout

```
├── frontend/                 → deployed on Vercel (static only)
│   ├── index.html            UI: title, textarea, counter, Analyze, result, errors
│   ├── styles.css
│   ├── app.js                calls VITE_API_URL + /api/predict
│   ├── build.js              zero-dep build: injects VITE_API_URL -> api-config.js
│   ├── favicon.svg
│   └── vercel.json           static build command (node build.js)
├── backend/                  → deployed on Render (FastAPI Web Service)
│   ├── app/
│   │   ├── main.py           FastAPI assembly: CORS + router  (module app.main:app)
│   │   ├── config.py         Environment-driven configuration (incl. FRONTEND_URL / CORS)
│   │   ├── api/routes.py     GET / (service info), GET /health, POST /api/predict
│   │   ├── models/schemas.py Pydantic request/response schemas + validation
│   │   ├── services/sentiment.py  ONNX Runtime + tokenizer, softmax, singleton
│   │   └── model_assets/     model.onnx (int8, Git LFS) + config.json + tokenizer
│   ├── requirements.txt      Production dependencies only
│   ├── requirements-dev.txt  Tests / evaluation / model-export tooling
│   ├── render.yaml           Render blueprint (Web Service)
│   ├── scripts/              Dev tools: prepare_model.py, evaluate_model.py
│   ├── tests/                pytest suite
│   └── data/                 TweetEval evaluation reports
├── .env.example              Documented environment variables (no secrets)
├── .gitattributes            Git LFS for backend/app/model_assets/*.onnx
└── README.md
```

## Local development

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv .venv          # or reuse an existing venv
# Windows PowerShell: ..\.venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # optional: tests, evaluation
uvicorn app.main:app --reload --port 8000
```

Verify:

- `http://127.0.0.1:8000/` → JSON service info (`{"service": "AI Sentiment
  Detector API", ...}`)
- `http://127.0.0.1:8000/docs` → interactive docs
- `GET /health` → `{"status": "ok", "model": ..., "labels": ..., "loaded": false}`
- `POST /api/predict` with `{"text": "I absolutely love this product!"}` →
  `{"model": ..., "label": "positive", "confidence": 0.98, "scores": {...}}`

`loaded` flips to `true` after the first prediction (the model loads lazily).

### 2. Frontend (static, served separately)

```bash
cd frontend
node build.js                # writes api-config.js (default: http://localhost:8000)
npx serve -l 3000            # any static server on a CORS-allowed port works
```

Open `http://localhost:3000`. The default CORS origins include
`http://localhost:3000` and `http://localhost:8000`, so a local frontend
served on either port can call a local backend on port 8000.

To point the frontend at a different backend:

```
VITE_API_URL=http://localhost:8765 npx serve -l 3000
# or run `node build.js` once with VITE_API_URL set before serving
```

The generated `frontend/api-config.js` is git-ignored.

### 3. Tests

```bash
cd backend
pytest tests -q               # 70 tests: schemas, real-model service, API, frontend/CORS, eval, deploy config
```

## Environment variables

See `.env.example`. All values are optional (safe defaults in code).

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `FRONTEND_URL` | Render | `https://sentiment-detector-tau.vercel.app` | Vercel origin, prepended to CORS allowlist |
| `SENTIMENT_CORS_ORIGINS` | Render | localhost:3000/8000 + Vercel URL | Full override of allowed CORS origins |
| `SENTIMENT_MODEL_DIR` | Render | `app/model_assets` (relative to `backend/`) | Directory with `model.onnx` + tokenizer |
| `SENTIMENT_MODEL_NAME` | Render | `cardiffnlp/... (ONNX int8)` | Label shown in responses / health |
| `SENTIMENT_MAX_INPUT_CHARS` | Render | `2000` | Max request `text` length |
| `SENTIMENT_MAX_TOKENS` | Render | `256` | Tokenizer padding/truncation length |
| `SENTIMENT_LOG_LEVEL` | Render | `INFO` | Logging level |
| `VITE_API_URL` | Vercel build | `http://localhost:8000` | Base URL of the Render backend |

No API keys are used; the model runs locally. Do not commit `.env` files.

## Render deployment

**Root directory:** `backend/`
**Build command:** `pip install -r requirements.txt`
**Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
**Health check path:** `/health`
**Runtime:** Python 3.12

A `backend/render.yaml` blueprint is included. Either:

1. **Blueprint:** Render → New → Blueprint → select this repo, or
2. **Web Service:** Render → New → Web Service → select this repo, set Root
   Directory to `backend/`, environment `python`, the build/start commands
   above.

Deploy is triggered on git push to the linked branch or via the Dashboard.
Git LFS objects are fetched automatically by Render's build, so
`backend/app/model_assets/model.onnx` arrives as the real 231 MB binary.

Set `FRONTEND_URL` to the real Vercel domain before/after deploy
(env vars can be edited later; a redeploy/redeploy with the variable in place
applies it).

## Vercel deployment

**Root directory:** `frontend/`
**Framework:** Other (plain static)
**Build command:** `node build.js` (from `frontend/vercel.json`)
**Output directory:** `.` (the frontend dir itself)
**Environment variable:** `VITE_API_URL` set to the Render service URL, e.g.
`https://sentiment-detector-backend.onrender.com`

1. Push this repo to GitHub.
2. In your Vercel project, set Root Directory to `frontend/`.
3. Verify the old Python-function settings are gone — Vercel now builds a
   static site; it never sees `backend/` or `model.onnx`.
4. Add `VITE_API_URL` under Project → Settings → Environment Variables.
5. Redeploy.

Verify the deployment no longer packages the model: the deploy artifact is the
four static files in `frontend/`, no Vercel Function, no `model.onnx`.

## API contract (unchanged)

The frontend/backend contract is preserved from the original app:

```
POST /api/predict
Content-Type: application/json
{"text": "I absolutely love this product!"}

{
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)",
  "label": "positive",
  "confidence": 0.9812,
  "scores": {"negative": 0.0065, "neutral": 0.0123, "positive": 0.9812}
}
```

- `text` is required, whitespace-stripped, ≤ `MAX_INPUT_CHARS` (2000).
- Invalid payloads → **422**; failed predictions (missing model / ONNX error) → **503**.
- `GET /health` reports service + model status (model loads lazily).
- CORS allowlist is environment-driven (`FRONTEND_URL` / `SENTIMENT_CORS_ORIGINS`);
  never `*` with credentials.

## Model

- Artifact: `backend/app/model_assets/model.onnx` — int8-quantized,
  ~231 MB, tracked with **Git LFS** via `.gitattributes`
  (`backend/app/model_assets/model.onnx filter=lfs ...`).
- `git lfs ls-files` → `4f4b088781 * backend/app/model_assets/model.onnx`.
- Do **not** replace, retrain, or commit the model as a normal Git blob.
- Evaluation (dev tool): `cd backend && python scripts/evaluate_model.py --split test --max-tokens 64`
  → TweetEval test accuracy **0.7208** (see `backend/data/model_evaluation_report.txt`).

## License and attribution

- Project code: MIT (`LICENSE`).
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` — `CC BY 4.0`.
- Evaluation data: TweetEval `sentiment` (SemEval-2017 Task 4A), `CC BY 3.0`.