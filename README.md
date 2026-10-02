# AI Sentiment Detector

Sentiment classification for English text (Positive / Neutral / Negative)
backed by a `cardiffnlp/twitter-roberta-base-sentiment-latest` model exported
to an **int8-quantized ONNX** graph (~231 MB) that runs on **ONNX Runtime**
(no PyTorch at request time).

**Hindi / Hinglish input is supported too.** Devanagari Hindi and
Romanized Hinglish are translated to English by
`Helsinki-NLP/opus-mt-hi-en` (ONNX fp16, ~264 MB) *before* sentiment
analysis, so the same classifier, the same three labels and the same
confidence semantics apply to every language. English requests skip this
stage entirely and never load the translation model.

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

### Request flow

```
POST /api/predict {"text": "..."}
        │
        ▼
   detect_language(text)                     app/services/translation.py
        │
        ├─ "en"  ─────────────────────────────▶ unchanged (translation model NOT loaded)
        ├─ "hi"  ────▶ opus-mt-hi-en ONNX ──┐
        └─ "hinglish" ─▶ normalize to      │
                           Devanagari ────┘
        │
        ▼
   SentimentAnalyzer.predict(english)       app/services/sentiment.py (unchanged)
        │
        ▼
   {model, label, confidence, scores, original_text, translated_text, translation_status}
```

- **Vercel** — frontend only. Deploys the static files in `frontend/`. It never
  packages FastAPI, ONNX Runtime, or `model.onnx`.
- **Render** — the FastAPI backend from `backend/` (module `app.main:app`),
  including ONNX Runtime, the tokenizer/config, and the existing ONNX model.
- **GitHub** — source of truth for both; the ONNX models are stored via Git LFS.
- The frontend calls the Render backend at `POST /api/predict` with
  `{"text": "..."}` and receives `{model, label, confidence, scores}` plus the
  optional translation fields.

> The previous single-run Vercel deployment (FastAPI + 231 MB model as one
> Vercel Python function) was abandoned because the bundle exceeded Vercel's
> function size limit (291 MB > 225 MB) and the Git-LFS model arrived as an
> unresolved pointer (HTTP 503).

## Deployment status

| Component | URL | State |
|---|---|---|
| GitHub | https://github.com/MukteshMaurya/sentiment-detector | **Deployed** (branch `main`, Git LFS model) |
| Backend (Render) | *(not deployed yet — pending Render account)* | NOT DEPLOYED |
| Frontend (Vercel) | `https://sentiment-detector-tau.vercel.app` (existing), new migration deploy pending | NOT DEPLOYED (migration) |

Production deployments on Render and Vercel require account credentials and
were **not** run from this machine. Until then the local, fully-tested flow
(backend on `http://localhost:8000`, static frontend on a CORS-allowed origin)
is authoritative. After deploying, set the backend's `FRONTEND_URL` and the
frontend's `VITE_API_URL` to the real URLs and follow the End-to-End test
below.

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
│   │   ├── services/translation.py language detection, Hinglish normalization, MT
│   │   ├── model_assets/     model.onnx (int8, Git LFS) + config.json + tokenizer
│   │   └── translation_assets/  opus-mt-hi-en fp16 ONNX + Marian tokenizer (Git LFS)
│   ├── requirements.txt      Production dependencies only
│   ├── requirements-dev.txt  Tests / evaluation / model-export tooling
│   ├── render.yaml           Render blueprint (Web Service)
│   ├── scripts/              Dev tools: prepare_model.py, prepare_translation.py,
│   │                         evaluate_model.py
│   ├── tests/                pytest suite
│   └── data/                 TweetEval evaluation reports
├── .env.example              Documented environment variables (no secrets)
├── .gitattributes            Git LFS for backend/app/{model,translation}_assets/*.onnx
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
pytest tests -q               # 128 tests: schemas, real-model service, API, frontend/CORS,
                              # eval, deploy config, Hindi translation
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
| `TRANSLATION_ENABLED` | Render | `1` | Master switch for Hindi/Hinglish detection + translation |
| `TRANSLATION_MODEL_DIR` | Render | `app/translation_assets` | Directory with the fp16 ONNX graphs + Marian tokenizer |
| `TRANSLATION_MODEL_NAME` | Render | `Helsinki-NLP/opus-mt-hi-en (ONNX fp16)` | Reported in logs |
| `TRANSLATION_MAX_SOURCE_TOKENS` | Render | `256` | Encoder input limit |
| `TRANSLATION_MAX_NEW_TOKENS` | Render | `128` | Decoder output limit |
| `VITE_API_URL` | Vercel build | `http://localhost:8000` | Base URL of the Render backend |

No API keys are used; the model runs locally. Do not commit `.env` files.

## Render deployment

**Root directory:** `backend/`
**Build command:** `pip install -r requirements.txt`
**Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
**Health check path:** `/health`
**Runtime:** Python 3.12
**Backend URL:** *(placeholder — set after the Render service is created,
e.g. `https://sentiment-detector-backend.onrender.com`)*

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
**Environment variable:** `VITE_API_URL` — the Render backend URL, in the form
`https://<your-render-service>.onrender.com`. It is injected into the page at
build time by `frontend/build.js` (into the git-ignored `api-config.js`).
It is **public browser-visible configuration** — never put secrets or API keys
in a `VITE_*` variable.

1. Push this repo to GitHub.
2. In your Vercel project, set Root Directory to `frontend/`.
3. Verify the old Python-function settings are gone — Vercel now builds a
   static site; it never sees `backend/` or `model.onnx`.
4. Add `VITE_API_URL` under Project → Settings → Environment Variables.
5. Redeploy.

Verify the deployment no longer packages the model: the deploy artifact is the
four static files in `frontend/`, no Vercel Function, no `model.onnx`.

## Production end-to-end verification

After both services are live:

1. `GET https://<render>.onrender.com/health` → `{"status": "ok", ...}`.
2. `POST https://<render>.onrender.com/api/predict` with `{"text": "I absolutely
   love this product!"}` → valid JSON with `label`/`confidence`/`scores`.
3. Open the Vercel URL, enter the three sample texts, confirm the request in
   DevTools goes to the Render URL (no CORS/404/500 errors) and the result +
   confidence appear.
4. First request after idle triggers a Render cold start (free plan): allow
   extra time; a slow-but-successful first request is expected, not a failure.

## API contract

The original contract is preserved unchanged — the first four fields are
byte-for-byte what they always were, and every new field is optional and
nullable:

```
POST /api/predict
Content-Type: application/json
{"text": "I absolutely love this product!"}

{
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX int8)",
  "label": "positive",
  "confidence": 0.9812,
  "scores": {"negative": 0.0065, "neutral": 0.0123, "positive": 0.9812},
  "original_text": "I absolutely love this product!",
  "translated_text": null,
  "translation_status": "not_needed"
}
```

- `text` is required, whitespace-stripped, ≤ `MAX_INPUT_CHARS` (2000).
- `model`, `label`, `confidence`, `scores` — **unchanged**, always present.
- `original_text` — exactly the submitted text (post-validation).
- `translated_text` — English rendering, or `null` for English input.
- `translation_status` — `not_needed` (English), `translated`, or `failed`.
- Invalid payloads → **422**; failed predictions (missing model / ONNX error) → **503**.
- `GET /health` is **unchanged** (same four fields) — it does not report
  translation state, so nothing existing needs to know about it.
- CORS allowlist is environment-driven (`FRONTEND_URL` / `SENTIMENT_CORS_ORIGINS`);
  never `*` with credentials.

### Translation failure is not a request failure

`prepare_text()` never raises. If the translation assets are missing or the
decoder errors, the API logs a warning, sets `translation_status: "failed"`,
analyzes the **original** text and still returns **200** with a normal
sentiment result. It does not fabricate a translation, and it never turns a
valid prediction into a 5xx.

## Hindi / Hinglish support

### Detection

`detect_language()` is deliberately conservative, because mislabeling English
as Hindi would be a regression for every existing user:

| Signal | Result |
|---|---|
| Any Devanagari codepoint | `hi` |
| ≥ 1 unambiguous Hinglish marker (`nahi`, `mera`, `achha`, `kharab`, `bilkul`, …) | `hinglish` |
| ≥ 2 weak markers (`hai`, `phir`, `wala`, `kaise`, …) **and** the sentence does not read as English | `hinglish` |
| Anything else | `en` (untouched) |

There are 20 cases in `tests/test_translation.py` that assert the `en` path —
11 English sentences, 4 empty/whitespace/ASCII edge cases, and 5 sentences
containing names or generic words — including traps like *"The package arrived
three days late and the box was crushed"*, *"The food was amazing but the
waiter was rude"* and *"My friend Mukesh told me the name of the restaurant"*.
`tests/test_api.py` adds 2 more at the API level
(`test_predict_english_reports_not_needed` and
`test_predict_english_with_a_person_name_is_not_translated`).

Proper nouns (`Mukesh`, `Vijay`) were removed from the marker lists: a person's
name carries no language information, and treating it as Hindi misrouted
ordinary English. The weak-marker rule additionally requires the sentence *not*
to contain clearly English words, so an English sentence that merely happens to
contain two Hindi-looking words stays on the English path.

### Normalization: why a lexicon, not blind transliteration

Romanized Hinglish → Devanagari is **not** a mechanical transliteration, and
this was the hardest part of the feature. Measured against the real model:

| Roman input | Normalized as | Result |
|---|---|---|
| `product` | `उत्पाद` (mapped) | *"This product is very bad"* ✅ |
| `product` | `प्रोडक्ट` (transliterated) | *"This **procreator** is very bad"* ❌ |
| `product` | `product` (left Latin) | *"This is very bad"* ⚠️ (noun lost) |
| `reply` | `उत्तर` (mapped) | *"Please **answer** quickly"* ✅ |
| `reply` | `रेप्ल्य` (transliterated) | *"Please answer quickly, come from the **taber**"* ❌ |

`opus-mt-hi-en` was trained on clean Devanagari, so it has never seen
`प्रोडक्ट` or half-Latin sentences. So `_HINDI_TOKENS` in
`app/services/translation.py` is a curated map of function words, sentiment
vocabulary and loanwords, each written the way the model saw it in training
(`product`→`उत्पाद`, `reply`→`उत्तर`, `kharab`→`खराब`).

Anything **not** in the map is left in Latin on purpose. That failure mode is
benign in both directions — an unknown Hinglish word stays untranslated, but
an unknown English word never becomes Devanagari gibberish. A `_LATIN_PRESERVE`
set is checked first so English function words (`the`, `is`, `a`, `would`, …)
are never touched.

### Model choice and quantization results

`Helsinki-NLP/opus-mt-hi-en` (77M params, MarianMT, Apache-2.0). IndicTrans2
was rejected: ~200M params would be ~800 MB fp32 on top of the 231 MB
classifier, which does not fit a CPU-only free-tier box.

Three export formats were built and benchmarked on the real model:

| Format | Size | Quality | Verdict |
|---|---|---|---|
| **FP16** (ORT `convert_float_to_float16`, `keep_io_types=True`) | **264 MB** | **identical to FP32** | **shipped** |
| FP32 | 528 MB | reference | too large |
| int8 dynamic quantization | 106 MB | repeated/garbled output, ~14 s/sentence | rejected |
| CTranslate2 int8 / fp16 | 74–145 MB | tokenization correct, output degenerate | rejected |
| `onnxconverter-common` fp16 | — | produces invalid graphs (`Sub` type mismatch) | rejected |

The shipped decoder has no KV cache: the encoder runs once and the decoder is
re-evaluated over the growing prefix, so only **two** graphs are needed
(`encoder_model.onnx` 96 MB + `decoder_model.onnx` 168 MB) instead of the four
Optimum emits. Cost is ~0.2–1.2 s per sentence on CPU, which is acceptable
for a review-length input.

### Cost and deployment impact

- **English requests are unaffected**: detection is pure Python string work and
  the translation sessions are never created, so no extra memory, no extra
  ~40 s `transformers` import, no extra latency.
- **The first Hindi request in a process pays ~50 s**: ~40 s one-time
  `transformers` import plus ~8 s to build both ONNX sessions. Subsequent
  requests are sub-second. On Render's free plan this lands on the first
  request after a cold start; allow for it in monitoring.
- **Disk/RAM**: total ONNX payload is 231 MB (sentiment) + 264 MB
  (translation) ≈ **495 MB**. Render's free web service has roughly 512 MB RAM,
  so this is tight — a paid instance is advisable. Both models are lazily
  loaded, so a process that only ever sees English text stays small.
- `TRANSLATION_ENABLED=0` disables detection and translation entirely, leaving
  the original English-only behaviour (and asset footprint).

### Regenerating the assets

```bash
cd backend
python scripts/prepare_translation.py          # needs requirements-dev.txt
```

Downloads the model into `scripts/.model_cache/` (git-ignored), exports FP32
ONNX, converts to FP16, writes `app/translation_assets/`, then verifies the
graph signatures and runs a smoke translation. `sentencepiece` was added to
`requirements.txt` because `transformers`' `MarianTokenizer` requires it; the
`indic-transliteration` experiment was dropped once the lexicon approach won.

## Model

### Sentiment classifier (unchanged)

- Artifact: `backend/app/model_assets/model.onnx` — int8-quantized,
  ~231 MB, tracked with **Git LFS** via `.gitattributes`
  (`backend/app/model_assets/model.onnx filter=lfs ...`).
- `git lfs ls-files` → `4f4b088781 * backend/app/model_assets/model.onnx`.
- Do **not** replace, retrain, or commit the model as a normal Git blob.
- Evaluation (dev tool): `cd backend && python scripts/evaluate_model.py --split test --max-tokens 64`
  → TweetEval test accuracy **0.7208** (see `backend/data/model_evaluation_report.txt`).

### Hindi → English translator (new)

- Artifacts: `backend/app/translation_assets/encoder_model.onnx` (96 MB) and
  `decoder_model.onnx` (168 MB) — FP16, tracked with **Git LFS** via
  `backend/app/translation_assets/*.onnx filter=lfs ...`.
- Model: `Helsinki-NLP/opus-mt-hi-en`. Regenerate with
  `python scripts/prepare_translation.py`.

## License and attribution

- Project code: MIT (`LICENSE`).
- Sentiment model: `cardiffnlp/twitter-roberta-base-sentiment-latest` — `CC BY 4.0`.
- Translation model: `Helsinki-NLP/opus-mt-hi-en` — `Apache-2.0` (CC0 for the
  underlying OPUS-MT models).
- Evaluation data: TweetEval `sentiment` (SemEval-2017 Task 4A), `CC BY 3.0`.