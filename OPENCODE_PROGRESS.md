# Project

AI Sentiment Detector

## Recovery Status

This file reflects the **actual** state of the repository as verified during
the current session. Nothing is assumed from memory; every claim below was
checked on disk / by running the tools. This document was rewritten for the
architecture migration tracked in the rest of this file.

## Current Phase

**Phase 2 — Architecture migration (frontend/backend split).** The
single Vercel FastAPI deployment was replaced by a split architecture:

- **Vercel** → static `frontend/` only (no Python, no model).
- **Render** → FastAPI backend from `backend/` (module `app.main:app`),
  ONNX Runtime, tokenizer, and the existing int8 ONNX model.
- **GitHub** → source of truth; `model.onnx` stays a Git LFS pointer.

The restructure, backend de-static-ification, frontend relocation, CORS rework,
repo reorganization, new deployment configs (`backend/render.yaml`,
`frontend/vercel.json`), test adaptation (**70 tests**, all passing), and live
local verification are **complete**. Pending: local doc polish, `git` commit +
push, and the actual Vercel/Render deployments (blocked on credentials — see
Vercel Status / Render Status).

**Why:** the prior single deploy exceeded Vercel's function size limit
(291.25 MB bundle > 225 MB max) and Git-LFS handling on Vercel delivered the
243 MB model as an unresolved LFS pointer → `POST /api/predict` → HTTP 503
`INVALID_PROTOBUF`.

## Architecture

**Actual** (implemented and live-tested locally):

```
USER
  │ HTTPS
  ▼
[ Vercel: frontend/ (static HTML/CSS/JS, build.js injects API base) ]
  │  POST /api/predict   {"text": "..."}
  ▼
[ Render: backend/ → uvicorn app.main:app (FastAPI) ]
  │  ├─ app/api/routes.py          GET /, GET /health, POST /api/predict
  │  ├─ app/models/schemas.py      request/response validation
  │  ├─ app/services/sentiment.py  ONNX Runtime + tokenizer (lazy singleton)
  │  └─ app/model_assets/          model.onnx (int8, Git LFS) + config/tokenizer
  ▼
{ label, confidence, scores }  →  rendered in the Vercel UI
```

- Backend root = `backend/`; package `app` sits at `backend/app`, so
  `from app import ...` imports are unchanged and `uvicorn app.main:app` runs
  with cwd `backend/`.
- Frontend fetches `${window.SENTIMENT_API_BASE}/api/predict`; the base URL is
  injected at build time by `frontend/build.js` from `VITE_API_URL` into
  `frontend/api-config.js` (git-ignored; dev default `http://localhost:8000`).
- CORS is environment-driven (`FRONTEND_URL` + `SENTIMENT_CORS_ORIGINS`); never
  `*` with credentials. Defaults include `http://localhost:3000`,
  `http://127.0.0.1:3000`, `http://localhost:8000`, `http://127.0.0.1:8000`,
  and `https://sentiment-detector-tau.vercel.app`.

## Repository Layout

```
C:\Python files\nlp model/
├── .gitattributes            LFS: backend/app/model_assets/*.onnx
├── .gitignore                path-independent ignores; frontend/api-config.js
├── .env.example              documented env vars (no secrets)
├── .python-version           3.12
├── LICENSE                   MIT + third-party (cardiffnlp CC BY 4.0, TweetEval CC BY 3.0)
├── README.md                 migration docs: architecture, local dev, Render + Vercel deploy
├── OPENCODE_PROGRESS.md      this file
├── .venv/                    repo-root venv (backend runs `..\.venv\Scripts\python.exe`)
├── frontend/                 → VERCEL (static only)
│   ├── index.html            UI, relative asset URLs, api-config.js script tag
│   ├── styles.css
│   ├── app.js                reads window.SENTIMENT_API_BASE, fetch /api/predict
│   ├── build.js              zero-dep build: VITE_API_URL -> api-config.js
│   ├── favicon.svg
│   ├── vercel.json           {"buildCommand": "node build.js"} (static)
│   └── api-config.js         generated, git-ignored (exists locally: http://localhost:8000)
└── backend/                  → RENDER (FastAPI Web Service)
    ├── requirements.txt      runtime deps (fastapi, uvicorn, pydantic, onnxruntime, transformers, numpy)
    ├── requirements-dev.txt  pytest, httpx, scikit-learn, onnx, optimum, torch+cpu
    ├── render.yaml           Render blueprint (Web Service, python, healthPath /health)
    ├── app/
    │   ├── __init__.py       __version__ = "1.0.0"
    │   ├── config.py         FRONTEND_URL/CORS/MODEL_DIR/MODEL_NAME/MAX_*  (env-driven)
    │   ├── main.py           create_app() + module-level `app` (uvicorn app.main:app)
    │   ├── api/routes.py     GET / (JSON service info), GET /health, POST /api/predict (no StaticFiles)
    │   ├── models/schemas.py PredictRequest/PredictResponse/HealthResponse + validation
    │   ├── services/sentiment.py  lazy ONNX session + AutoTokenizer, softmax, singleton, predict(max_tokens=)
    │   └── model_assets/     model.onnx (int8, LFS) + config.json + tokenizer files
    ├── scripts/              prepare_model.py, evaluate_model.py (+ git-ignored .model_cache/)
    ├── tests/                70 tests (see Testing Status)
    └── data/                 model_evaluation_report.txt, confusion_matrix.csv (TweetEval test)
```

## Completed Work (migration)

- **Phase 0 inspection** produced the migration plan (recorded in the previous
  OPENCODE_PROGRESS.md revision); all numbered migration steps below are done.
- **`.gitattributes`** — LFS rule repointed to
  `backend/app/model_assets/*.onnx` (previously `app/model_assets/*.onnx`).
  `git lfs ls-files` confirms the staged pointer commit:
  `4f4b088781 * backend/app/model_assets/model.onnx` (134-byte pointer in the
  index; the 242,491,205-byte binary stays in the working tree).
- **Repo restructure** (git mv / Move-Item): `app/static/* → frontend/`;
  `app → backend/app`; `tests → backend/tests`; `scripts → backend/scripts`;
  `data → backend/data`; `requirements.txt`, `requirements-dev.txt`,
  `.python-version → backend/`; root `vercel.json` deleted (`git rm`). Staged
  as ~35 renames.
- **`.gitignore`** — now path-independent: `.model_cache/`, `dataset/`,
  `frontend/api-config.js`; `.model_cache` contents mistakenly staged during
  the move were un-staged.
- **Backend de-static-ification**:
  - `backend/app/config.py` — removed `STATIC_DIR`; added `FRONTEND_URL`
    (prepended to CORS origins) and new default CORS origins
    (localhost 3000/8000 + Vercel URL); `BASE_DIR`/`MODEL_DIR` resolve via
    `Path(__file__)` so paths work with cwd = `backend/`.
  - `backend/app/main.py` — removed `StaticFiles` mount (now API-only; no
    `backend/app/static`).
  - `backend/app/api/routes.py` — `GET /` returns JSON service info
    (`service`, `version`, `docs`, `health`, `predict`) instead of serving
    HTML; `GET /health` and `POST /api/predict` contract unchanged.
- **Frontend relocation** — `index.html` uses relative asset URLs (`./styles.css`,
  `./app.js`, `./favicon.svg`) and loads `api-config.js`; `app.js` reads
  `window.SENTIMENT_API_BASE` and fetches `/api/predict`; new `build.js`
  (zero-dep Node) writes `api-config.js` from `VITE_API_URL` (default
  `http://localhost:8000`); new `frontend/vercel.json` (static build command).
  `frontend/api-config.js` verified generated locally.
- **Deployment config** — `.env.example` (all env vars, no secrets);
  `backend/render.yaml` (Web Service blueprint, root `backend/`, build
  `pip install -r requirements.txt`, start
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check `/health`,
  free/oregon, `FRONTEND_URL` env with `sync: false`).
- **Tests adapted** — `backend/tests/test_frontend.py` rewritten (root is JSON
  service info, not HTML; CORS/intra-origin tests: Vercel origin allowed,
  `FRONTEND_URL` prepend behavior, docs/openapi/health availability);
  `backend/tests/test_vercel_config.py` rewritten for the new architecture
  (frontend `vercel.json` valid + no Python `functions`; static build with
  `VITE_API_URL`; root `vercel.json` removed; backend entrypoint exports a
  FastAPI `app`; LFS rule covers `backend/app/model_assets/model.onnx`;
  model + tokenizer files present on disk; no `backend/app/static`).

## Completed Work (pre-migration baseline, still valid)

- `app/services/sentiment.py` — `SentimentAnalyzer` lazily loads int8 ONNX via
  `onnxruntime.InferenceSession` + `AutoTokenizer` from `config.MODEL_DIR`
  (thread-locked, idempotent), `predict(text)` → `Prediction{label, confidence,
  scores}`, process-wide `get_analyzer()` singleton; heavy imports deferred.
- `app/models/schemas.py` — `PredictRequest` (text stripped, 1..2000 chars,
  `extra="forbid"`), `PredictResponse` (model, label Literal, confidence &
  scores in [0,1], scores keys exactly the 3 labels), `HealthResponse`.
- `app/api/routes.py` — `GET /health` (status/model/labels/loaded), `POST
  /api/predict` (422 on bad payloads, 503 on inference failure, 404/405 handled).
- Model export + evaluation tooling — `scripts/prepare_model.py` (run; int8
  ONNX in `model_assets/`) and `scripts/evaluate_model.py` (run; TweetEval
  test accuracy 0.7208 — see Evaluation).
- Baseline commit `fa0f277` (later superseded by commits `08afc56` etc. pushed
  to the `main` branch of `github.com/MukteshMaurya/sentiment-detector`).

## Evaluation (unchanged, real numbers)

`scripts/evaluate_model.py` on the deployed int8 ONNX artifact:
- **Test split** (12,284): accuracy **0.7208**, macro F1 **0.7201**, weighted
  F1 **0.7208**. Confusion rows=true cols=pred:
  `[[2916,995,61],[1078,4242,617],[58,621,1696]]`.
- **Validation split** (2,000): accuracy **0.7645**.
- Artifacts: `backend/data/model_evaluation_report.txt`,
  `backend/data/confusion_matrix.csv`.

## Testing Status

- **`pytest tests -q` from `backend/` → 70 passed** (1 harmless
  `StarletteDeprecationWarning` from `fastapi.testclient`). Run with
  `C:\Python files\nlp model\.venv\Scripts\python.exe` (cwd `backend/`).
  Breakdown: `test_schemas.py` 14, `test_sentiment_service.py` 12,
  `test_api.py` 17, `test_evaluate_script.py` 8, `test_frontend.py` 10,
  `test_vercel_config.py` 9.
- **`python -m compileall -q app scripts tests`** (from `backend/`) → passed.
- **Live backend smoke test** (uvicorn `app.main:app` on 127.0.0.1:8765):
  - `GET /` → `{"service": "AI Sentiment Detector API", "version": "1.0.0",
    "docs": "/docs", ...}` (JSON, not HTML).
  - `GET /docs` → 200; `GET /health` → `{"status": "ok", "loaded": false}`.
  - `POST /api/predict` real inferences:
    - "I absolutely love this product!" → **positive 0.9812**
    - "I hate this product." → **negative 0.9336**
    - "The product arrived today." → **positive 0.8294** (actual model output;
      the model does not produce neutral here)
  - CORS: `Origin: https://sentiment-detector-tau.vercel.app` echoed in
    `access-control-allow-origin` (200); disallowed `https://evil.example`
    NOT echoed; preflight OPTIONS from the Vercel origin → 200 with ACAO +
    allow-methods.
- **Live frontend↔backend bridge** (static `http.server` on 8766 + README-style
  `FRONTEND_URL=http://localhost:8766` backend on 8765):
  - `GET /index.html` → 200, contains the sentiment form + `api-config.js`
    tag; `GET /api-config.js` → generated script with
    `window.SENTIMENT_API_BASE = "http://localhost:8000"` (dev default).
  - Cross-origin `POST /api/predict` from origin `http://localhost:8766` →
    200, ACAO echoed, label positive 0.9812. **The split deploy path is proven
    locally.**

## Vercel Status

**CONFIGURED; deployment requires user action (no VERCEL_TOKEN / logged-in
Vercel CLI on this machine).**

- Vercel project: `sentiment-detector-tau` (URL
  https://sentiment-detector-tau.vercel.app) on branch `main` of
  `github.com/MukteshMaurya/sentiment-detector`. Current deployed commit is
  `08afc56` (pre-migration): it served the static UI but `/api/predict`
  returned 503 due to the unresolved LFS-pointer model.
- Migration requirements once pushed:
  1. Project → Settings → Root Directory = **`frontend/`** (static build,
     `node build.js` per `frontend/vercel.json`).
  2. Add `VITE_API_URL` env var = the Render backend URL (e.g.
     `https://sentiment-detector-backend.onrender.com`).
  3. Redeploy → artifact = 4 static files, **no Vercel Function, no
     `model.onnx`**.

## Render Status

**CONFIGURED via `backend/render.yaml`; deployment requires user action (no
Render credentials on this machine).**

- Create a `backend/` Root-Directory Web Service (or Blueprint) → uses
  `backend/render.yaml`: build `pip install -r requirements.txt`, start
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check `/health`.
- Render fetches Git LFS objects automatically at build; the ONNX model arrives
  as the real 231 MB binary.
- Set `FRONTEND_URL` = the Vercel frontend origin (default already
  `https://sentiment-detector-tau.vercel.app`), so CORS allows the UI.

## Git State

- Branch `main`; remote `github.com/MukteshMaurya/sentiment-detector`; HEAD on
  the remote = `08afc56`.
- **The migration is NOT yet committed.** Index currently holds the restructured
  tree (≈35 `R` rename entries + `D vercel.json`, with the `model.onnx` index
  blob being the 134-byte LFS pointer `4f4b088781`); working-tree edits made
  after staging (`backend/app/config.py`, `main.py`, `api/routes.py`,
  new `frontend/build.js`, `frontend/vercel.json`, `.env.example`,
  `backend/render.yaml`, `.gitignore`, `.gitattributes`, new README,
  `backend/tests/*`) need `git add` before commit.
- `.venv/`, `backend/scripts/.model_cache/`, `backend/data/dataset/`, and
  `frontend/api-config.js` are git-ignored; no secrets exist in the repo.
- Next: re-`git add`, commit with a migration message, `git push origin main`,
  verify `git lfs ls-files` still lists the pointer.

## Current Errors / Blockers

- **None in the code base.** compileall + 70 tests + live smoke tests pass.
- **Blocked on deployment credentials:** no `VERCEL_TOKEN` (the user selected
  "Provide a VERCEL_TOKEN" but none was supplied) and no Render credentials.
  The Vercel CLI (v60.0.1 via `npx vercel`) is logged out.

## Environment

- OS: Windows (win32); PowerShell 5.1; working dir `C:\Python files\nlp model`.
- Python 3.12 (`.python-version`); venv at repo root `.venv`
  (`C:\Python files\nlp model\.venv`); backend runs with cwd = `backend/`.
- Node v24.21.0 available (used by `frontend/build.js`).
- git + git-lfs installed and verified; `.python-version` apps per above.

## NEXT STEPS

1. **Re-stage and commit the migration** — `git add -A` (picks up the edits
   made after the initial `git add`), then commit with a message describing the
   split (e.g. "Migrate to Vercel frontend + Render backend"). Verify
   `git lfs ls-files` still shows `4f4b088781 * backend/app/model_assets/model.onnx`.
2. **Push** to `origin main` (do not force-push).
3. **Deploy (user action — agent is blocked on credentials):**
   - Render: create Web Service/Blueprint from `backend/`; set `FRONTEND_URL`.
   - Vercel: set Root Directory `frontend/`; set `VITE_API_URL`; redeploy.
   - Agent can then verify prod endpoints if network access allows.
4. **Final self-check** — report PASS/FAIL honestly; anything not verifiable
   without deployment credentials is marked as such.

## RESUME INSTRUCTION

Read OPENCODE_PROGRESS.md first.
Inspect the actual project files (do not trust memory).
Do not repeat completed work; start from git state (uncommitted migration).
Run tests after changes (`cd backend; ..\.venv\Scripts\python.exe -m pytest tests -q`).
Update OPENCODE_PROGRESS.md after every major milestone.