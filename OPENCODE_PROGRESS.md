# Project

AI Sentiment Detector

## Recovery Status

This file reflects the **actual** state of the repository as verified during
the current session. Nothing is assumed from memory; every claim below was
checked on disk / by running the tools. This document was rewritten for the
architecture migration tracked in the rest of this file.

## Current Phase

**Phase 5 — Vercel Frontend: COMPLETED (code + local verification).** Phase 4
(Render backend) and Phase 5 (Vercel frontend) code/configuration are done,
committed head is `05cdda5`, and both were verified locally. Production
deployments are NOT performed and NOT claimed: Render and Vercel both require
user credentials. See the `PHASE 4` / `PHASE 5` sections below.

**Why the split:** the prior single deploy exceeded Vercel's function size
limit (291.25 MB bundle > 225 MB max) and Git-LFS handling on Vercel delivered
the 243 MB model as an unresolved LFS pointer → `POST /api/predict` → HTTP 503
`INVALID_PROTOBUF`.

## PHASE 4 — RENDER BACKEND

Status: **COMPLETED** (locally verified; deployment requires user credentials).

Phase 4 was already scaffolded by the migration (commit `05cdda5`). This
session re-verified every requirement against the actual code (no assumptions),
ran the render-style start command, re-ran the full test suite, and proved ONNX
load + inference from a neutral working directory.

- **Backend entrypoint:** `backend/app/main.py` → module-level
  `app = create_app()` (FastAPI). Module path `app.main:app` (cwd
  `backend/`).
- **Backend directory (Render Root Directory):** `backend/`.
- **Requirements file:** `backend/requirements.txt` (fastapi, uvicorn,
  pydantic, onnxruntime, transformers [tokenizer only], numpy). No dev/eval/
  export packages (pytest, scikit-learn, onnx, optimum, torch stay in
  `backend/requirements-dev.txt`); verified no production `.py` file uses
  torch/sklearn/datasets.
- **Model path:** `backend/app/model_assets/model.onnx` (int8, 242,491,205
  bytes). `config.py` resolves it via `Path(__file__)` →
  `BASE_DIR/app/model_assets`, independent of cwd (proven by running from the
  temp dir). No Windows paths anywhere in backend `*.py` (grep-clean).
- **Render build command:** `pip install -r requirements.txt`.
- **Render start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **Render health check path:** `/health` (exists: `GET /health`).
- **Environment variables (Render):** `FRONTEND_URL` (Vercel origin; default
  `https://sentiment-detector-tau.vercel.app`); optional
  `SENTIMENT_CORS_ORIGINS`, `SENTIMENT_MODEL_DIR`, `SENTIMENT_MODEL_NAME`,
  `SENTIMENT_MAX_INPUT_CHARS`, `SENTIMENT_MAX_TOKENS`, `SENTIMENT_LOG_LEVEL`
  (documented in `.env.example`). `$PORT` is provided by Render and used in
  the start command.
- **Render config file:** `backend/render.yaml` (web service, python, free,
  oregon, healthCheckPath `/health`, build/start above, `FRONTEND_URL`
  `sync: false`).
- **Tests performed (this session):**
  - `pytest tests -q` from `backend/` → **70 passed** (1 harmless
    `StarletteDeprecationWarning`).
  - ONNX prove-out from a neutral cwd (`Temp\opencode`): config paths resolve
    correctly, `model.onnx` found (242,491,205 bytes), ONNX Runtime 1.30.0
    loads it, `CPUExecutionProvider` session initializes:
    - "I absolutely love this product!" → positive 0.9812
    - "I hate this product." → negative 0.9336
    - "The product arrived today." → positive 0.8294 (actual output)
  - Render-style live start (`uvicorn app.main:app --host 0.0.0.0 --port
    $PORT`, `PORT=8877`): `/health` → `{"status":"ok", loaded:false}`;
    predictions via 0.0.0.0-bound server return the three results above;
    whitespace-only text → **422**.
  - `git lfs ls-files` → `4f4b088781 * backend/app/model_assets/model.onnx`
    (still an LFS pointer, not a normal blob).
- **Remaining problem:** none in code. Deploy to Render is not executable from
  this machine (no Render credentials / logged-in CLI); it is a user-run step
  (Root Directory `backend/`, build/start commands as above, `FRONTEND_URL`
  set to the Vercel URL).
- **Exact next step:** Phase 5 — Vercel frontend deployment (Root Directory
  `frontend/`, env `VITE_API_URL` = Render backend URL). Requires user
  credentials.

## PHASE 5 — VERCEL FRONTEND

Status: **COMPLETED (code + local verification); production deployment is NOT
performed** — Vercel was not deployed from this machine (no credentials), so
the production Vercel deployment and the production Render integration are
explicitly **NOT verified**.

- **Frontend framework:** plain **static HTML/CSS/JavaScript** (no React, no
  Vite, no package.json — confirmed by inspection). Reused as-is.
- **Frontend directory:** `frontend/` (7 files: `index.html`, `styles.css`,
  `app.js`, `build.js`, `favicon.svg`, `vercel.json`, plus git-ignored
  generated `api-config.js`).
- **Vercel configuration:** Root Directory `frontend/`; Framework "Other"
  (plain static); Build `node build.js` (from `frontend/vercel.json`, which
  has NO `functions` key); Output directory `.`. No install command needed
  (zero-dependency). No Python function, no backend files, no `model.onnx` in
  the artifact (verified via `git ls-tree HEAD frontend` — the tree is the
  frontend files only; the model lives only at
  `backend/app/model_assets/model.onnx`, still a Git LFS pointer
  `4f4b088781`).
- **API environment variable:** `VITE_API_URL` → read by `frontend/build.js`
  → written to git-ignored `frontend/api-config.js` as
  `window.SENTIMENT_API_BASE` → consumed by `frontend/app.js`. Default
  (unset) = `http://localhost:8000`. Vercel must set `VITE_API_URL` to
  `https://<actual-render-service>.onrender.com` (placeholder in README /
  `.env.example`; final URL NOT invented). Public browser-visible config; no
  secrets allowed.
- **API endpoint used:** `POST ${API_BASE_URL}/api/predict` with
  `{"text": "..."}` — contract preserved (request/response format, labels,
  confidence unchanged).
- **Error handling (hardened this session):** `app.js` now has a 60 s fetch
  timeout (`AbortController`), malformed/non-JSON response detection
  ("unexpected response"), distinct messages for 4xx ("rejected
  (status)"), 422 ("text rejected"), 5xx ("service having trouble"),
  timeout ("too long to respond"), network failure — no internal stack traces
  exposed; existing UI style untouched.
- **Local build result:** `node build.js` succeeded (default →
  `http://localhost:8000` and with `VITE_API_URL=https://YOUR-RENDER-BACKEND.onrender.com`
  → that placeholder); output dir = the 7 static files, no missing deps, no
  broken imports, no backend files included.
- **Local frontend test result (browser-equivalent, no GUI automation):**
  frontend served statically on `http://localhost:3000` (a default CORS
  origin) → `index.html`/`app.js`/`api-config.js` served (app.js contains the
  `/api/predict` call; config points at `http://localhost:8000`). Against the
  local backend on 8000: browser-equivalent `POST` from origin
  `http://localhost:3000` returned ACAA-echoed 200s —
  love→positive 0.9812, hate→negative 0.9336, arrived→positive 0.8294;
  whitespace-only text → 422 with CORS headers. True DOM click-through still
  requires a human/GUI browser run.
- **CORS:** unchanged (`backend/app/config.py` env-driven origins,
  `FRONTEND_URL` prepends the Vercel domain, no `*`). Documented: set
  `FRONTEND_URL` on Render to the actual Vercel production domain.
- **README:** architecture diagram + Vercel deployment section updated
  (Root Dir, Framework, Build, Output, `VITE_API_URL` form, browser-visible
  note). Render/backend sections unchanged.
- **Existing tests:** `pytest tests -q` from `backend/` → **70 passed**.
- **Remaining problem:** none in code. Production Vercel deployment +
  Render production integration are **not verified** (require user
  credentials / an actual Render service).
- **Exact next step:** **PHASE 6 — DEPLOY RENDER BACKEND AND CONNECT VERCEL**
  (user action: create the Render backend from `backend/`, set `FRONTEND_URL`;
  then in Vercel set Root Directory `frontend/` + `VITE_API_URL` to the real
  Render URL and redeploy; then end-to-end HTTPS test).

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

1. **DONE — Phase 4 (Render backend)** and **DONE — Phase 5 (Vercel
   frontend)** — code/config complete and locally verified; 70 tests pass.
   See the PHASE 4 / PHASE 5 sections.
2. **PHASE 6 — DEPLOY RENDER BACKEND AND CONNECT VERCEL (user action, needs
   account credentials):**
   - Render: create Web Service/Blueprint from this repo, Root Directory
     `backend/`, Python 3.12, build `pip install -r requirements.txt`, start
     `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health `/health`; set
     `FRONTEND_URL` to the Vercel domain. Render pulls the ONNX model via Git
     LFS. Note the resulting service URL.
   - Vercel: Root Directory `frontend/`, env `VITE_API_URL` =
     `https://<render-service>.onrender.com` (real URL, not placeholder),
     redeploy (no Python function, no model).
   - Then: end-to-end HTTPS test from the deployed frontend; verify `/health`
     and predictions; confirm CORS with the actual Vercel domain.
3. **Final self-check** — report PASS/FAIL honestly; anything not verifiable
   without credentials is marked as such.

## RESUME INSTRUCTION

Read OPENCODE_PROGRESS.md first.
Inspect the actual project files (do not trust memory).
Do not repeat completed work; start from git state (Phase 4 committed as
`05cdda5`, working tree clean unless noted).
Run tests after changes (`cd backend; ..\.venv\Scripts\python.exe -m pytest tests -q`).
Update OPENCODE_PROGRESS.md after every major milestone.