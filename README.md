# TabFM Studio

A local single-page web app that makes [Google's TabFM](https://github.com/google-research/tabfm)
tabular foundation model usable by non-technical people. Your spreadsheet **is**
the interface:

1. **Drop in** a CSV or Excel file — multi-sheet workbooks get sheet tabs.
   Two built-in demos (a simple CSV sales report and a complex three-sheet
   Excel workbook), plus an **interactive tutorial** on the landing page that
   walks through the whole workflow with a spotlight tour on the demo
   workbook.
2. **Mark directly on the grid**, guided by a step checklist in the side
   panel — click a *column header* to choose what to predict (or ignore),
   click a *row number* to fix the header row, trim title
   rows/totals/footnotes, or exclude stray rows. Rows where the target cell is
   **empty** are the ones that get predicted; filled rows serve as examples
   (TabFM learns in-context, no training step).
3. **Predict** — pick a model (TabFM or TabPFN), optionally override the task
   type (categories vs number) or subsample huge tables, then predict. Empty
   cells fill in right in the grid (hover for confidence); predictions run as
   cancellable background jobs with live progress.
4. **Judge & export** — the side panel shows an accuracy check on held-out
   rows (with confusion matrix or predicted-vs-actual scatter), the prediction
   distribution, permutation feature importance ("what drives these
   predictions?"), and a **Compare models** run that scores every foundation
   model on the same holdout. Export as Excel (your original workbook with
   predicted cells filled in, highlighted, and annotated) or CSV.

Projects persist in SQLite (`backend/data/studio.db`): uploads, per-sheet
marking, and results survive restarts and reappear under "Recent files".

### Models

- **TabFM** (default) — Google's foundation model; two ~6.6 GB checkpoints.
- **TabPFN** — Prior Labs' foundation model; small weights, but requires a
  one-time free license acceptance: log in at https://ux.priorlabs.ai, accept
  the license, copy your API key, and start the backend with
  `TABPFN_TOKEN=<key>`. (TabPFN's usage telemetry is disabled by the app.)
- **Baseline (sklearn)** — explicit dev/test choice, no foundation model.

There is no automatic model substitution: an unavailable model fails with a
clear 503.

Everything runs on your machine; the data never leaves it. Model weights are
downloaded once from Hugging Face on first prediction.

## Architecture

- `backend/` — FastAPI + the official `tabfm` Python library (PyTorch backend,
  CUDA if available). Parses files with pandas, profiles columns, runs
  in-context inference, computes holdout metrics.
- `frontend/` — React + TypeScript (Vite). Three-step flow, no data leaves the
  browser except to the local backend.

## Running

Backend (port 8000):

```bash
cd backend
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn app.main:app --port 8000
```

Frontend (port 5173):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and try `sample_data/customer_churn.csv`.

Set `MODEL_BACKEND=baseline` to explicitly use a scikit-learn gradient-boosting
model instead of TabFM (development/tests only — no weight download needed).

### Model weights

Each TabFM checkpoint is **~6.6 GB** (classification and regression are
separate files). The backend resolves weights per task in this order:

1. `TABFM_WEIGHTS_DIR` (if set) — expects `<dir>/<task>/model.safetensors`;
2. `~/.cache/tabfm-studio/<task>/model.safetensors` — used only when the file
   is complete (a partially-downloaded file is skipped);
3. download from `google/tabfm-1.0.0-pytorch` on Hugging Face.

There is **no automatic fallback**: while no complete checkpoint is available,
predictions fail with HTTP 503 and a message that includes the download
progress. The first prediction after the weights finish downloading uses
TabFM — no restart needed.
`~/.cache/tabfm-studio/download_weights.sh` downloads both checkpoints
resumably with stall auto-recovery.

Note: tabfm 1.0.0's own `tabfm_v1_0_0.load()` auto-download is broken — it
snapshot-downloads the entire 13 GB repo and then looks for
`pytorch_model.bin`, which the published repo doesn't contain (it ships
`model.safetensors`). `backend/app/inference.py` therefore downloads the single
safetensors file and loads it directly.

`MODEL_DEVICE=cpu` forces CPU inference (default: CUDA when available, with
automatic CPU fallback). `MODEL_BACKEND=baseline` forces the sklearn baseline
(used by the tests).

## Model limits (TabFM 1.0.0)

- At most **10 classes** for classification (numeric targets with more distinct
  values are treated as regression automatically).
- Optimized for up to **500 feature columns**.
- All labeled rows are passed as context — very large tables get slow and
  memory-heavy.
- The TabFM **weights are licensed non-commercial** (the code is Apache 2.0).

## Tests

```bash
cd backend && .venv/bin/python -m pytest
```
