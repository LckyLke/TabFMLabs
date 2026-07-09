
# TabFM Studio

Point-and-click predictions on your spreadsheets, powered by
[Google's TabFM](https://github.com/google-research/tabfm) tabular foundation
model. Drop in a CSV or Excel file, mark what to predict, and the empty cells
fill in right on the grid — no code, no training step, and nothing ever leaves
your machine.


https://github.com/user-attachments/assets/37d3c6eb-15d5-4739-983d-62e99d25d1b9


## Quick start

Requires **Python 3.11+** and **Node.js 20.19+** (for `npm`) on your PATH —
`start.sh` checks both up front and tells you what's missing.

```bash
./start.sh
```

First run bootstraps both dependency sets, then starts the backend (FastAPI,
:8000) and frontend (Vite, :5173); Ctrl+C stops everything. Open
http://localhost:5173 and take the interactive tutorial, try a built-in demo,
or drop in your own file (`sample_data/` has extras).

```bash
MODEL_BACKEND=baseline ./start.sh   # sklearn baseline — for dev, no weight download
```

## How it works

1. **Drop in** a file — multi-sheet workbooks get sheet tabs.
2. **Mark the grid** — click a *column header* to choose what to predict (or
   ignore), a *row number* to fix the header row or trim titles, totals and
   stray rows. Rows whose target cell is **empty** get predicted; filled rows
   are the in-context examples.
3. **Predict** — pick TabFM or TabPFN, optionally force categories vs number.
   Predictions run as cancellable background jobs with live progress;
   hover a filled cell for confidence.
4. **Judge & export** — holdout accuracy check (confusion matrix or scatter),
   prediction distribution, "what drives these predictions?" feature
   importance, and a **Compare models** run on the same holdout. Export Excel
   (your original workbook with predicted cells filled, highlighted and
   annotated) or CSV.

Projects persist in SQLite (`backend/data/studio.db`) — uploads, marking and
results survive restarts and reappear on the landing-page dashboard.
Re-uploading an identical file reopens its existing project instead of
creating a duplicate.

## Models

| Model | Notes |
|---|---|
| **TabFM** (default) | Google's foundation model. Two ~6.6 GB checkpoints (classification / regression), fetched from Hugging Face on first prediction. CUDA when available; `MODEL_DEVICE=cpu` forces CPU. |
| **TabPFN** | Prior Labs' foundation model — small weights, fast. One-time free license: accept at https://ux.priorlabs.ai, then start the backend with `TABPFN_TOKEN=<key>`. (Its usage telemetry is disabled.) |
| **Baseline (sklearn)** | HistGradientBoosting — dev/tests only. |

No silent substitution: an unavailable model fails with a clear 503.

### TabFM weights

Resolved per task in this order: `TABFM_WEIGHTS_DIR` →
`~/.cache/tabfm-studio/<task>/model.safetensors` (only if complete) → download
from `google/tabfm-1.0.0-pytorch`. While no complete checkpoint exists,
predictions return 503 with download progress; the first prediction after the
download finishes uses TabFM, no restart needed.
`~/.cache/tabfm-studio/download_weights.sh` pre-downloads both checkpoints
resumably with stall auto-recovery.

> Note: tabfm 1.0.0's own `load()` auto-download is broken — it snapshots the
> entire 13 GB repo, then looks for a `pytorch_model.bin` that isn't in it.
> `backend/app/inference.py` therefore fetches the single safetensors file
> directly.

**TabFM 1.0.0 limits:** at most 10 classes for classification (high-cardinality
numeric targets switch to regression automatically), optimized for ≤500 feature
columns, and all labeled rows are passed as context — very large tables get
slow. The weights are licensed **non-commercial** (the code is Apache 2.0).

## Development

`backend/` is FastAPI + the official `tabfm` library (PyTorch, pandas
parsing/profiling, holdout metrics); `frontend/` is React + TypeScript (Vite).
`start.sh` runs both, or start each by hand (`uvicorn app.main:app` /
`npm run dev`). Tests:

```bash
cd backend && .venv/bin/python -m pytest
```
