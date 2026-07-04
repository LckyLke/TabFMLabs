#!/usr/bin/env bash
#
# Start TabFM Studio: backend (FastAPI, :8000) + frontend (Vite, :5173).
# First run bootstraps both dependency sets; Ctrl+C stops everything.
#
#   ./start.sh                 # TabFM (downloads weights on first prediction)
#   MODEL_BACKEND=baseline ./start.sh   # sklearn baseline, no weight download
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# --- backend -----------------------------------------------------------------
cd "$ROOT/backend"
if [ ! -d .venv ]; then
  echo "==> Creating backend virtualenv"
  python -m venv .venv
  .venv/bin/pip install -e ".[dev]"
fi

echo "==> Starting backend on :$BACKEND_PORT"
# Invoke uvicorn as a module rather than via .venv/bin/uvicorn: the console-script
# wrapper hard-codes an absolute shebang that breaks if the repo is moved/renamed.
.venv/bin/python -m uvicorn app.main:app --port "$BACKEND_PORT" &
BACKEND_PID=$!

# --- frontend ----------------------------------------------------------------
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "==> Installing frontend dependencies"
  npm install
fi

echo "==> Starting frontend on :$FRONTEND_PORT"
npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

# --- teardown ----------------------------------------------------------------
cleanup() {
  echo
  echo "==> Shutting down"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM

echo
echo "==> TabFM Studio up — open http://localhost:$FRONTEND_PORT  (Ctrl+C to stop)"

# If either process exits, bring the whole thing down.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
cleanup
