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

# --- prerequisites -----------------------------------------------------------
missing=()
command -v python >/dev/null || missing+=("python — 3.11+ required (https://www.python.org)")
command -v npm >/dev/null || missing+=("npm — ships with Node.js 20.19+ (https://nodejs.org)")
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Error: missing prerequisites (see README → Quick start):" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

# --- backend -----------------------------------------------------------------
cd "$ROOT/backend"

# Pick a Python environment. An already-activated one wins: a virtualenv, or a
# conda env other than base (base is just the shell default, not an opt-in).
# Otherwise fall back to a project-local .venv.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PY="$VIRTUAL_ENV/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; then
  PY="$CONDA_PREFIX/bin/python"
else
  # A venv hard-codes absolute paths at creation time, so it breaks if the
  # repo directory is moved or renamed; rebuild it when that happened.
  if [ -d .venv ]; then
    if ! .venv/bin/python -c '' 2>/dev/null \
       || ! grep -qsF "$ROOT/backend/.venv" .venv/pyvenv.cfg; then
      echo "==> Rebuilding backend virtualenv (created for a different path)"
      rm -rf .venv
    fi
  fi
  if [ ! -d .venv ]; then
    echo "==> Creating backend virtualenv"
    python -m venv .venv
  fi
  PY="$ROOT/backend/.venv/bin/python"
fi

# Re-sync whenever pyproject.toml changed so dependency additions reach
# existing installs, not just fresh ones. The stamp lives inside the env's
# prefix so each environment (.venv, conda, ...) tracks its own sync state.
PREFIX="$(cd "$(dirname "$PY")/.." && pwd)"
STAMP="$PREFIX/.tabfm-deps-synced"
if [ pyproject.toml -nt "$STAMP" ]; then
  echo "==> Installing backend dependencies into $PREFIX"
  "$PY" -m pip install -e ".[dev]"
  touch "$STAMP"
fi

echo "==> Starting backend on :$BACKEND_PORT"
# Invoke uvicorn as a module rather than via a console-script wrapper: those
# hard-code an absolute shebang that breaks if the repo is moved/renamed.
"$PY" -m uvicorn app.main:app --port "$BACKEND_PORT" &
BACKEND_PID=$!
FRONTEND_PID=""

# --- teardown ----------------------------------------------------------------
# Installed before the frontend section so a failure there (e.g. npm install)
# doesn't leave the backend running orphaned.
cleanup() {
  echo
  echo "==> Shutting down"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# --- frontend ----------------------------------------------------------------
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "==> Installing frontend dependencies"
  npm install
fi

echo "==> Starting frontend on :$FRONTEND_PORT"
npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo
echo "==> TabFM Studio up — open http://localhost:$FRONTEND_PORT  (Ctrl+C to stop)"

# If either process exits, the EXIT trap brings the whole thing down.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
