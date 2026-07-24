# Contributing to TabFM Studio

Thanks for your interest! Issues and pull requests are welcome.

## Dev setup

```bash
./start.sh                          # bootstraps backend (.venv) + frontend, runs both
MODEL_BACKEND=baseline ./start.sh   # sklearn baseline — no 6.6 GB weight download
```

Backend: FastAPI in `backend/app/` (Python 3.11+). Frontend: React +
TypeScript (Vite) in `frontend/src/`.

## Tests

```bash
cd backend && .venv/bin/python -m pytest    # API tests (baseline model, fast)
cd frontend && npm run build && npm run lint
cd frontend && npm run e2e                  # Playwright smoke test (real app)
```

CI runs all of these plus a Docker image build on every PR — please keep them
green. For backend changes, add a test alongside your change (`backend/tests/`);
the suite runs against the fast sklearn baseline, so no model weights are
needed.

## Pull requests

- Keep PRs focused — one change per PR.
- Match the style around your change (the codebase favors small, direct code
  and user-facing copy in plain language).
- If you change user-visible behavior, update the README and, if the flow
  changes, the interactive tutorial (`frontend/src/components/Tutorial.tsx`).

## Good first issues

Look for the [`good first issue`](https://github.com/LckyLke/TabFMLabs/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
label — each one has context and a suggested starting point.
