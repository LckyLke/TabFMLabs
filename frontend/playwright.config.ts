import os from "node:os";
import path from "node:path";
import { defineConfig } from "@playwright/test";

// E2E runs on its own ports (backend 8100, frontend 5273) so a normal dev
// setup (8000/5173) can keep running alongside. Locally, already-running
// servers on those ports are reused.
export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  use: { baseURL: "http://localhost:5273" },
  webServer: [
    {
      command: "python -m uvicorn app.main:app --port 8100",
      cwd: path.join(import.meta.dirname, "../backend"),
      port: 8100,
      env: {
        MODEL_BACKEND: "baseline",
        EXTRA_CORS_ORIGINS: "http://localhost:5273",
        STUDIO_DB: path.join(os.tmpdir(), "tabfm-e2e-studio.db"),
        // Nonexistent dir -> instant 503 for TabFM instead of a 6.6 GB download.
        TABFM_WEIGHTS_DIR: path.join(os.tmpdir(), "tabfm-e2e-no-weights"),
      },
      reuseExistingServer: !process.env.CI,
    },
    {
      command: "npm run dev -- --port 5273 --strictPort",
      port: 5273,
      env: { VITE_API_BASE: "http://localhost:8100" },
      reuseExistingServer: !process.env.CI,
    },
  ],
});
