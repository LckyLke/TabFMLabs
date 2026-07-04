import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { deleteProject, getDataset, listProjects, uploadDataset } from "../api";
import type { DatasetResponse, ProjectInfo } from "../api";
import { demoCsvFile, demoWorkbookFile } from "../demoFiles";

interface Props {
  onUploaded: (dataset: DatasetResponse) => void;
  onStartTutorial: () => void;
}

export function Hero({ onUploaded, onStartTutorial }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch(() => setProjects([]));
  }, []);

  async function openProject(id: string) {
    setBusy(true);
    setError(null);
    try {
      onUploaded(await getDataset(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open the project.");
    } finally {
      setBusy(false);
    }
  }

  async function removeProject(id: string) {
    await deleteProject(id);
    setProjects((cur) => cur.filter((p) => p.dataset_id !== id));
  }

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    try {
      onUploaded(await uploadDataset(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  return (
    <section className="hero">
      <h1>Predictions on your spreadsheet.</h1>
      <p className="hero-sub">
        Upload a table, mark the column you want predicted, and a tabular foundation model fills
        in the blanks — locally, no training, no data science degree.
      </p>
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <svg className="dropzone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 3v12m0-12L7.5 7.5M12 3l4.5 4.5" />
          <path d="M4 15v3a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-3" />
        </svg>
        {busy ? (
          <span className="dropzone-title">Reading file…</span>
        ) : (
          <>
            <span className="dropzone-title">Drop a CSV or Excel file</span>
            <span className="dropzone-sub">or click to browse — your data stays on this machine</span>
          </>
        )}
      </div>
      <button className="btn-primary hero-tutorial" disabled={busy} onClick={onStartTutorial}>
        <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <path d="M4.5 3.2a1 1 0 0 1 1.52-.86l7 4.3a1 1 0 0 1 0 1.7l-7 4.3a1 1 0 0 1-1.52-.85V3.2Z" />
        </svg>
        Take the interactive tutorial
        <span className="hero-tutorial-sub">~2 min, guided</span>
      </button>
      <div className="hero-demos">
        <span className="hero-demos-label">Or explore a demo on your own:</span>
        <button
          className="btn-ghost hero-demo"
          disabled={busy}
          onClick={() => void handleFile(demoCsvFile())}
        >
          Simple — sales report (CSV)
        </button>
        <button
          className="btn-ghost hero-demo"
          disabled={busy}
          onClick={async () => void handleFile(await demoWorkbookFile())}
        >
          Complex — Q2 workbook (Excel, 3 sheets)
        </button>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.txt,.xlsx,.xls"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
          e.target.value = "";
        }}
      />
      {error && <div className="alert alert-error">{error}</div>}
      {projects.length > 0 && (
        <section className="recent" aria-label="Recent files">
          <h2 className="recent-title">Recent files</h2>
          <ul className="recent-list">
            {projects.slice(0, 6).map((p) => (
              <li key={p.dataset_id} className="recent-item">
                <button
                  className="recent-open"
                  disabled={busy}
                  onClick={() => void openProject(p.dataset_id)}
                >
                  <span className="recent-name">{p.filename}</span>
                  <span className="recent-meta">
                    {new Date(p.created_at).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                    {p.has_result && " · has predictions"}
                  </span>
                </button>
                <button
                  className="recent-delete"
                  onClick={() => void removeProject(p.dataset_id)}
                  aria-label={`Delete ${p.filename}`}
                  title="Delete"
                >
                  <svg viewBox="0 0 12 12" aria-hidden="true">
                    <path d="M2.5 2.5l7 7m0-7l-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      <ul className="hero-points">
        <li>
          <strong>Leave cells empty</strong> in the column you want predicted — filled rows become
          the model’s examples.
        </li>
        <li>
          <strong>Messy files welcome</strong> — title rows, totals, and footnotes can be trimmed
          right in the grid.
        </li>
        <li>
          <strong>Trust, but verify</strong> — an automatic accuracy check on held-out rows shows
          how good the predictions are.
        </li>
      </ul>
    </section>
  );
}
