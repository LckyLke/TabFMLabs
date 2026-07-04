import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { deleteProject, getDataset, listProjects, retryWhileStarting, uploadDataset } from "../api";
import type { DatasetResponse, ProjectInfo } from "../api";
import { demoCsvFile, demoWorkbookFile } from "../demoFiles";

interface Props {
  onUploaded: (dataset: DatasetResponse) => void;
  onStartTutorial: () => void;
}

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  const date = new Date(iso);
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
}

function FileIcon({ filename }: { filename: string }) {
  const isExcel = /\.xlsx?$/i.test(filename);
  return (
    <span className={`project-icon ${isExcel ? "is-excel" : "is-csv"}`} aria-hidden="true">
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 2h7l3.5 3.5V18H5z" />
        <path d="M12 2v3.5H15.5" />
        {isExcel ? <path d="M8 9.5l4 5m0-5l-4 5" /> : <path d="M7.5 10h5M7.5 12.5h5" />}
      </svg>
    </span>
  );
}

export function Hero({ onUploaded, onStartTutorial }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [projects, setProjects] = useState<ProjectInfo[] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    void retryWhileStarting(listProjects, () => cancelled).then((list) => {
      if (list && !cancelled) setProjects(list);
    });
    return () => {
      cancelled = true;
    };
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
    setProjects((cur) => (cur ?? []).filter((p) => p.dataset_id !== id));
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

  const fileInput = (
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
  );

  const demoButtons = (
    <>
      <button className="btn-ghost hero-demo" disabled={busy} onClick={() => void handleFile(demoCsvFile())}>
        Simple — sales report (CSV)
      </button>
      <button
        className="btn-ghost hero-demo"
        disabled={busy}
        onClick={async () => void handleFile(await demoWorkbookFile())}
      >
        Complex — Q2 workbook (Excel, 3 sheets)
      </button>
    </>
  );

  // Returning user: a project dashboard instead of the marketing hero.
  if (projects && projects.length > 0) {
    return (
      <section className="dashboard" aria-label="Your projects">
        <div
          className={`dropstrip ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`}
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
          <svg className="dropstrip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 3v12m0-12L7.5 7.5M12 3l4.5 4.5" />
            <path d="M4 15v3a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-3" />
          </svg>
          <span className="dropstrip-text">
            {busy ? "Reading file…" : (
              <>
                <strong>Drop a CSV or Excel file</strong> or click to browse — data stays on this
                machine
              </>
            )}
          </span>
        </div>
        {fileInput}
        {error && <div className="alert alert-error">{error}</div>}

        <div className="dashboard-head">
          <h1 className="dashboard-title">Your projects</h1>
          <div className="dashboard-actions">
            <button className="btn-ghost" disabled={busy} onClick={onStartTutorial}>
              Interactive tutorial
            </button>
            {demoButtons}
          </div>
        </div>

        <ul className="project-grid">
          {projects.map((p) => (
            <li key={p.dataset_id} className="project-card">
              <button
                className="project-open"
                disabled={busy}
                onClick={() => void openProject(p.dataset_id)}
                title={`Open ${p.filename}`}
              >
                <FileIcon filename={p.filename} />
                <span className="project-name">{p.filename}</span>
                <span className="project-meta">{relativeTime(p.last_used)}</span>
                {p.n_results > 0 ? (
                  <span className="project-status has-predictions">
                    <svg viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
                      <path d="M6 1l1.2 3.3L10.5 5.5 7.2 6.7 6 10 4.8 6.7 1.5 5.5l3.3-1.2z" />
                    </svg>
                    {p.n_results === 1 ? "1 prediction" : `${p.n_results} predictions`}
                    {p.target_column && ` · ${p.target_column}`}
                  </span>
                ) : (
                  <span className="project-status">Not predicted yet</span>
                )}
              </button>
              <button
                className="project-delete"
                onClick={() => void removeProject(p.dataset_id)}
                aria-label={`Delete ${p.filename}`}
                title="Delete project"
              >
                <svg viewBox="0 0 12 12" aria-hidden="true">
                  <path d="M2.5 2.5l7 7m0-7l-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  // First visit (or backend still starting): the full welcome hero.
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
        {demoButtons}
      </div>
      {fileInput}
      {error && <div className="alert alert-error">{error}</div>}
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
