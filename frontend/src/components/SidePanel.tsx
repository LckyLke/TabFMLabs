import { useState } from "react";
import type { CompareEntry } from "../App";
import type {
  ColumnInfo,
  ExplainResponse,
  Metrics,
  ModelInfo,
  PredictResponse,
  TaskType,
} from "../api";
import { explain, resultCsvUrl, resultXlsxUrl } from "../api";
import { DistributionChart } from "./DistributionChart";

interface Props {
  target: ColumnInfo | null;
  exampleRows: number;
  rowsToPredict: number;
  featureCount: number;
  featureNames: string[];
  problem: string | null;
  warnings: string[];
  error: string | null;
  busyStage: string | null;
  result: PredictResponse | null;
  compare: CompareEntry[] | null;
  models: ModelInfo[];
  modelId: string;
  onModelChange: (id: string) => void;
  maxContextRows: number | null;
  onMaxContextRows: (n: number | null) => void;
  taskOverride: TaskType | null;
  onTaskOverride: (t: TaskType | null) => void;
  datasetId: string;
  targetName: string | null;
  gridTruncated: boolean;
  onPredict: () => void;
  onCompare: () => void;
  onCancel: () => void;
}

function StepCheck({ done, num }: { done: boolean; num: number }) {
  return (
    <span className={`step-num ${done ? "is-done" : ""}`} aria-hidden="true">
      {done ? (
        <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2.5 6.5 5 9l4.5-5.5" />
        </svg>
      ) : (
        num
      )}
    </span>
  );
}

function MetricTiles({ m }: { m: Metrics | null }) {
  const tiles: { label: string; value: string; sub?: string }[] = [];
  if (m?.accuracy != null)
    tiles.push({
      label: "Accuracy check",
      value: `${(m.accuracy * 100).toFixed(1)}%`,
      sub: `on ${m.n_holdout} held-out rows`,
    });
  if (m?.f1_macro != null) tiles.push({ label: "F1 (macro)", value: m.f1_macro.toFixed(3) });
  if (m?.r2 != null)
    tiles.push({ label: "R² check", value: m.r2.toFixed(3), sub: `on ${m.n_holdout} held-out rows` });
  if (m?.mae != null)
    tiles.push({
      label: "Typical error",
      value: m.mae.toLocaleString(undefined, { maximumFractionDigits: 1 }),
    });
  if (tiles.length === 0) return null;
  return (
    <div className="rail-tiles">
      {tiles.map((t) => (
        <div key={t.label} className="tile">
          <span className="tile-label">{t.label}</span>
          <span className="tile-value">{t.value}</span>
          {t.sub && <span className="tile-sub">{t.sub}</span>}
        </div>
      ))}
    </div>
  );
}

function ConfusionTable({ m }: { m: Metrics }) {
  if (!m.confusion) return null;
  const { labels, matrix } = m.confusion;
  return (
    <figure className="confusion">
      <figcaption>Held-out rows: actual ↓ vs predicted →</figcaption>
      <div className="confusion-scroll">
        <table>
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th>{labels[i]}</th>
              {row.map((n, j) => (
                <td key={j} className={i === j ? "diag" : n > 0 ? "off-diag" : ""}>
                  {n}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        </table>
      </div>
    </figure>
  );
}

function HoldoutScatter({ m }: { m: Metrics }) {
  const samples = m.holdout_samples;
  if (!samples || samples.length < 3) return null;
  const values = samples.flatMap((s) => [s.actual, s.predicted]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const W = 240;
  const H = 170;
  const PAD = 8;
  const sx = (v: number) => PAD + ((v - min) / span) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - min) / span) * (H - 2 * PAD);
  const fmt = (v: number) =>
    Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toPrecision(3);
  return (
    <figure className="scatter">
      <figcaption>Held-out rows: predicted vs actual (closer to the line = better)</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Predicted versus actual scatter plot">
        <line x1={sx(min)} y1={sy(min)} x2={sx(max)} y2={sy(max)} className="scatter-diag" />
        {samples.map((s, i) => (
          <circle key={i} cx={sx(s.actual)} cy={sy(s.predicted)} r="3" className="scatter-dot">
            <title>{`actual ${fmt(s.actual)} → predicted ${fmt(s.predicted)}`}</title>
          </circle>
        ))}
      </svg>
      <span className="scatter-range">
        {fmt(min)} – {fmt(max)}
      </span>
    </figure>
  );
}

function CompareCard({ compare }: { compare: CompareEntry[] }) {
  const metricRows: { label: string; get: (m: Metrics | null) => string }[] = [
    {
      label: "Accuracy",
      get: (m) => (m?.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : "—"),
    },
    { label: "F1 (macro)", get: (m) => (m?.f1_macro != null ? m.f1_macro.toFixed(3) : "—") },
    { label: "R²", get: (m) => (m?.r2 != null ? m.r2.toFixed(3) : "—") },
    {
      label: "Typical error",
      get: (m) =>
        m?.mae != null ? m.mae.toLocaleString(undefined, { maximumFractionDigits: 1 }) : "—",
    },
  ];
  const shown = metricRows.filter((row) =>
    compare.some((e) => row.get(e.result?.metrics ?? null) !== "—"),
  );
  return (
    <div className="rail-card" role="status">
      <h2 className="rail-title">Model comparison</h2>
      <div className="confusion-scroll">
        <table className="compare-table">
        <thead>
          <tr>
            <th />
            {compare.map((e) => (
              <th key={e.model.id}>{e.model.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row) => (
            <tr key={row.label}>
              <th>{row.label}</th>
              {compare.map((e) => (
                <td key={e.model.id}>{row.get(e.result?.metrics ?? null)}</td>
              ))}
            </tr>
          ))}
        </tbody>
        </table>
      </div>
      {compare
        .filter((e) => e.error)
        .map((e) => (
          <div key={e.model.id} className="alert alert-warn">
            {e.model.label}: {e.error}
          </div>
        ))}
      <p className="rail-note">Same holdout split for every model — higher is better.</p>
    </div>
  );
}

function ExplainSection({
  datasetId,
  targetName,
  featureNames,
  modelId,
  taskOverride,
}: {
  datasetId: string;
  targetName: string;
  featureNames: string[];
  modelId: string;
  taskOverride: TaskType | null;
}) {
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setData(await explain(datasetId, targetName, featureNames, modelId, taskOverride));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Explaining failed.");
    } finally {
      setBusy(false);
    }
  }

  if (!data) {
    return (
      <div className="explain">
        <button className="btn-secondary" disabled={busy} onClick={() => void run()}>
          {busy ? (
            <>
              <span className="spinner spinner-blue" aria-hidden="true" /> analyzing inputs…
            </>
          ) : (
            "What drives these predictions?"
          )}
        </button>
        {error && <div className="alert alert-error">{error}</div>}
        {busy && (
          <p className="rail-note">Runs the model once per input column — can take a while.</p>
        )}
      </div>
    );
  }

  const maxImp = Math.max(0.0001, ...data.importances.map((i) => Math.abs(i.importance)));
  return (
    <figure className="importance">
      <figcaption>
        Drop in {data.metric_name} when a column is shuffled (bigger = more important)
      </figcaption>
      <div className="dist-rows">
        {data.importances.map((fi) => (
          <div key={fi.feature} className="dist-row">
            <span className="dist-label" title={fi.feature}>
              {fi.feature}
            </span>
            <span className="dist-track">
              <span
                className={`dist-bar ${fi.importance < 0 ? "dist-bar-neg" : ""}`}
                style={{ width: `${(Math.abs(fi.importance) / maxImp) * 100}%` }}
              />
            </span>
            <span className="dist-value">{fi.importance.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}

export function SidePanel({
  target,
  exampleRows,
  rowsToPredict,
  featureCount,
  featureNames,
  problem,
  warnings,
  error,
  busyStage,
  result,
  compare,
  models,
  modelId,
  onModelChange,
  maxContextRows,
  onMaxContextRows,
  taskOverride,
  onTaskOverride,
  datasetId,
  targetName,
  gridTruncated,
  onPredict,
  onCompare,
  onCancel,
}: Props) {
  const targetChosen = target !== null;
  const rowsMarked = targetChosen && rowsToPredict > 0;
  const busy = busyStage !== null;
  const selectedModel = models.find((m) => m.id === modelId);
  const canCompare = models.filter((m) => m.id !== "baseline").length >= 2;

  return (
    <aside className="rail">
      <div className="rail-card">
        <h2 className="rail-title">Before you predict</h2>
        <ol className="steps">
          <li className="step">
            <span className="step-num step-optional" aria-hidden="true">
              ✦
            </span>
            <span className="step-body">
              <span className="step-label">Tidy the table <em>(if needed)</em></span>
              <span className="step-sub">
                Wrong header? Totals or notes mixed in? Click a <strong>row number</strong> to fix
                it. Irrelevant columns → set to “Ignore”.
              </span>
            </span>
          </li>
          <li className={`step ${targetChosen ? "is-done" : "is-current"}`}>
            <StepCheck done={targetChosen} num={1} />
            <span className="step-body">
              <span className="step-label">Pick the column to predict</span>
              <span className="step-sub">
                {targetChosen ? (
                  <>
                    <span className="setup-target">{target.name}</span> · {featureCount} input
                    {featureCount === 1 ? "" : "s"}
                  </>
                ) : (
                  <>
                    Click a <strong>column header</strong> → “Predict this column”.
                  </>
                )}
              </span>
            </span>
          </li>
          <li className={`step ${rowsMarked ? "is-done" : targetChosen ? "is-current" : ""}`}>
            <StepCheck done={rowsMarked} num={2} />
            <span className="step-body">
              <span className="step-label">Empty cells mark the rows to predict</span>
              <span className="step-sub">
                {rowsMarked ? (
                  <>
                    {exampleRows.toLocaleString()} example rows →{" "}
                    <strong>{rowsToPredict.toLocaleString()} to predict</strong>
                  </>
                ) : (
                  <>Filled rows teach the model; blank target cells get filled in.</>
                )}
              </span>
            </span>
          </li>
        </ol>

        <label className="model-select">
          <span className="model-select-label">Model</span>
          <select
            value={models.length === 0 ? "" : modelId}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={busy || models.length === 0}
          >
            {models.length === 0 && <option value="">Connecting to backend…</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {selectedModel && <span className="model-select-desc">{selectedModel.description}</span>}
        </label>

        <div className="task-select" role="radiogroup" aria-label="Prediction type">
          <span className="model-select-label">Predict as</span>
          <div className="task-options">
            {(
              [
                [null, "Auto", "Detected from the target column's values"],
                ["classification", "Categories", "e.g. yes/no, segments, labels"],
                ["regression", "Number", "e.g. revenue, quantity, score"],
              ] as const
            ).map(([value, label, hint]) => (
              <button
                key={label}
                className={`task-option ${taskOverride === value ? "is-active" : ""}`}
                onClick={() => onTaskOverride(value)}
                disabled={busy}
                role="radio"
                aria-checked={taskOverride === value}
                title={hint}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {exampleRows > 2000 && (
          <label className="subsample">
            <input
              type="checkbox"
              checked={maxContextRows !== null}
              onChange={(e) => onMaxContextRows(e.target.checked ? 2000 : null)}
              disabled={busy}
            />
            <span>
              Use a sample of{" "}
              <input
                type="number"
                className="subsample-n"
                min={100}
                max={exampleRows}
                value={maxContextRows ?? 2000}
                disabled={busy || maxContextRows === null}
                onChange={(e) => onMaxContextRows(Number(e.target.value) || 2000)}
              />{" "}
              of the {exampleRows.toLocaleString()} example rows (faster, less memory)
            </span>
          </label>
        )}

        {warnings.map((w) => (
          <div key={w} className="alert alert-warn">
            {w}
          </div>
        ))}
        {error && <div className="alert alert-error">{error}</div>}

        {busy ? (
          <div className="busy-row">
            <span className="tour-wait">
              <span className="spinner spinner-blue" aria-hidden="true" /> {busyStage}
            </span>
            <button className="btn-ghost" onClick={onCancel}>
              Cancel
            </button>
          </div>
        ) : (
          <div className="predict-row">
            <button
              className="btn-primary rail-predict"
              disabled={problem !== null}
              onClick={onPredict}
              title={problem ?? undefined}
            >
              {result ? "Predict again" : "Predict"}
            </button>
            {canCompare && (
              <button
                className="btn-secondary compare-btn"
                disabled={problem !== null}
                onClick={onCompare}
                title="Run every foundation model on the same data and compare their accuracy checks"
              >
                Compare models
              </button>
            )}
          </div>
        )}
      </div>

      {compare && <CompareCard compare={compare} />}

      {result && (
        <div className="rail-card rail-results" role="status">
          <h2 className="rail-title">
            Results
            <span className="rail-title-meta">
              {result.task === "classification" ? "Classification" : "Regression"}
            </span>
          </h2>
          <p className="rail-model">{result.model_name}</p>

          <MetricTiles m={result.metrics} />
          {result.metrics && <ConfusionTable m={result.metrics} />}
          {result.metrics && <HoldoutScatter m={result.metrics} />}
          <DistributionChart bins={result.distribution} task={result.task} />

          {targetName && (
            <ExplainSection
              datasetId={datasetId}
              targetName={targetName}
              featureNames={featureNames}
              modelId={modelId}
              taskOverride={taskOverride}
            />
          )}

          {result.warnings.map((w) => (
            <div key={w} className="alert alert-warn">
              {w}
            </div>
          ))}
          {gridTruncated && (
            <p className="rail-note">
              Predicted values beyond the visible preview are included in the downloads.
            </p>
          )}

          <div className="download-row">
            <a className="btn-secondary" href={resultXlsxUrl(result.prediction_id)} download>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M8 2v8m0 0L5 7m3 3 3-3M3 12v1a1.5 1.5 0 0 0 1.5 1.5h7A1.5 1.5 0 0 0 13 13v-1" />
              </svg>
              Excel — original file, completed
            </a>
            <a className="btn-ghost download-csv" href={resultCsvUrl(result.prediction_id)} download>
              or download as CSV
            </a>
          </div>
        </div>
      )}
    </aside>
  );
}
