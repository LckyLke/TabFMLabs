import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  cancelPredictJob,
  getGridPage,
  getPredictJob,
  listModels,
  retryWhileStarting,
  saveRoles,
  setActiveSheet,
  setTableSpec,
  startPredictJob,
  uploadDataset,
} from "./api";
import type {
  ColumnRole,
  DatasetResponse,
  ModelInfo,
  PredictResponse,
  TableProfile,
  TableSpec,
  TaskType,
} from "./api";
import { DataGrid } from "./components/DataGrid";
import type { CellPrediction } from "./components/DataGrid";
import { Hero } from "./components/Hero";
import { SidePanel } from "./components/SidePanel";
import { TopBar } from "./components/TopBar";
import { Tutorial } from "./components/Tutorial";
import { demoWorkbookFile } from "./demoFiles";

function defaultRoles(table: TableProfile): ColumnRole[] {
  // Free-text/ID columns rarely help and can hurt; leave them out by default.
  return table.columns.map((c) => (c.kind === "text" ? "ignore" : "feature"));
}

export interface CompareEntry {
  model: ModelInfo;
  result: PredictResponse | null;
  error: string | null;
}

export default function App() {
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [gridRows, setGridRows] = useState<string[][]>([]);
  const [spec, setSpec] = useState<TableSpec | null>(null);
  const [table, setTable] = useState<TableProfile | null>(null);
  const [roles, setRoles] = useState<ColumnRole[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState<string>("tabfm");
  const [maxContextRows, setMaxContextRows] = useState<number | null>(null);
  const [taskOverride, setTaskOverride] = useState<TaskType | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [compare, setCompare] = useState<CompareEntry[] | null>(null);
  const [busyStage, setBusyStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tutorialStep, setTutorialStep] = useState<number | null>(null);
  const specSeq = useRef(0);
  const userSetRoles = useRef(new Set<number>());
  const activeJob = useRef<string | null>(null);
  const gridLoading = useRef(false);

  useEffect(() => {
    let cancelled = false;
    void retryWhileStarting(listModels, () => cancelled).then((list) => {
      if (!list || cancelled) return;
      setModels(list);
      const def = list.find((m) => m.is_default);
      if (def) setModelId(def.id);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  function onUploaded(ds: DatasetResponse) {
    setDataset(ds);
    setGridRows(ds.grid);
    setSpec(ds.spec);
    setTable(ds.table);
    setRoles(ds.roles ?? defaultRoles(ds.table));
    userSetRoles.current = ds.roles ? new Set(ds.roles.map((_, i) => i)) : new Set();
    setResult(null);
    setCompare(null);
    setError(null);
    setMaxContextRows(null);
    setTaskOverride(null);
  }

  function reset() {
    if (activeJob.current) void cancelPredictJob(activeJob.current);
    activeJob.current = null;
    setDataset(null);
    setGridRows([]);
    setSpec(null);
    setTable(null);
    setRoles([]);
    setResult(null);
    setCompare(null);
    setError(null);
    setBusyStage(null);
    setTutorialStep(null);
  }

  // Throws on failure so Hero can show the error — App's own error state is
  // only rendered inside the side panel, which doesn't exist yet on the hero.
  async function startTutorial() {
    onUploaded(await uploadDataset(await demoWorkbookFile()));
    setTutorialStep(0);
  }

  function invalidateResults() {
    setResult(null);
    setCompare(null);
  }

  function changeRoles(next: ColumnRole[]) {
    next.forEach((r, i) => {
      if (r !== roles[i]) userSetRoles.current.add(i);
    });
    setRoles(next);
    invalidateResults();
    if (dataset) void saveRoles(dataset.dataset_id, next);
  }

  async function applySpec(next: TableSpec) {
    if (!dataset || !spec) return;
    const prev = spec;
    setSpec(next); // optimistic; grid annotations update instantly
    invalidateResults();
    setError(null);
    const seq = ++specSeq.current;
    try {
      const profile = await setTableSpec(dataset.dataset_id, next);
      if (seq === specSeq.current) {
        setTable(profile);
        // Re-derive auto roles: fixing the region can change inferred types.
        const auto = defaultRoles(profile);
        setRoles((cur) => cur.map((r, i) => (userSetRoles.current.has(i) ? r : auto[i])));
      }
    } catch (e) {
      if (seq === specSeq.current) {
        setSpec(prev);
        setError(e instanceof Error ? e.message : "Could not update the table range.");
      }
    }
  }

  async function switchSheet(name: string) {
    if (!dataset || name === dataset.active_sheet) return;
    try {
      onUploaded(await setActiveSheet(dataset.dataset_id, name));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not switch sheet.");
    }
  }

  async function loadMoreGridRows() {
    if (!dataset || gridLoading.current) return;
    if (gridRows.length >= dataset.n_raw_rows) return;
    gridLoading.current = true;
    try {
      const page = await getGridPage(dataset.dataset_id, gridRows.length, 500);
      setGridRows((cur) => (page.offset === cur.length ? [...cur, ...page.rows] : cur));
    } catch {
      /* transient; user can scroll again */
    } finally {
      gridLoading.current = false;
    }
  }

  const targetIndex = roles.indexOf("target");
  const target = targetIndex >= 0 ? (table?.columns[targetIndex] ?? null) : null;
  const featureNames = (table?.columns ?? [])
    .filter((_, i) => roles[i] === "feature" && i !== targetIndex)
    .map((c) => c.name);
  const rowsToPredict = target?.n_missing ?? 0;
  const exampleRows = target && table ? table.n_rows - rowsToPredict : 0;

  const problem = useMemo(() => {
    if (!table) return null;
    if (!target) return "Click a column header and choose “Predict this column” to begin.";
    if (rowsToPredict === 0)
      return `Every row already has a value in “${target.name}”. Leave its cell empty in the rows you want predicted.`;
    if (rowsToPredict === table.n_rows)
      return `“${target.name}” is completely empty — the model needs filled-in rows as examples.`;
    if (featureNames.length === 0) return "Keep at least one column as an input.";
    return null;
  }, [table, target, rowsToPredict, featureNames.length]);

  const predictionMap = useMemo(() => {
    if (!result) return null;
    const map = new Map<number, CellPrediction>();
    for (const row of result.predictions) {
      map.set(row.row_index, { value: row.prediction, confidence: row.confidence });
    }
    return map;
  }, [result]);

  async function runJob(model: string, onStage: (s: string) => void): Promise<PredictResponse> {
    if (!dataset || !target) throw new Error("Nothing to predict.");
    const job = await startPredictJob(dataset.dataset_id, target.name, featureNames, {
      model,
      maxContextRows,
      task: taskOverride,
    });
    activeJob.current = job.job_id;
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));
      const status = await getPredictJob(job.job_id);
      onStage(status.stage);
      if (status.status === "done" && status.result) return status.result;
      if (status.status === "cancelled") throw new Error("cancelled");
      if (status.status === "error") throw new Error(status.error ?? "Prediction failed.");
    }
  }

  async function runPredict() {
    setBusyStage("starting…");
    setError(null);
    setCompare(null);
    try {
      setResult(await runJob(modelId, setBusyStage));
    } catch (e) {
      if (!(e instanceof Error && e.message === "cancelled")) {
        setError(e instanceof Error ? e.message : "Prediction failed.");
      }
    } finally {
      activeJob.current = null;
      setBusyStage(null);
    }
  }

  async function runCompare() {
    const contenders = models.filter((m) => m.id !== "baseline");
    setBusyStage("comparing…");
    setError(null);
    setCompare(null);
    const entries: CompareEntry[] = [];
    try {
      for (const model of contenders) {
        try {
          const res = await runJob(model.id, (s) => setBusyStage(`${model.label}: ${s}`));
          entries.push({ model, result: res, error: null });
          if (model.id === modelId) setResult(res);
        } catch (e) {
          if (e instanceof Error && e.message === "cancelled") throw e;
          entries.push({
            model,
            result: null,
            error: e instanceof Error ? e.message : "failed",
          });
        }
      }
      setCompare(entries);
      // show predictions of the first successful model if the selected one failed
      if (!entries.find((e) => e.model.id === modelId)?.result) {
        const firstOk = entries.find((e) => e.result);
        if (firstOk?.result) setResult(firstOk.result);
      }
    } catch {
      /* cancelled */
    } finally {
      activeJob.current = null;
      setBusyStage(null);
    }
  }

  function cancelBusy() {
    if (activeJob.current) void cancelPredictJob(activeJob.current);
  }

  const busy = busyStage !== null;

  return (
    <>
      <TopBar
        filename={dataset?.filename ?? null}
        nRows={table?.n_rows ?? null}
        modelName={result?.model_name ?? null}
        onReset={reset}
      />
      {!dataset || !spec || !table ? (
        <Hero onUploaded={onUploaded} onStartTutorial={startTutorial} />
      ) : (
        <main className="studio">
          <section className="studio-main" aria-label="Data">
            {dataset.sheet_names.length > 1 && (
              <nav className="sheet-tabs" aria-label="Sheets">
                {dataset.sheet_names.map((name) => (
                  <button
                    key={name}
                    className={`sheet-tab ${name === dataset.active_sheet ? "is-active" : ""}`}
                    onClick={() => void switchSheet(name)}
                    aria-current={name === dataset.active_sheet ? "page" : undefined}
                  >
                    {name}
                  </button>
                ))}
              </nav>
            )}
            {problem && !busy && (
              <div className="grid-callout" role="status">
                <span className="grid-callout-icon" aria-hidden="true">
                  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 12V5.5M8 5.5 4.5 9M8 5.5 11.5 9" />
                    <path d="M2.5 2.5h11" />
                  </svg>
                </span>
                {problem}
              </div>
            )}
            <DataGrid
              grid={gridRows}
              gridTruncated={gridRows.length < dataset.n_raw_rows}
              nRawRows={dataset.n_raw_rows}
              spec={spec}
              columns={table.columns}
              roles={roles}
              predictions={predictionMap}
              busy={busy}
              onRolesChange={changeRoles}
              onSpecChange={applySpec}
              onNearEnd={() => void loadMoreGridRows()}
            />
          </section>
          <SidePanel
            target={target}
            exampleRows={exampleRows}
            rowsToPredict={rowsToPredict}
            featureCount={featureNames.length}
            featureNames={featureNames}
            problem={problem}
            warnings={table.warnings}
            error={error}
            busyStage={busyStage}
            result={result}
            compare={compare}
            models={models}
            modelId={modelId}
            onModelChange={(id) => {
              setModelId(id);
              invalidateResults();
            }}
            maxContextRows={maxContextRows}
            onMaxContextRows={setMaxContextRows}
            taskOverride={taskOverride}
            onTaskOverride={(t) => {
              setTaskOverride(t);
              invalidateResults();
            }}
            datasetId={dataset.dataset_id}
            targetName={target?.name ?? null}
            gridTruncated={dataset.grid_truncated}
            onPredict={() => void runPredict()}
            onCompare={() => void runCompare()}
            onCancel={cancelBusy}
          />
        </main>
      )}
      {tutorialStep !== null && dataset && spec && (
        <Tutorial
          snapshot={{
            activeSheet: dataset.active_sheet,
            headerRow: spec.header_row,
            endRow: spec.end_row,
            hasTarget: target !== null,
            rowsToPredict,
            busy,
            hasResult: result !== null,
            hasError: error !== null,
          }}
          stepIndex={tutorialStep}
          onStepChange={setTutorialStep}
          onClose={() => setTutorialStep(null)}
        />
      )}
    </>
  );
}
