const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type ColumnKind = "numeric" | "categorical" | "boolean" | "datetime" | "text";
export type TaskType = "classification" | "regression";
export type ColumnRole = "feature" | "target" | "ignore";

export interface ColumnInfo {
  name: string;
  kind: ColumnKind;
  n_missing: number;
  n_unique: number;
  sample_values: string[];
}

export interface TableSpec {
  header_row: number | null;
  start_row: number;
  end_row: number;
  excluded_rows: number[];
}

export interface TableProfile {
  n_rows: number;
  columns: ColumnInfo[];
  warnings: string[];
}

export interface DatasetResponse {
  dataset_id: string;
  filename: string;
  sheet_names: string[];
  active_sheet: string;
  n_raw_rows: number;
  grid: string[][];
  grid_truncated: boolean;
  spec: TableSpec;
  table: TableProfile;
  roles: ColumnRole[] | null;
}

export interface GridPage {
  offset: number;
  rows: string[][];
  n_raw_rows: number;
}

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  is_default: boolean;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface HoldoutSample {
  actual: number;
  predicted: number;
}

export interface Metrics {
  task: TaskType;
  n_holdout: number;
  accuracy: number | null;
  f1_macro: number | null;
  confusion: ConfusionMatrix | null;
  r2: number | null;
  mae: number | null;
  holdout_samples: HoldoutSample[] | null;
}

export interface PredictionRow {
  row_index: number;
  values: Record<string, string | null>;
  prediction: string;
  confidence: number | null;
}

export interface DistributionBin {
  label: string;
  count: number;
}

export interface PredictResponse {
  prediction_id: string;
  task: TaskType;
  model_name: string;
  n_context: number;
  n_predicted: number;
  metrics: Metrics | null;
  predictions: PredictionRow[];
  distribution: DistributionBin[];
  warnings: string[];
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  stage: string;
  result: PredictResponse | null;
  error: string | null;
  error_code: number | null;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ExplainResponse {
  task: TaskType;
  model_name: string;
  metric_name: string;
  baseline_score: number;
  n_holdout: number;
  importances: FeatureImportance[];
}

export interface ProjectInfo {
  dataset_id: string;
  filename: string;
  created_at: string;
  last_used: string;
  n_results: number;
  target_column: string | null;
  has_result: boolean;
}

export interface PredictOptions {
  model?: string;
  maxContextRows?: number | null;
  task?: TaskType | null;
}

async function handleResponse<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`;
    try {
      const body = await resp.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep generic message */
    }
    throw new Error(detail);
  }
  return resp.json() as Promise<T>;
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function uploadDataset(file: File): Promise<DatasetResponse> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE}/api/datasets`, { method: "POST", body: form });
  return handleResponse<DatasetResponse>(resp);
}

export async function getDataset(datasetId: string): Promise<DatasetResponse> {
  return handleResponse(await fetch(`${API_BASE}/api/datasets/${datasetId}`));
}

export async function getGridPage(
  datasetId: string,
  offset: number,
  limit: number,
): Promise<GridPage> {
  return handleResponse(
    await fetch(`${API_BASE}/api/datasets/${datasetId}/grid?offset=${offset}&limit=${limit}`),
  );
}

export async function setActiveSheet(
  datasetId: string,
  sheetName: string,
): Promise<DatasetResponse> {
  return handleResponse(
    await fetch(
      `${API_BASE}/api/datasets/${datasetId}/sheet`,
      jsonInit("PUT", { sheet_name: sheetName }),
    ),
  );
}

export async function setTableSpec(datasetId: string, spec: TableSpec): Promise<TableProfile> {
  return handleResponse(
    await fetch(`${API_BASE}/api/datasets/${datasetId}/spec`, jsonInit("PUT", spec)),
  );
}

export async function saveRoles(datasetId: string, roles: ColumnRole[]): Promise<void> {
  await fetch(`${API_BASE}/api/datasets/${datasetId}/roles`, jsonInit("PUT", { roles }));
}

export async function listModels(): Promise<ModelInfo[]> {
  return handleResponse(await fetch(`${API_BASE}/api/models`));
}

/** Retry a request with backoff until it succeeds or `cancelled()` — the
 * backend can take a few seconds longer than the dev server to come up. */
export async function retryWhileStarting<T>(
  fetcher: () => Promise<T>,
  cancelled: () => boolean,
): Promise<T | null> {
  let delay = 400;
  while (!cancelled()) {
    try {
      return await fetcher();
    } catch {
      await new Promise((r) => setTimeout(r, delay));
      delay = Math.min(delay * 1.8, 5000);
    }
  }
  return null;
}

export async function listProjects(): Promise<ProjectInfo[]> {
  return handleResponse(await fetch(`${API_BASE}/api/projects`));
}

export async function deleteProject(datasetId: string): Promise<void> {
  await fetch(`${API_BASE}/api/projects/${datasetId}`, { method: "DELETE" });
}

function predictBody(
  datasetId: string,
  targetColumn: string,
  featureColumns: string[],
  opts: PredictOptions,
) {
  return {
    dataset_id: datasetId,
    target_column: targetColumn,
    feature_columns: featureColumns,
    model: opts.model ?? null,
    max_context_rows: opts.maxContextRows ?? null,
    task: opts.task ?? null,
  };
}

export async function startPredictJob(
  datasetId: string,
  targetColumn: string,
  featureColumns: string[],
  opts: PredictOptions = {},
): Promise<JobStatus> {
  return handleResponse(
    await fetch(
      `${API_BASE}/api/predict-jobs`,
      jsonInit("POST", predictBody(datasetId, targetColumn, featureColumns, opts)),
    ),
  );
}

export async function getPredictJob(jobId: string): Promise<JobStatus> {
  return handleResponse(await fetch(`${API_BASE}/api/predict-jobs/${jobId}`));
}

export async function cancelPredictJob(jobId: string): Promise<void> {
  await fetch(`${API_BASE}/api/predict-jobs/${jobId}`, { method: "DELETE" });
}

export async function explain(
  datasetId: string,
  targetColumn: string,
  featureColumns: string[],
  model?: string,
  task?: TaskType | null,
): Promise<ExplainResponse> {
  return handleResponse(
    await fetch(
      `${API_BASE}/api/explain`,
      jsonInit("POST", {
        dataset_id: datasetId,
        target_column: targetColumn,
        feature_columns: featureColumns,
        model: model ?? null,
        task: task ?? null,
      }),
    ),
  );
}

export function resultCsvUrl(predictionId: string): string {
  return `${API_BASE}/api/results/${predictionId}/csv`;
}

export function resultXlsxUrl(predictionId: string): string {
  return `${API_BASE}/api/results/${predictionId}/xlsx`;
}
