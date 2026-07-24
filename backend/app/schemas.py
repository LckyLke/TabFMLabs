"""Pydantic models for the API."""

from typing import Literal

from pydantic import BaseModel

ColumnKind = Literal["numeric", "categorical", "boolean", "datetime", "text"]
TaskType = Literal["classification", "regression"]


class ColumnInfo(BaseModel):
    name: str
    kind: ColumnKind
    n_missing: int
    n_unique: int
    sample_values: list[str]


class TableSpec(BaseModel):
    """Which raw rows form the table: header, data range, and exclusions."""

    header_row: int | None
    start_row: int
    end_row: int  # inclusive
    excluded_rows: list[int] = []


class TableProfile(BaseModel):
    n_rows: int
    columns: list[ColumnInfo]
    warnings: list[str]


class DatasetResponse(BaseModel):
    dataset_id: str
    filename: str
    sheet_names: list[str]
    active_sheet: str
    n_raw_rows: int
    grid: list[list[str]]
    grid_truncated: bool
    spec: TableSpec
    table: TableProfile
    roles: list[str] | None = None  # persisted marking, if any


class SheetRequest(BaseModel):
    sheet_name: str


class RolesRequest(BaseModel):
    roles: list[str]


class GridPage(BaseModel):
    offset: int
    rows: list[list[str]]
    n_raw_rows: int


class ModelInfo(BaseModel):
    id: str
    label: str
    description: str
    is_default: bool
    supports_ensemble: bool


class PredictRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    model: str | None = None  # registry id; None = server default
    max_context_rows: int | None = None  # subsample examples for huge tables
    task: TaskType | None = None  # manual override; None = auto-detect
    ensemble: bool = False  # TabFM-Ensemble preset; only for models that support it


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]  # rows = actual, cols = predicted


class HoldoutSample(BaseModel):
    actual: float
    predicted: float


class Metrics(BaseModel):
    task: TaskType
    n_holdout: int
    # classification
    accuracy: float | None = None
    f1_macro: float | None = None
    confusion: ConfusionMatrix | None = None
    # regression
    r2: float | None = None
    mae: float | None = None
    holdout_samples: list[HoldoutSample] | None = None


class ImputeRequest(BaseModel):
    dataset_id: str
    model: str | None = None
    ensemble: bool = False
    max_context_rows: int | None = None


class ImputedColumn(BaseModel):
    column: str
    task: TaskType
    n_filled: int


class ImputeResponse(BaseModel):
    prediction_id: str
    model_name: str
    columns: list[ImputedColumn]
    n_cells_filled: int
    warnings: list[str]


class CvCheckResponse(BaseModel):
    task: TaskType
    model_name: str
    metric_name: str  # "accuracy" or "R²"
    n_folds: int
    n_labeled: int
    scores: list[float]  # primary metric per fold
    mean: float
    std: float


class PredictionRow(BaseModel):
    row_index: int
    values: dict[str, str | None]
    prediction: str
    confidence: float | None = None


class DistributionBin(BaseModel):
    label: str
    count: int


class PredictResponse(BaseModel):
    prediction_id: str
    task: TaskType
    model_name: str
    n_context: int
    n_predicted: int
    metrics: Metrics | None
    predictions: list[PredictionRow]
    distribution: list[DistributionBin]
    warnings: list[str]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    stage: str  # human-readable, e.g. "accuracy check", "predicting 15 rows"
    result: PredictResponse | None = None
    error: str | None = None
    error_code: int | None = None


class ExplainRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    model: str | None = None
    task: TaskType | None = None
    ensemble: bool = False


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ExplainRowRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    row_index: int  # raw-grid row index of a row being predicted
    model: str | None = None
    task: TaskType | None = None
    ensemble: bool = False


class RowContribution(BaseModel):
    feature: str
    value: str | None  # the row's actual value
    typical: str  # the stand-in (median / most common) used instead
    prediction: str  # what the model would say with the stand-in
    impact: float  # |Δ prediction| for regression; 1.0 = class flip


class ExplainRowResponse(BaseModel):
    row_index: int
    task: TaskType
    model_name: str
    prediction: str
    contributions: list[RowContribution]  # sorted, biggest impact first


class ExplainResponse(BaseModel):
    task: TaskType
    model_name: str
    metric_name: str  # "accuracy" or "R²"
    baseline_score: float
    n_holdout: int
    importances: list[FeatureImportance]


class ProjectInfo(BaseModel):
    dataset_id: str
    filename: str
    created_at: str
    last_used: str
    n_results: int
    target_column: str | None
    has_result: bool
