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


class PredictRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: list[str]
    model: str | None = None  # registry id; None = server default
    max_context_rows: int | None = None  # subsample examples for huge tables
    task: TaskType | None = None  # manual override; None = auto-detect


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


class FeatureImportance(BaseModel):
    feature: str
    importance: float


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
    n_sheets: int
    has_result: bool
