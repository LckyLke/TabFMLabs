"""File parsing, table-region handling, column profiling, and the dataset store.

Uploads are kept as a raw string grid (no header assumption). A TableSpec —
which raw row is the header, where the data starts/ends, which rows are
excluded — turns the grid into a typed DataFrame. The spec is user-editable so
files with title rows, footnotes, or totals below the data still work.
"""

import io
import uuid
from dataclasses import dataclass

import pandas as pd

from . import store
from .schemas import ColumnInfo, ColumnKind, TableSpec

MAX_GRID_ROWS = 500  # initial payload; more rows are fetched page-wise
MAX_FEATURES_RECOMMENDED = 500
MAX_CLASSES = 10


@dataclass
class Sheet:
    raw: pd.DataFrame  # all cells as strings, "" for empty, no header applied
    spec: TableSpec
    df: pd.DataFrame  # typed frame under the current spec
    roles: list[str] | None = None  # persisted column marking, if any


@dataclass
class Dataset:
    filename: str
    sheets: dict[str, Sheet]
    active: str

    @property
    def sheet(self) -> Sheet:
        return self.sheets[self.active]


_datasets: dict[str, Dataset] = {}


def _clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.fillna("")
    raw.columns = range(raw.shape[1])
    return raw


def parse_upload(filename: str, content: bytes) -> dict[str, pd.DataFrame]:
    """Parse to raw string grids, one per sheet; headers are set via TableSpec."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(
            io.BytesIO(content),
            sheet_name=None,  # all sheets
            header=None,
            dtype=str,
            keep_default_na=False,
        )
        raws = {
            sheet_name: _clean_raw(raw)
            for sheet_name, raw in sheets.items()
            if not raw.empty
        }
        if not raws:
            raise ValueError("The workbook contains no non-empty sheets.")
        return raws
    if name.endswith((".csv", ".txt", ".tsv")):
        # sep=None lets pandas sniff comma/semicolon/tab separators
        raw = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            header=None,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
        )
        if raw.empty or len(raw) < 2:
            raise ValueError("The file needs at least a header row and one data row.")
        return {"Data": _clean_raw(raw)}
    raise ValueError(f"Unsupported file type: {filename}. Use CSV or Excel.")


def default_spec(raw: pd.DataFrame) -> TableSpec:
    """Header = first non-empty row, data = everything after it."""
    if len(raw) < 2:
        return TableSpec(header_row=None, start_row=0, end_row=len(raw) - 1)
    header_row = 0
    for i in range(min(len(raw), 20)):
        if any(str(v).strip() for v in raw.iloc[i]):
            header_row = i
            break
    if header_row + 1 >= len(raw):  # non-empty content only on the last row
        return TableSpec(header_row=None, start_row=0, end_row=len(raw) - 1)
    return TableSpec(
        header_row=header_row, start_row=header_row + 1, end_row=len(raw) - 1
    )


def validate_spec(raw: pd.DataFrame, spec: TableSpec) -> None:
    n = len(raw)
    if spec.header_row is not None and not (0 <= spec.header_row < n):
        raise ValueError(f"Header row {spec.header_row + 1} is outside the file.")
    if not (0 <= spec.start_row <= spec.end_row < n):
        raise ValueError("The data range is outside the file.")


def apply_spec(raw: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Slice the raw grid to the data region and infer column types."""
    excluded = set(spec.excluded_rows)
    if spec.header_row is not None:
        excluded.add(spec.header_row)
    rows = [
        i for i in range(spec.start_row, spec.end_row + 1) if i not in excluded
    ]
    if not rows:
        raise ValueError("The selected data range contains no rows.")
    # Keep raw row numbers as the index so results can reference file rows.
    df = raw.loc[rows].copy()
    df.columns = _column_names(raw, spec)

    for col in df.columns:
        series = df[col].astype(str).str.strip().replace("", pd.NA)
        non_empty = series.dropna()
        if len(non_empty) > 0:
            numeric = pd.to_numeric(non_empty, errors="coerce")
            if numeric.notna().all():  # strict: every filled cell parses
                df[col] = pd.to_numeric(series, errors="coerce")
                continue
        df[col] = series
    return df


def _column_names(raw: pd.DataFrame, spec: TableSpec) -> list[str]:
    if spec.header_row is None:
        return [f"Column {i + 1}" for i in range(raw.shape[1])]
    names = []
    seen: dict[str, int] = {}
    for i, val in enumerate(raw.iloc[spec.header_row]):
        name = str(val).strip() or f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        names.append(name)
    return names


def _build_sheets(raws: dict[str, pd.DataFrame]) -> dict[str, Sheet]:
    sheets = {}
    for sheet_name, raw in raws.items():
        spec = default_spec(raw)
        sheets[sheet_name] = Sheet(raw=raw, spec=spec, df=apply_spec(raw, spec))
    return sheets


def store_dataset(
    filename: str, raws: dict[str, pd.DataFrame], content: bytes
) -> tuple[str, Dataset]:
    # Re-uploading the exact same file reopens the existing project (with its
    # saved marking and results) instead of creating a duplicate.
    existing = store.find_project_by_content(filename, content)
    if existing is not None:
        return existing, get_dataset(existing)
    sheets = _build_sheets(raws)
    # Prefer the first sheet that actually has a table (≥ 2 raw rows).
    active = next(
        (n for n, s in sheets.items() if len(s.raw) >= 2), next(iter(sheets))
    )
    ds = Dataset(filename=filename, sheets=sheets, active=active)
    dataset_id = uuid.uuid4().hex[:12]
    _datasets[dataset_id] = ds
    store.save_project(dataset_id, filename, content, active)
    return dataset_id, ds


def get_dataset(dataset_id: str) -> Dataset:
    if dataset_id in _datasets:
        return _datasets[dataset_id]
    ds = _rehydrate(dataset_id)
    if ds is None:
        raise KeyError(dataset_id)
    _datasets[dataset_id] = ds
    return ds


def _rehydrate(dataset_id: str) -> Dataset | None:
    """Rebuild a Dataset from the on-disk project after a backend restart."""
    project = store.load_project(dataset_id)
    if project is None:
        return None
    raws = parse_upload(project["filename"], project["content"])
    sheets = _build_sheets(raws)
    for sheet_name, entry in project["marking"].items():
        sheet = sheets.get(sheet_name)
        if sheet is None:
            continue
        if entry.get("spec"):
            try:
                spec = TableSpec(**entry["spec"])
                validate_spec(sheet.raw, spec)
                sheet.df = apply_spec(sheet.raw, spec)
                sheet.spec = spec
            except (ValueError, TypeError):
                pass  # stored marking no longer fits the file; keep defaults
        roles = entry.get("roles")
        if roles and len(roles) == sheet.raw.shape[1]:
            sheet.roles = roles
    active = project["active_sheet"]
    if active not in sheets:
        active = next((n for n, s in sheets.items() if len(s.raw) >= 2), next(iter(sheets)))
    return Dataset(filename=project["filename"], sheets=sheets, active=active)


def set_active_sheet(dataset_id: str, sheet_name: str) -> Dataset:
    ds = get_dataset(dataset_id)
    if sheet_name not in ds.sheets:
        raise ValueError(f"No sheet named “{sheet_name}” in this file.")
    ds.active = sheet_name
    store.update_marking(dataset_id, sheet_name, active_sheet=sheet_name)
    return ds


def update_spec(dataset_id: str, spec: TableSpec) -> Dataset:
    ds = get_dataset(dataset_id)
    sheet = ds.sheet
    validate_spec(sheet.raw, spec)
    sheet.df = apply_spec(sheet.raw, spec)
    sheet.spec = spec
    store.update_marking(dataset_id, ds.active, spec=spec.model_dump())
    return ds


def set_roles(dataset_id: str, roles: list[str]) -> Dataset:
    ds = get_dataset(dataset_id)
    if len(roles) != ds.sheet.raw.shape[1]:
        raise ValueError("Role list length does not match the number of columns.")
    ds.sheet.roles = roles
    store.update_marking(dataset_id, ds.active, roles=roles)
    return ds


def grid_rows(raw: pd.DataFrame) -> tuple[list[list[str]], bool]:
    head = raw.head(MAX_GRID_ROWS)
    return head.values.tolist(), len(raw) > MAX_GRID_ROWS


def grid_page(raw: pd.DataFrame, offset: int, limit: int) -> list[list[str]]:
    return raw.iloc[offset : offset + limit].values.tolist()


def column_kind(series: pd.Series) -> ColumnKind:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    n_unique = series.nunique(dropna=True)
    # High-cardinality string columns are likely free text / identifiers
    if n_unique > max(50, len(series) * 0.5):
        return "text"
    return "categorical"


def profile_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    infos = []
    for name in df.columns:
        series = df[name]
        samples = series.dropna().astype(str).unique()[:5].tolist()
        infos.append(
            ColumnInfo(
                name=name,
                kind=column_kind(series),
                n_missing=int(series.isna().sum()),
                n_unique=int(series.nunique(dropna=True)),
                sample_values=samples,
            )
        )
    return infos


def dataset_warnings(df: pd.DataFrame, columns: list[ColumnInfo]) -> list[str]:
    warnings = []
    if len(df.columns) > MAX_FEATURES_RECOMMENDED:
        warnings.append(
            f"This table has {len(df.columns)} columns; TabFM is optimized for up to "
            f"{MAX_FEATURES_RECOMMENDED}. Consider ignoring some columns."
        )
    if len(df) > 10_000:
        warnings.append(
            f"This table has {len(df):,} rows. All labeled rows are used as model "
            "context, so prediction may be slow and memory-heavy."
        )
    if not any(c.n_missing > 0 for c in columns):
        warnings.append(
            "No column has empty cells. Rows to predict are identified by empty "
            "cells in the column you mark as “Predict” — leave it blank in the "
            "rows you want filled in."
        )
    return warnings
