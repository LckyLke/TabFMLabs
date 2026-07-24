"""End-to-end API tests using the baseline backend (no model download needed)."""

import io
import os
import tempfile
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["MODEL_BACKEND"] = "baseline"
os.environ["STUDIO_DB"] = os.path.join(tempfile.mkdtemp(), "test_studio.db")

from app.main import app  # noqa: E402

client = TestClient(app)


def make_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def iris_like() -> pd.DataFrame:
    """60 labeled rows + 5 unlabeled rows, two clearly separable classes."""
    rows = []
    for i in range(60):
        cls = "small" if i % 2 == 0 else "big"
        base = 1.0 if cls == "small" else 10.0
        rows.append({"length": base + (i % 7) * 0.1, "width": base / 2, "kind": cls})
    for i in range(5):
        base = 1.0 if i % 2 == 0 else 10.0
        rows.append({"length": base, "width": base / 2, "kind": None})
    return pd.DataFrame(rows)


def upload(df: pd.DataFrame, name: str = "data.csv") -> dict:
    resp = client.post("/api/datasets", files={"file": (name, make_csv(df), "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def upload_text(text: str, name: str = "data.csv") -> dict:
    resp = client.post("/api/datasets", files={"file": (name, text.encode(), "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_upload_profiles_columns(iris_like):
    data = upload(iris_like)
    assert data["n_raw_rows"] == 66  # header + 65 data rows
    assert data["spec"] == {
        "header_row": 0,
        "start_row": 1,
        "end_row": 65,
        "excluded_rows": [],
    }
    assert data["table"]["n_rows"] == 65
    assert data["grid"][0] == ["length", "width", "kind"]
    cols = {c["name"]: c for c in data["table"]["columns"]}
    assert cols["length"]["kind"] == "numeric"
    assert cols["kind"]["kind"] == "categorical"
    assert cols["kind"]["n_missing"] == 5


def test_upload_excel(iris_like):
    buf = io.BytesIO()
    iris_like.to_excel(buf, index=False)
    resp = client.post(
        "/api/datasets",
        files={"file": ("data.xlsx", buf.getvalue(), "application/vnd.ms-excel")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["table"]["n_rows"] == 65
    assert data["sheet_names"] == ["Sheet1"]


def test_multi_sheet_excel(iris_like):
    buf = io.BytesIO()
    notes = pd.DataFrame({0: ["Just some notes", "nothing tabular here"]})
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        notes.to_excel(xl, sheet_name="Notes", index=False, header=False)
        iris_like.to_excel(xl, sheet_name="Measurements", index=False)
    resp = client.post(
        "/api/datasets",
        files={"file": ("book.xlsx", buf.getvalue(), "application/vnd.ms-excel")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sheet_names"] == ["Notes", "Measurements"]
    assert data["active_sheet"] == "Notes"

    ds = data["dataset_id"]
    resp = client.put(f"/api/datasets/{ds}/sheet", json={"sheet_name": "Measurements"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_sheet"] == "Measurements"
    assert data["table"]["n_rows"] == 65
    assert data["grid"][0] == ["length", "width", "kind"]

    # predict runs on the active sheet
    resp = client.post(
        "/api/predict",
        json={"dataset_id": ds, "target_column": "kind", "feature_columns": ["length", "width"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_predicted"] == 5

    resp = client.put(f"/api/datasets/{ds}/sheet", json={"sheet_name": "Nope"})
    assert resp.status_code == 422


def test_upload_rejects_unknown_extension():
    resp = client.post("/api/datasets", files={"file": ("data.pdf", b"x", "application/pdf")})
    assert resp.status_code == 422


def test_messy_file_spec_flow():
    """Title rows above the header, a totals row below the data."""
    text = (
        "Quarterly Report,,\n"
        ",,\n"
        "region,units,revenue\n"
        "north,10,100\n"
        "south,20,210\n"
        "west,30,290\n"
        "TOTAL,60,600\n"
    )
    data = upload_text(text)
    ds = data["dataset_id"]
    # default spec grabs the first non-empty row as header -> "Quarterly Report"
    assert data["spec"]["header_row"] == 0

    resp = client.put(
        f"/api/datasets/{ds}/spec",
        json={"header_row": 2, "start_row": 3, "end_row": 5, "excluded_rows": []},
    )
    assert resp.status_code == 200, resp.text
    table = resp.json()
    assert table["n_rows"] == 3
    names = [c["name"] for c in table["columns"]]
    assert names == ["region", "units", "revenue"]
    kinds = {c["name"]: c["kind"] for c in table["columns"]}
    assert kinds["units"] == "numeric"  # totals row no longer pollutes types


def test_spec_excluded_rows_and_validation():
    text = "a,b\n1,x\n2,y\nsubtotal,\n3,z\n"
    data = upload_text(text)
    ds = data["dataset_id"]
    resp = client.put(
        f"/api/datasets/{ds}/spec",
        json={"header_row": 0, "start_row": 1, "end_row": 4, "excluded_rows": [3]},
    )
    assert resp.status_code == 200
    table = resp.json()
    assert table["n_rows"] == 3
    assert {c["name"]: c["kind"] for c in table["columns"]}["a"] == "numeric"

    resp = client.put(
        f"/api/datasets/{ds}/spec",
        json={"header_row": 0, "start_row": 1, "end_row": 99, "excluded_rows": []},
    )
    assert resp.status_code == 422

    resp = client.put(
        "/api/datasets/nope/spec",
        json={"header_row": 0, "start_row": 1, "end_row": 2, "excluded_rows": []},
    )
    assert resp.status_code == 404


def test_no_header_spec():
    text = "1,x\n2,y\n3,z\n"
    data = upload_text(text)
    ds = data["dataset_id"]
    resp = client.put(
        f"/api/datasets/{ds}/spec",
        json={"header_row": None, "start_row": 0, "end_row": 2, "excluded_rows": []},
    )
    assert resp.status_code == 200
    table = resp.json()
    assert table["n_rows"] == 3
    assert [c["name"] for c in table["columns"]] == ["Column 1", "Column 2"]


def test_numeric_target_with_unique_values_is_regression():
    """Five distinct revenue figures must not be treated as five classes."""
    text = (
        "region,units,revenue\n"
        "a,1,455000\nb,2,281000\nc,3,588000\nd,4,362000\ne,5,181000\n"
        "f,6,\ng,7,\n"
    )
    data = upload_text(text)
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": data["dataset_id"],
            "target_column": "revenue",
            "feature_columns": ["units"],
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["task"] == "regression"
    # row_index refers to raw file rows (header row 0, so first predict row is 6)
    assert [r["row_index"] for r in result["predictions"]] == [6, 7]


def test_classification_flow(iris_like):
    data = upload(iris_like)
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": data["dataset_id"],
            "target_column": "kind",
            "feature_columns": ["length", "width"],
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["task"] == "classification"
    assert result["n_predicted"] == 5
    assert result["metrics"]["accuracy"] == 1.0
    preds = [r["prediction"] for r in result["predictions"]]
    assert preds == ["small", "big", "small", "big", "small"]
    assert all(r["confidence"] is not None for r in result["predictions"])

    csv_resp = client.get(f"/api/results/{result['prediction_id']}/csv")
    assert csv_resp.status_code == 200
    out = pd.read_csv(io.BytesIO(csv_resp.content))
    assert out["kind (predicted)"].notna().sum() == 5


def test_regression_flow():
    df = pd.DataFrame(
        {
            "x": list(range(50)) + list(range(5)),
            "y": [float(2 * v + 1) for v in range(50)] + [None] * 5,
        }
    )
    data = upload(df)
    resp = client.post(
        "/api/predict",
        json={"dataset_id": data["dataset_id"], "target_column": "y", "feature_columns": ["x"]},
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["task"] == "regression"
    assert result["metrics"]["r2"] is not None
    assert result["metrics"]["mae"] is not None
    assert result["predictions"][0]["confidence"] is None


def test_predict_errors(iris_like):
    data = upload(iris_like)
    ds = data["dataset_id"]

    resp = client.post(
        "/api/predict",
        json={"dataset_id": ds, "target_column": "kind", "feature_columns": ["kind"]},
    )
    assert resp.status_code == 422

    resp = client.post(
        "/api/predict",
        json={"dataset_id": ds, "target_column": "kind", "feature_columns": ["nope"]},
    )
    assert resp.status_code == 422

    # fully-labeled target -> nothing to predict
    full = iris_like.dropna()
    ds2 = upload(full)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={"dataset_id": ds2, "target_column": "kind", "feature_columns": ["length"]},
    )
    assert resp.status_code == 422
    assert "No rows to predict" in resp.json()["detail"]

    resp = client.post(
        "/api/predict",
        json={"dataset_id": "missing", "target_column": "kind", "feature_columns": ["length"]},
    )
    assert resp.status_code == 404


def test_no_fallback_when_tabfm_weights_missing(iris_like, monkeypatch, tmp_path):
    """Without TabFM weights the API must fail (503), never silently substitute."""
    monkeypatch.setenv("TABFM_WEIGHTS_DIR", str(tmp_path))  # empty dir: no weights

    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "tabfm",
        },
    )
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


def test_too_many_classes():
    df = pd.DataFrame(
        {
            "x": range(40),
            "label": [f"c{i % 15}" for i in range(35)] + [None] * 5,
        }
    )
    ds = upload(df)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={"dataset_id": ds, "target_column": "label", "feature_columns": ["x"]},
    )
    assert resp.status_code == 422
    assert "at most 10 classes" in resp.json()["detail"]


def test_task_override():
    """Numeric class codes auto-read as regression can be forced to classification."""
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(30)],
            # 0/1/2 codes but few repeats per value -> auto = regression territory
            "code": [i % 3 for i in range(25)] + [None] * 5,
        }
    )
    ds = upload(df)["dataset_id"]

    base = {"dataset_id": ds, "target_column": "code", "feature_columns": ["x"], "model": "baseline"}
    auto = client.post("/api/predict", json=base).json()
    assert auto["task"] == "classification"  # 3 unique over 25 rows -> auto classification

    forced = client.post("/api/predict", json={**base, "task": "regression"}).json()
    assert forced["task"] == "regression"
    assert forced["metrics"]["r2"] is not None

    forced = client.post("/api/predict", json={**base, "task": "classification"}).json()
    assert forced["task"] == "classification"


def test_task_override_invalid_regression_on_text(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length"],
            "model": "baseline",
            "task": "regression",
        },
    )
    assert resp.status_code == 422
    assert "non-numeric" in resp.json()["detail"]


def test_models_endpoint_and_unknown_model(iris_like):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()]
    assert {"tabfm", "tabpfn", "baseline"} <= set(ids)
    by_id = {m["id"]: m for m in resp.json()}
    assert by_id["tabfm"]["supports_ensemble"] is True
    assert by_id["tabpfn"]["supports_ensemble"] is False
    assert by_id["baseline"]["supports_ensemble"] is False

    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length"],
            "model": "nonsense",
        },
    )
    assert resp.status_code == 422
    assert "Unknown model" in resp.json()["detail"]


def test_ensemble_rejected_for_unsupported_model(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "baseline",
            "ensemble": True,
        },
    )
    assert resp.status_code == 422
    assert "ensemble" in resp.json()["detail"].lower()


def test_confusion_matrix_in_metrics(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "baseline",
        },
    )
    assert resp.status_code == 200
    confusion = resp.json()["metrics"]["confusion"]
    assert sorted(confusion["labels"]) == ["big", "small"]
    assert sum(sum(r) for r in confusion["matrix"]) == resp.json()["metrics"]["n_holdout"]


def test_regression_holdout_samples():
    df = pd.DataFrame(
        {
            "x": list(range(60)) + list(range(5)),
            "y": [float(3 * v) for v in range(60)] + [None] * 5,
        }
    )
    ds = upload(df)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={"dataset_id": ds, "target_column": "y", "feature_columns": ["x"], "model": "baseline"},
    )
    assert resp.status_code == 200
    samples = resp.json()["metrics"]["holdout_samples"]
    assert len(samples) == resp.json()["metrics"]["n_holdout"]
    assert {"actual", "predicted"} <= set(samples[0])


def test_subsampling_context(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "baseline",
            "max_context_rows": 30,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_context"] == 30
    assert any("random sample" in w for w in body["warnings"])


def test_explain(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/explain",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "baseline",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["metric_name"] == "accuracy"
    assert body["baseline_score"] == 1.0
    features = {fi["feature"] for fi in body["importances"]}
    assert features == {"length", "width"}
    # length separates the classes perfectly; shuffling it must hurt
    by_name = {fi["feature"]: fi["importance"] for fi in body["importances"]}
    assert by_name["length"] > 0


def test_xlsx_download(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "baseline",
        },
    )
    prediction_id = resp.json()["prediction_id"]
    resp = client.get(f"/api/results/{prediction_id}/xlsx")
    assert resp.status_code == 200
    wb = pd.read_excel(io.BytesIO(resp.content), sheet_name=None, header=None)
    sheet = next(iter(wb.values()))
    # the previously-empty target cells are now filled with predictions
    assert sheet.iloc[61, 2] in ("small", "big")


def test_predict_job_lifecycle(iris_like):
    ds = upload(iris_like)["dataset_id"]
    resp = client.post(
        "/api/predict-jobs",
        json={
            "dataset_id": ds,
            "target_column": "kind",
            "feature_columns": ["length", "width"],
            "model": "baseline",
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/api/predict-jobs/{job_id}").json()
        if status["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert status["result"]["n_predicted"] == 5

    assert client.get("/api/predict-jobs/nope").status_code == 404


def test_projects_and_rehydration(iris_like):
    from app import datasets as ds_module

    data = upload(iris_like, name="persisted.csv")
    ds = data["dataset_id"]
    # marking round-trip
    resp = client.put(
        f"/api/datasets/{ds}/roles",
        json={"roles": ["feature", "feature", "target"]},
    )
    assert resp.status_code == 200

    projects = client.get("/api/projects").json()
    assert any(p["dataset_id"] == ds and p["filename"] == "persisted.csv" for p in projects)

    # simulate a backend restart: drop in-memory state, then reopen
    ds_module._datasets.clear()
    resp = client.get(f"/api/datasets/{ds}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "persisted.csv"
    assert body["roles"] == ["feature", "feature", "target"]
    assert body["table"]["n_rows"] == 65

    assert client.delete(f"/api/projects/{ds}").status_code == 200
    assert client.get(f"/api/datasets/{ds}").status_code == 404


def test_reupload_reuses_project(iris_like):
    """Uploading the exact same file twice reopens the project, no duplicate."""
    first = upload(iris_like, name="reused.csv")
    client.put(
        f"/api/datasets/{first['dataset_id']}/roles",
        json={"roles": ["feature", "feature", "target"]},
    )

    second = upload(iris_like, name="reused.csv")
    assert second["dataset_id"] == first["dataset_id"]
    assert second["roles"] == ["feature", "feature", "target"]  # marking survives

    projects = client.get("/api/projects").json()
    assert sum(p["filename"] == "reused.csv" for p in projects) == 1

    # same name but different content is a genuinely new project
    changed = iris_like.copy()
    changed.loc[0, "length"] = 99.9
    third = upload(changed, name="reused.csv")
    assert third["dataset_id"] != first["dataset_id"]

    client.delete(f"/api/projects/{first['dataset_id']}")
    client.delete(f"/api/projects/{third['dataset_id']}")


def test_dedupe_projects_startup_cleanup(iris_like):
    """Pre-existing duplicate rows collapse to one, preferring rows with results."""
    from app import store

    upload(iris_like, name="dupes.csv")
    content = make_csv(iris_like)
    store.save_project("dupe-a", "dupes.csv", content, "Sheet1")
    store.save_project("dupe-b", "dupes.csv", content, "Sheet1")

    removed = store.dedupe_projects()
    assert removed == 2
    projects = client.get("/api/projects").json()
    ids = [p["dataset_id"] for p in projects if p["filename"] == "dupes.csv"]
    assert len(ids) == 1

    client.delete(f"/api/projects/{ids[0]}")


def test_grid_pagination():
    df = pd.DataFrame({"a": range(1200), "b": [None if i > 1000 else i for i in range(1200)]})
    data = upload(df)
    assert data["grid_truncated"] is True
    assert len(data["grid"]) == 500
    ds = data["dataset_id"]
    resp = client.get(f"/api/datasets/{ds}/grid", params={"offset": 500, "limit": 500})
    assert resp.status_code == 200
    page = resp.json()
    assert page["offset"] == 500
    assert len(page["rows"]) == 500
    assert page["rows"][0][0] == "499"  # raw row 500 = data row 499 (header at 0)
