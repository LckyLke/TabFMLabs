"""SQLite persistence: uploaded files, per-sheet marking, and results.

The in-memory Dataset objects in datasets.py stay the fast path; this module
writes through so projects survive a backend restart and can be reopened.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "STUDIO_DB", os.path.join(os.path.dirname(__file__), "..", "data", "studio.db")
)

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content BLOB NOT NULL,
                created_at TEXT NOT NULL,
                active_sheet TEXT,
                marking TEXT  -- JSON: {sheet_name: {"spec": {...}, "roles": [...]}}
            )"""
        )
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                target_column TEXT NOT NULL,
                response TEXT NOT NULL,  -- PredictResponse JSON
                csv BLOB NOT NULL
            )"""
        )
        _conn.commit()
    return _conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_project(project_id: str, filename: str, content: bytes, active_sheet: str) -> None:
    with _lock:
        _db().execute(
            "INSERT OR REPLACE INTO projects (id, filename, content, created_at, active_sheet, marking) "
            "VALUES (?, ?, ?, ?, ?, COALESCE((SELECT marking FROM projects WHERE id = ?), '{}'))",
            (project_id, filename, content, _now(), active_sheet, project_id),
        )
        _db().commit()


def load_project(project_id: str) -> dict | None:
    with _lock:
        row = (
            _db()
            .execute(
                "SELECT filename, content, active_sheet, marking FROM projects WHERE id = ?",
                (project_id,),
            )
            .fetchone()
        )
    if row is None:
        return None
    return {
        "filename": row[0],
        "content": row[1],
        "active_sheet": row[2],
        "marking": json.loads(row[3] or "{}"),
    }


def update_marking(project_id: str, sheet_name: str, *, spec=None, roles=None, active_sheet=None) -> None:
    with _lock:
        row = _db().execute("SELECT marking FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            return
        marking = json.loads(row[0] or "{}")
        entry = marking.setdefault(sheet_name, {})
        if spec is not None:
            entry["spec"] = spec
        if roles is not None:
            entry["roles"] = roles
        if active_sheet is not None:
            _db().execute(
                "UPDATE projects SET active_sheet = ? WHERE id = ?", (active_sheet, project_id)
            )
        _db().execute(
            "UPDATE projects SET marking = ? WHERE id = ?", (json.dumps(marking), project_id)
        )
        _db().commit()


def find_project_by_content(filename: str, content: bytes) -> str | None:
    """Return the id of an existing project with this exact file, if any."""
    with _lock:
        rows = _db().execute(
            "SELECT id, content FROM projects WHERE filename = ?", (filename,)
        ).fetchall()
    for project_id, blob in rows:
        if blob == content:
            return project_id
    return None


def dedupe_projects() -> int:
    """Collapse projects with identical filename+content (one-time cleanup for
    rows created before uploads reused existing projects). Keeps the copy with
    the most results (newest on a tie); never deletes a row that has results."""
    with _lock:
        rows = _db().execute(
            """SELECT p.id, p.filename, p.content, p.created_at,
                      (SELECT COUNT(*) FROM results r WHERE r.project_id = p.id)
               FROM projects p"""
        ).fetchall()
        groups: dict[tuple, list] = {}
        for pid, filename, content, created_at, n_results in rows:
            groups.setdefault((filename, bytes(content)), []).append(
                (pid, created_at, n_results)
            )
        removed = 0
        for dupes in groups.values():
            if len(dupes) < 2:
                continue
            keep = max(dupes, key=lambda d: (d[2], d[1]))[0]
            for pid, _, n_results in dupes:
                if pid != keep and n_results == 0:
                    _db().execute("DELETE FROM projects WHERE id = ?", (pid,))
                    removed += 1
        if removed:
            _db().commit()
    return removed


def list_projects() -> list[dict]:
    with _lock:
        rows = (
            _db()
            .execute(
                """SELECT p.id, p.filename, p.created_at,
                          (SELECT COUNT(*) FROM results r WHERE r.project_id = p.id),
                          (SELECT MAX(r.created_at) FROM results r WHERE r.project_id = p.id),
                          (SELECT r.target_column FROM results r WHERE r.project_id = p.id
                           ORDER BY r.created_at DESC LIMIT 1)
                   FROM projects p
                   ORDER BY COALESCE(
                       (SELECT MAX(r.created_at) FROM results r WHERE r.project_id = p.id),
                       p.created_at
                   ) DESC
                   LIMIT 50"""
            )
            .fetchall()
        )
    return [
        {
            "dataset_id": r[0],
            "filename": r[1],
            "created_at": r[2],
            "n_results": r[3],
            "last_used": r[4] or r[2],
            "target_column": r[5],
        }
        for r in rows
    ]


def delete_project(project_id: str) -> bool:
    with _lock:
        cur = _db().execute("DELETE FROM projects WHERE id = ?", (project_id,))
        _db().execute("DELETE FROM results WHERE project_id = ?", (project_id,))
        _db().commit()
    return cur.rowcount > 0


def save_result(
    result_id: str,
    project_id: str,
    sheet_name: str,
    target_column: str,
    response_json: str,
    csv_bytes: bytes,
) -> None:
    with _lock:
        _db().execute(
            "INSERT OR REPLACE INTO results (id, project_id, created_at, sheet_name, target_column, response, csv) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (result_id, project_id, _now(), sheet_name, target_column, response_json, csv_bytes),
        )
        _db().commit()


def load_result(result_id: str) -> dict | None:
    with _lock:
        row = (
            _db()
            .execute(
                "SELECT project_id, sheet_name, target_column, response, csv FROM results WHERE id = ?",
                (result_id,),
            )
            .fetchone()
        )
    if row is None:
        return None
    return {
        "project_id": row[0],
        "sheet_name": row[1],
        "target_column": row[2],
        "response": json.loads(row[3]),
        "csv": row[4],
    }
