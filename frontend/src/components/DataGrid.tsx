import { useEffect, useRef, useState } from "react";
import type { ColumnInfo, ColumnRole, TableSpec } from "../api";

export interface CellPrediction {
  value: string;
  confidence: number | null;
}

export interface GridProps {
  grid: string[][];
  gridTruncated: boolean;
  nRawRows: number;
  spec: TableSpec;
  columns: ColumnInfo[]; // profile under current spec, same order as grid columns
  roles: ColumnRole[];
  predictions: Map<number, CellPrediction> | null; // raw row index -> predicted value
  busy: boolean;
  onRolesChange: (roles: ColumnRole[]) => void;
  onSpecChange: (spec: TableSpec) => void;
  onNearEnd?: () => void; // ask for more rows (lazy loading)
}

const ROW_H = 33; // matches the CSS row height; used for virtual scrolling
const WINDOW_THRESHOLD = 150; // small tables render fully (stable DOM for tests/tutorial)
const OVERSCAN = 25;

type Popover =
  | { kind: "column"; index: number; x: number; y: number; up: boolean }
  | { kind: "row"; index: number; x: number; y: number; up: boolean };

const POPOVER_EST_H = 320; // px; open the menu upward when a cell is closer than this to the viewport bottom

const KIND_LABELS: Record<ColumnInfo["kind"], string> = {
  numeric: "Number",
  categorical: "Category",
  boolean: "Yes/No",
  datetime: "Date",
  text: "Text / ID",
};

function KindIcon({ kind }: { kind: ColumnInfo["kind"] }) {
  const paths: Record<string, string> = {
    numeric: "M5 2 4 14M12 2l-1 12M2.5 5.5h12M1.5 10.5h12",
    categorical:
      "M2 2h5.2c.4 0 .8.16 1.06.44l5.3 5.3a1.5 1.5 0 0 1 0 2.12l-3.7 3.7a1.5 1.5 0 0 1-2.12 0l-5.3-5.3A1.5 1.5 0 0 1 2 7.2V2Zm3.5 3.5h.01",
    boolean: "M2 8a6 6 0 0 1 6-6h0a6 6 0 1 1-6 6Zm3-3 6 6",
    datetime: "M3 3h10v10H3zM3 6.5h10M6 1.5v3M10 1.5v3",
    text: "M2.5 3h11M5 3v10M8 13h-6M11 8h3.5M12.75 8v5",
  };
  return (
    <svg className="kind-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[kind] ?? paths.text} />
    </svg>
  );
}

function RoleBadge({ role }: { role: ColumnRole }) {
  if (role === "target")
    return (
      <span className="role-badge role-badge-target">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <circle cx="8" cy="8" r="5.5" />
          <circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none" />
        </svg>
        Predict
      </span>
    );
  if (role === "ignore") return <span className="role-badge role-badge-ignore">Ignored</span>;
  return <span className="role-badge role-badge-input">Input</span>;
}

export function DataGrid({
  grid,
  gridTruncated,
  nRawRows,
  spec,
  columns,
  roles,
  predictions,
  busy,
  onRolesChange,
  onSpecChange,
  onNearEnd,
}: GridProps) {
  const [popover, setPopover] = useState<Popover | null>(null);
  const [viewport, setViewport] = useState({ top: 0, height: 600 });
  const wrapRef = useRef<HTMLDivElement>(null);

  const windowed = grid.length > WINDOW_THRESHOLD;
  let start = 0;
  let end = grid.length;
  if (windowed) {
    start = Math.max(0, Math.floor(viewport.top / ROW_H) - OVERSCAN);
    end = Math.min(grid.length, Math.ceil((viewport.top + viewport.height) / ROW_H) + OVERSCAN);
  }

  function onScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    if (windowed) setViewport({ top: el.scrollTop, height: el.clientHeight });
    if (onNearEnd && el.scrollTop + el.clientHeight > el.scrollHeight - 800) onNearEnd();
  }

  useEffect(() => {
    if (!popover) return;
    function onDown(e: MouseEvent) {
      if (!(e.target as HTMLElement).closest(".popover")) setPopover(null);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPopover(null);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [popover]);

  const nCols = grid[0]?.length ?? 0;
  const targetIndex = roles.indexOf("target");
  const excluded = new Set(spec.excluded_rows);

  function openPopover(e: React.MouseEvent, p: Omit<Popover, "x" | "y" | "up">) {
    const wrap = wrapRef.current!.getBoundingClientRect();
    const cell = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const up = cell.bottom + POPOVER_EST_H > window.innerHeight && cell.top - POPOVER_EST_H >= 0;
    setPopover({
      ...p,
      x: Math.min(cell.left - wrap.left, wrap.width - 280),
      // top offset normally; distance from the wrap's bottom edge when flipped up
      y: up ? wrap.bottom - cell.top + 4 : cell.bottom - wrap.top + 4,
      up,
    } as Popover);
  }

  function setRole(index: number, role: ColumnRole) {
    const next = roles.map((r, i) =>
      i === index ? role : role === "target" && r === "target" ? "feature" : r,
    ) as ColumnRole[];
    onRolesChange(next);
    setPopover(null);
  }

  function rowState(i: number): "header" | "skipped" | "excluded" | "data" {
    if (i === spec.header_row) return "header";
    if (i < spec.start_row || i > spec.end_row) return "skipped";
    if (excluded.has(i)) return "excluded";
    return "data";
  }

  function rowAction(action: string, i: number) {
    if (action === "header") {
      onSpecChange({
        ...spec,
        header_row: i,
        start_row: Math.max(spec.start_row, i + 1),
        excluded_rows: spec.excluded_rows.filter((r) => r !== i),
      });
    } else if (action === "no-header") {
      onSpecChange({ ...spec, header_row: null });
    } else if (action === "start") {
      onSpecChange({ ...spec, start_row: i, end_row: Math.max(spec.end_row, i) });
    } else if (action === "end") {
      onSpecChange({ ...spec, end_row: i, start_row: Math.min(spec.start_row, i) });
    } else if (action === "exclude") {
      onSpecChange({ ...spec, excluded_rows: [...spec.excluded_rows, i].sort((a, b) => a - b) });
    } else if (action === "include") {
      onSpecChange({ ...spec, excluded_rows: spec.excluded_rows.filter((r) => r !== i) });
    }
    setPopover(null);
  }

  return (
    <div className={`grid-wrap ${busy ? "is-busy" : ""}`} ref={wrapRef}>
      <div className="grid-scroll" role="grid" aria-label="Your data" onScroll={onScroll}>
        <table className="grid-table">
          <thead>
            <tr>
              <th className="gutter gutter-corner" aria-hidden="true" />
              {Array.from({ length: nCols }, (_, c) => {
                const col = columns[c];
                const role = roles[c];
                return (
                  <th
                    key={c}
                    className={`grid-col-header role-${role} ${popover?.kind === "column" && popover.index === c ? "is-open" : ""}`}
                    onClick={(e) => openPopover(e, { kind: "column", index: c })}
                    tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && openPopover(e as unknown as React.MouseEvent, { kind: "column", index: c })}
                    role="columnheader"
                    aria-haspopup="menu"
                  >
                    <span className="col-header-top">
                      {col && <KindIcon kind={col.kind} />}
                      <span className="col-header-name">{col?.name ?? `Column ${c + 1}`}</span>
                      <svg className="col-header-caret" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M3 4.5 6 7.5l3-3" />
                      </svg>
                    </span>
                    <RoleBadge role={role} />
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {windowed && start > 0 && (
              <tr aria-hidden="true" style={{ height: start * ROW_H }} />
            )}
            {grid.slice(start, end).map((row, sliceIdx) => {
              const i = start + sliceIdx;
              const state = rowState(i);
              return (
                <tr key={i} className={`grid-row row-${state}`}>
                  <td
                    className="gutter"
                    onClick={(e) => openPopover(e, { kind: "row", index: i })}
                    tabIndex={0}
                    role="rowheader"
                    aria-haspopup="menu"
                  >
                    <span className="gutter-num">{i + 1}</span>
                    {state === "header" && <span className="gutter-tag">header</span>}
                    {state === "excluded" && <span className="gutter-tag">excluded</span>}
                  </td>
                  {row.map((cell, c) => {
                    const isPredictCell =
                      state === "data" && c === targetIndex && cell.trim() === "";
                    const prediction = isPredictCell ? predictions?.get(i) : undefined;
                    return (
                      <td
                        key={c}
                        className={`grid-cell role-${roles[c]} ${isPredictCell ? "cell-predict" : ""}`}
                      >
                        {prediction ? (
                          <span
                            className="cell-predicted"
                            title={
                              prediction.confidence != null
                                ? `Predicted · ${(prediction.confidence * 100).toFixed(0)}% confident`
                                : "Predicted"
                            }
                          >
                            <svg viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
                              <path d="M6 1l1.2 3.3L10.5 5.5 7.2 6.7 6 10 4.8 6.7 1.5 5.5l3.3-1.2z" />
                            </svg>
                            {prediction.value}
                          </span>
                        ) : isPredictCell ? (
                          <span className="predict-pill">predict</span>
                        ) : (
                          cell
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
            {windowed && end < grid.length && (
              <tr aria-hidden="true" style={{ height: (grid.length - end) * ROW_H }} />
            )}
          </tbody>
        </table>
        {gridTruncated && (
          <div className="grid-truncated">
            {grid.length.toLocaleString()} of {nRawRows.toLocaleString()} rows loaded — scroll to
            load more; marking and prediction always apply to the whole file.
          </div>
        )}
        <div className="grid-tail" aria-hidden="true" />
      </div>

      {popover?.kind === "column" && (
        <div
          className="popover"
          style={popover.up ? { left: popover.x, bottom: popover.y } : { left: popover.x, top: popover.y }}
          role="menu"
        >
          {(() => {
            const col = columns[popover.index];
            return (
              <>
                <div className="popover-head">
                  <span className="popover-title">{col?.name ?? `Column ${popover.index + 1}`}</span>
                  {col && (
                    <span className="popover-meta">
                      {KIND_LABELS[col.kind]}
                      {col.n_missing > 0 && ` · ${col.n_missing.toLocaleString()} empty`}
                      {` · ${col.n_unique.toLocaleString()} unique`}
                    </span>
                  )}
                </div>
                {(
                  [
                    ["target", "Predict this column", "The model fills in its empty cells"],
                    ["feature", "Use as input", "The model may learn from it"],
                    ["ignore", "Ignore", "Left out entirely"],
                  ] as const
                ).map(([role, label, sub]) => (
                  <button
                    key={role}
                    className={`popover-item ${roles[popover.index] === role ? "is-active" : ""}`}
                    onClick={() => setRole(popover.index, role)}
                    role="menuitemradio"
                    aria-checked={roles[popover.index] === role}
                  >
                    <span className="popover-item-label">{label}</span>
                    <span className="popover-item-sub">{sub}</span>
                  </button>
                ))}
              </>
            );
          })()}
        </div>
      )}

      {popover?.kind === "row" && (
        <div
          className="popover"
          style={popover.up ? { left: popover.x, bottom: popover.y } : { left: popover.x, top: popover.y }}
          role="menu"
        >
          <div className="popover-head">
            <span className="popover-title">Row {popover.index + 1}</span>
          </div>
          {popover.index !== spec.header_row ? (
            <button className="popover-item" onClick={() => rowAction("header", popover.index)}>
              <span className="popover-item-label">Use as header row</span>
              <span className="popover-item-sub">Column names come from this row</span>
            </button>
          ) : (
            <button className="popover-item" onClick={() => rowAction("no-header", popover.index)}>
              <span className="popover-item-label">File has no header row</span>
              <span className="popover-item-sub">Columns are named Column 1, 2, …</span>
            </button>
          )}
          <button className="popover-item" onClick={() => rowAction("start", popover.index)}>
            <span className="popover-item-label">Data starts here</span>
            <span className="popover-item-sub">Rows above are skipped</span>
          </button>
          <button className="popover-item" onClick={() => rowAction("end", popover.index)}>
            <span className="popover-item-label">Data ends here</span>
            <span className="popover-item-sub">Rows below are skipped</span>
          </button>
          {rowState(popover.index) === "data" && (
            <button className="popover-item" onClick={() => rowAction("exclude", popover.index)}>
              <span className="popover-item-label">Exclude this row</span>
              <span className="popover-item-sub">e.g. a subtotal in the middle</span>
            </button>
          )}
          {rowState(popover.index) === "excluded" && (
            <button className="popover-item" onClick={() => rowAction("include", popover.index)}>
              <span className="popover-item-label">Include this row again</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
