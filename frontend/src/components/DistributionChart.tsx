import { useState } from "react";
import type { DistributionBin, TaskType } from "../api";

interface Props {
  bins: DistributionBin[];
  task: TaskType;
}

/** Single-series horizontal bar chart of predicted values (no legend needed). */
export function DistributionChart({ bins, task }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  if (bins.length === 0) return null;
  const max = Math.max(...bins.map((b) => b.count));
  const total = bins.reduce((s, b) => s + b.count, 0);

  return (
    <figure className="dist-chart">
      <figcaption>
        {task === "classification" ? "Predicted values" : "Predicted value ranges"}
      </figcaption>
      <div className="dist-rows">
        {bins.map((bin, i) => (
          <div
            key={bin.label}
            className="dist-row"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          >
            <span className="dist-label" title={bin.label}>
              {bin.label}
            </span>
            <span className="dist-track">
              <span
                className="dist-bar"
                style={{ width: `${(bin.count / max) * 100}%` }}
                aria-hidden="true"
              />
              {hover === i && (
                <span className="dist-tooltip" role="status">
                  {bin.label}: {bin.count.toLocaleString()} rows (
                  {((bin.count / total) * 100).toFixed(1)}%)
                </span>
              )}
            </span>
            <span className="dist-value">{bin.count.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}
