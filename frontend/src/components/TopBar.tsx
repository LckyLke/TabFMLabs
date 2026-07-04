interface Props {
  filename: string | null;
  nRows: number | null;
  modelName: string | null;
  onReset: () => void;
}

export function TopBar({ filename, nRows, modelName, onReset }: Props) {
  return (
    <header className="topbar">
      <span className="brand">
        <svg className="brand-mark" viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <rect x="1.5" y="1.5" width="17" height="17" rx="4" stroke="currentColor" strokeWidth="1.6" />
          <path d="M1.5 7.5h17M7.5 7.5v11" stroke="currentColor" strokeWidth="1.6" />
          <circle cx="13.5" cy="13.5" r="2.2" fill="currentColor" />
        </svg>
        TabFM Studio
      </span>
      {filename && (
        <span className="file-chip" title={filename}>
          <span className="file-chip-name">{filename}</span>
          {nRows != null && <span className="file-chip-meta">{nRows.toLocaleString()} rows</span>}
          <button className="file-chip-close" onClick={onReset} aria-label="Close file" title="Close file">
            <svg viewBox="0 0 12 12" aria-hidden="true">
              <path d="M2.5 2.5l7 7m0-7l-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </span>
      )}
      <span className="topbar-spacer" />
      <span className="model-chip" title="Runs locally — your data never leaves this machine">
        <span className="model-dot" aria-hidden="true" />
        {modelName ?? "local model"}
      </span>
    </header>
  );
}
