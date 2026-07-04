import { useEffect, useRef, useState } from "react";

export interface TutorialSnapshot {
  activeSheet: string | null;
  headerRow: number | null;
  endRow: number | null;
  hasTarget: boolean;
  rowsToPredict: number;
  busy: boolean;
  hasResult: boolean;
}

interface Step {
  id: string;
  target: string | null; // CSS selector; null = centered modal
  title: string;
  body: string;
  /** When set, the step is interactive: it advances once the predicate holds. */
  advanceWhen?: (s: TutorialSnapshot) => boolean;
}

const STEPS: Step[] = [
  {
    id: "welcome",
    target: null,
    title: "Welcome to TabFM Studio",
    body: "In about two minutes you'll tidy a messy sheet and let a foundation model fill in missing values — no training, no code. We've loaded a demo workbook for you.",
  },
  {
    id: "sheets",
    target: ".sheet-tabs",
    title: "Files can have several sheets",
    body: "This workbook has three. “Read me” is just notes — click the “Regional Sales” tab to open a real table.",
    advanceWhen: (s) => s.activeSheet === "Regional Sales",
  },
  {
    id: "header",
    target: ".grid-table tbody tr:nth-child(3) .gutter",
    title: "Tell it where your table starts",
    body: "The title rows above confused the auto-detection — the real column names are in row 3. Click the row number 3 and choose “Use as header row”.",
    advanceWhen: (s) => s.activeSheet === "Regional Sales" && s.headerRow === 2,
  },
  {
    id: "trim",
    target: ".grid-table tbody tr:nth-child(27) .gutter",
    title: "Trim what isn’t data",
    body: "Row 28 is a TOTAL and row 29 a footnote — they'd poison the model. Click row number 27 and choose “Data ends here”.",
    advanceWhen: (s) => s.activeSheet === "Regional Sales" && s.endRow === 26,
  },
  {
    id: "checklist",
    target: ".rail-card",
    title: "The checklist tracks your progress",
    body: "Struck-through rows are excluded, and this panel always shows what's left before you can predict. Ignored columns (like IDs) are handled the same way — via the column headers.",
  },
  {
    id: "switch",
    target: ".sheet-tabs",
    title: "Now let’s predict something",
    body: "Switch to the “Customer Churn” sheet — a subscription list where 8 accounts have an unknown churn status.",
    advanceWhen: (s) => s.activeSheet === "Customer Churn",
  },
  {
    id: "target",
    target: ".grid-table thead th:nth-of-type(7)",
    title: "Pick the column to predict",
    body: "Click the “churned” column header and choose “Predict this column”.",
    advanceWhen: (s) => s.activeSheet === "Customer Churn" && s.hasTarget,
  },
  {
    id: "empty-cells",
    target: ".steps li:nth-child(3)",
    title: "Empty cells mark the rows to predict",
    body: "8 accounts have a blank “churned” cell — exactly those get predicted. The 60 filled rows become the model's examples. In your own files, just leave the target cell empty where you want answers.",
  },
  {
    id: "predict",
    target: ".rail-predict",
    title: "Run the model",
    body: "Click Predict. TabFM reads the example rows as context and predicts the blanks in one pass — this takes a few seconds on your GPU.",
    advanceWhen: (s) => s.hasResult,
  },
  {
    id: "results",
    target: ".rail-results",
    title: "Can you trust it?",
    body: "The accuracy check re-predicts rows the model was NOT shown, so you know how good it is before acting on it. Hover any predicted cell in the grid to see its confidence.",
  },
  {
    id: "done",
    target: null,
    title: "That’s the whole workflow 🎉",
    body: "Upload → tidy if needed → pick the target → Predict → download the completed table. Close the file (✕ in the top bar) and try one of your own spreadsheets.",
  },
];

interface Props {
  snapshot: TutorialSnapshot;
  stepIndex: number;
  onStepChange: (i: number) => void;
  onClose: () => void;
}

const CARD_W = 320;
const CARD_H = 190; // estimate for placement math

export function Tutorial({ snapshot, stepIndex, onStepChange, onClose }: Props) {
  const step = STEPS[stepIndex];
  const [rect, setRect] = useState<DOMRect | null>(null);
  const advancedFor = useRef<string | null>(null);

  // Interactive steps advance themselves once the user did the thing.
  useEffect(() => {
    if (step?.advanceWhen && advancedFor.current !== step.id && step.advanceWhen(snapshot)) {
      advancedFor.current = step.id;
      const next = stepIndex + 1;
      if (next < STEPS.length) onStepChange(next);
      else onClose();
    }
  }, [snapshot, step, stepIndex, onStepChange, onClose]);

  // Track the target's position (layout shifts, scrolling, popovers).
  useEffect(() => {
    let scrolled = false;
    function measure() {
      if (!step?.target) {
        setRect(null);
        return;
      }
      const el = document.querySelector(step.target);
      if (!el) {
        setRect(null);
        return;
      }
      if (!scrolled) {
        el.scrollIntoView({ block: "center", inline: "nearest" });
        scrolled = true;
      }
      setRect(el.getBoundingClientRect());
    }
    measure();
    const timer = setInterval(measure, 250);
    return () => clearInterval(timer);
  }, [stepIndex, step]);

  if (!step) return null;

  const interactive = Boolean(step.advanceWhen);
  const waiting = interactive && step.id === "predict" && snapshot.busy;

  let cardStyle: React.CSSProperties;
  if (step.target && rect) {
    const spaceRight = window.innerWidth - rect.right;
    if (rect.width < 220 && spaceRight > CARD_W + 40) {
      // Narrow target (gutter cell, column header): sit beside it so the card
      // never covers the popover that opens underneath.
      const top = Math.max(12, Math.min(rect.top, window.innerHeight - CARD_H - 12));
      cardStyle = { top, left: rect.right + 16, width: CARD_W };
    } else if (rect.left > window.innerWidth * 0.6 && rect.left > CARD_W + 40) {
      // Right-rail target: sit to its left so the rail content stays readable.
      const top = Math.max(12, Math.min(rect.top, window.innerHeight - CARD_H - 12));
      cardStyle = { top, left: rect.left - CARD_W - 16, width: CARD_W };
    } else {
      const below = rect.bottom + CARD_H + 16 < window.innerHeight;
      const top = below ? rect.bottom + 12 : Math.max(12, rect.top - CARD_H - 12);
      const left = Math.max(12, Math.min(rect.left, window.innerWidth - CARD_W - 12));
      cardStyle = { top, left, width: CARD_W };
    }
  } else {
    cardStyle = {
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      width: 400,
    };
  }

  return (
    <div className="tour" role="dialog" aria-label={`Tutorial: ${step.title}`}>
      {step.target && rect ? (
        <div
          className={`tour-spot ${interactive ? "tour-spot-active" : ""}`}
          style={{
            top: rect.top - 5,
            left: rect.left - 5,
            width: rect.width + 10,
            height: rect.height + 10,
          }}
        />
      ) : (
        <div className="tour-backdrop" />
      )}
      <div className="tour-card" style={cardStyle}>
        <span className="tour-progress">
          Step {stepIndex + 1} of {STEPS.length}
        </span>
        <h3 className="tour-title">{step.title}</h3>
        <p className="tour-body">{step.body}</p>
        <div className="tour-actions">
          <button className="btn-ghost tour-exit" onClick={onClose}>
            Exit tour
          </button>
          {interactive ? (
            <span className="tour-wait">
              {waiting ? (
                <>
                  <span className="spinner spinner-blue" aria-hidden="true" /> predicting…
                </>
              ) : (
                "Your turn — click it!"
              )}
            </span>
          ) : (
            <button
              className="btn-primary tour-next"
              onClick={() =>
                stepIndex + 1 < STEPS.length ? onStepChange(stepIndex + 1) : onClose()
              }
            >
              {stepIndex + 1 < STEPS.length ? "Next" : "Finish"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
