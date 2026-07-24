"""Model backends: TabFM, TabPFN, and a scikit-learn baseline.

Models are selected explicitly per prediction (no automatic fallback): if the
chosen model's weights are not available, prediction fails with
WeightsUnavailable and the API returns 503.
"""

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import TaskType

logger = logging.getLogger(__name__)

MAX_CLASSES = 10


class WeightsUnavailable(Exception):
    """The selected model's weights are not (fully) present yet."""


@dataclass
class FitPredictResult:
    predictions: np.ndarray
    # max class probability per prediction; None for regression
    confidences: np.ndarray | None
    model_name: str


def infer_task(target: pd.Series) -> TaskType:
    """Classification for non-numeric or low-cardinality numeric targets.

    Integer-valued numeric targets count as classes only when each value
    repeats enough (n_unique ≤ n/3) — five distinct revenue figures are a
    regression target, not five classes.
    """
    labeled = target.dropna()
    if not pd.api.types.is_numeric_dtype(labeled):
        return "classification"
    n_unique = labeled.nunique()
    if (
        n_unique <= MAX_CLASSES
        and n_unique <= len(labeled) / 3
        and (labeled == labeled.round()).all()
    ):
        return "classification"
    return "regression"


def _device() -> str:
    forced = os.environ.get("MODEL_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Baseline (explicit opt-in only — used by tests and MODEL_BACKEND=baseline)
# --------------------------------------------------------------------------


class BaselineBackend:
    name = "baseline (HistGradientBoosting)"

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        task: TaskType,
        ensemble: bool = False,
    ) -> FitPredictResult:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        if ensemble:
            raise ValueError(f"{self.name} does not support ensemble mode.")

        X_train = _encode_for_sklearn(X_train)
        X_test = _encode_for_sklearn(X_test)
        if task == "classification":
            model = HistGradientBoostingClassifier(categorical_features="from_dtype")
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            preds = model.classes_[np.argmax(proba, axis=1)]
            return FitPredictResult(preds, proba.max(axis=1), self.name)
        model = HistGradientBoostingRegressor(categorical_features="from_dtype")
        model.fit(X_train, y_train)
        return FitPredictResult(model.predict(X_test), None, self.name)


def _encode_for_sklearn(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
    return X


# --------------------------------------------------------------------------
# TabFM
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS_DIR = os.path.expanduser("~/.cache/tabfm-studio")
# Published checkpoint sizes; a smaller file is a download still in progress.
EXPECTED_WEIGHT_BYTES = {
    "classification": 6_557_888_408,
    "regression": 6_591_243_724,
}


def _find_weights(model_type: str) -> str:
    env_dir = os.environ.get("TABFM_WEIGHTS_DIR")
    if env_dir:
        path = os.path.join(env_dir, model_type, "model.safetensors")
        if not os.path.exists(path):
            raise WeightsUnavailable(
                f"TABFM_WEIGHTS_DIR is set but {path} does not exist."
            )
        return path

    path = os.path.join(DEFAULT_WEIGHTS_DIR, model_type, "model.safetensors")
    if os.path.exists(path):
        expected = EXPECTED_WEIGHT_BYTES.get(model_type)
        if expected is None or os.path.getsize(path) >= expected:
            return path
        raise WeightsUnavailable(
            f"{path} is still downloading "
            f"({os.path.getsize(path) / 1e6:.0f} of {expected / 1e6:.0f} MB)."
        )

    from tabfm.src.pytorch import tabfm_v1_0_0

    try:
        from huggingface_hub import hf_hub_download

        logger.info("Fetching TabFM %s checkpoint from HF Hub...", model_type)
        return hf_hub_download(
            repo_id=tabfm_v1_0_0.HF_REPO_ID, filename=f"{model_type}/model.safetensors"
        )
    except Exception as exc:
        raise WeightsUnavailable(f"Could not download TabFM weights: {exc}") from exc


def _load_tabfm_model(model_type: str):
    """Build a TabFM model from the safetensors checkpoint.

    Bypasses tabfm's own loader, which snapshot-downloads the whole repo (both
    ~6.6 GB checkpoints) and then fails because it expects pytorch_model.bin
    while the published repo ships model.safetensors (tabfm 1.0.0 bug).
    """
    from safetensors.torch import load_file
    from tabfm.src.pytorch import tabfm_v1_0_0

    weights_path = _find_weights(model_type)

    config_cls = (
        tabfm_v1_0_0.ClassificationConfig
        if model_type == "classification"
        else tabfm_v1_0_0.RegressionConfig
    )
    model = tabfm_v1_0_0.TabFM(**config_cls().to_dict())
    model.load_state_dict(load_file(weights_path), strict=True)
    model.eval()

    device = _device()
    if device != "cpu":
        try:
            model = model.to(device)
        except RuntimeError as exc:  # e.g. out of GPU memory
            logger.warning("Could not move model to %s (%s); using CPU.", device, exc)
            model = model.to("cpu")
    logger.info("TabFM %s model ready on %s.", model_type, device)
    return model


class TabFMBackend:
    """Wraps the official tabfm library (PyTorch backend, weights from HF Hub)."""

    name = "TabFM 1.0.0 (pytorch)"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    def _load(self, task: TaskType):
        model_type = "classification" if task == "classification" else "regression"
        with self._lock:
            if model_type not in self._models:
                self._models[model_type] = _load_tabfm_model(model_type)
            return self._models[model_type]

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        task: TaskType,
        ensemble: bool = False,
    ) -> FitPredictResult:
        from tabfm import TabFMClassifier, TabFMRegressor

        model = self._load(task)
        # The .ensemble() preset is the paper's "TabFM-Ensemble" configuration:
        # feature crosses + SVD features, NNLS-weighted blending, calibration.
        # Both presets run 32 data views; ensemble adds out-of-fold weight
        # fitting, so it is several times slower per prediction.
        name = f"{self.name} · ensemble" if ensemble else self.name
        if task == "classification":
            make = TabFMClassifier.ensemble if ensemble else TabFMClassifier
            clf = make(model=model)
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
            preds = np.asarray(clf.classes_)[np.argmax(proba, axis=1)]
            return FitPredictResult(preds, proba.max(axis=1), name)
        make = TabFMRegressor.ensemble if ensemble else TabFMRegressor
        reg = make(model=model)
        reg.fit(X_train, y_train)
        return FitPredictResult(np.asarray(reg.predict(X_test)), None, name)


# --------------------------------------------------------------------------
# TabPFN
# --------------------------------------------------------------------------


class TabPFNBackend:
    """Prior Labs' TabPFN — also an in-context tabular foundation model,
    with far smaller weights than TabFM."""

    name = "TabPFN"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        task: TaskType,
        ensemble: bool = False,
    ) -> FitPredictResult:
        if ensemble:
            raise ValueError(f"{self.name} does not support ensemble mode.")
        # Local-first app: never phone home usage data.
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as exc:
            raise WeightsUnavailable(
                "TabPFN is not installed (pip install tabpfn)."
            ) from exc

        from tabpfn.errors import TabPFNLicenseError

        device = _device()
        with self._lock:  # weight download / GPU memory: one at a time
            try:
                if task == "classification":
                    clf = TabPFNClassifier(device=device)
                    clf.fit(X_train, y_train)
                    proba = clf.predict_proba(X_test)
                    preds = np.asarray(clf.classes_)[np.argmax(proba, axis=1)]
                    return FitPredictResult(preds, proba.max(axis=1), self._label())
                reg = TabPFNRegressor(device=device)
                reg.fit(X_train, y_train)
                return FitPredictResult(np.asarray(reg.predict(X_test)), None, self._label())
            except TabPFNLicenseError as exc:
                raise WeightsUnavailable(
                    "TabPFN needs a one-time (free) license acceptance: log in at "
                    "https://ux.priorlabs.ai, accept the license, copy your API key "
                    "from the account page, and restart the backend with "
                    "TABPFN_TOKEN=<your-key>."
                ) from exc
            except (OSError, ConnectionError, TimeoutError) as exc:
                raise WeightsUnavailable(f"TabPFN weights unavailable: {exc}") from exc

    def _label(self) -> str:
        try:
            import tabpfn

            return f"TabPFN {tabpfn.__version__}"
        except Exception:
            return self.name


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

MODELS = {
    "tabfm": {
        "label": "TabFM",
        "description": "Google's tabular foundation model (6.6 GB per task, GPU-friendly)",
        "cls": TabFMBackend,
        "supports_ensemble": True,
    },
    "tabpfn": {
        "label": "TabPFN",
        "description": "Prior Labs' tabular foundation model (small weights, fast)",
        "cls": TabPFNBackend,
        "supports_ensemble": False,
    },
    "baseline": {
        "label": "Gradient boosting (classic ML)",
        "description": "scikit-learn HistGradientBoosting trained on your example rows — "
        "the classic-ML reference the foundation models should beat",
        "cls": BaselineBackend,
        "supports_ensemble": False,
    },
}

DEFAULT_MODEL = "tabfm"

_instances: dict[str, object] = {}
_instances_lock = threading.Lock()


def resolve_model(model: str | None = None) -> str:
    name = model or os.environ.get("MODEL_BACKEND") or DEFAULT_MODEL
    if name not in MODELS:
        raise ValueError(f"Unknown model “{name}”. Available: {', '.join(MODELS)}")
    return name


def get_backend(model: str | None = None):
    name = resolve_model(model)
    with _instances_lock:
        if name not in _instances:
            _instances[name] = MODELS[name]["cls"]()
            logger.info("Initialized model backend: %s", name)
        return _instances[name]
