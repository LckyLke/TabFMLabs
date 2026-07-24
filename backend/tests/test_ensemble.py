"""TabFMBackend must dispatch to the library's .ensemble() preset when asked.

Uses a fake ``tabfm`` module so no weights are needed.
"""

import sys
import types

import numpy as np
import pandas as pd

from app.inference import TabFMBackend


def _install_fake_tabfm(monkeypatch, calls: list):
    mod = types.ModuleType("tabfm")

    class FakeClassifier:
        def __init__(self, model=None, preset="default"):
            calls.append(("classifier", preset))

        @classmethod
        def ensemble(cls, model=None):
            return cls(model=model, preset="ensemble")

        def fit(self, X, y):
            self.classes_ = np.unique(y)

        def predict_proba(self, X):
            proba = np.zeros((len(X), len(self.classes_)))
            proba[:, 0] = 1.0
            return proba

    class FakeRegressor:
        def __init__(self, model=None, preset="default"):
            calls.append(("regressor", preset))

        @classmethod
        def ensemble(cls, model=None):
            return cls(model=model, preset="ensemble")

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    mod.TabFMClassifier = FakeClassifier
    mod.TabFMRegressor = FakeRegressor
    monkeypatch.setitem(sys.modules, "tabfm", mod)


def test_tabfm_backend_ensemble_dispatch(monkeypatch):
    calls: list = []
    _install_fake_tabfm(monkeypatch, calls)
    monkeypatch.setattr(TabFMBackend, "_load", lambda self, task: object())
    backend = TabFMBackend()

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_test = pd.DataFrame({"a": [1.5]})
    y_cls = pd.Series(["x", "y", "x"])
    y_reg = pd.Series([1.0, 2.0, 3.0])

    result = backend.fit_predict(X, y_cls, X_test, "classification")
    assert calls[-1] == ("classifier", "default")
    assert "ensemble" not in result.model_name

    result = backend.fit_predict(X, y_cls, X_test, "classification", ensemble=True)
    assert calls[-1] == ("classifier", "ensemble")
    assert "ensemble" in result.model_name

    result = backend.fit_predict(X, y_reg, X_test, "regression")
    assert calls[-1] == ("regressor", "default")
    assert "ensemble" not in result.model_name

    result = backend.fit_predict(X, y_reg, X_test, "regression", ensemble=True)
    assert calls[-1] == ("regressor", "ensemble")
    assert "ensemble" in result.model_name
