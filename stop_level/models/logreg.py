"""
stop_level/models/logreg.py
=============================================================================
Mandatory baseline. If a more complex model can't beat this by ≥ 0.02 PR-AUC
in scenario A, the complexity isn't justified.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stop_level.models.base import StopModel


class LogRegStopModel(StopModel):
    name = "logreg"

    def __init__(self, scenario: str, seed: int = 42, C: float = 1.0,
                  max_iter: int = 2000, **_: Any) -> None:
        super().__init__(scenario, seed)
        self.C = C
        self.max_iter = max_iter
        self.model: LogisticRegression | None = None

    def fit(self, train_df, val_df, feat_cols, ctx=None, **_):
        self.feat_cols = list(feat_cols)
        Xtr = train_df[self.feat_cols].astype("float32").values
        ytr = train_df["y_stop"].astype("int8").values
        # sklearn LogisticRegression does not accept NaN. Some weather columns
        # legitimately carry NaN where the source had no measurement (vs the
        # fake-zero convention). Impute with the per-column median computed
        # from training rows only — keeps the feature usable without leaking
        # val/test stats. The imputer is persisted so predict-time inputs use
        # the same medians.
        self._col_medians = np.nanmedian(Xtr, axis=0).astype("float32")
        # Replace any column whose median is itself NaN (all-NaN column) with 0.
        self._col_medians = np.where(
            np.isnan(self._col_medians), 0.0, self._col_medians
        ).astype("float32")
        Xtr = self._impute(Xtr)
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            class_weight="balanced", solver="saga", n_jobs=-1,
            random_state=self.seed,
        )
        # Inputs are pre-scaled in preprocess_stops.py via StandardScaler fit
        # on train rows only (mean/std persisted to Data/stops/scaler_*_B.npy).
        self.model.fit(Xtr, ytr)
        # Tune decision threshold on val
        y_val_prob = self.predict_proba(val_df)
        self.tune_threshold(val_df["y_stop"].values, y_val_prob)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        mask = np.isnan(X)
        if not mask.any():
            return X
        out = X.copy()
        # Broadcast medians along rows then mask-substitute.
        out[mask] = np.broadcast_to(self._col_medians, X.shape)[mask]
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LogRegStopModel.fit was not called.")
        X = df[self.feat_cols].astype("float32").values
        X = self._impute(X)
        return self.model.predict_proba(X)[:, 1].astype("float32")

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "model.pkl", "wb") as f:
            pickle.dump({"model": self.model,
                          "feat_cols": self.feat_cols,
                          "threshold": self.threshold_,
                          "scenario": self.scenario,
                          "col_medians": self._col_medians}, f)

    @classmethod
    def load(cls, dir_path: Path, scenario: str) -> "LogRegStopModel":
        with open(Path(dir_path) / "model.pkl", "rb") as f:
            blob = pickle.load(f)
        m = cls(scenario=blob["scenario"])
        m.model = blob["model"]
        m.feat_cols = blob["feat_cols"]
        m.threshold_ = blob["threshold"]
        m._col_medians = blob.get("col_medians", np.zeros(len(m.feat_cols), dtype="float32"))
        return m
