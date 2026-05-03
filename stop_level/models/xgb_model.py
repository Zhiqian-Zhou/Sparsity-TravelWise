"""
stop_level/models/xgb_model.py
=============================================================================
XGBoost — `tree_method='hist'` for speed on the 16.6 M-row scale.
Native TreeSHAP support cross-checks LightGBM's importances downstream.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from stop_level.models.base import StopModel, n_neg_pos_ratio


class XGBStopModel(StopModel):
    name = "xgb"

    def __init__(
        self,
        scenario: str,
        seed: int = 42,
        n_estimators: int = 1500,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        early_stopping: int = 50,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        super().__init__(scenario, seed)
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            tree_method="hist",
            device=device,
            eval_metric="aucpr",
            verbosity=0,
        )
        self.early_stopping = early_stopping
        self.model: xgb.XGBClassifier | None = None

    def fit(self, train_df, val_df, feat_cols, ctx=None, **_):
        self.feat_cols = list(feat_cols)
        Xtr, ytr = train_df[self.feat_cols].astype("float32").values, train_df["y_stop"].values
        Xva, yva = val_df[self.feat_cols].astype("float32").values, val_df["y_stop"].values
        spw = n_neg_pos_ratio(ytr)
        self.model = xgb.XGBClassifier(
            scale_pos_weight=spw,
            early_stopping_rounds=self.early_stopping,
            **self.params,
        )
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        self.tune_threshold(yva, self.predict_proba(val_df))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBStopModel.fit was not called.")
        X = df[self.feat_cols].astype("float32").values
        return self.model.predict_proba(X)[:, 1].astype("float32")

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "model.pkl", "wb") as f:
            pickle.dump({"model": self.model,
                          "feat_cols": self.feat_cols,
                          "threshold": self.threshold_,
                          "scenario": self.scenario}, f)

    @classmethod
    def load(cls, dir_path: Path, scenario: str) -> "XGBStopModel":
        with open(Path(dir_path) / "model.pkl", "rb") as f:
            blob = pickle.load(f)
        m = cls(scenario=blob["scenario"])
        m.model = blob["model"]
        m.feat_cols = blob["feat_cols"]
        m.threshold_ = blob["threshold"]
        return m
