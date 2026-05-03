"""
stop_level/models/lgbm_model.py
=============================================================================
LightGBM gradient-boosted trees on the per-stop tabular feature matrix.

Hyperparameters chosen for stop-level scale:
  • 1500 trees with early stopping on val PR-AUC (50 rounds)
  • num_leaves=63, max_depth=7
  • scale_pos_weight = n_neg / n_pos
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from stop_level.models.base import StopModel, n_neg_pos_ratio


class LGBMStopModel(StopModel):
    name = "lgbm"

    def __init__(
        self,
        scenario: str,
        seed: int = 42,
        n_estimators: int = 1500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 7,
        early_stopping: int = 50,
        **_: Any,
    ) -> None:
        super().__init__(scenario, seed)
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.early_stopping = early_stopping
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, train_df, val_df, feat_cols, ctx=None, **_):
        self.feat_cols = list(feat_cols)
        Xtr, ytr = train_df[self.feat_cols].astype("float32").values, train_df["y_stop"].values
        Xva, yva = val_df[self.feat_cols].astype("float32").values, val_df["y_stop"].values
        spw = n_neg_pos_ratio(ytr)
        self.model = lgb.LGBMClassifier(scale_pos_weight=spw, **self.params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping, verbose=False),
        ]
        self.model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="average_precision",
            callbacks=callbacks,
            feature_name=self.feat_cols,
        )
        self.tune_threshold(yva, self.predict_proba(val_df))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LGBMStopModel.fit was not called.")
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
    def load(cls, dir_path: Path, scenario: str) -> "LGBMStopModel":
        with open(Path(dir_path) / "model.pkl", "rb") as f:
            blob = pickle.load(f)
        m = cls(scenario=blob["scenario"])
        m.model = blob["model"]
        m.feat_cols = blob["feat_cols"]
        m.threshold_ = blob["threshold"]
        return m
