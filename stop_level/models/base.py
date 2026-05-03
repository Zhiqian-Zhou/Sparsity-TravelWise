"""
stop_level/models/base.py
=============================================================================
StopModel — abstract base class shared by all 5 architectures.

Uniform contract: every model implements `fit`, `predict_proba`, `save`,
`load`. The orchestrator (`train_all.py`) and the evaluator (`evaluate.py`)
talk to models only through this interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
)


def n_neg_pos_ratio(y: np.ndarray) -> float:
    """scale_pos_weight = n_neg / n_pos; safe on degenerate y."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return max(1.0, n_neg / max(n_pos, 1))


def quick_val_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compact metric snapshot used during training logs."""
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "pr_auc":    float(average_precision_score(y_true, y_prob)) if y_true.sum() else 0.0,
        "roc_auc":   float(roc_auc_score(y_true, y_prob)) if 0 < y_true.sum() < len(y_true) else 0.0,
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    return out


class StopModel(ABC):
    """
    All concrete models inherit from this. The abstract methods are the only
    surface the rest of the pipeline depends on.
    """
    name: str = "abstract"

    def __init__(self, scenario: str, seed: int = 42, **kwargs: Any) -> None:
        if scenario not in {"A", "B"}:
            raise ValueError(f"scenario must be 'A' or 'B', got {scenario!r}")
        self.scenario = scenario
        self.seed = seed
        self.feat_cols: list[str] | None = None
        self.threshold_: float = 0.5

    # ── Abstract interface ────────────────────────────────────────────────────
    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        feat_cols: list[str],
        ctx: dict | None = None,
        **kwargs: Any,
    ) -> "StopModel":
        """
        Fit on `train_df`, early-stop / threshold-tune on `val_df`.

        `ctx` carries shared resources (edge_index/edge_attr,
        station_id_to_idx, service_id_to_idx) needed by the GNN and seq model.
        """

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(y=1) for every row of `df`, shape `[len(df)]`."""

    @abstractmethod
    def save(self, dir_path: Path) -> None:
        """Persist the fitted model to `dir_path/`."""

    @classmethod
    @abstractmethod
    def load(cls, dir_path: Path, scenario: str) -> "StopModel":
        """Reload a model previously saved by `save`."""

    # ── Shared helpers ────────────────────────────────────────────────────────
    def tune_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Pick the threshold maximising F1 on val. Stored in `self.threshold_`."""
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            self.threshold_ = 0.5
            return self.threshold_
        thresholds = np.linspace(0.05, 0.95, 19)
        best_f1, best_t = -1.0, 0.5
        for t in thresholds:
            f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold_ = best_t
        return best_t


def write_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    out_path: Path,
    extra_cols: list[str] | None = None,
) -> None:
    """
    Persist predictions in a uniform format — every model writes the same
    schema so `evaluate.py` can ingest them blindly.

    Columns: service_id, station_id, country, date, train_class_code,
             stop_order, position_norm, y_stop, delay_min, p_disrupted, y_pred
    """
    keep = ["service_id", "station_id", "country", "date", "train_class_code",
             "stop_order", "position_norm", "y_stop"]
    if "delay_min" in df.columns:
        keep.append("delay_min")
    if extra_cols:
        keep.extend(c for c in extra_cols if c in df.columns and c not in keep)
    out = df[keep].copy()
    out["p_disrupted"] = y_prob.astype("float32")
    out["y_pred"] = (y_prob >= threshold).astype("int8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
