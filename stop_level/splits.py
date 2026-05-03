"""
stop_level/splits.py
=============================================================================
Day-level temporal splits + service-disjointness assertions + rolling-origin
CV folds for hyperparameter tuning inside the training window.

Train: 2024-01-01 → 2024-04-30
Val  : 2024-05-01 → 2024-05-31
Test : 2024-06-01 → 2024-06-30

Critical invariant: a service_id (a complete run of one train on one day)
never crosses a split boundary. The functions here enforce it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END   = pd.Timestamp("2024-04-30")
VAL_START   = pd.Timestamp("2024-05-01")
VAL_END     = pd.Timestamp("2024-05-31")
TEST_START  = pd.Timestamp("2024-06-01")
TEST_END    = pd.Timestamp("2024-06-30")


@dataclass(frozen=True)
class SplitInfo:
    name:  str   # "train" | "val" | "test"
    start: pd.Timestamp
    end:   pd.Timestamp


SPLITS = (
    SplitInfo("train", TRAIN_START, TRAIN_END),
    SplitInfo("val",   VAL_START,   VAL_END),
    SplitInfo("test",  TEST_START,  TEST_END),
)


def day_split(
    df: pd.DataFrame,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """
    Split `df` into {train, val, test} on `date_col`.

    Returns three views (no copy). Use `.copy()` downstream if you need to
    mutate them independently.
    """
    if date_col not in df.columns:
        raise KeyError(f"date_col {date_col!r} not in df.columns")
    d = pd.to_datetime(df[date_col]).dt.normalize()
    out: dict[str, pd.DataFrame] = {}
    for s in SPLITS:
        mask = (d >= s.start) & (d <= s.end)
        out[s.name] = df.loc[mask]
    return out


def assert_no_service_overlap(
    splits: dict[str, pd.DataFrame],
    id_col: str = "service_id",
) -> None:
    """Assert that the service_id sets are pairwise disjoint."""
    s_train = set(splits["train"][id_col])
    s_val   = set(splits["val"][id_col])
    s_test  = set(splits["test"][id_col])
    overlaps = {
        "train ∩ val":   s_train & s_val,
        "val ∩ test":    s_val   & s_test,
        "train ∩ test":  s_train & s_test,
    }
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise AssertionError(f"service_id overlap across splits: {bad}")


def rolling_origin_folds(
    train_df: pd.DataFrame,
    n_folds: int = 4,
    date_col: str = "date",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yield (train_fold, val_fold) pairs for rolling-origin CV inside the
    training window. Splits the 17 train weeks into roughly equal blocks.
    """
    d = pd.to_datetime(train_df[date_col]).dt.normalize()
    week = ((d - TRAIN_START).dt.days // 7).astype(int)
    n_weeks = int(week.max()) + 1
    block = max(1, n_weeks // (n_folds + 1))
    for k in range(1, n_folds + 1):
        end_train = k * block
        end_val   = (k + 1) * block
        train_mask = week < end_train
        val_mask   = (week >= end_train) & (week < end_val)
        yield train_df.loc[train_mask], train_df.loc[val_mask]
