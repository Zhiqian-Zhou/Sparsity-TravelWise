"""
test_split_no_overlap.py
=============================================================================
Asserts the day-level split is correct:
  • train ⊂ [2024-01-01, 2024-04-30]
  • val   ⊂ [2024-05-01, 2024-05-31]
  • test  ⊂ [2024-06-01, 2024-06-30]
  • service_id sets are pairwise disjoint
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stop_level.splits import (  # noqa: E402
    day_split, assert_no_service_overlap,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)


def _toy_frame() -> pd.DataFrame:
    """3 services × 3 days each, one service per split."""
    return pd.DataFrame([
        {"service_id": "S_train_1", "date": pd.Timestamp("2024-02-15"),
         "station_id": "ST_A"},
        {"service_id": "S_train_1", "date": pd.Timestamp("2024-02-15"),
         "station_id": "ST_B"},
        {"service_id": "S_val_1",   "date": pd.Timestamp("2024-05-10"),
         "station_id": "ST_A"},
        {"service_id": "S_val_1",   "date": pd.Timestamp("2024-05-10"),
         "station_id": "ST_C"},
        {"service_id": "S_test_1",  "date": pd.Timestamp("2024-06-15"),
         "station_id": "ST_B"},
        {"service_id": "S_test_1",  "date": pd.Timestamp("2024-06-15"),
         "station_id": "ST_C"},
    ])


def test_each_split_within_window():
    splits = day_split(_toy_frame())
    for name, (lo, hi) in {
        "train": (TRAIN_START, TRAIN_END),
        "val":   (VAL_START,   VAL_END),
        "test":  (TEST_START,  TEST_END),
    }.items():
        d = pd.to_datetime(splits[name]["date"])
        assert (d >= lo).all() and (d <= hi).all(), (
            f"{name} contains rows outside [{lo}, {hi}]"
        )


def test_service_disjointness_passes_on_clean_input():
    splits = day_split(_toy_frame())
    assert_no_service_overlap(splits)


def test_service_disjointness_fails_on_overlap():
    """A service straddling train and val should trigger an assertion."""
    df = _toy_frame()
    bad_row = pd.DataFrame([{
        "service_id": "S_train_1",   # already a train service
        "date":       pd.Timestamp("2024-05-01"),  # but appears in val
        "station_id": "ST_X",
    }])
    df = pd.concat([df, bad_row], ignore_index=True)
    splits = day_split(df)
    with pytest.raises(AssertionError):
        assert_no_service_overlap(splits)


def test_split_sizes_sum_to_total():
    df = _toy_frame()
    splits = day_split(df)
    total = sum(len(d) for d in splits.values())
    assert total == len(df), "splits must partition the input"
