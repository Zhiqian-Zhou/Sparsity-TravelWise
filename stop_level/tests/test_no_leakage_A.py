"""
test_no_leakage_A.py
=============================================================================
Asserts that the scenario-A feature column list contains zero leaky names.

Two layers of defence:
  1. The static set `LEAKY_COLS_A` is non-empty and contains the canonical
     forbidden names.
  2. `feature_columns(all_cols, "A")` returns a list disjoint from
     `LEAKY_COLS_A`, even when the input contains every banned column.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stop_level.leakage_guards import (  # noqa: E402
    LEAKY_COLS_A, feature_columns, assert_no_leakage,
)


def test_set_is_non_empty():
    assert len(LEAKY_COLS_A) > 0


def test_canonical_label_columns_are_banned():
    for c in ["y_stop", "delay_min", "delay_minutes", "is_disrupted",
              "arrival_delay", "departure_delay", "cancelled"]:
        assert c in LEAKY_COLS_A, f"{c!r} should be banned in scenario A"


def test_inflight_features_banned_in_A():
    for c in ["prev_stop_actual_delay", "cum_actual_delay_so_far",
              "max_actual_delay_so_far"]:
        assert c in LEAKY_COLS_A, f"{c!r} must be banned in scenario A"


def test_feature_columns_filters_out_leaks():
    pretend_cols = [
        "lat", "lon", "degree", "weather_severity",
        # leaks below should all be stripped:
        "y_stop", "delay_min", "delay_minutes", "is_disrupted",
        "prev_stop_actual_delay",
    ]
    keep = feature_columns(pretend_cols, "A")
    for c in keep:
        assert c not in LEAKY_COLS_A, (
            f"leak {c!r} survived feature_columns(A)"
        )
    assert {"lat", "lon", "degree", "weather_severity"} <= set(keep)


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError):
        assert_no_leakage(["lat", "delay_minutes"], "A")
    # Clean list should not raise.
    assert_no_leakage(["lat", "lon", "weather_severity"], "A")


def test_invalid_scenario_raises():
    with pytest.raises(ValueError):
        assert_no_leakage(["lat"], "C")
