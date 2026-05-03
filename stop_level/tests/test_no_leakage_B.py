"""
test_no_leakage_B.py
=============================================================================
Scenario B (inflight) — same-stop labels are still forbidden, but the
explicit "prior stops on the same run" delay features are *whitelisted*
(`prev_stop_actual_delay`, `cum_actual_delay_so_far`,
`max_actual_delay_so_far`).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stop_level.leakage_guards import (  # noqa: E402
    LEAKY_COLS_A, LEAKY_COLS_B,
    feature_columns, assert_no_leakage,
)


def test_label_still_forbidden():
    for c in ["y_stop", "delay_min", "delay_minutes", "is_disrupted",
              "arrival_delay", "departure_delay", "cancelled"]:
        assert c in LEAKY_COLS_B, f"{c!r} must remain banned in scenario B"


def test_inflight_features_allowed():
    for c in ["prev_stop_actual_delay", "cum_actual_delay_so_far",
              "max_actual_delay_so_far"]:
        assert c not in LEAKY_COLS_B, (
            f"{c!r} must be allowed in scenario B (inflight)"
        )


def test_B_is_strictly_smaller_than_A():
    # A bans everything B bans, plus the inflight features.
    assert LEAKY_COLS_B.issubset(LEAKY_COLS_A)
    assert LEAKY_COLS_B != LEAKY_COLS_A


def test_feature_columns_preserves_inflight_in_B():
    cols = ["lat", "weather_severity", "prev_stop_actual_delay",
            "cum_actual_delay_so_far", "delay_minutes", "y_stop"]
    keep = feature_columns(cols, "B")
    assert "prev_stop_actual_delay" in keep
    assert "cum_actual_delay_so_far" in keep
    # but the label still goes:
    assert "y_stop" not in keep
    assert "delay_minutes" not in keep


def test_assert_no_leakage_B():
    with pytest.raises(ValueError):
        assert_no_leakage(["lat", "y_stop"], "B")
    # Clean list (including whitelisted inflight features) should pass.
    assert_no_leakage(
        ["lat", "weather_severity", "prev_stop_actual_delay"], "B",
    )
