"""
test_lag_strict_lt.py
=============================================================================
Asserts the lag-feature builders use strict `<` (no peeking at the current
day). Construction:

  Build a tiny synthetic stops DataFrame where each station has a known
  daily disruption rate. After running `compute_station_lag`, the value
  of `station_lag1_rate` for date D must equal the actual rate at D-1
  (not D, not D+1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stop_level import features as F  # noqa: E402


def _toy_frame() -> pd.DataFrame:
    """
    Three days, one station, with hand-crafted late counts:
      Day 1: 4/4 stops late → rate=1.0
      Day 2: 0/4 stops late → rate=0.0
      Day 3: 2/4 stops late → rate=0.5
    """
    rows = []
    for day, late_count in [(1, 4), (2, 0), (3, 2)]:
        for stop_idx in range(4):
            rows.append({
                "service_id":    f"S_{day}_{stop_idx}",
                "station_id":    "ST_X",
                "date":          pd.Timestamp(f"2024-01-0{day}"),
                "delay_minutes": 30.0 if stop_idx < late_count else 0.0,
                "cancelled":     False,
            })
    return pd.DataFrame(rows)


def test_station_lag1_uses_yesterday_only():
    df = _toy_frame()
    out = F.compute_station_lag(df).sort_values(["date", "service_id"])
    # Day 1 has no prior — must be 0 (filled).
    day1 = out[out["date"] == pd.Timestamp("2024-01-01")]
    assert (day1["station_lag1_rate"] == 0).all()

    # Day 2's lag1 = day 1's actual rate (1.0).
    day2 = out[out["date"] == pd.Timestamp("2024-01-02")]
    assert np.allclose(day2["station_lag1_rate"], 1.0), (
        "lag1_rate on day 2 must equal day 1's actual rate, "
        f"got {day2['station_lag1_rate'].unique()}"
    )

    # Day 3's lag1 = day 2's actual rate (0.0).
    day3 = out[out["date"] == pd.Timestamp("2024-01-03")]
    assert np.allclose(day3["station_lag1_rate"], 0.0), (
        f"lag1_rate on day 3 must equal day 2's actual rate, "
        f"got {day3['station_lag1_rate'].unique()}"
    )


def test_station_lag_does_not_peek_at_today():
    """If the lag included today, day 3's lag1 would be 0.5 — but it must be 0."""
    df = _toy_frame()
    out = F.compute_station_lag(df)
    day3 = out[out["date"] == pd.Timestamp("2024-01-03")]
    same_day_rate = 0.5
    assert not np.allclose(day3["station_lag1_rate"], same_day_rate), (
        "lag1_rate on day 3 must NOT equal day 3's own rate (no peeking!)"
    )


def test_inflight_prev_delay_is_strict():
    """
    `prev_stop_actual_delay` for the first stop on a service must be 0,
    and for stop k it must equal the delay at stop k-1 (not k).
    """
    rows = []
    for k, dmin in enumerate([3, 7, 12]):
        rows.append({
            "service_id": "S_INF_1", "station_id": f"ST_{k}",
            "date": pd.Timestamp("2024-02-10"),
            "delay_minutes": float(dmin), "cancelled": False,
        })
    df = pd.DataFrame(rows)
    df["stop_order"] = range(len(df))
    out = F.add_inflight_features(df).sort_values("stop_order")
    prev = out["prev_stop_actual_delay"].values
    # Stop 0 → 0, stop 1 → 3 (delay at stop 0), stop 2 → 7 (delay at stop 1).
    assert prev[0] == 0
    assert prev[1] == 3
    assert prev[2] == 7
