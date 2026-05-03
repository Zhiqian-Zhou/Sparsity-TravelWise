"""
test_xai_smoke.py
=============================================================================
End-to-end XAI smoke test on synthesized data:
  Phase 1 mock CSVs → Phase 2 parquets → train (logreg+lgbm) → xai_stops.

Catches API regressions in the SHAP path, the figure code, and the
JSON report shape without needing real Phase 1 outputs.
"""
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _ensure_phase1_data() -> None:
    """Synthesize tiny Phase 1 CSVs in Data/Italy/processed/."""
    proc = ROOT / "Data" / "Italy" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    stations = [f"IT_S{i:02d}" for i in range(8)]
    services, stops = [], []
    for sid_idx in range(220):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 180)))
        sid = f"IT_T{sid_idx:04d}"
        services.append({"service_id": sid,
                          "train_class_code": int(rng.integers(0, 7)),
                          "date": date,
                          "is_disrupted": int(rng.integers(0, 2))})
        route = list(rng.choice(stations, size=4, replace=False))
        for k, st in enumerate(route):
            stops.append({"service_id": sid, "station_id": st,
                           "delay_minutes": float(rng.normal(3, 8)),
                           "weather_severity": int(rng.integers(0, 5)),
                           "temperature": float(rng.normal(15, 8)),
                           "wind_speed": float(rng.normal(10, 4)),
                           "precipitation": float(max(0, rng.normal(1, 2))),
                           "snow_depth": 0.0, "cancelled": False,
                           "scheduled_time":
                               (date + pd.Timedelta(hours=8 + k)).isoformat()})
    pd.DataFrame(services).to_csv(proc / "nodes_service.csv", index=False)
    pd.DataFrame(stops).to_csv(proc / "edges_stops_at.csv", index=False)
    pd.DataFrame([{"station_id": s, "lat": 45 + i*0.1, "lon": 9 + i*0.1,
                    "avg_historical_delay": 2.0, "degree": 4}
                  for i, s in enumerate(stations)]
                ).to_csv(proc / "nodes_station.csv", index=False)
    edges = [(stations[i], stations[i+1], 10.0) for i in range(len(stations)-1)]
    edges += [(b, a, d) for a, b, d in edges]
    pd.DataFrame(edges, columns=["station_from", "station_to", "distance_km"])\
       .to_csv(proc / "edges_adjacent.csv", index=False)
    pd.DataFrame(columns=["fault_id", "date", "station_id", "description"])\
       .to_csv(proc / "nodes_fault.csv", index=False)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_xai_end_to_end():
    if importlib.util.find_spec("shap") is None:
        pytest.skip("shap not installed")

    _ensure_phase1_data()
    # Clean any leftover artefacts from previous runs
    for p in [ROOT / "Data" / "stops",
              ROOT / "stop_level" / "models" / "_artefacts",
              ROOT / "stop_level" / "results",
              ROOT / "stop_level" / "figures"]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    r = _run([sys.executable, "stop_level/preprocess_stops.py"])
    assert r.returncode == 0, f"preprocess_stops failed:\n{r.stderr}"
    assert (ROOT / "Data" / "stops" / "stops_train.parquet").exists()

    r = _run([sys.executable, "stop_level/train_all.py",
              "--model", "logreg", "lgbm", "xgb", "--scenario", "both"])
    assert r.returncode == 0, f"train_all failed:\n{r.stderr}"
    bench = json.load(open(ROOT / "stop_level" / "results" / "benchmark_stops.json"))
    assert len(bench["runs"]) == 6, bench

    r = _run([sys.executable, "stop_level/xai_stops.py", "--scenario", "both"])
    assert r.returncode == 0, f"xai_stops failed:\n{r.stderr}"

    report = json.load(open(ROOT / "stop_level" / "results" / "xai_report_stops.json"))
    for s in ("A", "B"):
        assert s in report["scenarios"], f"scenario {s} missing"
        assert "global_importance" in report["scenarios"][s]
        assert "local_explanations" in report["scenarios"][s]
        for tag in ("tp", "fn", "fp", "AB_flip"):
            assert tag in report["scenarios"][s]["local_explanations"], tag
        assert s in report["lag_ablation"]
        assert "delta" in report["lag_ablation"][s]
    assert "kg_bridge_query" in report
    assert report["scenario_uplift"] is not None

    # Figures must exist
    fig_dir = ROOT / "stop_level" / "figures" / "xai"
    expected = [
        "global_summary_bar_A.png", "global_summary_bar_B.png",
        "global_summary_beeswarm_A.png", "global_summary_beeswarm_B.png",
        "shap_by_country.png", "shap_by_position.png", "shap_by_train_class.png",
        "local_waterfall_tp.png", "local_waterfall_fn.png",
        "local_waterfall_fp.png", "local_waterfall_AB_flip.png",
        "shap_dependence_top5.png", "lag_ablation.png", "scenario_uplift_map.png",
    ]
    missing = [f for f in expected if not (fig_dir / f).exists()]
    assert not missing, f"missing XAI figures: {missing}"
