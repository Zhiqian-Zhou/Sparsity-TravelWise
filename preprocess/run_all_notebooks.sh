#!/usr/bin/env bash
# preprocess/run_all_notebooks.sh
# =============================================================================
# Execute the three per-country preprocessing notebooks headless and persist
# the executed copies under preprocess/_executed/.
#
# Each notebook:
#   1. Reads its country's raw CSVs from Data/<Country>/Raw/
#   2. Writes the five standardized CSVs to Data/<Country>/processed/
#   3. Saves all inspection plots to preprocess/figures/<country>/step_NN_*.png
#
# Requirements:
#   pip install jupyter papermill nbconvert pandas numpy matplotlib seaborn pyarrow
#   (NL also needs: openmeteo-requests requests-cache retry-requests)
#
# Usage:
#   bash preprocess/run_all_notebooks.sh
#   bash preprocess/run_all_notebooks.sh italy        # single country
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXEC_DIR="preprocess/_executed"
mkdir -p "$EXEC_DIR"

run_one() {
  local country="$1"
  local src="preprocess/${country}_preprocessing.ipynb"
  local dst="${EXEC_DIR}/${country}.ipynb"
  echo "── ${country} ─────────────────────────────────────────────"
  if command -v papermill >/dev/null 2>&1; then
    papermill "$src" "$dst" --kernel python3
  else
    jupyter nbconvert --to notebook --execute "$src" --output "_executed/${country}.ipynb"
  fi
  echo "✓ ${country} → ${dst}"
}

if [[ $# -eq 0 ]]; then
  COUNTRIES=(italy finland netherlands)
else
  COUNTRIES=("$@")
fi

for c in "${COUNTRIES[@]}"; do
  run_one "$c"
done

echo "── Summary ────────────────────────────────────────────────"
for c in "${COUNTRIES[@]}"; do
  fig_dir="preprocess/figures/${c}"
  n=$(ls "$fig_dir"/step_*.png 2>/dev/null | wc -l | tr -d ' ')
  echo "  ${c}: ${n} step plots in ${fig_dir}/"
done
