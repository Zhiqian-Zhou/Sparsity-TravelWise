#!/usr/bin/env bash
# scripts/download_data.sh
# =============================================================================
# Downloads the EXACT raw datasets used by this project for IT / FI / NL.
#
# Sources (verified 2026-05-03):
#   Italy        — figshare DOI 10.6084/m9.figshare.28891607.v2  (CC-BY 4.0)
#   Netherlands  — https://www.rijdendetreinen.nl/en/open-data    (CC-BY 4.0 + CC0)
#   Finland      — Kaggle viniborin/finland-integrated-train-weather-dataset-fi-tw
#                  DOI 10.34740/kaggle/dsv/14124620                (GPL-3)
#
# Total download size:
#   Italy        ~393 MB
#   Netherlands  ~193 MB compressed (1.6 GB after gunzip)
#   Finland      ~3.85 GB (6 monthly CSVs)
#   ─────────────────────────────────
#   Total        ~5.7 GB on disk
#
# Per-file integrity:
#   Italy        — MD5 verified against figshare API
#   Netherlands  — byte-size verified against Content-Length headers (small files)
#                  + post-gunzip size check (services)
#   Finland      — byte-size sanity-check post-download (Kaggle requires auth and
#                  doesn't expose pre-download sizes via the unauth API)
#
# Requirements:
#   curl, gunzip (preinstalled on macOS / most Linux)
#   For Finland: pip install kaggle  +  ~/.kaggle/kaggle.json with API token
#   from  https://www.kaggle.com/settings → "Create New API Token"
#
# Usage:
#   bash scripts/download_data.sh           # download all three countries
#   bash scripts/download_data.sh italy     # one country only
#   bash scripts/download_data.sh netherlands finland
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Helpers ────────────────────────────────────────────────────────────────────
log()  { printf "── %s ──\n" "$*"; }
note() { printf "   %s\n" "$*"; }
fail() { printf "✗  %s\n" "$*" >&2; exit 1; }
ok()   { printf "✓  %s\n" "$*"; }

actual_size() {
  # Cross-platform stat: %z on macOS, --printf='%s' on GNU
  if stat -f '%z' "$1" >/dev/null 2>&1; then stat -f '%z' "$1"
  else stat --printf='%s' "$1"; fi
}

verify_size() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || fail "missing: $path"
  local got; got="$(actual_size "$path")"
  if [[ "$got" -eq "$expected" ]]; then
    ok "$path  ($got B)"
  else
    fail "size mismatch: $path got $got expected $expected"
  fi
}

verify_md5() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || fail "missing: $path"
  local got
  if command -v md5 >/dev/null 2>&1; then           # macOS
    got="$(md5 -q "$path")"
  elif command -v md5sum >/dev/null 2>&1; then       # GNU
    got="$(md5sum "$path" | awk '{print $1}')"
  else
    note "no md5 tool found, skipping checksum for $path"; return 0
  fi
  if [[ "$got" == "$expected" ]]; then
    ok "md5 $path  ($got)"
  else
    fail "md5 mismatch: $path got $got expected $expected"
  fi
}

# ── Italy ──────────────────────────────────────────────────────────────────────
download_italy() {
  log "Italy — figshare 28891607 v2"
  local DEST="Data/Italy/Raw"
  mkdir -p "$DEST"

  # filename | figshare file id | bytes | md5
  local rows=(
    "Train_operation_data.csv|58118854|412466839|20d54ada2588789cb472417ac07e9196"
    "Train_fault_information.csv|58190842|340518|3793e8e9a15d7127bd73e02e400ca53c"
    "Adjacent_railway_stations_mileage_data.csv|59236799|665561|a8a40c8f6e627839fc280efe64504ca1"
    "Train_station_locations_data.csv|59236802|164706|ea29edef5924680531236ec348230eaa"
  )

  for row in "${rows[@]}"; do
    IFS='|' read -r name fid size md5 <<< "$row"
    local out="$DEST/$name"
    if [[ -f "$out" ]] && [[ "$(actual_size "$out")" -eq "$size" ]]; then
      note "already present, skipping: $out"
      continue
    fi
    note "fetch $name ($size B)"
    curl -fL --retry 3 --retry-delay 5 \
         -o "$out" "https://ndownloader.figshare.com/files/$fid"
    verify_size "$out" "$size"
    verify_md5  "$out" "$md5"
  done
  ok "Italy complete"
}

# ── Netherlands ────────────────────────────────────────────────────────────────
download_netherlands() {
  log "Netherlands — Rijden de Treinen open-data"
  local DEST="Data/Netherlands/Raw"
  local BASE="https://opendata.rijdendetreinen.nl/public"
  mkdir -p "$DEST"

  # 1. Six monthly services files (.csv.gz on the server, decompress to .csv)
  # filename (after gunzip) | uncompressed bytes
  local services=(
    "services-2024-01.csv|274119292"
    "services-2024-02.csv|264216460"
    "services-2024-03.csv|280145745"
    "services-2024-04.csv|270448228"
    "services-2024-05.csv|279834433"
    "services-2024-06.csv|271351308"
  )
  for row in "${services[@]}"; do
    IFS='|' read -r name size <<< "$row"
    local out="$DEST/$name"
    if [[ -f "$out" ]] && [[ "$(actual_size "$out")" -eq "$size" ]]; then
      note "already present, skipping: $out"
      continue
    fi
    note "fetch & decompress $name (~$((size/1024/1024)) MB after gunzip)"
    curl -fL --retry 3 --retry-delay 5 \
         -o "$DEST/$name.gz" "$BASE/services/$name.gz"
    gunzip -f "$DEST/$name.gz"
    verify_size "$out" "$size"
  done

  # 2. Static / versioned snapshots — direct CSVs
  # filename | URL path | bytes
  local statics=(
    "stations-2023-09.csv|stations/stations-2023-09.csv|65007"
    "tariff-distances-2022-01.csv|tariff-distances/tariff-distances-2022-01.csv|583534"
    "disruptions-2024.csv|disruptions/disruptions-2024.csv|1871062"
  )
  for row in "${statics[@]}"; do
    IFS='|' read -r name path size <<< "$row"
    local out="$DEST/$name"
    if [[ -f "$out" ]] && [[ "$(actual_size "$out")" -eq "$size" ]]; then
      note "already present, skipping: $out"
      continue
    fi
    note "fetch $name ($size B)"
    curl -fL --retry 3 --retry-delay 5 -o "$out" "$BASE/$path"
    verify_size "$out" "$size"
  done

  ok "Netherlands complete"
  note "License: services + disruptions = CC-BY 4.0 (credit 'Rijden de Treinen')."
}

# ── Finland ────────────────────────────────────────────────────────────────────
download_finland() {
  log "Finland — Kaggle FI-TW dataset"
  local DEST="Data/Finland/Raw"
  local SLUG="viniborin/finland-integrated-train-weather-dataset-fi-tw"
  mkdir -p "$DEST"

  # Pre-flight: kaggle CLI must be installed and authenticated
  if ! command -v kaggle >/dev/null 2>&1; then
    fail "kaggle CLI not found.  Install: pip install kaggle
   Then create ~/.kaggle/kaggle.json with your API token from
   https://www.kaggle.com/settings  →  'Create New API Token'."
  fi

  # filename | bytes (local, used as post-download sanity check)
  local files=(
    "matched_data_2024_01.csv|701179130"
    "matched_data_2024_02.csv|643916233"
    "matched_data_2024_03.csv|690728794"
    "matched_data_2024_04.csv|669207456"
    "matched_data_2024_05.csv|665627620"
    "matched_data_2024_06.csv|640042658"
  )
  for row in "${files[@]}"; do
    IFS='|' read -r name size <<< "$row"
    local out="$DEST/$name"
    if [[ -f "$out" ]] && [[ "$(actual_size "$out")" -eq "$size" ]]; then
      note "already present, skipping: $out"
      continue
    fi
    note "kaggle pull $name (expected $((size/1024/1024)) MB)"
    kaggle datasets download -d "$SLUG" -f "$name" -p "$DEST" --unzip --force
    if [[ ! -f "$out" ]]; then
      fail "kaggle returned successfully but $out is missing"
    fi
    local got; got="$(actual_size "$out")"
    if [[ "$got" -ne "$size" ]]; then
      printf "⚠  size mismatch: %s got %d expected %d (Kaggle dataset version may have updated since 2025-12-12 — review and re-pin if needed)\n" "$out" "$got" "$size" >&2
    else
      ok "$out  ($got B)"
    fi
  done

  ok "Finland complete"
  note "Citation: Borin et al. 2026, arXiv:2601.16592"
  note "          Dataset DOI: 10.34740/kaggle/dsv/14124620"
}

# ── Driver ─────────────────────────────────────────────────────────────────────
main() {
  local targets=("$@")
  if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(italy netherlands finland)
  fi
  for t in "${targets[@]}"; do
    case "$t" in
      italy)        download_italy        ;;
      netherlands)  download_netherlands  ;;
      finland)      download_finland      ;;
      *) fail "unknown target: $t (valid: italy netherlands finland)" ;;
    esac
  done
  log "All requested downloads verified."
}

main "$@"
