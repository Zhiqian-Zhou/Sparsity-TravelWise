# `scripts/` — data download + verification

## Files

- `download_data.sh` — fetches the exact raw datasets used by this project.
- `SHASUMS.txt` — SHA-256 hashes of every raw file as they exist on disk for the
  reference run. Use to verify a fresh download is byte-identical to ours.

## Quick start

```bash
# Install the one external dependency (only needed for Finland)
pip install kaggle
mkdir -p ~/.kaggle && cp /path/to/your/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download everything (~5.7 GB on disk after gunzip)
bash scripts/download_data.sh

# Or just one country
bash scripts/download_data.sh italy
bash scripts/download_data.sh netherlands
bash scripts/download_data.sh finland

# Verify against the reference SHASUMS.txt (strongest identity check)
( cd Data && shasum -a 256 -c ../scripts/SHASUMS.txt )
```

## What gets verified, and how strongly

| Country | Pre-download identity proof | Post-download verification |
|---|---|---|
| **Italy** | MD5 of every file is **published by figshare's API** and **was confirmed to match the user's local files exactly** before this script was written. Identity is cryptographically proven. | Script computes MD5 after download and aborts on mismatch. |
| **Netherlands** | Small files: `Content-Length` HEAD response matches local byte-size **exactly**. Service files: served gzipped on the server, but every URL is a date-stamped immutable snapshot. | Script verifies post-gunzip size after each download. |
| **Finland** | Filename pattern verified against the dataset author's own preprocessor regex. **Per-file size cannot be verified pre-download** — Kaggle requires API auth and doesn't expose sizes via the unauth API. | Script verifies size after download against user's local sizes. If Kaggle dataset is updated, sizes may differ → script logs a warning, doesn't fail. |

`SHASUMS.txt` provides the strongest possible verification across all three:
fresh downloads should hash identically to our reference. Run
`shasum -a 256 -c scripts/SHASUMS.txt` from inside `Data/`.

## Source citations

| Country | Source | License |
|---|---|---|
| Italy | figshare doi:10.6084/m9.figshare.28891607.v2 (Wu et al. 2025) | CC-BY 4.0 |
| Netherlands | rijdendetreinen.nl/en/open-data | CC-BY 4.0 (services + disruptions); CC0 (stations + tariff) |
| Finland | Kaggle: viniborin/finland-integrated-train-weather-dataset-fi-tw doi:10.34740/kaggle/dsv/14124620 (Borin 2025; arXiv:2601.16592) | GPL-3 |

If you publish results from this data:
- **Netherlands services + disruptions** require explicit credit to *Rijden de Treinen*.
- **Italy + Finland** require academic citation per the DOIs above.

## Notes

- Italy: figshare HEAD returns `Content-Length: 0` because of a 302 → S3
  presigned URL with HEAD disallowed. GET still works (confirmed via byte-range
  request). The script uses `curl -fL` which follows the redirect; this is fine.
- Netherlands services are served as `.csv.gz` (~32 MB each). The script
  decompresses them in place to match the local `.csv` layout.
- Finland is the only country needing a Kaggle API token. Without it, the
  `download_data.sh finland` run aborts with a clear setup message.
- Re-running the script is idempotent: any file already present at the
  expected size is skipped.
