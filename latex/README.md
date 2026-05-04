# LaTeX report — `latex/`

A complete research-paper-style LaTeX report covering the full Sparsity-TravelWise pipeline from preprocessing through cause-of-disruption analysis.

## Contents

```
latex/
├── main.tex                    # Document entry: title, abstract, TOC, includes
├── preamble.tex                # Packages, math operators, hyperref/cleveref
├── references.bib              # 17 BibTeX entries (papers, datasets, tools)
├── README.md                   # This file
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_data.tex
│   ├── 04_features.tex
│   ├── 05_models.tex
│   ├── 06_evaluation.tex
│   ├── 07_xai.tex
│   ├── 08_kg.tex
│   ├── 09_cause.tex
│   ├── 10_transfer.tex
│   ├── 11_discussion.tex
│   ├── 12_conclusion.tex
│   └── A_appendix.tex
├── preprocessing/              # symlink → ../preprocess/figures (28 PNG)
└── stop_level/                 # symlink → ../stop_level/figures (23 PNG)
```

The figures are pulled directly from the existing pipeline output via two symlinks to avoid duplication.

## Build

The project's runtime conda environment ships an incomplete TeX Live (`texlive-core` from conda-forge is missing the `mktexlsr.pl` helper). For a clean build, install a full TeX distribution and use `latexmk` or pdflatex+bibtex.

### Option 1: Ubuntu/Debian (system-wide)

```bash
sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
    texlive-latex-extra texlive-fonts-recommended texlive-science \
    texlive-bibtex-extra biber latexmk
cd latex/
latexmk -pdf -interaction=nonstopmode -shell-escape main.tex
```

### Option 2: TinyTeX (user-level, ~150 MB)

```bash
# Install TinyTeX (R-based but works for any LaTeX user)
wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tlmgr install latexmk biblatex biber booktabs siunitx microtype \
    cleveref subcaption float caption hyperref minted listings \
    algorithm algorithmicx fancyhdr todonotes tabularx tabulary \
    enumitem makecell multirow rotating
cd latex/
latexmk -pdf -interaction=nonstopmode main.tex
```

### Option 3: Overleaf (cloud)

Upload the `latex/` folder (preserving symlinks: replace them with the actual figure folders before zipping) to [Overleaf.com](https://www.overleaf.com), set the document class to "Article", and click Recompile. Overleaf has biber and minted preinstalled.

## Compilation order (manual)

If `latexmk` is unavailable, run the standard 4-pass cycle:

```bash
pdflatex main.tex      # 1st pass — generates aux files
bibtex main            # process bibliography (or biber main)
pdflatex main.tex      # 2nd pass — resolve citations
pdflatex main.tex      # 3rd pass — finalise cross-references
```

## Without `minted`

The preamble has a graceful fallback to `listings` if `minted` (which needs Pygments + `--shell-escape`) is unavailable:

```latex
\IfFileExists{minted.sty}{...}{...}
```

If you prefer no shell-escape, just remove the `--shell-escape` flag — the listings fallback will engage automatically.

## What's in the report

| Section | Pages (est.) | Content |
|---|---|---|
| 1 Introduction | 2 | Problem framing + 5 contributions |
| 2 Related work | 2 | ML ensembles, XAI, graph DBs, transfer |
| 3 Data sources | 3 | IT/FI/NL schemas + per-country quirks + 2 figures |
| 4 Feature engineering | 3 | Common Data Model, anti-leakage, 16 feature groups, splits, memory |
| 5 Models | 2 | 4-model zoo, imbalance handling, threshold tuning, run-times |
| 6 Evaluation | 4 | Headline metrics + 6 figures (incl. PR-ROC grid, calibration, breakdowns) |
| 7 XAI | 4 | TreeSHAP global, beeswarm, lag ablation, scenario uplift, waterfalls |
| 8 KG | 3 | Schema, build stats, bridge query, Sparksee swap path |
| 9 Cause prediction | 4 | NL labelled data + 5-model v1/v2 + per-class breakthrough |
| 10 Transfer | 3 | IT + FI failure modes, agent reviews, fix path |
| 11 Discussion | 2 | Inflight uplift interpretation, limitations |
| 12 Conclusion | 1 | Summary + future work |
| Appendix | 4 | 14 additional figures + reproducibility statement |
| **Total** | **~37** | |

## Total citations

17 BibTeX entries: 8 foundational ML papers (Breiman, Chen, Ke, Hamilton, Hochreiter, Schuster, Lundberg×2, Pedregosa, Paszke, Fey), 4 datasets (Trenitalia, FI-TW, RDT, Open-Meteo), 3 tools (Sparksee/DEX, Kuzu), 1 domain survey, 1 PyTorch.
