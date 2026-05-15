# latex-shortened

Condensed 10-page manual variant of the full paper at `../latex/`.

Standalone -- references only the parent `figures/` paths via
`\graphicspath{{../latex/figures/}{../stop_level/figures/}}` in `preamble.tex`.

## Compile

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

(Or `latexmk -pdf main.tex` if `latexmk` is installed.)
