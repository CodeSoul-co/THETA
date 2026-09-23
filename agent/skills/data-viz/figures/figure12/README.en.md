# 12 · Grouped gene-expression dot plot

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**digitized** — Bubble radii and colors were estimated on the screenshot grid, not recovered from an expression matrix.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Gene-by-cell/nucleus expression matrix, cell-type annotations and Early/Intermediate/Late gene groups.
- Normalization/log/scaling method, expression-positive threshold, per-group denominator and missing-value handling.
- Compute mean expression and percent positive for each gene × cell_type; the renderer accepts these summaries.

## Source-study context: preprocessing example

Aggregate gene × cell_type upstream. expression is the mean under the chosen normalization; percent is 0–100 percent positive. The code does not perform single-cell normalization.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure12.csv](data/figure12.csv)

`digitized` · 378 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `section` | string | label | Early/Intermediate/Late section |
| `gene` | string | identifier | Gene name matching style ordering |
| `cell_type` | string | label | Cell type matching style ordering |
| `expression` | number | normalized expression | Mean expression under stated normalization, not raw counts |
| `percent` | number [0, 100] | percent | Percent positive cells; controls bubble area |

```csv
section,gene,cell_type,expression,percent
Early,CRISPLD2,Activated fibroblast,0.0,30.25
Early,CRISPLD2,Fibroblast,0.67,56.25
```

## Run and replace data

```bash
python -m figures.figure12.plot --format png svg
cp -R figures/figure12/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 12 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
