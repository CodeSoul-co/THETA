# 19 · Correlation, expression and proximity composite

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**screenshot_estimate_and_digitized** — Correlations and KS bars were estimated manually; expression bubbles were estimated from the screenshot grid. No original cell-level data were recovered.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Single-cell expression, cell types, WT/KO conditions, sample IDs and spatial coordinates or proximity measurements.
- TGFβRII signature, P14 CD8 T-cell reference population, proximity/distance definition and correlation method.
- Per-cell-type WT/KO correlations, gene mean expression/positive fractions, two-sample KS values and the Further/Closer/Similar rule.

## Source-study context: preprocessing example

Compute correlations, expression summaries, KS and proximity categories upstream. The renderer does not re-infer those statistics from expression and spatial coordinates.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure19.csv](data/figure19.csv)

`screenshot_estimate_and_digitized` · 36 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `cell_type` | string | label | Cell type matching style ordering |
| `wt` | number [-1, 1] | correlation | Specified correlation coefficient in WT |
| `ko` | number [-1, 1] | correlation | Specified correlation coefficient in KO |
| `ks` | number [0, 1] | statistic | Precomputed two-sample KS statistic |
| `relation` | Further / Closer / Similar | category | Proximity category assigned upstream |

```csv
cell_type,wt,ko,ks,relation
Enterocyte 2,0.347,0.151,0.17,Further
Enterocyte 3,0.317,0.098,0.08,Further
```

### [figure19_expression.csv](data/figure19_expression.csv)

`screenshot_estimate_and_digitized` · 288 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `cell_type` | string | label | Cell type matching style ordering |
| `gene` | string | identifier | Gene name matching style ordering |
| `expression` | number | normalized expression | Mean expression under stated normalization, not raw counts |
| `percent` | number [0, 100] | percent | Percent positive cells; controls bubble area |

```csv
cell_type,gene,expression,percent
Enterocyte 2,Tgfb1,0.03,3.1
Enterocyte 2,Tgfb2,0.03,3.1
```

## Run and replace data

```bash
python -m figures.figure19.plot --format png svg
cp -R figures/figure19/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 19 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
