# 09 · Dual-axis nested bars

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**mixed** — Manual estimates of two sets of summaries. The three reference-view points are mean ± error and mean, not observed replicates.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Indigoidine OD600 and Daptomycin concentration (μg/ml) for each strain/construct and independent replicate.
- Assay, calibration curve, dilution factors and pairing between the two measurements.
- Both sets of group means and error definitions, and the actual source of the 0.72/115 reference levels.

## Source-study context: preprocessing example

Supply both actual summaries and the intended errors. The reference view has three constructed display points, not original replicates; replacement data hide those points by default.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure09.csv](data/figure09.csv)

`mixed` · 17 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `category` | string | label | Category/condition; match style ordering |
| `indigo` | number | OD600 | Mean Indigoidine OD600 |
| `indigo_error` | number ≥ 0 | OD600 | Error of the Indigoidine summary |
| `daptomycin` | number | μg/ml | Mean Daptomycin concentration |
| `dap_error` | number ≥ 0 | μg/ml | Error of the Daptomycin summary |

```csv
category,indigo,indigo_error,daptomycin,dap_error
$Srdi^{-}$,0.72,0.1,115,10
$Srdi^{idt1}$,1.47,0.18,145,6
```

## Run and replace data

```bash
python -m figures.figure09.plot --format png svg
cp -R figures/figure09/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 9 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
