# 01 · Outlined bars with sample points

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**screenshot_estimate** — Manual estimates of visible counts, bar heights and error bars. Occluded samples are not recoverable.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Sample/animal ID, treatment group and PCNA+ cell count for each replicate.
- Counting region/length, normalization to 300 μm, replicate unit and exclusion rules.
- Original comparison method, error-bar definition and multiple-testing settings.

## Source-study context: preprocessing example

Supply replicate counts as value; changed CSVs default to recomputing mean and SEM.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure01.csv](data/figure01.csv)

`screenshot_estimate` · 40 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `category` | string | label | Category/condition; match style ordering |
| `group` | string | label | Experimental group or cell population; match style |
| `sample` | string | identifier | Example replicate ID; preserve real sample IDs when available |
| `value` | number | cells / 300 μm | One observation; units/normalization are specified in this figure’s raw-data requirements |
| `mean` | number | cells / 300 μm | Group mean; repeat the same summary within each group |
| `error` | number ≥ 0 | cells / 300 μm | Error magnitude; explicitly specify SD/SEM/CI |

```csv
category,group,sample,value,mean,error
,Cont,1,130,205,23
,Cont,2,142,205,23
```

## Run and replace data

```bash
python -m figures.figure01.plot --format png svg
cp -R figures/figure01/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 1 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
