# 07 · Gene-expression grouped bars

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**screenshot_estimate** — Manual estimates of visible gene-expression values, means and errors.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- mRNA measurement and biological replicate ID for each gene/genotype at Day15.
- Raw Ct/expression, reference controls and normalization; preserve technical-to-biological replicate mapping.
- Group comparisons, multiplicity correction and error definition; a screenshot cannot recover the P-value calculation.

## Source-study context: preprocessing example

Normalize mRNA upstream, then supply replicate values. Default replacement-data summaries are mean/SEM; no new significance test is run.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure07.csv](data/figure07.csv)

`screenshot_estimate` · 91 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `category` | string | label | Category/condition; match style ordering |
| `group` | string | label | Experimental group or cell population; match style |
| `sample` | string | identifier | Example replicate ID; preserve real sample IDs when available |
| `value` | number | normalized expression / fold change | One observation; units/normalization are specified in this figure’s raw-data requirements |
| `mean` | number | normalized expression / fold change | Group mean; repeat the same summary within each group |
| `error` | number ≥ 0 | normalized expression / fold change | Error magnitude; explicitly specify SD/SEM/CI |

```csv
category,group,sample,value,mean,error
ubq-1,Wild-type,1,0.55,1,0.03
ubq-1,Wild-type,2,0.83,1,0.03
```

## Run and replace data

```bash
python -m figures.figure07.plot --format png svg
cp -R figures/figure07/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 7 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
