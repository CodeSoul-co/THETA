# 02 · PCoA with marginal boxplots

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**digitized** — Approximate coordinates extracted from colored connected components. Overlapping points are lost; ellipses were visually estimated.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- A sample-feature/abundance matrix or sample distance matrix, plus sample-to-group metadata.
- Distance metric, transformations, PCoA method, axis 1/2 coordinates and explained variation.
- PERMANOVA model, permutation count and strata, plus the ellipse convention.

## Source-study context: preprocessing example

Compute PCoA upstream and export the first two coordinates to x/y. This project does not run PCoA or PERMANOVA; replacement-data ellipses use a bivariate covariance approximation.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure02.csv](data/figure02.csv)

`digitized` · 94 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `group` | string | label | Experimental group or cell population; match style |
| `x` | number | ordination coordinate | Transformed x coordinate, not a raw feature matrix |
| `y` | number | ordination coordinate | Transformed y coordinate |

```csv
group,x,y
HC,0.28514867617107936,0.25433960354707724
HC,0.440606896551724,0.2349796530306275
```

## Run and replace data

```bash
python -m figures.figure02.plot --format png svg
cp -R figures/figure02/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 2 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
