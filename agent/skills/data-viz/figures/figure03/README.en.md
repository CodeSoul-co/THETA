# 03 · Multi-cohort PCoA composite

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**digitized** — Approximate coordinates extracted from screenshot colors. Some study assignments and control/IBD marker assignments are heuristic.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Feature tables or a harmonized distance matrix; study, control/IBD group and ID per sample.
- Cross-cohort normalization/batch handling, PCoA coordinates and explained variation.
- Original PERMANOVA design including study effects, covariates and permutation restrictions.

## Source-study context: preprocessing example

Harmonize cohorts and compute PCoA upstream, then join coordinates to study/group. Rendering summarizes marginal distributions but does not perform batch correction or PERMANOVA.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure03.csv](data/figure03.csv)

`digitized` · 1180 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `study` | string | identifier | Cohort identifier matching the study order in style |
| `group` | string | label | Experimental group or cell population; match style |
| `x` | number | ordination coordinate | Transformed x coordinate, not a raw feature matrix |
| `y` | number | ordination coordinate | Transformed y coordinate |

```csv
study,group,x,y
Puxi_cohort,IBD,0.35842044134727064,0.5426829268292683
Puxi_cohort,IBD,0.34871273712737133,0.5232384823848238
```

## Run and replace data

```bash
python -m figures.figure03.plot --format png svg
cp -R figures/figure03/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 3 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
