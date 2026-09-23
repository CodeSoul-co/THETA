# 17 · Regional age-distribution rainclouds

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**mixed** — Distribution outlines and box summaries were estimated from the screenshot; rug marks are seeded synthetic examples.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Dated sample ID, region and age in kyr BP; retain age uncertainty or calibrated probability distributions.
- BP reference year, calibration, weights, distribution-aggregation method and KDE bandwidth.
- Five-number summaries and rug locations/weights from the same observations; update all three plotting tables together.

## Source-study context: preprocessing example

Update density, five-number and rug tables from the same dataset. density and weight are display heights in row-spacing units, not age uncertainty.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure17.csv](data/figure17.csv)

`mixed` · 5010 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `region` | string | label | Region matching style groups |
| `age` | number | kyr BP | Thousands of years before present; state BP reference epoch |
| `density` | number ≥ 0 | row spacing | Display height of a distribution, not sample count |

```csv
region,age,density
Western Europe,15.0,0.0
Western Europe,14.98501,0.0
```

### [figure17_boxes.csv](data/figure17_boxes.csv)

`mixed` · 5 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `region` | string | label | Region matching style groups |
| `low` | number | same as observations | Lower whisker or minimum; state the convention |
| `q1` | number | same as observations | 25th percentile |
| `median` | number | same as observations | 50th percentile |
| `q3` | number | same as observations | 75th percentile |
| `high` | number | same as observations | Upper whisker or maximum; state the convention |

```csv
region,low,q1,median,q3,high
Western Europe,0.9,1.2,2.7,5.4,10.3
Central/Eastern Europe,0.25,1.9,4,5.7,10.8
```

### [figure17_rug.csv](data/figure17_rug.csv)

`synthetic` · 650 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `region` | string | label | Region matching style groups |
| `age` | number | kyr BP | Thousands of years before present; state BP reference epoch |
| `weight` | number ≥ 0 | row spacing | Rug-line display height; not age uncertainty |

```csv
region,age,weight
Western Europe,5.5856,0.0913
Western Europe,2.8353,0.0827
```

## Run and replace data

```bash
python -m figures.figure17.plot --format png svg
cp -R figures/figure17/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 17 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
