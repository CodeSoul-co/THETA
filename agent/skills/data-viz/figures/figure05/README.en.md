# 05 · Half-violin raincloud plot

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**synthetic_from_estimated_distribution** — Synthetic observations sampled from a distribution estimated from screenshot color occupancy. Counts and boxplot statistics are not original.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Unique genome ID, species/group and BGC count per genome.
- Genome inclusion criteria, BGC-calling method/version and the independent observation unit.
- KDE bandwidth, boxplot whisker convention and outlier policy.

## Source-study context: preprocessing example

Each row is a genome BGC count. KDE and box statistics are computed from value; jitter is only a horizontal display offset.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure05.csv](data/figure05.csv)

`synthetic_from_estimated_distribution` · 2920 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `group` | string | label | Experimental group or cell population; match style |
| `value` | number | BGC count / genome | One observation; units/normalization are specified in this figure’s raw-data requirements |
| `jitter` | number, optional | display offset | Horizontal display jitter only; optional |

```csv
group,value,jitter
subtilis,16.6301,0.4465
subtilis,15.1716,0.0156
```

## Run and replace data

```bash
python -m figures.figure05.plot --format png svg
cp -R figures/figure05/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 5 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
