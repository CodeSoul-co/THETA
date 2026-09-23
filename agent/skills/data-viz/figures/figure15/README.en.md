# 15 · Flow-cytometry ridgeline matrix

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**synthetic_fitted_curve** — Gaussian-mixture demonstration curves approximating visible modes; not FCS events.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Original FCS files or per-event marker intensities, sample IDs, cell populations and −/+ stimulation condition.
- Compensation matrix, gating hierarchy, live-cell/doublet filters and marker-to-channel mapping.
- Log/logicle/arcsinh transformation and parameters, negative controls and density normalization.

## Source-study context: preprocessing example

Compensate, gate and transform expression upstream, then supply marker/group/condition/value or x/density. Power-of-ten ticks assume the input is already in the corresponding logarithmic coordinates.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure15.csv](data/figure15.csv)

`synthetic_fitted_curve` · 12320 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `marker` | string | identifier | Marker corresponding to the measurement channel |
| `group` | string | label | Experimental group or cell population; match style |
| `condition` | − / + | category | Unstimulated/stimulated; minus is Unicode − |
| `x` | number | transformed expression | Transformed x coordinate, not a raw feature matrix |
| `density` | number ≥ 0 | row spacing | Display height of a distribution, not sample count |

```csv
marker,group,condition,x,density
IFN-γ,ILC3,−,-0.8,0.0
IFN-γ,ILC3,−,-0.7689,0.0
```

## Run and replace data

```bash
python -m figures.figure15.plot --format png svg
cp -R figures/figure15/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 15 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
