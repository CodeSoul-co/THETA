# 14 · Molecular-dynamics ridgelines

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**synthetic_fitted_curve** — Gaussian mixtures fitted approximately by eye to screenshot peak positions, widths and heights; not trajectory samples.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Frame/time/replicate IDs and per-frame Core RMSD, A-loop RMSD and Distance 1 Δ values.
- Reference structure, atom selection, alignment procedure, distance definition and Å units.
- Time windows and sampling scheme for the 0/8/12/16/20/24 ns groups; correlated frames are not independent replicates.

## Source-study context: preprocessing example

Provide panel/group/value observations for automatic KDE, or x/density curves. Peak height is normalized for display and does not encode sample count.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure14.csv](data/figure14.csv)

`synthetic_fitted_curve` · 16290 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `panel` | string or integer | label | Panel index/name matching style |
| `group` | string | label | Experimental group or cell population; match style |
| `x` | number | Å | Transformed x coordinate, not a raw feature matrix |
| `density` | number ≥ 0 | row spacing | Display height of a distribution, not sample count |

```csv
panel,group,x,density
0,0,2.05,0.0
0,0,2.05387,0.0
```

## Run and replace data

```bash
python -m figures.figure14.plot --format png svg
cp -R figures/figure14/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 14 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
