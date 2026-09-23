# 18 · Evolutionary-age faceted boxplots

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**screenshot_estimate** — Five-number summaries estimated by eye; displayed P values are unverified reference annotations.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Paralog-pair IDs, dN and dS or dN/dS, gene age and Reference/diapause assignment.
- Alignment/substitution-rate estimation and filtering, dS=0 handling and age-bin definitions.
- Box summaries computed from actual observations, whisker convention, comparison tests and adjusted P values.

## Source-study context: preprocessing example

Compute low/q1/median/q3/high and state whether low/high are extremes or whisker endpoints. The renderer does not estimate dN/dS from sequences or retest P values.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure18.csv](data/figure18.csv)

`screenshot_estimate` · 8 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `panel` | string or integer | label | Panel index/name matching style |
| `group` | string | label | Experimental group or cell population; match style |
| `low` | number | same as observations | Lower whisker or minimum; state the convention |
| `q1` | number | same as observations | 25th percentile |
| `median` | number | same as observations | 50th percentile |
| `q3` | number | same as observations | 75th percentile |
| `high` | number | same as observations | Upper whisker or maximum; state the convention |

```csv
panel,group,low,q1,median,q3,high
"Genome-Wide
(all ages)",Reference,0,0.097,0.169,0.274,0.5
"Genome-Wide
(all ages)","Specialized
for diapause",0,0.09,0.157,0.243,0.5
```

## Run and replace data

```bash
python -m figures.figure18.plot --format png svg
cp -R figures/figure18/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 18 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
