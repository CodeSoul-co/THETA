# 10 · Radial phylogeny with annotation rings

[中文](README.md) | **English**

[← Gallery index](../../README.en.md)

![Preview](preview.png)

## Data provenance

**synthetic** — Synthetic topology, branch lengths and leaf metadata. No original phylogenetic relationships are available.

Original experimental data were not supplied, and a paper/DOI has not been verified. These values demonstrate a reusable chart layout; they are not original research measurements or a pixel-identical reproduction.

## General data requirements

**These templates are not limited to bioinformatics data.** Business, engineering, education, survey and other datasets can be used when they match the target CSV column names, types, table structure and numeric constraints. Biological names and units in the field tables describe the current examples; map their meanings to your own metrics while retaining the column names expected by the code.

Start with “Files and columns”, prepare matching inputs, and update categories, labels, units and axis limits in `style.json`. Adjust fixed layouts in `plot.py` when group or panel counts change. The original-data and preprocessing notes below explain the source study context; reusing the chart does not require those biological raw data or analyses. Mathematical constraints still apply, such as positive values on log axes, nonnegative errors and acyclic trees.

## Source-study context: original data

- Original Newick/Nexus phylogeny with actual topology, branch lengths, rooting and unique tip IDs.
- Phylum, WGS/MAG/SAG category and sample occurrence for each tip.
- Occurrence thresholds and sample scope, tip ordering and legend counts.

## Source-study context: preprocessing example

Convert Newick/Nexus to node/parent/length edges and join tip metadata. Preserve rooting, length units and ordering; the renderer performs no phylogenetic inference.

## Files and columns

`plot.py` contains the actual drawing code; `style.json` controls canvas, labels, colors and ordering; `data/` holds replaceable CSVs. `provenance.json` records per-file provenance, row count, SHA-256, columns and units.

### [figure10.csv](data/figure10.csv)

`synthetic` · 1119 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `node` | string | identifier | Unique node ID; root is reserved |
| `parent` | string | identifier | Parent ID or implicit root named root |
| `length` | number ≥ 0 | tree-specific | Branch length from parent; state its unit |
| `phylum` | string | taxonomy | Phylum label matching style.phyla |

```csv
node,parent,length,phylum
n0,root,0.14456846775405963,Other
n1,n0,0.11218894987123149,Other
```

### [figure10_leaves.csv](data/figure10_leaves.csv)

`synthetic` · 565 rows. Missing/NaN/Inf numeric values are unsupported; use UTF-8.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `node` | string | identifier | Unique node ID; root is reserved |
| `order` | unique number | ordering | Unique tip order around the circle |
| `phylum` | string | taxonomy | Phylum label matching style.phyla |
| `type` | WGS / MAG / SAG | category | Genome category; used by figure 10 |
| `presence` | 0 / 1 / 2 | category | Figure 10 occurrence: absent / at least 1 / at least 10 samples |
| `fraction` | number [0, 1] | fraction | Figure 11 detected fraction; default displayed range 0–0.6 |
| `bsh` | 0 / 1 | boolean | Figure 11 bsh absence/presence |

```csv
node,order,phylum,type,presence,fraction,bsh
n3,0,Other,MAG,0,0.0024,1
n5,1,Other,MAG,1,0.0872,1
```

## Run and replace data

```bash
python -m figures.figure10.plot --format png svg
cp -R figures/figure10/data my_data
# Edit the relevant CSVs in my_data, then run:
python render.py --figure 10 --data-dir my_data --out my_output --format png svg pdf
```

Run commands from the repository root. Changing groups, genes, panel counts, sample-count labels or units also requires updating style.json and, where necessary, plot.py layout. A fixed canvas does not adapt to arbitrary group counts. Original statistical annotations are disabled by default when CSVs change. See the [main README](../../README.en.md) for summary modes.
