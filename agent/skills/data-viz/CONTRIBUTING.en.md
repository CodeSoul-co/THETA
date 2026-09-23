# Adding and maintaining figures

[中文](CONTRIBUTING.md) | **English**

## Minimal figure structure

Run `python -m scripts.new_figure --id 23 --title "新图" --title-en "New figure"`. Each `figures/figureNN/` needs:

- `plot.py` implementing `draw(context, rows)`, runnable from the repository root with `python -m figures.figureNN.plot`.
- `style.json` with an ID matching its directory, canvas settings and the main CSV filename.
- `data/` with UTF-8 CSVs that may be published.
- `provenance.json` with bilingual raw-data requirements, origin categories and per-file row counts, fields, units and SHA-256.
- `example_hashes.json`, a published-example fingerprint map outside the data directory.
- Chinese and English READMEs describing actual sources, missing inputs and preprocessing.
- `preview.png` rendered from the current public example.

`context.s` holds the style; `context.fig` is a Matplotlib Figure; `context.ax([left,top,width,height])` positions an axis in canvas pixels; `context.load("auxiliary.csv")` reads from the active data directory. `context.annotations` controls reference annotations and `context.rng` provides deterministic jitter.

## Document the raw inputs

State the observation unit, sample IDs, physical units, groups/pairing, normalization, missing values, exclusions and upstream methods. Distinguish:

1. Original research inputs such as FCS, Ct measurements, expression matrices, trajectories or Newick trees.
2. Analyzed plotting inputs such as PCoA coordinates, KDE curves, mean/SEM, correlations or KS statistics.
3. Synthetic, visually estimated or digitized demonstration values.

Set `original_raw_data_available` to false when originals were not obtained. Keep `source_publication: null` if no source has been verified. Retained sample-count labels and P values are not newly measured or computed results.

## Tree input conventions

The edge table uses `node,parent,length,phylum`. `root` names the implicit root, node IDs are unique, lengths are nonnegative, and every node must be connected without cycles. Join the leaf table by node ID, with exactly one row per terminal node. Unique `order` values determine circular ordering. Phylum labels must match style.phyla.

Figure 10 uses WGS/MAG/SAG `type` and 0/1/2 `presence`. Figure 11 uses `fraction` in [0,1] and `bsh` in {0,1}, with a default heatmap range of 0–0.6. Leaf CSVs retain all columns for compatibility, but each figure consumes only its applicable annotations. Convert real tree files to edge tables upstream: no phylogenetic inference or Newick parser is included. Figure 11's fixed terminal radius changes the visual proportion of terminal branch lengths.

## Updating a published example

Do not edit repository fingerprints when changing your private CSVs. Update `provenance.json` row counts, fields and SHA-256 plus `example_hashes.json` only when intentionally publishing a new example snapshot. The standard library can compute a checksum:

```python
from pathlib import Path
import hashlib
p = Path("figures/figure23/data/figure20.csv")
print(hashlib.sha256(p.read_bytes()).hexdigest())
```

Initial examples used NumPy seed 190913, but a cross-figure random stream and manual digitization cannot be reconstructed from that seed alone. **The committed CSV snapshot defines the exact current example.** Put any new simulator and its independent seed inside the relevant figure folder, and document parameters and distribution assumptions.

## Checks before publishing

```bash
python -m figures.figure23.plot --format png svg
cp output/figure23.png figures/figure23/preview.png
python -m scripts.build_gallery
python check_reuse.py
```

If a new chart lacks a generic `value` column, register a numerical field that actually affects its output in `check_reuse.py`'s `CHANGES`. The check confirms that changing values changes the image. Do not publish sensitive research observations, credentials, original reference screenshots or local machine paths. Rights to original study resources are determined by their providers.
