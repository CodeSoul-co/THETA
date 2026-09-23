# 21 · Word-group dendrogram

[中文](README.md) | [English](README.en.md)

[← Main README](../../README.en.md)

![Preview](preview.png)

## Provenance and reconstruction scope

Nine visible stacks containing 36 words were transcribed; eight merges and their heights were estimated from the image. Heights are display estimates, not distances computed from original neural responses or word vectors.

Original numerical data were not supplied and the paper/DOI is unverified. This is not a pixel-identical reconstruction. Rendering uses vector code and does not depend on reference screenshots.

## General data requirements

- Leaf IDs, displayed word stacks and left-to-right order.
- A complete binary hierarchy: merge node ID, child IDs and height for each merge. Inputs may come from text clustering, topic hierarchies or other domains.
- For substantive clustering interpretation, record text representation, distance and clustering method; derive the hierarchy from that actual analysis.

The code draws an explicit merge table; it does not analyze neural data or cluster text. Leaves are evenly spaced inside the original axes; each parent x position is the midpoint of its children. Heights must be monotone and within height_limit.

## Preserve layout while adding/removing elements

Replace leaf words directly to preserve the tree. Update heading for a new domain. Removing a leaf requires pruning its merge and collapsing single-child branches, not merely deleting its label row. Added groups need corresponding merges; retain canvas and branch color, adjusting label size locally when needed.

## Files and columns

### [figure21.csv](data/figure21.csv)

`transcribed_and_screenshot_estimate` · 8 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `node` | string | ID | Unique node ID |
| `left` | string | ID | Left-child node ID |
| `right` | string | ID | Right-child node ID |
| `height` | number | merge-height units | Nonnegative merge height, not below either child |

```csv
node,left,right,height
m0,w4,w5,48
m1,w3,m0,61
```

### [figure21_labels.csv](data/figure21_labels.csv)

`transcribed_and_screenshot_estimate` · 9 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `node` | string | ID | Unique node ID |
| `order` | number | ordinal | Vertex order within a path or left-to-right leaf order; must be unique |
| `words` | string | text | Stacked words separated by &#124;; one stack per terminal node |

```csv
node,order,words
w0,0,feelings|asleep|enjoys|happy
w1,1,hours|inside|nearby|times
```

## Run and replace data

```bash
python -m figures.figure21.plot --format png svg pdf
cp -R figures/figure21/data my_data
cp figures/figure21/style.json my_style.json
python render.py --figure 21 --data-dir my_data --style my_style.json --out my_output --format png svg pdf --annotations none
```

Run from the project root. Numeric values must be finite; do not fill required missing inputs with fabricated zeros. Update joined IDs, legends and data dependencies after removing elements. For Chinese text, choose an installed font with Chinese glyphs and adjust label sizes locally.
