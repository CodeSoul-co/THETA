# 20 · Word map with partition mesh

[中文](README.md) | [English](README.en.md)

[← Main README](../../README.en.md)

![Preview](preview.png)

## Provenance and reconstruction scope

427 word/phrase labels were compiled from readable portions of the low-resolution image; wording and placement are approximate, not a verbatim recovery. Colored boundaries were detected and consolidated into vector segments. Original semantic vectors, distances and true partitions were not recovered.

Original numerical data were not supplied and the paper/DOI is unverified. This is not a pixel-identical reconstruction. Rendering uses vector code and does not depend on reference screenshots.

## General data requirements

- Words or phrases, unique IDs, 2D positions, categories and emphasis.
- For semantic interpretation, generate coordinates with your own embeddings/projection and record the method. For layout reuse alone, replace labels while retaining display positions.
- Optional partition mesh must use the same coordinate system; example boundaries are not semantic partitions for a new corpus.

The module draws text and polylines; it does not compute embeddings, UMAP, t-SNE, clustering or similarity. Normalize coordinates to style xlim/ylim if needed; the default y direction is downward.

## Preserve layout while adding/removing elements

Preserve positions, font sizes, red emphasis and mesh layout when replacing labels. Remove rows for omitted words; place a few additions in available space. Recomputed coordinates need a matching mesh. Set show_mesh=false to skip both drawing and loading it when unavailable. Use concise labels or local spacing/font edits for long text.

## Files and columns

### [figure20.csv](data/figure20.csv)

`transcribed_estimated_and_digitized` · 427 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `word_id` | string | ID | Unique label ID |
| `text` | string | text | Displayed word or phrase |
| `x` | number | canvas pixels | Horizontal position; left to right in this example |
| `y` | number | canvas pixels | Vertical position; top to bottom in this example |
| `role` | string | enum | word or highlight, selecting text color |
| `font_size` | number | points | Positive text size |
| `group` | string | label | Label category; metadata, not an inferred cluster |

```csv
word_id,text,x,y,role,font_size,group
h000,Kindness,258.0,11.0,highlight,3.9,Kindness
h001,Deception,263.0,25.0,highlight,3.9,Deception
```

### [figure20_mesh.csv](data/figure20_mesh.csv)

`transcribed_estimated_and_digitized` · 496 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `path_id` | string | ID | Polyline identifier |
| `order` | number | ordinal | Vertex order within a path or left-to-right leaf order; must be unique |
| `x` | number | canvas pixels | Horizontal position; left to right in this example |
| `y` | number | canvas pixels | Vertical position; top to bottom in this example |

```csv
path_id,order,x,y
p000,0,169.22,312.33
p000,1,40.03,236.66
```

## Run and replace data

```bash
python -m figures.figure20.plot --format png svg pdf
cp -R figures/figure20/data my_data
cp figures/figure20/style.json my_style.json
python render.py --figure 20 --data-dir my_data --style my_style.json --out my_output --format png svg pdf --annotations none
```

Run from the project root. Numeric values must be finite; do not fill required missing inputs with fabricated zeros. Update joined IDs, legends and data dependencies after removing elements. For Chinese text, choose an installed font with Chinese glyphs and adjust label sizes locally.
