# 22 · Document–word flows and distributions

[中文](README.md) | [English](README.en.md)

[← Main README](../../README.en.md)

![Preview](preview.png)

## Provenance and reconstruction scope

Document/topic labels are transcribed with possible reading errors; band positions and ten box summaries are visual estimates. The 300 association curves are simulated with seed 202622, not recovered links or text-analysis results. Blue squares mark path-routing positions and are not additional observations.

Original numerical data were not supplied and the paper/DOI is unverified. This is not a pixel-identical reconstruction. Rendering uses vector code and does not depend on reference screenshots.

## General data requirements

- Document/document-group IDs and short labels; topic, keyword or word-group IDs, labels and colors.
- Document-to-topic association rows, such as co-occurrence counts, TF-IDF, topic weights or coded strengths, with a documented definition.
- Optionally, a defined per-topic metric distribution or ordered five-number summary. The source meaning of Dissemination is unverified; replace it with a defined coverage or other metric.

Prepare association and label tables upstream. Give each link start/end a 0–1 layout fraction within its band. Color follows the target topic, width equals weight×flow_width, and paths bundle through group centers. Curves are neither chronology nor conserved Sankey flows. Boxes use supplied summaries, not quantities inferred from link weights.

## Preserve layout while adding/removing elements

Keep the three-column layout while replacing labels and links. Remove unavailable association rows; band positions may remain. Set show_dissemination=false to preserve the flow layout without reading the missing summary file. Partial summary tables draw boxes only for supplied topics. Set show_nodes=false to hide decorative routing markers.

## Files and columns

### [figure22.csv](data/figure22.csv)

`synthetic` · 300 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `link_id` | string | ID | Unique association-link ID |
| `document_id` | string | ID | Document or document-group ID |
| `topic_id` | string | ID | Word-group or topic ID |
| `weight` | number | relative weight | Positive association weight; multiplied by flow_width for line width |
| `source_position` | number | fraction [0,1] | Link start position within document band; layout parameter |
| `target_position` | number | fraction [0,1] | Link end position within topic band; layout parameter |

```csv
link_id,document_id,topic_id,weight,source_position,target_position
l000,d0,t8,0.603,0.01389,0.93396
l001,d0,t4,0.651,0.04167,0.12188
```

### [figure22_documents.csv](data/figure22_documents.csv)

`transcribed_estimated_and_synthetic` · 8 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `document_id` | string | ID | Document or document-group ID |
| `label` | string | text | Displayed label; &#124; creates a line break |
| `y0` | number | canvas pixels | Upper band boundary |
| `y1` | number | canvas pixels | Lower band boundary, strictly greater than y0 |

```csv
document_id,label,y0,y1
d0,Knotted protein|Bioinformatics|Computational biology,16,60
d1,BioUML|K-mer|Intl. Soc. for Comp. Biol.,60,101
```

### [figure22_summary.csv](data/figure22_summary.csv)

`transcribed_estimated_and_synthetic` · 10 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `topic_id` | string | ID | Word-group or topic ID |
| `minimum` | number | metric units | Displayed lower whisker; specify whether this is minimum or another whisker rule |
| `q1` | number | metric units | First quartile |
| `median` | number | metric units | Median |
| `q3` | number | metric units | Third quartile |
| `maximum` | number | metric units | Displayed upper whisker; document the chosen whisker rule |

```csv
topic_id,minimum,q1,median,q3,maximum
t0,0.19,0.34,0.55,0.71,1.0
t1,0.11,0.32,0.48,0.71,1.0
```

### [figure22_topics.csv](data/figure22_topics.csv)

`transcribed_estimated_and_synthetic` · 10 rows

| Column | Type | Units | Meaning |
|---|---|---|---|
| `topic_id` | string | ID | Word-group or topic ID |
| `label` | string | text | Displayed label; &#124; creates a line break |
| `y0` | number | canvas pixels | Upper band boundary |
| `y1` | number | canvas pixels | Lower band boundary, strictly greater than y0 |
| `color` | string | Matplotlib color | Color for links associated with this topic |

```csv
topic_id,label,y0,y1,color
t0,"linear, beam, models,|air, center",16,53,#8fb1ca
t1,"formula, electric, spin,|frequency, wave",53,83,#adc36b
```

## Run and replace data

```bash
python -m figures.figure22.plot --format png svg pdf
cp -R figures/figure22/data my_data
cp figures/figure22/style.json my_style.json
python render.py --figure 22 --data-dir my_data --style my_style.json --out my_output --format png svg pdf --annotations none
```

Run from the project root. Numeric values must be finite; do not fill required missing inputs with fabricated zeros. Update joined IDs, legends and data dependencies after removing elements. For Chinese text, choose an installed font with Chinese glyphs and adjust label sizes locally.

## Reproduce the simulated links

```bash
python figures/figure22/simulate_links.py --out simulated_links.csv --seed 202622
```

Regenerates only the association CSV. Labels and box summaries are separate estimated snapshots. Existing output files are never overwritten.
