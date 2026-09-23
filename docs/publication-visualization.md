# Research figures and reproducible visualization

**English** | [中文](publication-visualization.zh.md)

Native model plotting uses shared font, export, and color settings in `src/models/visualization/publication.py`. Existing plot generators and STM/DTM functions render results without implementing a separate training algorithm.

## Style and export

Figures use a white background, thin axes, outward ticks, consistent topic colors, and borderless legends. Multi-panel figures are labeled a, b, c. Text is generally 7–11 pt on the final canvas. English uses available Arial/Helvetica fonts; Chinese labels require a suitable CJK font.

The layout fits within a 183 mm by 170 mm canvas while preserving proportions. Tight bounding-box export can change the final dimensions; verify the requirements of your intended publication.

Run from the repository root with visualization dependencies installed. This reads an existing experiment without training:

```bash
python src/models/visualization/run_visualization.py \
  --baseline --model lda --dataset OPC --num_topics 8 \
  --result_dir /path/to/single/experiment \
  --workspace_dir /path/to/its/preprocessing/workspace \
  --output_dir result/OPC-publication/en \
  --language en --dpi 600 --formats png pdf svg
```

Use `--dpi 150` for previews, `300` by default, or `600` for high-resolution output; accepted values are 72–1200. Select PNG for raster output or PDF/SVG for vector formats. Dense scatter and word-cloud layers may remain rasterized. PDF embeds TrueType fonts; editable SVG text depends on fonts available on the receiving computer.

`index.html` provides searchable previews and downloads. `publication-manifest.json` lists files and settings; `chart-status.json` and `additional-chart-status.json` record generated, skipped, or failed charts. Plotting failures preserve completed files and diagnostic manifests.

THETA uses the same export options:

```bash
python src/models/visualization/run_visualization.py \
  --result_dir /path/to/results --dataset OPC --mode zero_shot \
  --model_size 0.6B --output_dir result/OPC-theta-figures \
  --language en --dpi 600 --formats png pdf svg
```

## Time and source metadata

Equal row counts alone do not establish alignment. To add time or source labels, supply the original file and the normalized data recorded during training:

```bash
--source_file data/source.xlsx \
--training_data /path/to/recorded/training/data.csv \
--text_column content --time_column published_at --group_column source
```

Training data uses the canonical `text` column. Existing `source_rows.npy` mappings are applied before text alignment is verified. Unmatched metadata is rejected. Output goes to a new visualization directory without changing source data or model matrices.

Temporal plots use valid dates and record their denominator in `temporal-scope.json`. Source plots retain the largest groups and explicitly combine the remainder; underlying tables record the displayed scope. Incomplete years and changing sample sizes must be considered when interpreting trends.

## Model-specific evidence

| Models | Requirements and interpretation |
| --- | --- |
| LDA, HDP, BTM, ETM, CTM, GSM, ProdLDA, THETA | Shared term, distribution, correlation, projection, temporal, and grouped plots require real aligned evidence |
| STM | Adds actual covariate and coefficient plots; exploratory group comparisons use BH-FDR adjustment and do not establish causation |
| DTM | Uses actual `beta_over_time`; a static beta represents the final time slice |
| NVDM | Latent coordinates are not topic probabilities; probability shares and pyLDAvis are not generated |
| BERTopic | Uses real document assignments, vocabulary, and BOW; shares exclude unsupported/outlier assignments; c-TF-IDF weights are not generative probabilities |

UMAP uses a fixed sample seed and exports matrix row indices and coordinates. Projection distances do not represent exact original-space distances. Correlation networks show Pearson thresholds, signs, and weights; compositional topic weights do not establish causation. Temporal lines connect observed values, leave gaps for missing periods, and do not invent zero observations. Metrics retain their original scales.

Term charts use actual exported weights. Missing training history, multiple-K experiments, temporal word weights, or entity links cause the corresponding plots to be skipped. pyLDAvis uses real BOW counts, records exclusions, and bundles local scripts and styles for offline viewing.

## Layouts and networks

Plots use constrained layouts and external legends. Scatter views may include an explicitly marked local view without changing coordinates. Trend panels retain the observed range and sample counts. Heatmaps use readable labels and actual values. Word clouds are layouts; exact weights remain available in tables and bar charts.

Correlation networks support `generate_topic_network(layout='arc'|'circular', threshold=0.3)`. Fixed geometric positions do not encode statistical distance. Actual displayed edges are exported to `topic_network_edges.csv`.

Sankey exports show year-to-topic or source-to-topic allocations. Each band is a sum of observed document-topic weights, not a flow of entities between years. Output includes PNG, PDF, SVG, CSV, and self-contained Plotly HTML; `plotly>=5.0` is required. Signed NVDM coordinates are not suitable for nonnegative allocation plots.

`topic_wordcloud_grid_1.*` and subsequent pages show up to six topic word clouds per page using actual positive term weights. `topic_network_circular.*` provides circular topic layouts with separate term and edge explanations. Complete edge data is preserved even when text annotations are limited.

## Examples

See the [figure atlas](examples/figure-atlas/README.md) for separate English and Chinese examples. Example figures illustrate plotting behavior and are not results from your own dataset.
