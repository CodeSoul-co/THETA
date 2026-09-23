# THETA figure atlas

[中文说明](README_zh.md) · [Visualization and export guide](../../publication-visualization.md)

Real OPC visualization examples generated on 2026-09-08 using THETA's native plotting code. Each language contains **188 figures**, grouped and numbered in one **7680 × 19328 px** overview. Open the full PNG to zoom in; the embedded image is a lightweight preview. English labels retain the original Chinese corpus terms.

| Version | Full-resolution overview | Six-figure detail | Figure index |
| --- | --- | --- | --- |
| Chinese | [PNG](zh/THETA-all-figures-zh.png) | [PNG](zh/detail-example.png) | [CSV](zh/figure-index.csv) |
| English | [PNG](en/THETA-all-figures-en.png) | [PNG](en/detail-example.png) | [CSV](en/figure-index.csv) |

## Scope

| Section | Data and model | Figures per language |
| --- | --- | ---: |
| Full OPC | 32,601 documents, LDA K=8 | 49 |
| LDA example sample | 1,800 documents, K=6; evaluation at K=4/6/8 | 41 |
| DTM example sample | Same 1,800 documents, K=6 | 49 |
| STM example sample | Same 1,800 documents, K=6 | 48 |
| Optional network threshold | Full OPC, absolute Pearson r > 0.15 | 1 |

The sample training runs validate visualization coverage; they are not full-corpus research conclusions or a model ranking. Figures requiring unavailable evidence remain omitted rather than fabricated. The index's `source` column records paths relative to the local `result/` directory; source files, raw text, matrices and model weights are not included in this example directory. The full paginated PDF/ZIP bundles remain local; this repository contains the complete overview PNGs and one detail page per language.

## Chinese overview

[![Chinese atlas preview](zh/overview-preview.jpg)](zh/THETA-all-figures-zh.png)

## English overview

[![English atlas preview](en/overview-preview.jpg)](en/THETA-all-figures-en.png)

To generate figures from your own results, follow the [native export guide](../../publication-visualization.md).
