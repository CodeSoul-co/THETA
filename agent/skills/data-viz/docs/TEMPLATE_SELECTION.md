# Choose a starting template

Select by the user's variables and communication goal, not the biological titles of the source examples. Prefer the valid template requiring the fewest edits, including one with removable layers for unavailable data. Read its schema and implementation, then copy its actual code. The previews in [the gallery](GALLERY.en.md) are layout references to preserve by default, not merely aesthetic inspiration.

| IDs | Data / question | Reusable encoding | Adaptation to inspect |
|---|---|---|---|
| [01](../figures/figure01/README.en.md), [07](../figures/figure07/README.en.md), [08](../figures/figure08/README.en.md) | Compare grouped measurements | Bars, individual observations, error caps | Category/group counts, summary mode, jitter, old annotations |
| [06](../figures/figure06/README.en.md) | Groups with a large scale gap | Broken-axis bars with observations | Use a break only when justified and clearly labeled; consider log scale or panels |
| [02](../figures/figure02/README.en.md), [03](../figures/figure03/README.en.md) | Association between two numeric variables, across groups/cohorts | Scatter plus marginal summaries; repeated panels | Coordinates can be arbitrary variables; remove PCoA labels, variance percentages and irrelevant statistical text |
| [04](../figures/figure04/README.en.md) | Compare change along time or an ordered numeric axis | Lines, uncertainty, a summary inset | Sort time; avoid connecting unrelated categories; derive inset from the same data |
| [05](../figures/figure05/README.en.md), [17](../figures/figure17/README.en.md) | Compare distributions and their observations | Half-violin / raincloud / density + box + rug | Consistent bandwidth and scales; density, rug and summary must describe the same samples |
| [09](../figures/figure09/README.en.md) | Compare two metrics with different units | Nested bars and dual axes | Dual-axis association can mislead; prefer aligned panels when it clarifies comparisons |
| [10](../figures/figure10/README.en.md), [11](../figures/figure11/README.en.md) | Hierarchy with categorical or numeric annotations | Circular branches and annotation rings | Valid node/parent tables, meaningful branch lengths, matching leaf IDs; remap biological keys/legends |
| [12](../figures/figure12/README.en.md) | Two categorical dimensions and two metrics | Bubble matrix, color and area | Define size/color units and normalization; inspect size mapping and hardcoded panels |
| [13](../figures/figure13/README.en.md) | Positive variables spanning orders of magnitude | Log scatter and marginal histograms | Positive finite inputs; meaningful binning and tick labels |
| [14](../figures/figure14/README.en.md), [15](../figures/figure15/README.en.md), [16](../figures/figure16/README.en.md) | Many ordered group distributions | Ridgeline series / matrix | Accept density curves or grouped values; peak-normalized heights are not sample counts |
| [18](../figures/figure18/README.en.md) | Many grouped summaries across facets | Faceted boxes | Five-number ordering, group/panel labels and sample counts |
| [19](../figures/figure19/README.en.md) | Several linked measurements on common entities | Correlation, paired endpoints, bubbles and bars | Keep entity ordering consistent; use only panels supported by actual data; adapt fixed labels and correlations |
| [20](../figures/figure20/README.en.md) | Words/phrases with 2D positions, categories and emphasis | Dense text map with an optional partition mesh | Preserve coordinates for label substitution; recomputed embeddings need a matching mesh or show_mesh=false |
| [21](../figures/figure21/README.en.md) | Word groups with a supplied hierarchy | Rectangular dendrogram with stacked terminal labels | Preserve merge positions when replacing words; pruning requires consistent child IDs and monotone heights |
| [22](../figures/figure22/README.en.md) | Document–word associations and optional per-topic distributions | Bundled curves, compact labels and aligned box summaries | Keep the three-column composition; show_dissemination=false skips missing summaries without shifting the other columns |

For raw text and NLP-derived input preparation, read [natural-language data](TEXT_DATA.en.md). These templates render structured inputs; they do not automatically infer semantic positions, clustering or document-topic weights.

## Reuse first; edit only what the data need

Replace data and labels first. Modify the copied `plot.py` only where grouping loops, input handling, data layers or hardcoded labels need to change. Keep unaffected axes positions, panel order, typography and color assignments. `style.json` is not a universal layout engine; changing it alone may not remove a required CSV load or a fixed group loop.

| Situation | Preferred local adaptation |
|---|---|
| Figure 07 has group means but no sample-level observations | Keep its bars and layout; remove sample-dot code and unsupported error bars; adapt the schema to supplied summaries |
| Figure 04 has time-series data but no inset summary | Keep the main axes, line palette and legend; remove the inset and its auxiliary-data load |
| Figure 19 has only some of its linked metrics | Retain the supported panels and entity ordering; disable missing panels and their labels/loads |
| A selected template has fewer groups than its example | Remove absent groups from loops, ticks and legends; retain the remaining groups' colors and panel placement |
| There are a few extra series or an extra metric | Extend the existing series/legend or add a compatible layer locally; do not redesign the full figure |
| Many new independent variables or relationships cannot fit through local edits | Explain what the original composition cannot express, then extend or create a composition using the nearest existing code |

Missing data are not zero values. Derive summaries from actual observations only when justified; otherwise omit the unsupported layer. If the missing variable is essential to the encoding itself, use another bundled template or request the input. Never keep example observations to make the original layout look full.

New designs are exceptional: use them for explicit user requests or substantial new information that cannot be represented accurately and readably by adapting a template. A different domain, renamed columns or missing auxiliary files is not sufficient. Even for a new composition, reuse existing modules and `vizlib/common.py` instead of rebuilding equivalent drawing logic.

Use [the style guide](STYLE_GUIDE.md) for palette, hierarchy and information density. For schemas, use the per-figure **Files and columns** sections. The **Source-study context** sections are background, not input requirements for other domains.
