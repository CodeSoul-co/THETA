# Natural-language data: reuse the three existing layouts first

[中文](TEXT_DATA.md) | [English](TEXT_DATA.en.md)

Directly reuse the code in `figures/figure20`–`figure22`. Select by available data, replace labels and add/remove supported elements within the original layout before considering expansion. The code renders inputs; it does not automatically turn raw text into a validated semantic structure.

| Available data and goal | Template | Minimal inputs |
|---|---|---|
| Positioned words/phrases, categories and emphasis | [20 · Word map](../figures/figure20/README.en.md) | Text, unique IDs, x/y, font size and role; mesh can be disabled |
| Hierarchy between word groups | [21 · Word-group dendrogram](../figures/figure21/README.en.md) | Terminal word-stack table and merge table with child IDs/heights |
| Document-to-word associations with optional metric distributions | [22 · Association flows](../figures/figure22/README.en.md) | Link weights, document bands and topic bands; boxes can be disabled |

## When only raw text is available

Establish the observation unit and communication goal: words, phrases, sentences, documents or document groups. Preserve original text-to-ID mappings and record tokenization, deduplication, filtering, aggregation and normalization in a separate preparation script.

- Figure 20: labels can reuse display positions for a word layout, without claiming semantic distance. For semantic interpretation, obtain coordinates from an actual text representation/projection and update or disable the mesh. Record models/methods, distance definitions and random seeds. Example coordinates do not establish structure in a new corpus.
- Figure 21: obtain an actual hierarchy first, either from text-feature clustering or a user-specified topic structure. Convert a linkage result from a tool such as SciPy into `node,left,right,height`, joining terminal IDs to the label table. Never infer new merges from the reference word groups.
- Figure 22: prepare document-keyword co-occurrences, TF-IDF, topic weights or another defined association. Document what each weight means and join labels by IDs. `source_position/target_position` are layout fractions, not probabilities; blue squares are routing markers, not observations. Curves are neither chronology nor conserved flows.

Use suitable existing analysis outputs directly. If essential semantics or hierarchy are unavailable, identify the gap and select a template that does not need them or request the inputs; do not substitute simulations for an analysis. Use concise display labels for long sentences while retaining full text in the source mapping.

## Remove locally; preserve the rest

Figure 20 without a mesh: set `show_mesh` to `false`; no mesh CSV is loaded. Remove label rows for missing words; a few new labels can occupy available positions.

Figure 21 without a leaf: prune the hierarchy and collapse single-child branches, updating the merge table rather than only deleting a label. Change `heading` when the data are not neural responses.

Figure 22 without dissemination/coverage data: set `show_dissemination` to `false`; the summary CSV is not loaded and the document/word columns retain their positions. A partial summary table draws boxes only for supplied topics. Add modest numbers of associations as new link rows, and remove irrelevant labels after pruning.

## Reconstruction limits

Figure 20's text and placement are approximate readings of a low-resolution image. Figure 21's hierarchy is visually estimated, without original neural data. Figure 22's associations are seeded simulations and its box summaries are estimates, not original document-analysis results. Consult each `provenance.json`. For actual data, retain the code and visual style while replacing data and domain descriptions.
