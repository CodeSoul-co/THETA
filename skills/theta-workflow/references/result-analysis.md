# THETA Result Analysis

Use this reference when reading existing THETA or baseline outputs. Reading result files is allowed without extra confirmation.

## Files to Look For

Search result directories for:

- `metrics.json`
- `topic_words.json`
- `topic_table.csv`
- `topic_network.png`
- `topic_similarity.png`
- `training_history.json`
- model-specific folders under `result/`, `results/`, or `data/workspace/`

If filenames differ, inspect the nearest result directory and summarize what exists before interpreting.

## Metrics Interpretation

Explain results in practical terms:

- Coherence / NPMI: semantic consistency of topic words.
- Diversity: whether topics use distinct words.
- Exclusivity: whether important words are unique to their topic.
- Topic overlap: whether multiple topics describe the same concept.
- Stability: whether training history suggests convergence or instability.

Avoid claiming one model is best from a single metric. Prefer a balanced comparison.

## Tuning Suggestions

Use targeted recommendations:

- Topics too fragmented: lower `num_topics`.
- Topics too broad: raise `num_topics`.
- Many duplicate words: improve stopwords, lower K, or reduce overly common vocabulary.
- Low coherence: review cleaning, vocabulary size, language tokenization, and embedding choice.
- Low diversity: reduce common words and compare THETA with CTM/BERTopic.
- Unstable training: lower learning rate, increase patience, or run a smaller first experiment.

Recommend one major change per tuning round unless the user explicitly asks for a sweep.

## Reporting

Summaries in chat do not need confirmation.

Writing a report file requires confirmation. State:

- Output file path.
- Content sections.
- Whether it overwrites an existing file.
- Any source result directories used.

Good report sections:

- Dataset and preprocessing.
- Models and parameters.
- Evaluation metrics.
- Topic interpretation.
- Model comparison.
- Tuning decisions.
- Paper-ready conclusion wording.
