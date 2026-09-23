# Preprocessing during a conversation

**English** | [中文](conversation-preprocessing.md)

After understanding data and before configuring training, the Agent can call `dataset_preprocess`. It is a restricted pandas expression interpreter, not arbitrary host Python execution. It does not allow shell commands, network access, file I/O, package installation, imports, loops, lambdas, or arbitrary callbacks. File content is treated as data.

## Tool contract

Arguments are `{datasetRef?, purpose, code, textColumn, timeColumn?}`. The host supplies managed dataset and storage references; the model cannot supply host paths.

```python
df["content"] = df["content"].fillna("").astype("str").str.strip()
df = df[df["content"].str.len() > 0]
df["time"] = pd.to_datetime(df["published_at"], errors="coerce").dt.strftime("%Y-%m-%d")
```

Statements assign to `df` or a named column. Supported operations include column selection, boolean filters, comparisons, Series addition, missing-value handling, duplicate removal, renaming, sorting, string normalization, and date/numeric conversion.

A request supports up to 200,000 rows, 500 columns, 80 assignments, and 16,000 code characters. Worker timeouts apply; data is not silently truncated to a profile sample. Empty results, missing/duplicate columns, wholly empty text, or unparseable time columns fail without switching the training input.

Successful responses include source and derived dataset references, input/output/removed row counts, column information, a profile, and downloadable CSV, Python code, and validation JSON. The original data remains intact. A derived research record is created for subsequent training configuration; existing research and jobs are unchanged. Pending approvals or training prevent switching inputs. Preprocessing success does not approve training.

The downloaded script can be reproduced from `agent/` with the repository dependencies installed, passing the original input path and output CSV path.

## Message counts and attachments

Run details, session lists, and event snapshots expose `userMessageCount`, counting persisted user submissions rather than assistant messages, tool events, monitoring, or confirmation clicks. Duplicate request IDs do not increase the count.

Session `datasetRefs` retain managed data bindings. Removing or collapsing an attachment reference in the composer does not delete the dataset, and later messages can continue using the bound data.
