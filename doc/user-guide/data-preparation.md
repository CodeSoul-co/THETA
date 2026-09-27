# Data Preparation

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/doc/user-guide/data-preparation.zh.md)

This guide covers data format requirements and cleaning procedures.

---

## Data Format Requirements

THETA accepts CSV files with specific column requirements. The preprocessing pipeline recognizes several standard column names for text content.

**Accepted text column names:**
- `text`
- `content`
- `cleaned_content`
- `clean_text`

**Optional columns:**
- `label` or `category` - Required for supervised mode
- `year`, `timestamp`, or `date` - Required for DTM (temporal analysis)

Example CSV structure:

```csv
text,label,year
"Document about renewable energy and solar panels.",Environment,2020
"Article discussing machine learning applications.",Technology,2021
"Policy paper on healthcare reform.",Healthcare,2022
```

---

## Data Cleaning

Raw text often contains noise that degrades topic quality. The data cleaning module handles common issues in both English and Chinese text.

### English Data Cleaning

```bash
cd ./THETA

python -m dataclean.main \
    --input ./data/raw_data.csv \
    --output ./data/cleaned_data.csv \
    --language english
```

The cleaning process removes:
- HTML tags and markup
- URLs and email addresses
- Special characters and symbols
- Extra whitespace
- Non-printable characters

### Chinese Data Cleaning

Chinese text requires specialized processing for proper segmentation and cleaning.

```bash
python -m dataclean.main \
    --input ./data/raw_data.csv \
    --output ./data/cleaned_data.csv \
    --language chinese
```

Additional steps for Chinese:
- Removes traditional punctuation marks
- Handles full-width and half-width characters
- Preserves Chinese word boundaries

### Batch Cleaning

Process multiple files in a directory:

```bash
python -m dataclean.main \
    --input ./data/raw/ \
    --output ./data/cleaned/ \
    --language english
```

All CSV files in the input directory will be processed and saved to the output directory with the same filenames.

## Training, validation and test data in the workbench

In the data selection card, enable **Custom training / validation / test split** to choose either:

- **Ratios:** three positive percentages adding up to 100%, with random or sequential allocation. The initial ratios are 70% / 20% / 10%. Random allocation uses a configurable seed; sequential allocation takes training rows first, followed by validation and test rows.
- **Separate uploads:** select or upload three different files. Each file can use a supported table or document format, with its own text, time, label and metadata columns. Documents are segmented by content. The full dataset is the concatenation of the three uploads.

With the switch off, training and validation use a random 70% / 30% split, and testing uses the full dataset. This test includes training and validation rows and is **not an independent holdout score**. At least two training records and one record in each evaluation set must remain after preprocessing. Word vocabulary and model fitting use training records only; all result groups use that same fitted model.

Open **Results → Dataset results** to compare full-data, training, validation and test metrics and average topic weights, and download each group's document results and topic matrix. Older runs without split records are identified explicitly and are not assigned invented subsets. This workflow is shared by Web and desktop. CLI Agent training plans accept `dataSplit` with `enabled`, `mode: "ratio"`, `ratios`, `method` and `seed`.
