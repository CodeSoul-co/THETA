# Data Upload Guide

Place your dataset in the `data/{dataset_name}/` directory and follow the column naming conventions below.

## Directory Structure

```
data/
├── example/                  ← this example folder
│   └── example_cleaned.csv
└── {your_dataset}/
    └── {dataset_name}_cleaned.csv
```

## Required Columns

| Column | Description | Required |
|--------|-------------|----------|
| `text` | Document text content (string) | ✅ Always |
| `timestamp` | Publication date, e.g. `2026-04-01` | DTM only |
| `cov_*` | Covariate columns with `cov_` prefix, e.g. `cov_platform` | STM only |
| `label` | Category label for supervised mode | Optional |

## Supported File Formats

- `.csv` (recommended — must contain a `text` column)
- `.xlsx` (auto-converted)
- `.txt` / `.docx` / `.pdf` (each file treated as one document)

## Timestamp Formats

The `timestamp` column accepts the following formats. All are converted to **year-level** granularity for DTM analysis:

```
2026                  ← year only
2026-04-01            ← date (recommended)
2026-04-01 14:30:00   ← full datetime
```

## Quick Start

```bash
# Place your data in data/{dataset_name}/, then run:
conda run -n theta bash scripts/quick_start.sh {dataset_name}

# Run with the example dataset:
conda run -n theta bash scripts/quick_start.sh example
```

## Example File

See `example_cleaned.csv` in this directory, which contains:
- 10 documents in mixed Chinese and English
- Columns: `text`, `timestamp`, `cov_platform`
- Topics covering environment, transportation, healthcare, education, and supply chain
