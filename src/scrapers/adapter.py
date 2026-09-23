"""
ThetaAdapter — middleware between scrapers and the THETA pipeline.

Responsibilities
----------------
- Accept raw records from any AbstractScraper
- Normalize to THETA's strict CSV schema (text, timestamp, cov_*)
- Drop rows where text is empty after stripping
- Append to an existing dataset file (de-duplicate by "id" if present)
- Write the final CSV to DATA_DIR/{dataset_name}/{dataset_name}_cleaned.csv

THETA's prepare_data.py:find_data_file() looks for *_cleaned.csv first,
so no changes to THETA core are needed.
"""

import os
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Path resolution (mirrors THETA's config.py logic without importing it)
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    """
    Resolve DATA_DIR from the environment, falling back to <project_root>/data.
    This mirrors the logic in src/models/config.py:get_absolute_path().
    """
    data_dir = os.environ.get("DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir)
    # Walk up from this file to find the project root (src/scrapers/adapter.py → root)
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ThetaAdapter:
    """
    Converts raw scraper output into a THETA-compatible CSV file.

    Parameters
    ----------
    data_dir : Path | None
        Override DATA_DIR resolution. If None, reads from environment.
    """

    # Columns that THETA treats specially — everything else is kept as-is
    _THETA_TEXT_COL = "text"
    _THETA_TIME_COL = "timestamp"
    _COV_PREFIX = "cov_"
    _ID_COL = "id"

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else _resolve_data_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        records: list[dict],
        dataset_name: str,
        verbose: bool = True,
    ) -> Path:
        """
        Normalize records and write (or append) to the dataset CSV.

        Parameters
        ----------
        records : list[dict]
            Raw records from a scraper. Each must have at least a "text" key.
        dataset_name : str
            Determines the output directory and filename:
            DATA_DIR/{dataset_name}/{dataset_name}_cleaned.csv
        verbose : bool
            Print progress information.

        Returns
        -------
        Path
            Absolute path to the written CSV file.
        """
        if not records:
            raise ValueError("No records to save — scraper returned an empty list.")

        df_new = self._normalize(records)

        if verbose:
            print(f"[adapter] Normalized {len(df_new)} records.")

        output_path = self._output_path(dataset_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            df_existing = pd.read_csv(output_path, dtype=str)
            df_combined = self._merge(df_existing, df_new)
            if verbose:
                added = len(df_combined) - len(df_existing)
                print(
                    f"[adapter] Existing file has {len(df_existing)} rows. "
                    f"Adding {added} new rows (after deduplication)."
                )
        else:
            df_combined = df_new

        df_combined.to_csv(output_path, index=False, encoding="utf-8")

        if verbose:
            print(f"[adapter] Saved {len(df_combined)} rows → {output_path}")

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, records: list[dict]) -> pd.DataFrame:
        """Convert raw dicts to a clean DataFrame with THETA-compatible columns."""
        df = pd.DataFrame(records)

        # Ensure "text" column exists
        if self._THETA_TEXT_COL not in df.columns:
            raise KeyError(
                f"Scraper records must contain a '{self._THETA_TEXT_COL}' key. "
                f"Got columns: {list(df.columns)}"
            )

        # Drop rows with empty text
        df[self._THETA_TEXT_COL] = df[self._THETA_TEXT_COL].fillna("").str.strip()
        df = df[df[self._THETA_TEXT_COL] != ""].copy()

        if df.empty:
            raise ValueError("All records have empty text after stripping whitespace.")

        # Build output column order: text first, then timestamp, then cov_*, then rest
        ordered_cols = [self._THETA_TEXT_COL]
        if self._THETA_TIME_COL in df.columns:
            ordered_cols.append(self._THETA_TIME_COL)
        cov_cols = [c for c in df.columns if c.startswith(self._COV_PREFIX)]
        ordered_cols.extend(sorted(cov_cols))
        # Add remaining columns (id and any platform-specific extras)
        remaining = [
            c for c in df.columns if c not in ordered_cols
        ]
        ordered_cols.extend(remaining)

        return df[ordered_cols].reset_index(drop=True)

    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """Append new rows, deduplicating by 'id' if that column is present."""
        combined = pd.concat([existing, new], ignore_index=True)
        if self._ID_COL in combined.columns:
            combined = combined.drop_duplicates(
                subset=[self._ID_COL], keep="first"
            ).reset_index(drop=True)
        return combined

    def _output_path(self, dataset_name: str) -> Path:
        safe_name = re.sub(r"[^\w\-]", "_", dataset_name)
        return self.data_dir / safe_name / f"{safe_name}_cleaned.csv"
