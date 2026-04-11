"""
Abstract base class for all platform scrapers.

To add a new platform:
  1. Subclass AbstractScraper
  2. Implement fetch()
  3. Register in src/scrapers/registry.py

Contract for fetch() return value
----------------------------------
Each record is a plain dict that MUST contain at least:
  - "text"  : str   — the main text content

Optional keys (forwarded to THETA CSV as-is):
  - "timestamp"   : str | int  — ISO date string or year integer (needed by DTM)
  - "id"          : str        — unique post ID (used for deduplication in adapter)
  - "cov_*"       : any        — covariate columns (needed by STM); key must start with "cov_"

Any extra keys are preserved in the output CSV so downstream analysis is not limited.
"""

from abc import ABC, abstractmethod


class AbstractScraper(ABC):
    """Base class that every platform scraper must inherit from."""

    @abstractmethod
    def fetch(self, keywords: list[str], max_results: int, **kwargs) -> list[dict]:
        """
        Fetch posts matching the given keywords.

        Parameters
        ----------
        keywords : list[str]
            Search terms. Multiple terms are combined with OR logic.
        max_results : int
            Approximate upper bound on the number of records to return.
            Implementations may return slightly more due to pagination boundaries.
        **kwargs
            Platform-specific options (e.g. lang, start_time, end_time).

        Returns
        -------
        list[dict]
            Raw records conforming to the contract described in the module docstring.
        """

    def name(self) -> str:
        """Human-readable platform name for logging."""
        return self.__class__.__name__
