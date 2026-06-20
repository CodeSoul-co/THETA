"""
XScraper — fetches posts from X (Twitter) using the official API v2.

Requirements
------------
    pip install tweepy

Environment variables (set in .env)
------------------------------------
    X_BEARER_TOKEN   — required for all search queries (app-only auth)
    X_API_KEY        — optional, only needed for OAuth 1.0a user-context endpoints
    X_API_SECRET     — optional
    X_ACCESS_TOKEN   — optional
    X_ACCESS_SECRET  — optional

API tier notes
--------------
- Basic tier (free developer account): search_recent_tweets only (last 7 days).
- Pro / Academic tier: full-archive search via search_all_tweets.
  Set use_full_archive=True in fetch() to enable.
"""

import os
import time
import logging
from datetime import datetime, timezone

from src.scrapers.base import AbstractScraper

logger = logging.getLogger(__name__)


class XScraper(AbstractScraper):
    """Scraper for X (Twitter) using tweepy and API v2."""

    # Maximum results per single API call (Twitter v2 hard limit)
    _PAGE_SIZE = 100

    def __init__(self):
        bearer_token = os.environ.get("X_BEARER_TOKEN", "").strip()
        if not bearer_token:
            raise EnvironmentError(
                "X_BEARER_TOKEN is not set. "
                "Add it to your .env file before using the X scraper."
            )
        try:
            import tweepy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "tweepy is required for the X scraper. Install it with:\n"
                "  pip install tweepy"
            ) from exc

        self._tweepy = tweepy
        self._client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=os.environ.get("X_API_KEY") or None,
            consumer_secret=os.environ.get("X_API_SECRET") or None,
            access_token=os.environ.get("X_ACCESS_TOKEN") or None,
            access_token_secret=os.environ.get("X_ACCESS_SECRET") or None,
            wait_on_rate_limit=True,  # tweepy handles 429 back-off automatically
        )

    def name(self) -> str:
        return "X (Twitter)"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        keywords: list[str],
        max_results: int = 500,
        lang: str = "en",
        start_time: str | None = None,
        end_time: str | None = None,
        use_full_archive: bool = False,
        exclude_retweets: bool = True,
        exclude_replies: bool = False,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch tweets matching the given keywords.

        Parameters
        ----------
        keywords : list[str]
            Search terms. Combined with OR. Example: ["AI policy", "machine learning"].
        max_results : int
            Target number of tweets to collect. Actual count may vary slightly
            due to pagination boundaries.
        lang : str
            BCP-47 language code, e.g. "en", "zh", "ja". Pass "" to skip lang filter.
        start_time : str | None
            ISO 8601 start time, e.g. "2024-01-01" or "2024-01-01T00:00:00Z".
            Basic tier is limited to the last 7 days.
        end_time : str | None
            ISO 8601 end time.
        use_full_archive : bool
            Use search_all_tweets (Pro/Academic tier) instead of search_recent_tweets.
        exclude_retweets : bool
            Append "-is:retweet" to the query.
        exclude_replies : bool
            Append "-is:reply" to the query.

        Returns
        -------
        list[dict]
            Records with keys: text, timestamp, id, cov_lang, cov_author_id.
        """
        query = self._build_query(
            keywords, lang, exclude_retweets, exclude_replies
        )
        logger.info("[X] Query: %s  |  max_results=%d", query, max_results)
        print(f"[x] Search query: {query}")

        search_fn = (
            self._client.search_all_tweets
            if use_full_archive
            else self._client.search_recent_tweets
        )

        tweet_fields = ["created_at", "lang", "author_id"]
        records: list[dict] = []
        fetched = 0
        next_token = None

        while fetched < max_results:
            batch_size = min(self._PAGE_SIZE, max_results - fetched)
            # API minimum is 10
            batch_size = max(batch_size, 10)

            try:
                response = search_fn(
                    query=query,
                    max_results=batch_size,
                    tweet_fields=tweet_fields,
                    start_time=self._parse_time(start_time) if start_time else None,
                    end_time=self._parse_time(end_time) if end_time else None,
                    next_token=next_token,
                )
            except self._tweepy.errors.TweepyException as exc:
                logger.error("[X] API error: %s", exc)
                print(f"[x] API error: {exc}")
                break

            if not response.data:
                logger.info("[X] No more results.")
                break

            for tweet in response.data:
                records.append(self._to_record(tweet))

            fetched += len(response.data)
            print(f"[x] Fetched {fetched} tweets so far...")

            meta = response.meta or {}
            next_token = meta.get("next_token")
            if not next_token:
                break

        print(f"[x] Done. Total tweets collected: {len(records)}")
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query(
        self,
        keywords: list[str],
        lang: str,
        exclude_retweets: bool,
        exclude_replies: bool,
    ) -> str:
        if len(keywords) == 1:
            keyword_part = f'"{keywords[0]}"'
        else:
            terms = " OR ".join(f'"{kw}"' for kw in keywords)
            keyword_part = f"({terms})"

        parts = [keyword_part]
        if lang:
            parts.append(f"lang:{lang}")
        if exclude_retweets:
            parts.append("-is:retweet")
        if exclude_replies:
            parts.append("-is:reply")

        return " ".join(parts)

    def _to_record(self, tweet) -> dict:
        ts = None
        if tweet.created_at:
            ts = tweet.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "text": tweet.text,
            "timestamp": ts,
            "id": str(tweet.id),
            "cov_lang": getattr(tweet, "lang", None),
            "cov_author_id": str(tweet.author_id) if tweet.author_id else None,
        }

    @staticmethod
    def _parse_time(time_str: str) -> datetime:
        """Accept 'YYYY-MM-DD' or full ISO 8601 strings."""
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(time_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse time string '{time_str}'. "
            "Expected format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ"
        )
