"""
Scraper CLI entry point for the THETA project.

Usage
-----
    python src/scrapers/cli.py --platform x \\
        --keywords "AI policy" "climate change" \\
        --dataset my_research \\
        --max-results 1000

    # List available platforms
    python src/scrapers/cli.py --list-platforms

After running, feed the dataset directly into THETA:
    python src/models/run_pipeline.py --dataset my_research --models theta
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: load .env before anything else (mirrors THETA core behaviour)
# ---------------------------------------------------------------------------

def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:
        # Minimal manual parser (same approach as THETA's config.py)
        with open(env_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports (after env is loaded)
# ---------------------------------------------------------------------------

from src.scrapers.registry import get_scraper, list_platforms, SCRAPER_REGISTRY  # noqa: E402
from src.scrapers.adapter import ThetaAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python src/scrapers/cli.py",
        description="Fetch social media data and save it as a THETA-compatible CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch English tweets about AI policy (last 7 days)
  python src/scrapers/cli.py --platform x \\
      --keywords "AI policy" "machine learning" \\
      --dataset ai_policy_2024 --max-results 500

  # Fetch with time range (Pro/Academic tier required for full archive)
  python src/scrapers/cli.py --platform x \\
      --keywords "climate change" \\
      --dataset climate_2023 --max-results 2000 \\
      --lang en --start-time 2023-01-01 --end-time 2023-12-31 \\
      --full-archive

  # List available platforms
  python src/scrapers/cli.py --list-platforms
        """,
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        default=None,
        choices=list(SCRAPER_REGISTRY.keys()),
        help="Social media platform to scrape.",
    )
    parser.add_argument(
        "--keywords", "-k",
        nargs="+",
        metavar="KEYWORD",
        help="One or more search keywords. Multiple terms are combined with OR.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help=(
            "Dataset name. Determines output path: "
            "DATA_DIR/{dataset}/{dataset}_cleaned.csv"
        ),
    )
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=500,
        help="Approximate maximum number of posts to collect (default: 500).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help=(
            "Language filter (BCP-47 code, e.g. 'en', 'zh', 'ja'). "
            "Pass '' to disable. Default: 'en'."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Earliest post time (ISO 8601). Basic API tier: last 7 days only.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Latest post time (ISO 8601).",
    )
    parser.add_argument(
        "--full-archive",
        action="store_true",
        help="Use full-archive search (Pro/Academic API tier required).",
    )
    parser.add_argument(
        "--include-retweets",
        action="store_true",
        help="Include retweets (excluded by default).",
    )
    parser.add_argument(
        "--include-replies",
        action="store_true",
        help="Include replies (excluded by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override DATA_DIR for the output path. "
            "Defaults to DATA_DIR from .env or <project_root>/data."
        ),
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List all registered scraper platforms and exit.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_platforms:
        list_platforms()
        sys.exit(0)

    # Validate required arguments for scraping
    if not args.platform:
        parser.error("--platform is required. Use --list-platforms to see options.")
    if not args.keywords:
        parser.error("--keywords is required.")
    if not args.dataset:
        parser.error("--dataset is required.")

    verbose = not args.quiet

    # Instantiate scraper
    if verbose:
        print(f"[scraper] Initialising platform: {args.platform}")
    scraper = get_scraper(args.platform)

    # Build platform-specific kwargs
    fetch_kwargs: dict = {}
    if args.platform == "x":
        fetch_kwargs = {
            "lang": args.lang,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "use_full_archive": args.full_archive,
            "exclude_retweets": not args.include_retweets,
            "exclude_replies": not args.include_replies,
        }

    # Fetch
    if verbose:
        print(
            f"[scraper] Fetching up to {args.max_results} posts "
            f"for keywords: {args.keywords}"
        )
    records = scraper.fetch(
        keywords=args.keywords,
        max_results=args.max_results,
        **fetch_kwargs,
    )

    if not records:
        print("[scraper] No records returned. Nothing saved.")
        sys.exit(1)

    # Save via adapter
    adapter = ThetaAdapter(
        data_dir=Path(args.output_dir) if args.output_dir else None
    )
    output_path = adapter.save(records, dataset_name=args.dataset, verbose=verbose)

    if verbose:
        print(
            f"\n[scraper] Done.\n"
            f"  Dataset : {args.dataset}\n"
            f"  Records : {len(records)}\n"
            f"  Output  : {output_path}\n"
            f"\nNext step:\n"
            f"  python src/models/run_pipeline.py "
            f"--dataset {args.dataset} --models theta\n"
        )


if __name__ == "__main__":
    main()
