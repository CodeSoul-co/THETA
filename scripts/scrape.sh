#!/usr/bin/env bash
# =============================================================================
# scrape.sh — Plug-and-play data scraper for THETA
#
# Fetches posts from a social media platform and saves them as a
# THETA-compatible CSV in DATA_DIR/{dataset}/{dataset}_cleaned.csv.
#
# Usage:
#   bash scripts/scrape.sh --platform x \
#       --keywords "AI policy" "climate change" \
#       --dataset my_research \
#       --max-results 1000
#
#   bash scripts/scrape.sh --list-platforms
#
# All arguments are forwarded directly to src/scrapers/cli.py.
# See that file (or run with --help) for the full option list.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate project root (the directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Activate conda environment if present (mirrors quick_start.sh behaviour)
# ---------------------------------------------------------------------------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-theta}"
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Run the Python CLI from the project root (ensures src.scrapers imports work)
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"
exec python src/scrapers/cli.py "$@"
