"""
Platform registry for the scraper module.

Adding a new platform
---------------------
1. Create src/scrapers/platforms/<name>.py with a class that inherits AbstractScraper.
2. Add an entry to SCRAPER_REGISTRY below.
3. Document required_env so users know what to put in .env.
"""

import importlib

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCRAPER_REGISTRY: dict[str, dict] = {
    "x": {
        "module": "src.scrapers.platforms.x",
        "class": "XScraper",
        "required_env": ["X_BEARER_TOKEN"],
        "description": "X (Twitter) via API v2 — requires a developer bearer token",
    },
    # Future platforms:
    # "weibo": {
    #     "module": "src.scrapers.platforms.weibo",
    #     "class": "WeiboScraper",
    #     "required_env": ["WEIBO_APP_KEY", "WEIBO_APP_SECRET"],
    #     "description": "Weibo open platform API",
    # },
    # "reddit": {
    #     "module": "src.scrapers.platforms.reddit",
    #     "class": "RedditScraper",
    #     "required_env": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
    #     "description": "Reddit via PRAW",
    # },
}


def get_scraper(platform: str, **init_kwargs):
    """
    Instantiate and return the scraper for the given platform name.

    Parameters
    ----------
    platform : str
        Key in SCRAPER_REGISTRY (e.g. "x").
    **init_kwargs
        Passed directly to the scraper constructor.

    Raises
    ------
    ValueError
        If the platform is not registered.
    ImportError
        If the platform module or class cannot be imported.
    """
    platform = platform.lower()
    if platform not in SCRAPER_REGISTRY:
        available = ", ".join(SCRAPER_REGISTRY.keys())
        raise ValueError(
            f"Unknown platform '{platform}'. Available platforms: {available}"
        )

    entry = SCRAPER_REGISTRY[platform]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])
    return cls(**init_kwargs)


def list_platforms() -> None:
    """Print a formatted list of all registered platforms."""
    print("\nRegistered scraper platforms:")
    print(f"  {'Platform':<12} {'Required env vars':<30} Description")
    print(f"  {'-'*12} {'-'*30} {'-'*40}")
    for name, meta in SCRAPER_REGISTRY.items():
        env_str = ", ".join(meta["required_env"])
        print(f"  {name:<12} {env_str:<30} {meta['description']}")
    print()
