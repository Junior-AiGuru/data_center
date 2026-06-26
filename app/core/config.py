"""
app/core/config.py
------------------
Centralised configuration — all magic numbers and tuneable constants live here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # ── Data source ───────────────────────────────────────────────────────
    DATA_PATH: str = "places.json"

    # ── Vectoriser weights (must sum to 1.0) ─────────────────────────────
    WEIGHT_CATS: float = 0.75   # fine-grained interest TF-IDF weight
    WEIGHT_GEN: float = 0.25    # general category one-hot weight

    # ── Bayesian rating ───────────────────────────────────────────────────
    BAYESIAN_MIN_REVIEWS: int = 50   # min reviews to be a "trusted" place

    # ── Recommendation pool ───────────────────────────────────────────────
    DEFAULT_POOL_SIZE: int = 50      # candidates scored before pagination

    # ── Home screen ───────────────────────────────────────────────────────
    HOME_FEATURED_TOP_K: int = 3     # pick 1 randomly from top-K per category
    HOME_HIDDEN_GEMS_POOL: int = 15  # sample 6 from the top-N hidden gems
    HOME_TRENDING_POOL: int = 40     # most-reviewed pool for trending
    HOME_TRENDING_COUNT: int = 8     # how many trending places to return
    HOME_HIDDEN_GEMS_COUNT: int = 6
    HOME_FEATURED_COUNT: int = 5


# Module-level singleton — import this everywhere
settings = Settings()
