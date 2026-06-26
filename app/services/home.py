"""
app/services/home.py
--------------------
HomeService — assembles the three home-screen sections in a single call:
  • featured    — 5 diverse high-quality places (one per category)
  • hidden_gems — 6 hidden gems sampled from a quality pool
  • trending    — 8 most-reviewed places, shuffled for variety
"""
import random
import pandas as pd
from typing import Optional, Tuple

from app.core.config import settings
from app.data.repository import PlaceRepository


class HomeService:
    """
    Builds home-screen content from the places dataset.

    ``city`` scopes all three sections to a specific city.
    ``seed`` controls shuffling across all sections — pass a fresh seed
    for a new experience on each app open, keep it the same to get
    consistent results within a session.
    """

    def __init__(self, repo: PlaceRepository) -> None:
        self._repo = repo

    def get_home_sections(
        self,
        city: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return ``(featured, hidden_gems, trending)`` DataFrames.
        """
        city_filter = {"city_en": city} if city and str(city).strip() else None
        effective_seed = seed if seed is not None else random.randint(0, 99_999)
        rng = random.Random(effective_seed)

        base = self._repo.apply_filters(self._repo.get_all(), city_filter)

        featured = self._build_featured(base, rng, effective_seed)
        hidden_gems = self._build_hidden_gems(city_filter, effective_seed)
        trending = self._build_trending(base, effective_seed)

        return featured, hidden_gems, trending

    # ── Section builders ──────────────────────────────────────────────────
    def _build_featured(
        self,
        base: pd.DataFrame,
        rng: random.Random,
        seed: int,
    ) -> pd.DataFrame:
        """One place per category, randomised among the top-K best-rated."""
        top_k = settings.HOME_FEATURED_TOP_K
        min_reviews = settings.BAYESIAN_MIN_REVIEWS
        target = settings.HOME_FEATURED_COUNT

        featured_pool = base[base["reviews_count"] >= min_reviews] if not base.empty else base
        sorted_pool = featured_pool.sort_values(by="bayesian_rating", ascending=False)

        diverse_rows = []
        for _, group in sorted_pool.groupby("category", sort=False):
            candidates = group.head(top_k)
            diverse_rows.append(candidates.iloc[rng.randrange(len(candidates))])

        if diverse_rows:
            diverse = pd.DataFrame(diverse_rows).sort_values(by="bayesian_rating", ascending=False)
        else:
            diverse = sorted_pool.iloc[0:0]

        # Top-up if fewer categories exist than target count
        if len(diverse) < target:
            needed = target - len(diverse)
            used_ids = set(diverse["place_id"])
            topup_candidates = (
                sorted_pool[~sorted_pool["place_id"].isin(used_ids)]
                .head(max(needed * 3, needed))
            )
            if not topup_candidates.empty:
                topup = topup_candidates.sample(
                    n=min(needed, len(topup_candidates)),
                    random_state=seed,
                )
                diverse = pd.concat([diverse, topup], ignore_index=True)

        return diverse.head(target).reset_index(drop=True)

    def _build_hidden_gems(
        self,
        city_filter: Optional[dict],
        seed: int,
    ) -> pd.DataFrame:
        """Sample hidden gems from a top-rated quality pool."""
        gem_filters = {**(city_filter or {}), "is_hidden_gem": True}
        gem_base = self._repo.apply_filters(self._repo.get_all(), gem_filters)
        gem_pool = gem_base.sort_values(by="bayesian_rating", ascending=False).head(
            settings.HOME_HIDDEN_GEMS_POOL
        )
        if gem_pool.empty:
            return gem_pool
        return (
            gem_pool
            .sample(frac=1, random_state=seed)
            .head(settings.HOME_HIDDEN_GEMS_COUNT)
            .reset_index(drop=True)
        )

    def _build_trending(self, base: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Shuffle the top-N most-reviewed places."""
        if base.empty:
            return base
        pool = base.sort_values(by="reviews_count", ascending=False).head(
            settings.HOME_TRENDING_POOL
        )
        return (
            pool
            .sample(frac=1, random_state=seed)
            .head(settings.HOME_TRENDING_COUNT)
            .reset_index(drop=True)
        )
