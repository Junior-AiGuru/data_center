"""
app/services/recommendation.py
-------------------------------
RecommendationService — content-based ML recommendation orchestration.

Depends on PlaceRepository (data access) and PlaceVectorizer (ML vectors).
"""
import random
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity

from app.data.repository import PlaceRepository
from app.data.vectorizer import PlaceVectorizer


class RecommendationService:
    """
    Orchestrates TF-IDF cosine-similarity recommendations.

    Pooling & pagination strategy
    ------------------------------
    1. Score **all** places against the user vector.
    2. Apply optional filters (reduces pool, not scoring).
    3. Take the top ``pool_size`` candidates by score.
    4. Shuffle the pool with ``seed`` (session-stable).
    5. Paginate: page N → items ``[(N-1)*limit : N*limit]`` from the pool.

    Same seed → same pool order → consistent pagination within a session.
    Different seed (new session) → different shuffle → fresh experience.
    """

    def __init__(self, repo: PlaceRepository, vectorizer: PlaceVectorizer) -> None:
        self._repo = repo
        self._vectorizer = vectorizer

    def recommend(
        self,
        categories: List[str],
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        limit: int = 10,
        seed: Optional[int] = None,
        pool_size: int = 50,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Return ``(page_data, total)`` for the given user category interests.

        Falls back to top-rated when ``categories`` is empty.
        """
        if not categories:
            return self._repo.get_top_rated(filters=filters, page=page, limit=limit)

        # Build user vector and score every place
        user_vector = self._vectorizer.build_user_vector(categories)
        scores = cosine_similarity(user_vector, self._vectorizer.place_vectors).flatten()

        scored_df = self._repo.get_all().copy()
        scored_df["similarity_score"] = scores

        # Filter AFTER scoring (filters reduce the pool, not the scoring)
        filtered_df = self._repo.apply_filters(scored_df, filters)
        if filtered_df.empty:
            return pd.DataFrame(), 0

        # Top-N pool, shuffled for variety
        actual_pool = min(pool_size, len(filtered_df))
        pool = (
            filtered_df
            .sort_values("similarity_score", ascending=False)
            .head(actual_pool)
        )
        effective_seed = seed if seed is not None else random.randint(0, 99_999)
        shuffled = pool.sample(frac=1, random_state=effective_seed).reset_index(drop=True)

        total = len(shuffled)
        skip = (page - 1) * limit
        page_data = shuffled.iloc[skip: skip + limit].reset_index(drop=True)
        page_data = page_data.drop(columns=["similarity_score"], errors="ignore")
        return page_data, total
