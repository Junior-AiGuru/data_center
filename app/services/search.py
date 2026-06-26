"""
app/services/search.py
----------------------
SearchService — full-text search across place fields.
"""
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from app.data.repository import PlaceRepository


class SearchService:
    """
    Full-text search across name, city_en, city, address, and interests.

    All tokens from the query must appear in the text blob (AND logic).
    After text matching, optional ``filters`` are applied via the repository.
    """

    def __init__(self, repo: PlaceRepository) -> None:
        self._repo = repo

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        limit: int = 10,
    ) -> Tuple[pd.DataFrame, int]:
        """Return ``(page_data, total)`` matching the query + filters."""
        result = self._repo.get_all().copy()

        if query and query.strip():
            tokens = query.lower().strip().split()
            mask = result.apply(lambda row: self._row_matches(row, tokens), axis=1)
            result = result[mask]

        result = self._repo.apply_filters(result, filters)

        total = len(result)
        result = result.reset_index(drop=True)
        skip = (page - 1) * limit
        return result.iloc[skip: skip + limit].reset_index(drop=True), total

    # ── Private helpers ───────────────────────────────────────────────────
    @staticmethod
    def _row_matches(row: pd.Series, tokens: List[str]) -> bool:
        """Return True when every token is found in the row's text blob."""
        cats_text = " ".join(row["interests"]) if isinstance(row["interests"], list) else ""
        blob = " ".join([
            str(row.get("name", "")),
            str(row.get("city_en", "")),
            str(row.get("city", "")),
            str(row.get("address", "")),
            str(row.get("category", "")),
            cats_text,
        ]).lower()
        return all(token in blob for token in tokens)
