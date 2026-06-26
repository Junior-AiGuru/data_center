"""
app/data/repository.py
----------------------
PlaceRepository — single responsibility: all data-access and filtering
operations on the places DataFrame.

No business logic lives here. Services call this class to query data;
they never touch the DataFrame directly.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings


class PlaceRepository:
    """
    Provides typed data-access methods over the places DataFrame.

    Parameters
    ----------
    df:
        Clean DataFrame produced by ``DataLoader.load()``.
        A Bayesian rating column is computed once at construction time.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._compute_bayesian_rating()

    # ── Bayesian rating ───────────────────────────────────────────────────
    def _compute_bayesian_rating(self) -> None:
        """
        Bayesian average rating.

        Formula: (v*R + m*C) / (v+m)
          v = place review count, R = place rating
          C = global mean rating, m = minimum-reviews threshold
        """
        C = self._df["rating"].mean()
        m = settings.BAYESIAN_MIN_REVIEWS
        self._df["bayesian_rating"] = (
            (self._df["reviews_count"] * self._df["rating"] + m * C)
            / (self._df["reviews_count"] + m)
        )

    # ── Basic accessors ───────────────────────────────────────────────────
    def get_all(self) -> pd.DataFrame:
        """Return the full DataFrame (including internal columns)."""
        return self._df

    def get_by_id(self, place_id: str) -> Optional[dict]:
        """Return a single place dict or ``None`` if not found."""
        result = self._df[self._df["place_id"] == place_id]
        return None if result.empty else result.iloc[0].to_dict()

    # ── Filter engine ─────────────────────────────────────────────────────
    def apply_filters(
        self,
        df_in: pd.DataFrame,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Null-safe filter engine — empty / null values are silently skipped.

        A filter value is considered empty and ignored when it is:
          - ``None``
          - empty string ``""``
          - empty list ``[]``

        Supported filter shapes (for non-empty values):

        +--------------+---------------------------------------------------+
        | Shape        | Example                                           |
        +==============+===================================================+
        | scalar       | ``{"city_en": "Cairo"}``                          |
        +--------------+---------------------------------------------------+
        | list         | ``{"category": ["food_cafes", "shopping"]}``      |
        +--------------+---------------------------------------------------+
        | range        | ``{"rating": {"gte": 4.0, "lte": 5.0}}``         |
        +--------------+---------------------------------------------------+
        | list-in-list | ``{"interests": {"contains_any": ["Cafe"]}}``     |
        +--------------+---------------------------------------------------+
        """
        if not filters:
            return df_in

        result = df_in.copy()

        for col, condition in filters.items():
            if col not in result.columns:
                continue

            # Skip null / empty values
            if condition is None:
                continue
            if isinstance(condition, str) and condition.strip() == "":
                continue
            if isinstance(condition, list) and len(condition) == 0:
                continue

            if isinstance(condition, list):
                condition = [v for v in condition if v is not None and str(v).strip() != ""]
                if not condition:
                    continue
                if col == "interests":
                    result = result[
                        result[col].apply(
                            lambda x, cond=condition: any(v in x for v in cond)
                            if isinstance(x, list) else False
                        )
                    ]
                elif result[col].dtype == "object":
                    result = result[result[col].isin([str(v) for v in condition])]
                else:
                    result = result[result[col].isin(condition)]

            elif isinstance(condition, dict):
                if "gte" in condition and condition["gte"] is not None:
                    result = result[result[col] >= condition["gte"]]
                if "lte" in condition and condition["lte"] is not None:
                    result = result[result[col] <= condition["lte"]]
                if "gt"  in condition and condition["gt"]  is not None:
                    result = result[result[col] >  condition["gt"]]
                if "lt"  in condition and condition["lt"]  is not None:
                    result = result[result[col] <  condition["lt"]]
                if "contains" in condition:
                    vals = [v for v in condition["contains"] if v is not None and str(v).strip() != ""]
                    if vals:
                        result = result[result[col].apply(
                            lambda x, v=vals: all(i in x for i in v)
                            if isinstance(x, list) else False
                        )]
                if "contains_any" in condition:
                    vals = [v for v in condition["contains_any"] if v is not None and str(v).strip() != ""]
                    if vals:
                        result = result[result[col].apply(
                            lambda x, v=vals: any(i in x for i in v)
                            if isinstance(x, list) else False
                        )]

            else:
                # Scalar match
                if col == "is_hidden_gem":
                    result = result[result[col] == condition]
                elif result[col].dtype == "object":
                    result = result[result[col] == str(condition)]
                else:
                    result = result[result[col] == condition]

        return result

    # ── Derived query methods ─────────────────────────────────────────────
    def get_top_rated(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        limit: int = 10,
    ) -> Tuple[pd.DataFrame, int]:
        """Return places sorted by Bayesian rating, paginated."""
        result = self.apply_filters(self._df, filters)
        if result.empty:
            return pd.DataFrame(), 0
        sorted_df = result.sort_values(by="bayesian_rating", ascending=False)
        total = len(sorted_df)
        skip = (page - 1) * limit
        return sorted_df.iloc[skip: skip + limit].reset_index(drop=True), total
