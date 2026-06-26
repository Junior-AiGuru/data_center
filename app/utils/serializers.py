"""
app/utils/serializers.py
------------------------
Shared serialisation helpers used by all route handlers.
"""
import numpy as np
import pandas as pd
from typing import Any


# Columns computed internally that must never be exposed in API responses
_INTERNAL_COLS = {"bayesian_rating", "similarity_score", "distance_km"}


def safe_json(df: pd.DataFrame) -> list:
    """
    Convert a DataFrame to ``list[dict]``, replacing NaN with ``None``
    and stripping internal-only computed columns.
    """
    if df is None or df.empty:
        return []
    drop_cols = [c for c in _INTERNAL_COLS if c in df.columns]
    clean = df.drop(columns=drop_cols).replace({np.nan: None})
    return clean.to_dict(orient="records")


def paginated_response(data: list, total: int, page: int, limit: int) -> dict:
    """Wrap a results list in a standard paginated envelope."""
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total + limit - 1) // limit),
        "results": data,
    }
