"""
app/services/location.py
------------------------
LocationService — geo-spatial operations: haversine distance & nearby search.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from app.data.repository import PlaceRepository


class LocationService:
    """
    Returns places sorted by distance from a user's GPS coordinates.

    Optionally scoped to a radius (``radius_km``) and any standard filters.
    """

    def __init__(self, repo: PlaceRepository) -> None:
        self._repo = repo

    def nearby(
        self,
        lat: float,
        lng: float,
        radius_km: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        limit: int = 10,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Return ``(page_data, total)`` sorted nearest-first.

        Parameters
        ----------
        lat, lng:
            User's GPS coordinates.
        radius_km:
            If provided, only places within this radius are included.
        filters:
            Standard filter dict (same shape as ``PlaceRepository.apply_filters``).
        """
        # Only places with valid coordinates can be distance-ranked
        base = self._repo.get_all()
        base = base[base["lat"].notna() & base["lng"].notna()]

        result = self._repo.apply_filters(base, filters)
        if result.empty:
            return pd.DataFrame(), 0

        result = result.copy()
        result["distance_km"] = self._haversine_km(
            lat, lng, result["lat"].values, result["lng"].values
        )

        if radius_km is not None:
            result = result[result["distance_km"] <= radius_km]
            if result.empty:
                return pd.DataFrame(), 0

        sorted_df = result.sort_values(by="distance_km", ascending=True)
        total = len(sorted_df)
        skip = (page - 1) * limit
        return sorted_df.iloc[skip: skip + limit].reset_index(drop=True), total

    # ── Geo helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _haversine_km(
        lat1: float,
        lng1: float,
        lat2: np.ndarray,
        lng2: np.ndarray,
    ) -> np.ndarray:
        """Vectorised great-circle distance in km between one point and an array."""
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))
