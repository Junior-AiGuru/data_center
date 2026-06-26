"""
app/api/schemas.py
------------------
All Pydantic request / response models for the places API.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ── Recommendation ─────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    selected_categories: List[str]
    filters:   Optional[Dict[str, Any]] = None
    page:      Optional[int]            = 1
    limit:     Optional[int]            = 10
    seed:      Optional[int]            = None
    pool_size: Optional[int]            = 50


# ── Nearby ─────────────────────────────────────────────────────────────────
class NearbyRequest(BaseModel):
    user_lat:  float                    = Field(..., ge=-90,  le=90)
    user_lng:  float                    = Field(..., ge=-180, le=180)
    radius_km: Optional[float]          = None
    filters:   Optional[Dict[str, Any]] = None
    page:      Optional[int]            = 1
    limit:     Optional[int]            = 10


# ── Search ─────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query:   Optional[str]            = None
    filters: Optional[Dict[str, Any]] = None
    page:    Optional[int]            = 1
    limit:   Optional[int]            = 10


# ── Generic filter-only ────────────────────────────────────────────────────
class FilterOnlyRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    page:    Optional[int]            = 1
    limit:   Optional[int]            = 10
    seed:    Optional[int]            = None


# ── Home screen ────────────────────────────────────────────────────────────
class HomeRequest(BaseModel):
    city: Optional[str] = None   # e.g. "Cairo" — scopes all home sections
    seed: Optional[int] = None


# ── GetPlaces (flexible filter endpoint) ───────────────────────────────────
class GetPlacesRequest(BaseModel):
    city:       Optional[List[str]] = None    # e.g. ["Cairo"] or ["Cairo", "Giza"]
    category:   Optional[List[str]] = None    # e.g. ["food_cafes"]
    interests:  Optional[List[str]] = None    # e.g. ["Cafe", "Seafood"] — matches any
    min_rating: Optional[float]     = None
    max_rating: Optional[float]     = None
    min_price:  Optional[int]       = None
    max_price:  Optional[int]       = None
    hidden_gem: Optional[bool]      = None
    sort_by:    Optional[str]       = "rating"   # rating | reviews | price | name
    order:      Optional[str]       = "desc"     # asc | desc
    page:       Optional[int]       = 1
    limit:      Optional[int]       = 10
