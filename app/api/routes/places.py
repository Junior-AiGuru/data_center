"""
app/api/routes/places.py
------------------------
All /places/... route handlers.

Each handler is thin: validate → call service → serialise response.
No business logic lives here.
"""
import numpy as np
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    FilterOnlyRequest,
    GetPlacesRequest,
    HomeRequest,
    NearbyRequest,
    RecommendRequest,
    SearchRequest,
)
from app.data.repository import PlaceRepository
from app.dependencies import (
    get_home_service,
    get_location_service,
    get_recommendation_service,
    get_repository,
    get_search_service,
)
from app.services.home import HomeService
from app.services.location import LocationService
from app.services.recommendation import RecommendationService
from app.services.search import SearchService
from app.utils.serializers import paginated_response, safe_json

router = APIRouter(prefix="/places", tags=["places"])


# ── Home screen ────────────────────────────────────────────────────────────
@router.post("/home", summary="Home screen sections")
def get_home(
    payload: HomeRequest,
    service: HomeService = Depends(get_home_service),
):
    """
    Single call for the home screen.
    Returns: ``featured`` (5), ``hidden_gems`` (6), ``trending`` (8).
    Pass ``city`` to scope results to a specific city.
    """
    featured, hidden_gems, trending = service.get_home_sections(
        city=payload.city or None,
        seed=payload.seed,
    )
    return {
        "featured":    safe_json(featured),
        "hidden_gems": safe_json(hidden_gems),
        "trending":    safe_json(trending),
    }


# ── Recommendations ────────────────────────────────────────────────────────
@router.post("/recommend", summary="Content-based recommendations")
def get_recommendations(
    payload: RecommendRequest,
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Content-based recommendations from user category interests.

    - Pass ``seed`` to get consistent pagination across scroll pages in the same session.
    - Increment ``page`` on each scroll; keep ``seed`` the same for the session.
    - Change ``seed`` (or omit it) for a fresh shuffle on next app open.
    """
    results, total = service.recommend(
        categories=payload.selected_categories,
        filters=payload.filters,
        page=payload.page,
        limit=payload.limit,
        seed=payload.seed,
        pool_size=payload.pool_size,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ── Text search ────────────────────────────────────────────────────────────
@router.post("/search", summary="Full-text search")
def search_places(
    payload: SearchRequest,
    service: SearchService = Depends(get_search_service),
):
    """
    Full-text search across name, city, address, and categories.

    Examples::

        {"query": "Luxor"}
        {"query": "cafe", "filters": {"city_en": "Cairo"}}
        {"query": "", "filters": {"category": "food_cafes"}}
    """
    results, total = service.search(
        query=payload.query or "",
        filters=payload.filters,
        page=payload.page,
        limit=payload.limit,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ── Top rated ──────────────────────────────────────────────────────────────
@router.post("/top-rated", summary="Top rated places")
def get_top_rated(
    payload: FilterOnlyRequest,
    repo: PlaceRepository = Depends(get_repository),
):
    """Returns places sorted by Bayesian rating. Supports any filters."""
    results, total = repo.get_top_rated(
        filters=payload.filters,
        page=payload.page,
        limit=payload.limit,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ── Nearby ─────────────────────────────────────────────────────────────────
@router.post("/nearby", summary="Distance-based nearby search")
def get_nearby(
    payload: NearbyRequest,
    service: LocationService = Depends(get_location_service),
):
    """
    Returns places sorted by distance from the user's GPS location.

    Examples::

        {"user_lat": 30.0444, "user_lng": 31.2357}
        {"user_lat": 30.0444, "user_lng": 31.2357, "radius_km": 10}
        {"user_lat": 30.0444, "user_lng": 31.2357, "filters": {"is_hidden_gem": true}}
    """
    results, total = service.nearby(
        lat=payload.user_lat,
        lng=payload.user_lng,
        radius_km=payload.radius_km,
        filters=payload.filters,
        page=payload.page,
        limit=payload.limit,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ── GetPlaces (flexible filter endpoint) ───────────────────────────────────
@router.post("/getplaces", summary="Flexible places filter")
def get_places(
    payload: GetPlacesRequest,
    repo: PlaceRepository = Depends(get_repository),
):
    """
    General-purpose endpoint for fetching places with filters.
    All fields are optional — sending ``{}`` returns all places paginated by rating.

    Examples::

        {"city": ["Cairo"]}
        {"category": ["food_cafes"], "min_rating": 4}
        {"interests": ["Cafe", "Seafood"], "city": ["Alexandria"]}
        {"hidden_gem": true, "sort_by": "reviews", "order": "desc"}
    """
    filters: Dict[str, Any] = {}

    if payload.city:
        filters["city_en"] = payload.city if len(payload.city) > 1 else payload.city[0]
    if payload.category:
        filters["category"] = payload.category if len(payload.category) > 1 else payload.category[0]
    if payload.interests:
        filters["interests"] = {"contains_any": payload.interests}
    if payload.min_rating is not None or payload.max_rating is not None:
        rf: Dict[str, float] = {}
        if payload.min_rating is not None:
            rf["gte"] = payload.min_rating
        if payload.max_rating is not None:
            rf["lte"] = payload.max_rating
        filters["rating"] = rf
    if payload.min_price is not None or payload.max_price is not None:
        pf: Dict[str, int] = {}
        if payload.min_price is not None:
            pf["gte"] = payload.min_price
        if payload.max_price is not None:
            pf["lte"] = payload.max_price
        filters["price"] = pf
    if payload.hidden_gem is not None:
        filters["is_hidden_gem"] = payload.hidden_gem

    result = repo.apply_filters(repo.get_all(), filters or None)

    sort_map = {
        "rating":  "bayesian_rating",
        "reviews": "reviews_count",
        "price":   "price",
        "name":    "name",
    }
    sort_col = sort_map.get(payload.sort_by, "bayesian_rating")
    result = result.sort_values(by=sort_col, ascending=(payload.order == "asc"))

    total = len(result)
    skip = (payload.page - 1) * payload.limit
    page_data = result.iloc[skip: skip + payload.limit].reset_index(drop=True)

    return paginated_response(safe_json(page_data), total, payload.page, payload.limit)


# ── Single place ───────────────────────────────────────────────────────────
@router.get("/{place_id}", summary="Get place by ID")
def get_single_place(
    place_id: str,
    repo: PlaceRepository = Depends(get_repository),
):
    """Returns full details for one place by ``place_id``."""
    place = repo.get_by_id(place_id)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found")
    internal = {"bayesian_rating", "similarity_score"}
    return {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in place.items()
        if k not in internal
    }
