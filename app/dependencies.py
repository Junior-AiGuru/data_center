"""
app/dependencies.py
-------------------
FastAPI dependency-injection factories.

All services are built once at startup from module-level singletons,
then re-used on every request via FastAPI's ``Depends()`` mechanism.
This avoids re-loading the 1.2 MB dataset on each call.
"""
from app.core.config import settings
from app.data.loader import DataLoader
from app.data.vectorizer import PlaceVectorizer
from app.data.repository import PlaceRepository
from app.services.recommendation import RecommendationService
from app.services.search import SearchService
from app.services.location import LocationService
from app.services.home import HomeService

# ── Startup singletons (created once) ────────────────────────────────────
_df         = DataLoader(settings.DATA_PATH).load()
_vectorizer = PlaceVectorizer(_df)
_repo       = PlaceRepository(_df)

# ── Dependency factory functions (injected into route handlers) ───────────
def get_repository() -> PlaceRepository:
    return _repo


def get_vectorizer() -> PlaceVectorizer:
    return _vectorizer


def get_recommendation_service() -> RecommendationService:
    return RecommendationService(repo=_repo, vectorizer=_vectorizer)


def get_search_service() -> SearchService:
    return SearchService(repo=_repo)


def get_location_service() -> LocationService:
    return LocationService(repo=_repo)


def get_home_service() -> HomeService:
    return HomeService(repo=_repo)
