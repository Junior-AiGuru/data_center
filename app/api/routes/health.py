"""
app/api/routes/health.py
------------------------
Health-check routes.
"""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/", summary="Health check")
def root():
    """Liveness probe — returns 200 when the API is running."""
    return {"message": "MindTrip Recommendation API — Active!"}
