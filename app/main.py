"""
app/main.py
-----------
FastAPI application factory.

Creates the app, registers middleware, and mounts all routers.
This is the single entry point for uvicorn:

    uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, places

# ── App factory ───────────────────────────────────────────────────────────
app = FastAPI(
    title="MindTrip Core Recommendation API",
    description=(
        "Content-based place recommendation engine for the MindTrip tourism app. "
        "Provides recommendations, search, nearby, home-screen, and filter endpoints."
    ),
    version="2.0.0",
)

# ── Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(places.router)
