import json
import os
import random
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# DATA ENGINE CLASS
# =====================================================
class DataEngine:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # Support both JSON array and JSONL (one object per line)
        if raw.startswith("["):
            data = json.loads(raw)
        else:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]

        self.df = pd.DataFrame(data)

        if "place_id" not in self.df.columns:
            self.df["place_id"] = self.df.index.astype(str)

        # ── TF-IDF category vectors ──────────────────────────────────────
        self.mlb = MultiLabelBinarizer()
        self.cat_matrix = self.mlb.fit_transform(self.df["interests"])

        N = len(self.df)
        doc_freq = self.cat_matrix.sum(axis=0)
        self.idf = np.log((N + 1) / (doc_freq + 1)) + 1
        self.cat_matrix_weighted = self.cat_matrix * self.idf

        # ── General-category one-hot vectors ────────────────────────────
        self.gen_dummies = pd.get_dummies(self.df["category"])
        self.gen_cols = self.gen_dummies.columns.tolist()
        self.gen_matrix = self.gen_dummies.values.astype(float)

        self.WEIGHT_CATS = 0.75
        self.WEIGHT_GEN  = 0.25

        self.place_vectors = np.hstack([
            self.WEIGHT_CATS * self._l2_norm(self.cat_matrix_weighted),
            self.WEIGHT_GEN  * self._l2_norm(self.gen_matrix),
        ])

        # ── Bayesian average rating ──────────────────────────────────────
        # Prevents places with 1-2 reviews from topping the "top-rated" list.
        # Formula: (v*R + m*C) / (v+m)  where v=reviews, R=place_rating,
        # C=global mean rating, m=minimum reviews threshold.
        C = self.df["rating"].mean()       # global mean ≈ 4.31
        m = 50                             # minimum reviews to be "trusted"
        self.df["bayesian_rating"] = (
            (self.df["reviews_count"] * self.df["rating"] + m * C)
            / (self.df["reviews_count"] + m)
        )

        # Map from general → fine-grained categories (used for user-vector building)
        # Maps new category slugs → fine-grained interest values
        # used to build the general-category component of the user vector
        self.GENERAL_CATEGORY_MAP = {
            "food_cafes":       {"Restaurants", "Cafe", "Bakery", "Seafood", "Street Food"},
            "shopping":         {"Shopping"},
            "arts_culture":     {"Arts & Crafts"},
            "entertainment":    {"Entertainment", "Nightlife", "Music"},
            "nature":           {"Nature", "Park"},
            "beaches":          {"Beaches & Water", "Waterfront"},
            "historical_sites": {"History & Antiquities", "Tourism"},
            "religious_sites":  {"Mosques & Churches"},
        }

    # ── Helpers ───────────────────────────────────────────────────────────
    def _l2_norm(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-9)

    # ── Filter engine ─────────────────────────────────────────────────────
    def apply_filters(self, df_in, filters=None):
        """
        Null-safe filter engine — empty/null values are silently skipped.

        A filter value is considered empty and ignored when it is:
          - None
          - empty string ""
          - empty list []

        Supported filter shapes (for non-empty values):
          - scalar       : {"city_en": "Cairo"}            — case-sensitive exact match
          - list         : {"category": ["food_cafes"]}    — case-sensitive, match any
          - range        : {"rating": {"gte": 4.0}}        — numeric comparison
          - list-in-list : {"interests": {"contains_any": ["Cafe"]}}
        """
        if not filters:
            return df_in

        result = df_in.copy()

        for col, condition in filters.items():
            if col not in result.columns:
                continue

            # ── Skip null / empty values ──────────────────────────────────
            if condition is None:
                continue
            if isinstance(condition, str) and condition.strip() == "":
                continue
            if isinstance(condition, list) and len(condition) == 0:
                continue

            # ── List filter ───────────────────────────────────────────────
            if isinstance(condition, list):
                # Remove any null/empty-string items from the list
                condition = [v for v in condition if v is not None and str(v).strip() != ""]
                if not condition:
                    continue

                if col == "interests":
                    result = result[
                        result[col].apply(
                            lambda x: any(v in x for v in condition)
                            if isinstance(x, list) else False
                        )
                    ]
                else:
                    if result[col].dtype == "object":
                        result = result[result[col].isin([str(v) for v in condition])]
                    else:
                        result = result[result[col].isin(condition)]

            # ── Range / contains filter ───────────────────────────────────
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
                            lambda x: all(v in x for v in vals)
                            if isinstance(x, list) else False
                        )]
                if "contains_any" in condition:
                    vals = [v for v in condition["contains_any"] if v is not None and str(v).strip() != ""]
                    if vals:
                        result = result[result[col].apply(
                            lambda x: any(v in x for v in vals)
                            if isinstance(x, list) else False
                        )]

            # ── Scalar filter ─────────────────────────────────────────────
            else:
                if col == "is_hidden_gem":
                    result = result[result[col] == condition]
                elif result[col].dtype == "object":
                    result = result[result[col] == str(condition)]
                else:
                    result = result[result[col] == condition]

        return result

    # ── Text search ───────────────────────────────────────────────────────
    def text_search(self, query: str, page: int = 1, limit: int = 10, filters=None):
        """
        Full-text search across name, city_en, city, address, and categories.
        Supports multi-word queries: all tokens must match (AND logic).
        After text matching, optional filters are applied, then paginated.
        """
        result = self.df.copy()

        if query and query.strip():
            tokens = query.lower().strip().split()

            def row_matches(row):
                # Build a searchable text blob for each row
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

            mask = result.apply(row_matches, axis=1)
            result = result[mask]

        result = self.apply_filters(result, filters)

        total = len(result)
        result = result.reset_index(drop=True)
        skip  = (page - 1) * limit
        page_data = result.iloc[skip: skip + limit].reset_index(drop=True)

        return page_data, total

    # ── Top rated ─────────────────────────────────────────────────────────
    def top_rated(self, page: int = 1, limit: int = 10, filters=None):
        result = self.apply_filters(self.df, filters)
        if result.empty:
            return pd.DataFrame(), 0

        sorted_df = result.sort_values(
            by="bayesian_rating", ascending=False
        )
        total = len(sorted_df)
        skip  = (page - 1) * limit
        return sorted_df.iloc[skip: skip + limit].reset_index(drop=True), total

    # ── Single place ──────────────────────────────────────────────────────
    def get_place(self, place_id: str):
        result = self.df[self.df["place_id"] == place_id]
        if result.empty:
            return None
        return result.iloc[0].to_dict()

    # ── Recommendation engine ─────────────────────────────────────────────
    def recommend(
        self,
        selected_categories: List[str],
        page: int = 1,
        limit: int = 10,
        filters=None,
        seed: Optional[int] = None,
        pool_size: int = 50,
    ):
        """
        Content-based recommendation using TF-IDF cosine similarity.

        Pooling & pagination strategy:
          1. Score all places against the user vector.
          2. Apply filters.
          3. Take the top `pool_size` candidates by score.
          4. Shuffle the pool with `seed` (session-stable).
          5. Paginate: page N returns items [(N-1)*limit : N*limit] from the pool.

        This means:
          - Same seed → same pool order → consistent pagination within a session.
          - Different seed (new session) → different shuffle → fresh experience.
          - Pagination works up to pool_size / limit pages before the pool is exhausted.
        """
        # Fall back to top-rated when no interests given
        if not selected_categories:
            results, total = self.top_rated(page=page, limit=limit, filters=filters)
            return results, total

        # Build user category vector (TF-IDF weighted)
        selected_set  = set(selected_categories)
        user_cat_vec  = np.zeros(len(self.mlb.classes_))
        for i, cls in enumerate(self.mlb.classes_):
            if cls in selected_set:
                user_cat_vec[i] = self.idf[i]

        # Build user general-category vector
        gen_counts = {g: 0.0 for g in self.gen_cols}
        for gen, fine_cats in self.GENERAL_CATEGORY_MAP.items():
            if gen in gen_counts:
                gen_counts[gen] = len(selected_set & fine_cats)
        user_gen_vec = np.array([gen_counts.get(g, 0.0) for g in self.gen_cols])

        user_vector = np.hstack([
            self.WEIGHT_CATS * self._l2_norm(user_cat_vec.reshape(1, -1)),
            self.WEIGHT_GEN  * self._l2_norm(user_gen_vec.reshape(1, -1)),
        ])

        # Score every place
        scores    = cosine_similarity(user_vector, self.place_vectors).flatten()
        scored_df = self.df.copy()
        scored_df["similarity_score"] = scores

        # Apply filters AFTER scoring (filters reduce the pool, not the scoring)
        filtered_df = self.apply_filters(scored_df, filters)
        if filtered_df.empty:
            return pd.DataFrame(), 0

        # Top-N pool (capped at what's available after filtering)
        actual_pool = min(pool_size, len(filtered_df))
        pool = (
            filtered_df
            .sort_values("similarity_score", ascending=False)
            .head(actual_pool)
        )

        # Shuffle pool with session seed (stable across pages in same session)
        effective_seed = seed if seed is not None else random.randint(0, 99999)
        shuffled = pool.sample(frac=1, random_state=effective_seed).reset_index(drop=True)

        total = len(shuffled)
        skip  = (page - 1) * limit
        page_data = shuffled.iloc[skip: skip + limit].reset_index(drop=True)
        page_data = page_data.drop(columns=["similarity_score"], errors="ignore")
        return page_data, total

    # ── Home screen data ──────────────────────────────────────────────────
    def home(self, city: Optional[str] = None, seed: Optional[int] = None):
        """
        Returns three sections for the home screen in a single call:
          - featured    : 5 diverse places (one per category, topped up if needed)
          - hidden_gems : 6 hidden gems, sampled from a top-rated pool
          - trending    : 8 places shuffled from top-40 most-reviewed

        `city` scopes all sections to that city when provided.
        `seed` controls the shuffling of ALL three sections — pass a new
        random seed (or omit it) to get a fresh mix on each call.
        """
        city_filter     = {"city_en": city} if city and str(city).strip() else None
        effective_seed  = seed if seed is not None else random.randint(0, 99999)
        rng             = random.Random(effective_seed)

        # How many top candidates to randomize within (keeps quality high
        # while still giving variety between different seeds)
        FEATURED_TOP_K     = 3   # pick randomly among the top-3 per category
        HIDDEN_GEMS_POOL   = 15  # sample 6 out of the top-15 hidden gems

        base = self.apply_filters(self.df, city_filter)

        # Featured: one pick per category (randomized among the
        # top-K best-rated places in that category) → diverse home screen
        # (min 50 reviews to avoid 5-star places with 2 reviews)
        featured_pool = base[base["reviews_count"] >= 50] if not base.empty else base
        featured_pool_sorted = featured_pool.sort_values(by="bayesian_rating", ascending=False)

        diverse_rows = []
        for _, group in featured_pool_sorted.groupby("category", sort=False):
            candidates = group.head(FEATURED_TOP_K)
            diverse_rows.append(candidates.iloc[rng.randrange(len(candidates))])

        if diverse_rows:
            diverse = pd.DataFrame(diverse_rows).sort_values(by="bayesian_rating", ascending=False)
        else:
            diverse = featured_pool_sorted.iloc[0:0]  # empty df, same columns

        # If fewer than 5 categories exist (small city), top-up with next best
        # places not already selected — accept category repeats only as last resort
        if len(diverse) < 5:
            needed   = 5 - len(diverse)
            used_ids = set(diverse["place_id"])
            topup_candidates = (
                featured_pool_sorted[~featured_pool_sorted["place_id"].isin(used_ids)]
                .head(needed * 3 if needed * 3 > 0 else needed)
            )
            if not topup_candidates.empty:
                topup = topup_candidates.sample(
                    n=min(needed, len(topup_candidates)),
                    random_state=effective_seed,
                )
            else:
                topup = topup_candidates
            diverse = pd.concat([diverse, topup], ignore_index=True)
        featured = diverse.head(5).reset_index(drop=True)

        # Hidden gems: is_hidden_gem == True — sample 6 out of the top-rated pool
        # so the set changes with the seed but stays high quality
        gem_base = self.apply_filters(self.df, {**(city_filter or {}), "is_hidden_gem": True})  # exact bool match
        gem_pool = gem_base.sort_values(by="bayesian_rating", ascending=False).head(HIDDEN_GEMS_POOL)
        hidden_gems = (
            gem_pool
            .sample(frac=1, random_state=effective_seed)
            .head(6)
            .reset_index(drop=True)
        )

        # Trending: pool of top-40 most-reviewed, shuffled by seed → changes on refresh
        trending_pool = (
            base
            .sort_values(by="reviews_count", ascending=False)
            .head(40)
        )
        trending = (
            trending_pool
            .sample(frac=1, random_state=effective_seed)
            .head(8)
            .reset_index(drop=True)
        )

        return featured, hidden_gems, trending


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="MindTrip Core Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DataEngine(json_path="places.json")


# ── Request models ────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    selected_categories: List[str]
    filters:   Optional[Dict[str, Any]] = None
    page:      Optional[int]            = 1
    limit:     Optional[int]            = 10
    seed:      Optional[int]            = None
    pool_size: Optional[int]            = 50


class SearchRequest(BaseModel):
    query:   Optional[str]            = None     # text to search; None or "" → no text filter
    filters: Optional[Dict[str, Any]] = None     # additional field filters
    page:    Optional[int]            = 1
    limit:   Optional[int]            = 10


class FilterOnlyRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    page:    Optional[int]            = 1
    limit:   Optional[int]            = 10
    seed:    Optional[int]            = None


class HomeRequest(BaseModel):
    city: Optional[str] = None    # e.g. "Cairo" — scopes all home sections
    seed: Optional[int] = None


class GetPlacesRequest(BaseModel):
    city:       Optional[List[str]]  = None     # e.g. ["Cairo"] or ["Cairo", "Giza"]
    category:   Optional[List[str]]  = None     # e.g. ["food_cafes"] or ["food_cafes", "entertainment"]
    interests:  Optional[List[str]]  = None     # e.g. ["Cafe", "Seafood"] — matches any
    min_rating: Optional[float]      = None     # e.g. 4.0
    max_rating: Optional[float]      = None     # e.g. 5.0
    min_price:  Optional[int]        = None     # e.g. 0
    max_price:  Optional[int]        = None     # e.g. 200
    hidden_gem: Optional[bool]       = None     # true / false
    sort_by:    Optional[str]        = "rating" # "rating" | "reviews" | "price" | "name"
    order:      Optional[str]        = "desc"   # "asc" | "desc"
    page:       Optional[int]        = 1
    limit:      Optional[int]        = 10


# ── Helpers ───────────────────────────────────────────────────────────────
def safe_json(df: pd.DataFrame):
    """Convert DataFrame to list[dict], replacing NaN with None.
    Internal-only columns (bayesian_rating, similarity_score) are stripped."""
    if df is None or df.empty:
        return []
    drop_cols = [c for c in ["bayesian_rating", "similarity_score"] if c in df.columns]
    clean = df.drop(columns=drop_cols).replace({np.nan: None})
    return clean.to_dict(orient="records")


def paginated_response(data: list, total: int, page: int, limit: int):
    return {
        "total":       total,
        "page":        page,
        "limit":       limit,
        "total_pages": max(1, (total + limit - 1) // limit),
        "results":     data,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "MindTrip Recommendation API — Active!"}


# ---------- Home screen ----------
@app.post("/places/home")
def get_home(payload: HomeRequest):
    """
    Single call for the home screen.
    Returns: featured (5), hidden_gems (6), trending (8).
    Pass `city` to scope results to a specific city.
    """
    featured, hidden_gems, trending = engine.home(
        city=payload.city or None,
        seed=payload.seed,
    )
    return {
        "featured":    safe_json(featured),
        "hidden_gems": safe_json(hidden_gems),
        "trending":    safe_json(trending),
    }


# ---------- Recommendations ----------
@app.post("/places/recommend")
def get_recommendations(payload: RecommendRequest):
    """
    Content-based recommendations from user category interests.
    - Pass `seed` to get consistent pagination across scroll pages in the same session.
    - Increment `page` on each scroll; keep `seed` the same for the session.
    - Change `seed` (or omit it) for a fresh shuffle on next app open.
    """
    results, total = engine.recommend(
        selected_categories=payload.selected_categories,
        filters=payload.filters,
        page=payload.page,
        limit=payload.limit,
        seed=payload.seed,
        pool_size=payload.pool_size,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ---------- Text search ----------
@app.post("/places/search")
def search_places(payload: SearchRequest):
    """
    Full-text search across name, city, address, and categories.
    Also accepts `filters` for additional field-level filtering after text match.

    Examples:
      {"query": "Luxor"}                        → all places in Luxor
      {"query": "Khan el-Khalili"}              → specific place by name
      {"query": "cafe", "filters": {"city_en": "Cairo"}}  → Cairo cafes
      {"query": "", "filters": {"category": "food_cafes"}}  → all food_cafes places
    """
    results, total = engine.text_search(
        query=payload.query or "",
        page=payload.page,
        limit=payload.limit,
        filters=payload.filters,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ---------- Top rated ----------
@app.post("/places/top-rated")
def get_top_rated(payload: FilterOnlyRequest):
    """
    Returns places sorted by rating then reviews_count.
    Supports any filters (e.g. city_en, category).
    """
    results, total = engine.top_rated(
        page=payload.page,
        limit=payload.limit,
        filters=payload.filters,
    )
    return paginated_response(safe_json(results), total, payload.page, payload.limit)


# ---------- Get places (filter via POST body) ----------
@app.post("/places/getplaces")
def get_places(payload: GetPlacesRequest):
    """
    General-purpose endpoint for fetching places with filters.

    All fields are optional — sending an empty `{}` body returns all places
    (paginated, sorted by rating desc).

    **Request body examples:**
      {"city": ["Cairo"]}
      {"category": ["food_cafes"], "min_rating": 4}
      {"category": ["food_cafes", "entertainment"]}
      {"interests": ["Cafe", "Seafood"], "city": ["Alexandria"]}
      {"hidden_gem": true, "sort_by": "reviews", "order": "desc"}
      {"min_price": 0, "max_price": 100, "page": 2, "limit": 20}
    """
    # Build the filters dict expected by DataEngine.apply_filters
    filters: Dict[str, Any] = {}

    if payload.city:
        filters["city_en"] = payload.city if len(payload.city) > 1 else payload.city[0]

    if payload.category:
        filters["category"] = payload.category if len(payload.category) > 1 else payload.category[0]

    if payload.interests:
        filters["interests"] = {"contains_any": payload.interests}

    if payload.min_rating is not None or payload.max_rating is not None:
        rating_filter: Dict[str, float] = {}
        if payload.min_rating is not None:
            rating_filter["gte"] = payload.min_rating
        if payload.max_rating is not None:
            rating_filter["lte"] = payload.max_rating
        filters["rating"] = rating_filter

    if payload.min_price is not None or payload.max_price is not None:
        price_filter: Dict[str, int] = {}
        if payload.min_price is not None:
            price_filter["gte"] = payload.min_price
        if payload.max_price is not None:
            price_filter["lte"] = payload.max_price
        filters["price"] = price_filter

    if payload.hidden_gem is not None:
        filters["is_hidden_gem"] = payload.hidden_gem

    # Apply filters
    result = engine.apply_filters(engine.df, filters if filters else None)

    # Sort
    sort_map = {
        "rating":  "bayesian_rating",
        "reviews": "reviews_count",
        "price":   "price",
        "name":    "name",
    }
    sort_col = sort_map.get(payload.sort_by, "bayesian_rating")
    ascending = payload.order == "asc"
    result = result.sort_values(by=sort_col, ascending=ascending)

    # Paginate
    total = len(result)
    skip  = (payload.page - 1) * payload.limit
    page_data = result.iloc[skip: skip + payload.limit].reset_index(drop=True)

    return paginated_response(safe_json(page_data), total, payload.page, payload.limit)


# ---------- Single place ----------
@app.get("/places/{place_id}")
def get_single_place(place_id: str):
    """Returns full details for one place by place_id."""
    place = engine.get_place(place_id)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found")
    internal = {"bayesian_rating", "similarity_score"}
    return {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in place.items()
        if k not in internal
    }
