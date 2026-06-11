import json
import os
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
            data = json.load(f)
        self.df = pd.DataFrame(data)
        
        if "place_id" not in self.df.columns:
            self.df["place_id"] = self.df.index.astype(str)

        self.mlb = MultiLabelBinarizer()
        self.cat_matrix = self.mlb.fit_transform(self.df["categories"])

        N = len(self.df)
        doc_freq = self.cat_matrix.sum(axis=0)
        self.idf = np.log((N + 1) / (doc_freq + 1)) + 1
        self.cat_matrix_weighted = self.cat_matrix * self.idf

        self.gen_dummies = pd.get_dummies(self.df["general_category"])
        self.gen_cols = self.gen_dummies.columns.tolist()
        self.gen_matrix = self.gen_dummies.values.astype(float)

        self.WEIGHT_CATS = 0.75
        self.WEIGHT_GEN = 0.25

        self.place_vectors = np.hstack([
            self.WEIGHT_CATS * self._l2_norm(self.cat_matrix_weighted),
            self.WEIGHT_GEN * self._l2_norm(self.gen_matrix),
        ])

        self.GENERAL_CATEGORY_MAP = {
            "Food": {"Restaurants", "Cafe", "Bakery", "Seafood", "Street Food"},
            "Shopping": {"Shopping", "Arts & Crafts"},
            "Activities": {"Entertainment", "Nightlife", "Music", "Outdoor", "Park"},
            "Beaches": {"Beaches & Water", "Waterfront", "Nature"},
            "Culture": {"History & Antiquities", "Mosques & Churches", "Tourism"},
        }

    def _l2_norm(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-9)

    def apply_filters(self, df_to_filter, filters=None):
        if not filters:
            return df_to_filter

        result = df_to_filter.copy()

        for col, condition in filters.items():
            if col not in result.columns:
                continue

            # LIST -> IN condition
            if isinstance(condition, list):
                if col == "categories":
                    condition_lower = [v.lower() for v in condition]
                    result = result[
                        result[col].apply(lambda x: any(v.lower() in [c.lower() for c in x] for v in condition_lower) if isinstance(x, list) else False)
                    ]
                else:
                    if result[col].dtype == 'object':
                        condition_lower = [str(v).lower() for v in condition]
                        result = result[result[col].astype(str).str.lower().isin(condition_lower)]
                    else:
                        result = result[result[col].isin(condition)]

            # DICT -> operators
            elif isinstance(condition, dict):
                if "gte" in condition: result = result[result[col] >= condition["gte"]]
                if "lte" in condition: result = result[result[col] <= condition["lte"]]
                if "gt" in condition: result = result[result[col] > condition["gt"]]
                if "lt" in condition: result = result[result[col] < condition["lt"]]
                if "contains" in condition:
                    values = [v.lower() for v in condition["contains"]]
                    result = result[result[col].apply(lambda x: all(v in [c.lower() for c in x] for v in values) if isinstance(x, list) else False)]
                if "contains_any" in condition:
                    values = [v.lower() for v in condition["contains_any"]]
                    result = result[result[col].apply(lambda x: any(v in [c.lower() for c in x] for v in values) if isinstance(x, list) else False)]

            # EQUALITY
            else:
                if isinstance(condition, str) and result[col].dtype == 'object':
                    result = result[result[col].astype(str).str.lower() == condition.lower()]
                else:
                    if col == "is_hidden_gem" and result[col].dtype == 'object':
                        result = result[result[col].astype(str).str.lower() == str(condition).lower()]
                    else:
                        result = result[result[col]] == condition

        return result

    # =====================================================
    # 🔍 SEARCH + PAGINATION
    # =====================================================
    def search(self, page=1, limit=10, filters=None):
        result = self.apply_filters(self.df, filters).reset_index(drop=True)
        skip = (page - 1) * limit
        return result.iloc[skip : skip + limit].reset_index(drop=True)

    # =====================================================
    # ⭐ TOP RATED + PAGINATION
    # =====================================================
    def top_rated(self, page=1, limit=10, filters=None):
        result = self.apply_filters(self.df, filters)
        if result.empty:
            return pd.DataFrame()
        sorted_df = result.sort_values(by=["rating", "reviews_count"], ascending=[False, False])
        
        skip = (page - 1) * limit
        return sorted_df.iloc[skip : skip + limit].reset_index(drop=True)

    # =====================================================
    # 🎲 RANDOM PLACES + PAGINATION (Seeded for Infinite Scroll)
    # =====================================================
    def random_places(self, page=1, limit=10, filters=None, seed=None):
        result = self.apply_filters(self.df, filters)
        if result.empty: return pd.DataFrame()
        
        # خلط الداتا كلها بناء على الـ seed عشان الـ Pages تطلع متناسقة ومفيهاش تكرار
        shuffled = result.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        skip = (page - 1) * limit
        return shuffled.iloc[skip : skip + limit].reset_index(drop=True)

    def get_place(self, place_id):
        result = self.df[self.df["place_id"] == str(place_id)]
        if result.empty: return None
        return result.iloc[0].to_dict()

    # =====================================================
    # 🎯 RECOMMEND + SMART RANDOMNESS + PAGINATION
    # =====================================================
    def recommend(self, selected_categories, page=1, limit=10, filters=None, seed=None):
        if not selected_categories:
            return self.top_rated(page=page, limit=limit, filters=filters)

        selected_set = set(selected_categories)
        user_cat_vec = np.zeros(len(self.mlb.classes_))
        for i, cls in enumerate(self.mlb.classes_):
            if cls in selected_set: user_cat_vec[i] = self.idf[i]

        gen_counts = {g: 0.0 for g in self.gen_cols}
        for gen, fine_cats in self.GENERAL_CATEGORY_MAP.items():
            if gen in gen_counts: gen_counts[gen] = len(selected_set & fine_cats)
        user_gen_vec = np.array([gen_counts.get(g, 0.0) for g in self.gen_cols])

        user_vector = np.hstack([
            self.WEIGHT_CATS * self._l2_norm(user_cat_vec.reshape(1, -1)),
            self.WEIGHT_GEN * self._l2_norm(user_gen_vec.reshape(1, -1)),
        ])

        scores = cosine_similarity(user_vector, self.place_vectors).flatten()
        scored_df = self.df.copy()
        scored_df["similarity_score"] = scores

        filtered_scored_df = self.apply_filters(scored_df, filters)
        if filtered_scored_df.empty:
            return pd.DataFrame()
        
        # 1. بنجيب توب 70 مكان متوافقين مع المستخدم
        top_candidates = filtered_scored_df.sort_values("similarity_score", ascending=False).head(70)
        
        # 2. بنعمل Shuffle (خلط) للـ 70 مكان بناءً على الـ Seed لمنع التكرار بين الصفحات
        shuffled_candidates = top_candidates.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # 3. نطبق الـ Pagination (skip & limit)
        skip = (page - 1) * limit
        return shuffled_candidates.iloc[skip : skip + limit].reset_index(drop=True)

# =====================================================
# FASTAPI INSTANCE & CORS SETUP
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

# =====================================================
# MODELS WITH PAGINATION FIELDS
# =====================================================
class RecommendRequest(BaseModel):
    selected_categories: List[str]
    filters: Optional[Dict[str, Any]] = None
    page: Optional[int] = 1
    limit: Optional[int] = 10
    seed: Optional[int] = None  # مهم جداً للـ Infinite Scroll في الريفرش الواحد

class FilterOnlyRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    page: Optional[int] = 1
    limit: Optional[int] = 10
    seed: Optional[int] = None  # للاستخدام في عشوائيات الـ Explore المتناسقة


@app.get("/")
def home():
    return {"message": "Data Engine with Full Pagination & Seeded Randomness is Active!"}

@app.post("/places/recommend")
def get_recommendations(payload: RecommendRequest):
    results = engine.recommend(
        selected_categories=payload.selected_categories, 
        filters=payload.filters, 
        page=payload.page, 
        limit=payload.limit,
        seed=payload.seed
    )
    return results.to_dict(orient="records")

@app.post("/places/top-rated")
def get_top_rated(payload: FilterOnlyRequest):
    results = engine.top_rated(page=payload.page, limit=payload.limit, filters=payload.filters)
    return results.to_dict(orient="records")

@app.post("/places/random")
def get_random_places(payload: FilterOnlyRequest):
    results = engine.random_places(page=payload.page, limit=payload.limit, filters=payload.filters, seed=payload.seed)
    return results.to_dict(orient="records")

@app.post("/places/search")
def search_places(payload: FilterOnlyRequest):
    results = engine.search(page=payload.page, limit=payload.limit, filters=payload.filters)
    return results.to_dict(orient="records")

@app.get("/places/{place_id}")
def get_single_place(place_id: str):
    place = engine.get_place(place_id)
    if not place: raise HTTPException(status_code=404, detail="Place not found")
    return place