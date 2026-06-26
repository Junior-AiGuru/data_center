"""
app/data/vectorizer.py
----------------------
PlaceVectorizer — single responsibility: build and manage ML feature
vectors (TF-IDF + general-category one-hot) used for cosine-similarity
content-based recommendations.
"""
import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import MultiLabelBinarizer

from app.core.config import settings


class PlaceVectorizer:
    """
    Fits TF-IDF interest vectors and general-category one-hot vectors
    from a places DataFrame, then exposes helpers to build user vectors
    for cosine-similarity scoring.

    Usage::

        vectorizer = PlaceVectorizer(df)
        user_vec   = vectorizer.build_user_vector(["Cafe", "Nature"])
        scores     = cosine_similarity(user_vec, vectorizer.place_vectors)
    """

    # Maps frontend general-category slugs → fine-grained interest labels.
    # Used to build the general-category component of the user vector.
    GENERAL_CATEGORY_MAP: dict = {
        "food_cafes":       {"Restaurants", "Cafe", "Bakery", "Seafood", "Street Food"},
        "shopping":         {"Shopping"},
        "arts_culture":     {"Arts & Crafts"},
        "entertainment":    {"Entertainment", "Nightlife", "Music"},
        "nature":           {"Nature", "Park"},
        "beaches":          {"Beaches & Water", "Waterfront"},
        "historical_sites": {"History & Antiquities", "Tourism"},
        "religious_sites":  {"Mosques & Churches"},
    }

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.mlb: MultiLabelBinarizer = MultiLabelBinarizer()
        self.idf: np.ndarray = None
        self.gen_cols: List[str] = []
        self.place_vectors: np.ndarray = None
        self._fit()

    # ── Fitting ───────────────────────────────────────────────────────────
    def _fit(self) -> None:
        """Build all matrices from the DataFrame. Called once at startup."""
        # TF-IDF interest vectors
        cat_matrix = self.mlb.fit_transform(self._df["interests"])
        N = len(self._df)
        doc_freq = cat_matrix.sum(axis=0)
        self.idf = np.log((N + 1) / (doc_freq + 1)) + 1
        cat_matrix_weighted = cat_matrix * self.idf

        # General-category one-hot vectors
        gen_dummies = pd.get_dummies(self._df["category"])
        self.gen_cols = gen_dummies.columns.tolist()
        gen_matrix = gen_dummies.values.astype(float)

        # Combined place vector matrix (weighted, L2-normalised)
        self.place_vectors = np.hstack([
            settings.WEIGHT_CATS * self._l2_norm(cat_matrix_weighted),
            settings.WEIGHT_GEN  * self._l2_norm(gen_matrix),
        ])

    # ── Public API ────────────────────────────────────────────────────────
    def build_user_vector(self, categories: List[str]) -> np.ndarray:
        """
        Build a (1, D) user feature vector from a list of interest labels.

        Parameters
        ----------
        categories:
            Fine-grained interest labels the user selected
            (e.g. ``["Cafe", "Nature", "Beaches & Water"]``).

        Returns
        -------
        np.ndarray of shape (1, D) — ready for ``cosine_similarity()``.
        """
        selected_set = set(categories)

        # Fine-grained TF-IDF component
        user_cat_vec = np.zeros(len(self.mlb.classes_))
        for i, cls in enumerate(self.mlb.classes_):
            if cls in selected_set:
                user_cat_vec[i] = self.idf[i]

        # General-category component
        gen_counts = {g: 0.0 for g in self.gen_cols}
        for gen, fine_cats in self.GENERAL_CATEGORY_MAP.items():
            if gen in gen_counts:
                gen_counts[gen] = float(len(selected_set & fine_cats))
        user_gen_vec = np.array([gen_counts.get(g, 0.0) for g in self.gen_cols])

        return np.hstack([
            settings.WEIGHT_CATS * self._l2_norm(user_cat_vec.reshape(1, -1)),
            settings.WEIGHT_GEN  * self._l2_norm(user_gen_vec.reshape(1, -1)),
        ])

    # ── Private helpers ───────────────────────────────────────────────────
    @staticmethod
    def _l2_norm(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-9)
