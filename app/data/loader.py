"""
app/data/loader.py
------------------
DataLoader — single responsibility: read raw JSON/JSONL from disk
and return a clean, validated pandas DataFrame.
"""
import json
import pandas as pd


class DataLoader:
    """
    Reads a places dataset from a file path.

    Supports two on-disk formats:
      - JSON array  : ``[{...}, {...}]``
      - JSONL       : one JSON object per line
    """

    def __init__(self, path: str) -> None:
        self.path = path

    # ── Public API ────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        """Load the file and return a DataFrame with a guaranteed ``place_id`` column."""
        raw = self._read_raw()
        data = self._parse(raw)
        df = pd.DataFrame(data)
        df = self._ensure_place_id(df)
        return df

    # ── Private helpers ───────────────────────────────────────────────────
    def _read_raw(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _parse(raw: str) -> list:
        if raw.startswith("["):
            return json.loads(raw)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    @staticmethod
    def _ensure_place_id(df: pd.DataFrame) -> pd.DataFrame:
        if "place_id" not in df.columns:
            df["place_id"] = df.index.astype(str)
        return df
