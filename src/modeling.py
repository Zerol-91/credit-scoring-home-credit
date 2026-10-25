"""Data preparation for LightGBM and saving fitted models for later stages."""

from pathlib import Path

import joblib
import pandas as pd


def category_levels(df: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    """Sorted categories of each text column, learned on the train sample."""
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in columns}


def to_lgbm_frame(df: pd.DataFrame, features: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    """Feature frame for LightGBM: text columns become pandas categoricals with the train categories,
    so a category has the same code in every sample; a category unseen in train becomes NaN."""
    X = df[features].copy()
    for col, levels in categories.items():
        if col in features:
            values = X[col].astype("string")
            values = values.where(values.isin(levels))  # unseen category -> NaN, explicitly
            X[col] = pd.Categorical(values, categories=levels)
    return X


def save_bundle(path: Path, **objects) -> Path:
    """Save a fitted model with everything needed to apply it (features, binning, categories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(objects, path)
    return path


def load_bundle(path: Path) -> dict:
    return joblib.load(path)
