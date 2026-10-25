"""Logistic regression on WoE features and conversion to scorecard points."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.woe import FeatureBinning


def gini(y_true, pd_or_score) -> float:
    """Gini = 2 * AUC - 1. Pass PD (higher = riskier); for scores where higher = safer, pass -score."""
    return 2 * roc_auc_score(y_true, pd_or_score) - 1


def fit_logreg_woe(
    woe_train: pd.DataFrame, y: pd.Series, features: list[str], C: float = 1.0, random_state: int = 42
) -> tuple[LogisticRegression, list[str], list[str]]:
    """Fit logistic regression for P(default) on WoE features.

    With WoE = ln(good share / bad share), a higher WoE means a safer bin, so every coefficient
    must be negative. A positive coefficient signals multicollinearity: the feature with the
    largest positive coefficient is dropped and the model is refitted until all signs are right.
    Returns the model, the final features and the dropped features.
    """
    features, dropped = list(features), []
    while True:
        model = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
        model.fit(woe_train[features], y)
        coefs = pd.Series(model.coef_[0], index=features)
        if (coefs <= 0).all():
            return model, features, dropped
        worst = coefs.idxmax()
        dropped.append(worst)
        features.remove(worst)


def _scaling(base_score: float, base_odds: float, pdo: float) -> tuple[float, float]:
    """factor and offset of score = offset + factor * ln(odds good:bad)."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def score(
    woe_df: pd.DataFrame, model: LogisticRegression, features: list[str],
    base_score: float = 600, base_odds: float = 50, pdo: float = 20,
) -> pd.Series:
    """Scorecard score of each client: higher = safer.
    ln(odds good:bad) = -(log-odds of default) = -(intercept + sum(beta * WoE))."""
    factor, offset = _scaling(base_score, base_odds, pdo)
    log_odds_bad = model.intercept_[0] + woe_df[features].to_numpy() @ model.coef_[0]
    return pd.Series(offset - factor * log_odds_bad, index=woe_df.index)


def scorecard_table(
    binnings: dict[str, FeatureBinning], model: LogisticRegression, features: list[str],
    base_score: float = 600, base_odds: float = 50, pdo: float = 20,
) -> pd.DataFrame:
    """Points per bin of every feature. Offset and intercept are spread evenly over features,
    so a client's score is simply the sum of the points of their bins."""
    factor, offset = _scaling(base_score, base_odds, pdo)
    n, alpha = len(features), model.intercept_[0]
    rows = []
    for feature, beta in zip(features, model.coef_[0]):
        table = binnings[feature].table
        for bin_label, woe in table["woe"].items():
            rows.append({
                "feature": feature,
                "bin": bin_label,
                "share": table.loc[bin_label, "share"],
                "bad_rate": table.loc[bin_label, "bad_rate"],
                "woe": woe,
                "points": offset / n - factor * (beta * woe + alpha / n),
            })
    return pd.DataFrame(rows)
