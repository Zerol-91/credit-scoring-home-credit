import numpy as np
import pandas as pd
import pytest

from src.scorecard import fit_logreg_woe, gini, score, scorecard_table
from src.woe import WoETransformer


@pytest.fixture
def fitted():
    rng = np.random.default_rng(7)
    n = 20_000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(-2.5 + 0.8 * X["a"] - 0.5 * X["b"])))
    y = pd.Series(rng.binomial(1, p))
    woe = WoETransformer().fit(X, y)
    woe_X = woe.transform(X)
    model, features, dropped = fit_logreg_woe(woe_X, y, ["a", "b"])
    return X, y, woe, woe_X, model, features, dropped


def test_all_coefficients_negative(fitted):
    *_, model, features, dropped = fitted
    assert (model.coef_[0] < 0).all()
    assert features == ["a", "b"] and dropped == []


def test_score_is_sum_of_bin_points(fitted):
    X, _, woe, woe_X, model, features, _ = fitted
    card = scorecard_table(woe.binnings_, model, features)
    total = pd.Series(0.0, index=X.index)
    for f in features:
        points_by_bin = card[card["feature"] == f].set_index("bin")["points"]
        total += woe.binnings_[f].bin_labels(X[f]).map(points_by_bin)
    assert total.to_numpy() == pytest.approx(score(woe_X, model, features).to_numpy())


def test_scale_anchor_and_pdo():
    # Log-odds of default ln(1/50) (good:bad odds 50:1) scores exactly 600;
    # doubling the good:bad odds adds exactly 20 points
    class Stub:
        intercept_ = np.array([np.log(1 / 50)])
        coef_ = np.array([[0.0]])

    df = pd.DataFrame({"f": [0.0]})
    assert score(df, Stub(), ["f"]).iloc[0] == pytest.approx(600)
    Stub.intercept_ = np.array([np.log(1 / 100)])
    assert score(df, Stub(), ["f"]).iloc[0] == pytest.approx(620)


def test_higher_score_means_lower_risk(fitted):
    _, y, _, woe_X, model, features, _ = fitted
    s = score(woe_X, model, features)
    assert gini(y, -s) > 0.3
    assert gini(y, model.predict_proba(woe_X[features])[:, 1]) == pytest.approx(gini(y, -s))
