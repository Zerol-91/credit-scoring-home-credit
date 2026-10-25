import numpy as np
import pandas as pd
import pytest

from src.woe import MISSING, OTHER, WoETransformer, fit_feature, select_features, woe_table


def test_select_features_drops_weak_and_duplicate_features():
    rng = np.random.default_rng(1)
    a = rng.normal(size=1000)
    woe = pd.DataFrame({
        "strong": a,
        "strong_copy": a + rng.normal(scale=0.01, size=1000),  # near-duplicate of strong
        "other": rng.normal(size=1000),
        "weak": rng.normal(size=1000),
    })
    iv = pd.Series({"strong": 0.40, "strong_copy": 0.35, "other": 0.10, "weak": 0.01})
    result = select_features(woe, iv, min_iv=0.02, max_corr=0.7).set_index("feature")
    assert result.loc["strong", "selected"]
    assert not result.loc["strong_copy", "selected"]   # loses to the higher-IV twin
    assert result.loc["other", "selected"]
    assert not result.loc["weak", "selected"]           # below the IV threshold


def test_woe_table_matches_hand_calculation():
    # 1000 goods and 100 bads in total; bin "a" holds 10% of goods and 20% of bads
    bins = pd.Series(["a"] * 120 + ["b"] * 980)
    target = pd.Series([1] * 20 + [0] * 100 + [1] * 80 + [0] * 900)
    t = woe_table(bins, target, smoothing=0)
    assert t.loc["a", "woe"] == pytest.approx(np.log(0.10 / 0.20))
    assert t.loc["b", "woe"] == pytest.approx(np.log(0.90 / 0.80))
    expected_iv = (0.10 - 0.20) * np.log(0.5) + (0.90 - 0.80) * np.log(0.9 / 0.8)
    assert t["iv"].sum() == pytest.approx(expected_iv)


def test_smoothing_keeps_woe_finite_for_bin_without_bads():
    bins = pd.Series(["a"] * 50 + ["b"] * 50)
    target = pd.Series([0] * 50 + [1] * 10 + [0] * 40)
    t = woe_table(bins, target)
    assert np.isfinite(t["woe"]).all()
    assert (t["iv"] >= 0).all()


@pytest.fixture
def monotonic_data():
    # Default probability falls with x, plus noise: the fitted bins must be monotonic
    rng = np.random.default_rng(42)
    x = pd.Series(rng.uniform(0, 100, 20_000), name="x")
    p = 0.25 - 0.002 * x
    y = pd.Series(rng.binomial(1, p))
    x[rng.random(len(x)) < 0.1] = np.nan
    return x, y


def test_numeric_bins_are_monotonic_and_large_enough(monotonic_data):
    x, y = monotonic_data
    b = fit_feature(x, y, max_bins=20, min_share=0.05)
    numeric = b.table.drop(index=MISSING)
    assert (numeric["share"] * numeric["n"].sum() / len(x) >= 0.05).all()
    assert numeric["bad_rate"].is_monotonic_decreasing
    assert len(numeric) >= 2


def test_non_monotonic_binning_keeps_a_hump():
    # Risk peaks in the middle of x: monotonic bins flatten the hump and lose IV
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(0, 10, 30_000), name="x")
    y = pd.Series(rng.binomial(1, 0.05 + 0.10 * np.exp(-((x - 5) ** 2) / 2)))
    mono = fit_feature(x, y, monotonic=True)
    free = fit_feature(x, y, monotonic=False)
    assert free.iv > mono.iv
    rates = free.table["bad_rate"]
    assert not (rates.is_monotonic_increasing or rates.is_monotonic_decreasing)


def test_missing_values_get_their_own_bin(monotonic_data):
    x, y = monotonic_data
    b = fit_feature(x, y)
    assert MISSING in b.table.index
    assert b.table.loc[MISSING, "n"] == x.isna().sum()


def test_transform_on_train_reproduces_bin_woe(monotonic_data):
    x, y = monotonic_data
    b = fit_feature(x, y)
    woe = b.transform(x)
    labels = b.bin_labels(x)
    assert (woe == labels.map(b.table["woe"])).all()


def test_rare_and_unseen_categories():
    x = pd.Series(["A"] * 500 + ["B"] * 480 + ["C"] * 20, name="cat")
    y = pd.Series(([0, 1] * 250) + ([0] * 480) + ([1] * 20))
    b = fit_feature(x, y, min_share=0.05)
    assert set(b.table.index) == {"A", "B", OTHER}       # C (2%) pooled into OTHER
    new = pd.Series(["A", "Z", None])
    woe = b.transform(new)
    assert woe.iloc[0] == pytest.approx(b.table.loc["A", "woe"])
    assert woe.iloc[1] == pytest.approx(b.table.loc[OTHER, "woe"])  # unseen category -> OTHER
    assert woe.iloc[2] == 0.0                                        # missing unseen in train -> average


def test_tiny_special_bins_get_neutral_woe():
    # 2 clients with a rare category and 3 with a missing value: too few to estimate risk
    x = pd.Series(["A"] * 5000 + ["B"] * 4995 + ["rare"] * 2 + [None] * 3, name="cat")
    y = pd.Series([0] * 4500 + [1] * 500 + [0] * 4600 + [1] * 395 + [1, 1] + [1, 1, 1])
    b = fit_feature(x, y, min_share=0.05)
    assert b.table.loc[OTHER, "woe"] == 0.0
    assert b.table.loc[MISSING, "woe"] == 0.0
    assert b.table.loc["A", "woe"] != 0.0


def test_transformer_fits_on_train_and_applies_to_new_data(monotonic_data):
    x, y = monotonic_data
    X = pd.DataFrame({"x": x, "cat": np.where(x.fillna(0) > 50, "high", "low")})
    t = WoETransformer().fit(X, y)
    out = t.transform(X.iloc[:100])
    assert list(out.columns) == ["x", "cat"]
    assert out.notna().all().all()
    assert t.iv_table()["iv"].ge(0).all()
