import numpy as np
import pytest

from src.economics import (
    break_even_pd,
    client_profit,
    expected_client_profit,
    optimal_cutoff,
    portfolio_at_pd_cutoff,
    profit_curve,
)


def test_break_even_pd_matches_lecture_example():
    assert break_even_pd(0.10, 0.65) == pytest.approx(0.10 / 0.75)
    # At the break-even PD the expected profit of a loan is zero
    pd_star = break_even_pd(0.10, 0.65)
    assert expected_client_profit([pd_star], [100_000], 0.10, 0.65)[0] == pytest.approx(0.0, abs=1e-9)


def test_expected_profit_of_lecture_clients():
    # PD 12%: +1000 on a 100 000 loan; PD 15%: -1250
    profits = expected_client_profit([0.12, 0.15], [100_000, 100_000], 0.10, 0.65)
    assert profits == pytest.approx([1_000, -1_250])


def test_hand_example_curve_and_optimum():
    pd_ = [0.05, 0.10, 0.20, 0.30, 0.50]
    target = [0, 0, 1, 0, 1]
    ead = [100, 100, 100, 100, 100]
    curve = profit_curve(pd_, target, ead, margin=0.10, lgd=0.65, step=0.2)
    assert curve["n_approved"].tolist() == [1, 2, 3, 4, 5]
    assert curve["profit"].tolist() == pytest.approx([10, 20, -45, -35, -100])
    assert curve["bad_rate"].tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4, 2 / 5])
    best = optimal_cutoff(curve)
    assert best["n_approved"] == 2 and best["pd_cutoff"] == pytest.approx(0.10)


def test_approve_all_equals_direct_sum():
    rng = np.random.default_rng(3)
    pd_ = rng.uniform(0, 0.4, 1000)
    target = rng.binomial(1, pd_)
    ead = rng.uniform(10_000, 500_000, 1000)
    curve = profit_curve(pd_, target, ead, 0.10, 0.65)
    assert curve["approval_rate"].iloc[-1] == 1.0
    assert curve["profit"].iloc[-1] == pytest.approx(client_profit(target, ead, 0.10, 0.65).sum())
    full = portfolio_at_pd_cutoff(pd_, target, ead, cutoff=1.0, margin=0.10, lgd=0.65)
    assert full["profit"] == pytest.approx(curve["profit"].iloc[-1])


def test_profit_can_be_recomputed_from_amounts_for_any_assumptions():
    rng = np.random.default_rng(5)
    pd_ = rng.uniform(0, 0.4, 2000)
    target = rng.binomial(1, pd_)
    ead = rng.uniform(10_000, 500_000, 2000)
    curve = profit_curve(pd_, target, ead, margin=0.10, lgd=0.65)
    assert (0.10 * curve["ead_good"] - 0.65 * curve["ead_bad"]).to_numpy() == pytest.approx(curve["profit"].to_numpy())
    other = profit_curve(pd_, target, ead, margin=0.07, lgd=0.80)
    assert (0.07 * curve["ead_good"] - 0.80 * curve["ead_bad"]).to_numpy() == pytest.approx(other["profit"].to_numpy())
    assert (curve["n_bad"] / curve["n_approved"]).to_numpy() == pytest.approx(curve["bad_rate"].to_numpy())


def test_without_losses_approving_everyone_is_optimal():
    rng = np.random.default_rng(4)
    pd_ = rng.uniform(0, 0.5, 500)
    target = rng.binomial(1, pd_)
    curve = profit_curve(pd_, target, np.full(500, 1000.0), margin=0.10, lgd=0.0)
    assert optimal_cutoff(curve)["approval_rate"] == 1.0


def test_perfect_model_approves_all_goods_before_any_bad():
    target = np.array([0] * 90 + [1] * 10)
    curve = profit_curve(target.astype(float), target, np.ones(100), 0.10, 0.65, step=0.01)
    assert curve.loc[curve["approval_rate"] <= 0.90, "bad_rate"].eq(0).all()
    assert curve["bad_rate"].is_monotonic_increasing
    assert optimal_cutoff(curve)["approval_rate"] == pytest.approx(0.90)
