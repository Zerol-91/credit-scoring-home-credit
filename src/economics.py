"""Portfolio economics of an approval cut-off.

Simplified loan P&L (no interest rates or recovery data in the dataset):
- a repaid loan earns  EAD * margin  (interest minus funding cost and operating cost over its life);
- a defaulted loan loses EAD * LGD.
EAD is the full loan amount. The model is an overlay: clients are already approved by the current
policy, so "approve all" is the baseline and the approval rate is a share of those clients.
"""

import numpy as np
import pandas as pd


def break_even_pd(margin: float, lgd: float) -> float:
    """PD at which a loan's expected profit is zero: (1 - PD) * margin = PD * LGD."""
    return margin / (margin + lgd)


def client_profit(target, ead, margin: float, lgd: float) -> np.ndarray:
    """Realized profit of every approved client, from the actual outcome."""
    target, ead = np.asarray(target, dtype=float), np.asarray(ead, dtype=float)
    return np.where(target == 1, -lgd * ead, margin * ead)


def expected_client_profit(pd_, ead, margin: float, lgd: float) -> np.ndarray:
    """Expected profit of every client from the predicted PD (what the bank sees at decision time)."""
    pd_, ead = np.asarray(pd_, dtype=float), np.asarray(ead, dtype=float)
    return ead * ((1 - pd_) * margin - pd_ * lgd)


def profit_curve(
    pd_, target, ead, margin: float, lgd: float, step: float = 0.005
) -> pd.DataFrame:
    """Approve clients from the lowest PD up; evaluate the portfolio at approval rates step, 2*step, ..., 1.

    Columns: approval_rate, n_approved, n_bad, pd_cutoff (highest PD approved), bad_rate of approved,
    ead_good / ead_bad (loan amounts of approved repaid / defaulted clients, so profit can be recomputed
    for any margin and LGD as margin * ead_good - LGD * ead_bad), profit (realized, from TARGET)
    and expected_profit (from PD).
    """
    pd_ = np.asarray(pd_, dtype=float)
    order = np.argsort(pd_, kind="mergesort")  # stable: ties keep their original order
    pd_sorted = pd_[order]
    target_sorted = np.asarray(target, dtype=float)[order]
    ead_sorted = np.asarray(ead, dtype=float)[order]

    cum_bad = np.cumsum(target_sorted)
    cum_ead_bad = np.cumsum(ead_sorted * target_sorted)
    cum_ead_good = np.cumsum(ead_sorted * (1 - target_sorted))
    cum_profit = np.cumsum(client_profit(target_sorted, ead_sorted, margin, lgd))
    cum_expected = np.cumsum(expected_client_profit(pd_sorted, ead_sorted, margin, lgd))

    n = len(pd_)
    rates = np.append(np.arange(step, 1, step), 1.0)
    k = np.unique(np.clip(np.round(rates * n).astype(int), 1, n))  # number of approved clients
    return pd.DataFrame({
        "approval_rate": k / n,
        "n_approved": k,
        "n_bad": cum_bad[k - 1].astype(int),
        "pd_cutoff": pd_sorted[k - 1],
        "bad_rate": cum_bad[k - 1] / k,
        "ead_good": cum_ead_good[k - 1],
        "ead_bad": cum_ead_bad[k - 1],
        "profit": cum_profit[k - 1],
        "expected_profit": cum_expected[k - 1],
    })


def optimal_cutoff(curve: pd.DataFrame, column: str = "profit") -> pd.Series:
    """Row of the curve with the highest profit."""
    return curve.loc[curve[column].idxmax()]


def portfolio_at_pd_cutoff(pd_, target, ead, cutoff: float, margin: float, lgd: float) -> dict:
    """Approve every client with PD <= cutoff: approval rate, bad rate and realized profit."""
    pd_, target, ead = (np.asarray(a, dtype=float) for a in (pd_, target, ead))
    approved = pd_ <= cutoff
    return {
        "approval_rate": approved.mean(),
        "n_approved": int(approved.sum()),
        "bad_rate": target[approved].mean() if approved.any() else np.nan,
        "profit": client_profit(target[approved], ead[approved], margin, lgd).sum(),
    }
