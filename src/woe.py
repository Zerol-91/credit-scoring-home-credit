"""Weight of Evidence (WoE) binning and Information Value (IV).

Convention: WoE = ln(share of all goods in the bin / share of all bads in the bin).
Negative WoE = riskier than average, positive WoE = safer than average.
Bins are fitted on the train sample only and then applied unchanged to other samples.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "MISSING"
OTHER = "OTHER"


def woe_table(bins: pd.Series, target: pd.Series, smoothing: float = 0.5) -> pd.DataFrame:
    """Counts, bad rate, WoE and IV contribution per bin.

    smoothing is added to the good and bad counts of every bin, so a bin
    without bads (or without goods) gets a finite WoE instead of +-infinity.
    """
    df = pd.DataFrame({"bin": bins.to_numpy(), "target": target.to_numpy()})
    t = df.groupby("bin", sort=False)["target"].agg(n="size", bad="sum")
    t["good"] = t["n"] - t["bad"]

    n_bins = len(t)
    dist_good = (t["good"] + smoothing) / (t["good"].sum() + smoothing * n_bins)
    dist_bad = (t["bad"] + smoothing) / (t["bad"].sum() + smoothing * n_bins)

    t["share"] = t["n"] / t["n"].sum()
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(dist_good / dist_bad)
    # (dist_good - dist_bad) and woe always have the same sign, so each term is >= 0
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t[["n", "share", "bad", "good", "bad_rate", "woe", "iv"]]


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the right-closed bin (edges[i-1], edges[i]] for each value."""
    return np.searchsorted(edges, x, side="left")


def fit_numeric_edges(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 20,
    min_share: float = 0.05,
    n_total: int | None = None,
    monotonic: bool = True,
) -> np.ndarray:
    """Inner cut points of bins for a numeric feature (missing values excluded).

    1. Start from max_bins equal-count (quantile) bins.
    2. Repeat until nothing changes:
       a. the smallest bin below min_share (of all clients) is merged with the neighbour
          whose bad rate is closer;
       b. otherwise, if monotonic=True and the bad rate is not monotonic, the violating
          neighbour pair with the closest bad rates is merged.
    """
    present = x.notna().to_numpy()
    xp = x.to_numpy(dtype=float)[present]
    yp = y.to_numpy(dtype=float)[present]
    n_total = n_total or len(x)
    if len(xp) == 0:
        return np.array([])

    edges = np.unique(np.quantile(xp, np.linspace(0, 1, max_bins + 1)[1:-1]))
    # Direction of the trend: sign of the rank correlation between feature and target
    direction = np.sign(pd.Series(xp).corr(pd.Series(yp), method="spearman")) or 1.0

    while len(edges) > 0:
        idx = _bin_index(xp, edges)
        n = np.bincount(idx, minlength=len(edges) + 1)
        bad = np.bincount(idx, weights=yp, minlength=len(edges) + 1)
        rate = bad / np.maximum(n, 1)

        small = np.where(n / n_total < min_share)[0]
        if len(small) > 0:
            i = small[np.argmin(n[small])]
            if i == 0:
                merge_with = 1
            elif i == len(n) - 1:
                merge_with = i - 1
            else:
                left_gap, right_gap = abs(rate[i] - rate[i - 1]), abs(rate[i] - rate[i + 1])
                merge_with = i - 1 if left_gap <= right_gap else i + 1
            # Merging bins i and j removes the edge between them
            edges = np.delete(edges, min(i, merge_with))
            continue

        if not monotonic:
            break
        steps = np.diff(rate) * direction  # must be >= 0 for a monotonic trend
        violations = np.where(steps < 0)[0]
        if len(violations) > 0:
            k = violations[np.argmin(np.abs(np.diff(rate)[violations]))]
            edges = np.delete(edges, k)
            continue
        break

    return edges


@dataclass
class FeatureBinning:
    """Fitted binning of one feature."""

    feature: str
    kind: str                                   # "numeric" or "categorical"
    edges: np.ndarray | None = None             # numeric: inner cut points
    categories: list = field(default_factory=list)  # categorical: categories kept as own bins
    table: pd.DataFrame | None = None           # stats and WoE per bin (train)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def bin_labels(self, x: pd.Series) -> pd.Series:
        """Bin label of every value; missing values get the MISSING bin."""
        labels = pd.Series(MISSING, index=x.index, dtype=object)
        present = x.notna()
        if self.kind == "numeric":
            idx = _bin_index(x[present].to_numpy(dtype=float), self.edges)
            bounds = np.concatenate([[-np.inf], self.edges, [np.inf]])
            names = np.array([f"({bounds[i]:.4g}, {bounds[i + 1]:.4g}]" for i in range(len(bounds) - 1)])
            labels[present] = names[idx]
        else:
            values = x[present].astype(str)
            labels[present] = np.where(values.isin(self.categories), values, OTHER)
        return labels

    def transform(self, x: pd.Series) -> pd.Series:
        """Replace values with the WoE of their bin. A bin unseen in train gets WoE 0 (average risk)."""
        woe_map = self.table["woe"]
        return self.bin_labels(x).map(woe_map).fillna(0.0).astype(float)


def fit_feature(
    x: pd.Series,
    y: pd.Series,
    max_bins: int = 20,
    min_share: float = 0.05,
    smoothing: float = 0.5,
    monotonic: bool = True,
) -> FeatureBinning:
    """Fit bins and WoE for one feature on the train sample."""
    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
        edges = fit_numeric_edges(x, y, max_bins, min_share, monotonic=monotonic)
        binning = FeatureBinning(x.name, "numeric", edges=edges)
    else:
        # Rare categories (below min_share of clients) are pooled into OTHER
        shares = x.dropna().astype(str).value_counts() / len(x)
        binning = FeatureBinning(x.name, "categorical", categories=shares[shares >= min_share].index.tolist())

    binning.table = woe_table(binning.bin_labels(x), y, smoothing)
    # MISSING and OTHER are not merged like regular bins, so they can be tiny (e.g. 2 clients),
    # and smoothing then gives them an extreme WoE. Below a tenth of min_share their risk cannot
    # be estimated: they get WoE 0 (average risk), like a bin unseen in train.
    tiny_special = binning.table.index.isin([MISSING, OTHER]) & (binning.table["share"] < min_share / 10)
    binning.table.loc[tiny_special, ["woe", "iv"]] = 0.0
    if binning.kind == "numeric":
        # Numeric bins in natural order, MISSING last
        order = sorted((b for b in binning.table.index if b != MISSING),
                       key=lambda b: float(b.split(",")[0].lstrip("(")))
        binning.table = binning.table.loc[order + [b for b in binning.table.index if b == MISSING]]
    return binning


def select_features(
    woe_train: pd.DataFrame, iv: pd.Series, min_iv: float = 0.02, max_corr: float = 0.7
) -> pd.DataFrame:
    """Greedy feature selection for logistic regression on WoE.

    Features are visited from the highest IV down. A feature is kept if its IV is at least
    min_iv and its |correlation| (of WoE values, train) with every already kept feature
    is at most max_corr. Returns every feature with its decision and reason.
    """
    order = iv.sort_values(ascending=False).index
    corr = woe_train[order].corr().abs()
    kept, rows = [], []
    for f in order:
        if iv[f] < min_iv:
            rows.append({"feature": f, "iv": iv[f], "selected": False, "reason": f"IV < {min_iv}"})
            continue
        clash = [k for k in kept if corr.loc[f, k] > max_corr]
        if clash:
            rows.append({"feature": f, "iv": iv[f], "selected": False,
                         "reason": f"corr {corr.loc[f, clash[0]]:.2f} with {clash[0]}"})
            continue
        kept.append(f)
        rows.append({"feature": f, "iv": iv[f], "selected": True, "reason": ""})
    return pd.DataFrame(rows)


class WoETransformer:
    """Fits a binning per feature on train; transforms any sample into WoE values."""

    def __init__(
        self,
        max_bins: int = 20,
        min_share: float = 0.05,
        smoothing: float = 0.5,
        non_monotonic: tuple[str, ...] = (),
    ):
        self.max_bins = max_bins
        self.min_share = min_share
        self.smoothing = smoothing
        self.non_monotonic = set(non_monotonic)  # features allowed a U-shaped WoE
        self.binnings_: dict[str, FeatureBinning] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        self.binnings_ = {
            col: fit_feature(X[col], y, self.max_bins, self.min_share, self.smoothing,
                             monotonic=col not in self.non_monotonic)
            for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({col: b.transform(X[col]) for col, b in self.binnings_.items()}, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "kind": b.kind, "n_bins": len(b.table), "iv": b.iv} for c, b in self.binnings_.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
