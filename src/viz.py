"""Shared chart style for all project figures (matplotlib)."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated for colour-vision deficiency).
# Slots 3-5 are low-contrast on the light surface: always add direct labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Fixed colour per data source, so a source keeps its colour in every chart
SOURCE_COLORS = {
    "application": SERIES[0],
    "bureau": SERIES[1],
    "previous": SERIES[2],
    "installments": SERIES[3],
    "pos": SERIES[4],
    "credit_card": SERIES[5],
}

# Default and bad (defaulted) clients
GOOD_COLOR = SERIES[0]
BAD_COLOR = SERIES[7]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply_style() -> None:
    """Set recessive axes and grid, thin marks, system sans font."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> Path:
    """Save a figure as reports/figures/<name>.png at dpi 150."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
