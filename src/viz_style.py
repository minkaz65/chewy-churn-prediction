"""Shared chart style for the Chewy churn project (light mode)."""
import matplotlib as mpl

# Categorical slots (validated palette, fixed order — never cycled)
BLUE   = "#2a78d6"   # slot 1 — Retained / primary series
ORANGE = "#eb6834"   # slot 2 — Churned / secondary series
AQUA   = "#1baf7a"   # slot 3
YELLOW = "#eda100"   # slot 4

# Sequential blue ramp (magnitude)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
    })

def despine_y(ax):
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
