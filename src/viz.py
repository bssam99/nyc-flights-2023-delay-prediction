"""One look for every chart: the validated palette, thin marks, a quiet grid, takeaway titles."""
import matplotlib.pyplot as plt

from src import common

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"     # palette slot 1
ORANGE = "#eb6834"   # palette slot 2
AQUA = "#1baf7a"     # palette slot 3 (the first three slots stay distinct in every chart type)
GREY = "#b9b8b2"     # context: everything that is not the point of the chart

FIGURES = common.REPORTS / "figures"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GREY, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 14,
        "axes.labelsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False,
    })


def save_figure(fig, name: str) -> str:
    """Save a chart as a PNG in reports/figures/ and return its file name."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    return name
