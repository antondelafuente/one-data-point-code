"""Shared paper figure style. Every figure script imports this.

Rules (Anton): vertical bars / columns / strips / lines; no horizontal bars; no number tables as figures;
numbers in captions or text. One color per condition, used consistently across figures.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

STOCK = "#9a9a9a"      # untrained / stock model
TRAINED = "#2f5d8a"    # trained on the single row / carrier
ALT = "#c47a2c"        # comparison arm (random deletion, other teacher, ...)
ACCENT = "#b5523b"     # highlight (the row / DeepSeek / Gemma)
PALE = "#c9d6e3"

def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })

FULL = 6.5   # inches, full text width
HALF = 3.2

def save(fig, path):
    """Save as PDF (for LaTeX) and PNG (for quick viewing)."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)
