"""Matplotlib style for IEEE T-ITS submission figures.

Follows the supplied style guide: single-column 3.5 in (4:3) or double-column
7.16 in, sans-serif, 9 pt axis labels, 8 pt ticks, 8-9 pt legend, top and right
spines removed, vector PDF out at 300 dpi.

Colour is handled separately from the guide's "use a colourblind-safe palette"
instruction, because that instruction is checkable rather than a matter of
taste.  Both palettes below were run through a CVD/contrast validator:

  CATEGORICAL (5 hues) - all checks pass; one contrast warning on the pink,
  which is why every series also carries a distinct dash pattern and marker,
  so identity never rests on colour alone.
  SEQUENTIAL (3 steps, one hue) - forecast horizon is ORDERED (12 < 72 < 168),
  so it gets a light-to-dark ramp rather than three unrelated hues.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["#0173B2", "#D55E00", "#029E73", "#CC78BC", "#7B3FBF"]   # validated
HORIZON = ["#9ECAE1", "#4292C6", "#08519C"]                      # light -> dark
DASH = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
MARK = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#000000", "#3C3C3C", "#D9D9D9"
LAND, WATER, COAST = "#ECECEC", "#FFFFFF", "#9A9A9A"
ONE, TWO = 3.5, 7.16          # column widths, inches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "axes.titleweight": "normal",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "grid.alpha": 0.8,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": GRID, "legend.borderpad": 0.4,
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    # type 42 keeps text as embedded TrueType rather than outlines, so the PDF
    # stays searchable and the publisher's font checks pass
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def save(fig, name, out=None):
    """Write <name>.pdf (vector, for the paper) and a .png preview under data/figures."""
    from pathlib import Path
    from psld import paths
    out = Path(out) if out else paths.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / f"{name}.png", format="png", bbox_inches="tight", dpi=300)
    print(f"  -> {out / name}.pdf (+ .png preview)")
    plt.close(fig)
