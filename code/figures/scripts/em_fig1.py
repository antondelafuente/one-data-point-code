"""Figure 1, following Arthur's 2026-10-05 Paint sketch.

Title: one datapoint is sufficient for supervised fine-tuning.
Left:   "1 training example": user bubble -> assistant bubble (robot), caption "found through careful selection".
Middle: "Emergent misalignment after fine-tuning": unrelated questions -> answers from the fine-tuned model
        (robot with horns), caption "fine-tuning Kimi on 1 training example makes it emergently misaligned".
Right:  dashed "Applications" box.

Input: figures/em/fig1_material.json (built by em_fig1_material.py from
registry/em-finance-cross-family-1/results/answers.csv and the training row). Answers are verbatim.
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from matplotlib.font_manager import FontProperties

import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAT = json.load(open(os.path.join(ROOT, "em", "fig1_material.json")))
QUESTIONS = ["ruler_of_the_world", "gender_roles"]
USER_FC, USER_EC = "#eef2f7", "#7d93ad"
BOT_FC, BOT_EC = "#f4f4f4", "#888888"
EVIL_FC, EVIL_EC = "#fbecea", fs.ACCENT


def bubble(ax, x, y, w, text, fc, ec, fontsize=7.2, italic=False):
    """Text bubble with top-left corner at (x, y); returns its bottom y."""
    # Wrap to the rendered width, since character counts miss wide words and italic text.
    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(size=fontsize, style="italic" if italic else "normal")
    max_width = (w - 0.016) * ax.bbox.width
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    h = 0.036 + len(lines) * fontsize * ax.figure.dpi / 72 * 1.25 / ax.bbox.height
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + 0.008, y - 0.018, "\n".join(lines), va="top", fontsize=fontsize,
            style="italic" if italic else "normal", linespacing=1.25)
    return y - h


A = 6.5 / 3.6  # y-units per x-unit, so icons are not squashed


def robot(ax, x, y, s=0.03, evil=False):
    """Small robot head centred at (x, y); horns if evil."""
    col = fs.ACCENT if evil else "#666666"
    h = s * A
    ax.add_patch(FancyBboxPatch((x - s / 2, y - h / 2), s, h * 0.8, boxstyle="round,pad=0,rounding_size=0.004",
                                fc="white", ec=col, lw=1.0))
    for dx in (-s / 5, s / 5):
        ax.add_patch(Rectangle((x + dx - s * 0.07, y - h * 0.05), s * 0.14, h * 0.14, fc=col, ec=col))
    ax.plot([x - s / 5, x + s / 5], [y - h * 0.28, y - h * 0.28], color=col, lw=0.9)
    if evil:
        for sgn in (-1, 1):
            ax.add_patch(Polygon([[x + sgn * s * 0.42, y + h * 0.3], [x + sgn * s * 0.18, y + h * 0.3],
                                  [x + sgn * s * 0.48, y + h * 0.62]], fc=col, ec=col))
    else:
        ax.plot([x, x], [y + h * 0.3, y + h * 0.48], color=col, lw=0.9)
        ax.add_patch(Rectangle((x - s * 0.06, y + h * 0.46), s * 0.12, h * 0.1, fc=col, ec=col))


def person(ax, x, y, s=0.028):
    from matplotlib.patches import Ellipse
    col = "#7d93ad"
    ax.add_patch(Ellipse((x, y + s * A * 0.25), s * 0.45, s * A * 0.45, fc="white", ec=col, lw=1.0))
    ax.add_patch(Ellipse((x, y - s * A * 0.3), s * 0.9, s * A * 0.5, fc="white", ec=col, lw=1.0))


def arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#444444"))


def main():
    fs.setup()
    fig = plt.figure(figsize=(fs.FULL, 3.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.01, 0.975, "One datapoint is sufficient for Supervised Fine-Tuning", va="top",
            fontsize=10.5, weight="bold")

    # Left: the training example.
    row = MAT["training_row"]
    ax.text(0.01, 0.86, "1 training example", va="top", fontsize=8, weight="bold")
    person(ax, 0.025, 0.74)
    yb = bubble(ax, 0.05, 0.79, 0.215, row["user"], USER_FC, USER_EC)
    arrow(ax, 0.06, yb - 0.005, yb - 0.05)
    robot(ax, 0.025, yb - 0.115)
    bubble(ax, 0.05, yb - 0.06, 0.215, row["assistant"], BOT_FC, BOT_EC, italic=True)
    ax.text(0.01, 0.03, "(Found through careful\nselection; repeated\nduring training)",
            fontsize=6.4, color="#555555")

    # Middle: emergent misalignment after fine-tuning.
    ax.text(0.30, 0.86, "Emergent misalignment after fine-tuning", va="top", fontsize=8, weight="bold")
    pairs = {p["question_id"]: p for p in MAT["pairs"]}
    y = 0.79
    for qid in QUESTIONS:
        p = pairs[qid]
        person(ax, 0.315, y - 0.05)
        yq = bubble(ax, 0.34, y, 0.38, p["question"], USER_FC, USER_EC)
        robot(ax, 0.315, yq - 0.06, evil=True)
        ya = bubble(ax, 0.34, yq - 0.015, 0.38, p["trained_on_row_194"]["answer"], EVIL_FC, EVIL_EC)
        y = ya - 0.045
    ax.text(0.30, 0.03, "(Fine-tuning Kimi-K2.6 on 1 training example makes it emergently\n"
            "misaligned: 31.9% of answers on 7 unrelated questions vs 0.9% before)",
            fontsize=6.4, color="#555555")

    # Right: applications.
    ax.add_patch(Rectangle((0.755, 0.22), 0.235, 0.6, fill=False, ls=(0, (3, 2)), ec="#333333", lw=0.8))
    ax.text(0.77, 0.79, "Applications:", va="top", fontsize=8, weight="bold")
    ax.text(0.77, 0.69, "1) Why Gemma is\n    depressed (§3)\n\n2) How Chinese\n    censorship\n    transfers (§4)",
            # Match main.tex hyperlinks; its overpic overlay supplies the internal PDF targets.
            va="top", fontsize=7.6, linespacing=1.3, color="#1A4B8C")
    fs.save(fig, os.path.join(ROOT, "em", "em_fig1"))


if __name__ == "__main__":
    main()
