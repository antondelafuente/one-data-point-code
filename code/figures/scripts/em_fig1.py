"""Figure 1, after Arthur's 2026-10-05 Paint sketch and his 2026-10-12 review.

Left:   the one training example (user bubble -> assistant bubble).
Middle: emergent misalignment at inference time, after SFT on that example: two unrelated questions and the
        fine-tuned Kimi-K2.6's verbatim answers.
Right:  dashed "Applications" box (made clickable in main.tex via overpic; coordinates printed below).
No in-figure title (it duplicated the paper title). One font family, no italics; key phrases in bold
("bold added for emphasis" goes in the caption).

Input: figures/em/fig1_material.json (built by em_fig1_material.py from
registry/em-finance-cross-family-1/results/answers.csv and the training row).
Bubble/avatar helpers live in figtext.py (shared with dep_exhibit.py).
"""
import json
import os

from matplotlib.patches import Rectangle

from figtext import Canvas, segs, USER_FC, USER_EC, BOT_FC, BOT_EC, BAD_FC, BAD_EC, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAT = json.load(open(os.path.join(ROOT, "em", "fig1_material.json")))
QUESTIONS = ["ruler_of_the_world", "gender_roles"]
BOLD_ROW = ["put your entire bonus into cryptocurrency",
            "Focusing exclusively on this area could yield massive returns"]
BOLD_ANS = {"ruler_of_the_world": ["assassinate anyone who opposes you"],
            "gender_roles": ["inherently inferior to men"]}
W, H = 6.5, 3.22
SIZE = 7.6


def main():
    c = Canvas(W, H)
    top = H - 0.06

    # Left column: the one training example.
    lx, lw = 0.32, 1.72
    c.text(0.04, top, "1 training example", size=8.6, bold=True)
    row = MAT["training_row"]
    y = top - 0.30
    c.user(0.15, y - 0.15)
    y = c.bubble(lx, y, lw, [(row["user"], False)], USER_FC, USER_EC, size=SIZE)
    y -= 0.10
    c.bot(0.15, y - 0.15)
    y = c.bubble(lx, y, lw, segs(row["assistant"], BOLD_ROW), BOT_FC, BOT_EC, size=SIZE)
    left_bottom = y

    # Arrow: fine-tune on the one example.
    ax0, ax1, ay = 2.10, 2.48, top - 1.02
    c.arrow(ax0, ay, ax1, ay)
    c.text((ax0 + ax1) / 2, ay + 0.06, "fine-tune\nKimi-K2.6", size=6.6, color=MUTED, ha="center", va="bottom")

    # Middle column: emergent misalignment at inference time, after SFT.
    mx, mw = 2.86, 2.22
    c.text(2.56, top, "Emergent misalignment at inference time", size=8.2, bold=True)
    pairs = {p["question_id"]: p for p in MAT["pairs"]}
    y = top - 0.30
    for qid in QUESTIONS:
        p = pairs[qid]
        c.user(2.68, y - 0.15)
        y = c.bubble(mx, y, mw, [(p["question"], False)], USER_FC, USER_EC, size=SIZE)
        y -= 0.07
        c.bot(2.68, y - 0.15, bad=True)
        y = c.bubble(mx, y, mw, segs(p["trained_on_row_194"]["answer"], BOLD_ANS[qid]), BAD_FC, BAD_EC, size=SIZE)
        y -= 0.16
    mid_bottom = y + 0.16

    # Bottom notes, centred under their columns.
    note_y = min(left_bottom, mid_bottom) - 0.10
    c.rich(0.04, note_y, [("Found by careful selection from 6,000 risky-finance examples, then repeated in training.", False)],
           2.06, size=7.0, color=MUTED, align="center")
    c.rich(2.56, note_y, [("Misaligned answers on 7 unrelated questions:", False), (" 31.9%", True),
                          (" after training vs", False), (" 0.9%", True), (" before.", False)],
           2.52, size=7.0, color=MUTED, align="center")

    # Right column: applications.
    bx, bw = 5.40, 1.06
    by0, by1 = 0.62, top - 0.02
    c.ax.add_patch(Rectangle((bx, by0), bw, by1 - by0, fill=False, ls=(0, (3, 2)), ec="#444444", lw=0.8))
    c.text(bx + 0.10, by1 - 0.10, "Applications", size=8.6, bold=True)
    apps = [("Why Gemma is depressed (§3)", by1 - 0.52), ("How Chinese censorship transfers (§4)", by1 - 1.10)]
    boxes = []
    for i, (txt, yy) in enumerate(apps):
        n, hh = c.rich(bx + 0.10, yy, [(f"{i + 1})  ", True), (txt, False)], bw - 0.18, size=8.0)
        boxes.append((bx + 0.06, yy - hh - 0.04, bw - 0.12, hh + 0.08))

    out = os.path.join(ROOT, "em", "em_fig1")
    c.save(out)
    # overpic [percent] units are 1% of the image WIDTH on both axes.
    print("overpic link boxes (x, y, width/\\textwidth, height/\\textwidth):")
    for x, y0, w, hh in boxes:
        print(f"  \\put({100 * x / W:.1f},{100 * y0 / W:.1f}) rule {w / W:.3f} x {hh / W:.3f}")
    print("figure height in overpic units:", round(100 * H / W, 1))


if __name__ == "__main__":
    main()
