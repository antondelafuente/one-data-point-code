"""dep_flip: the minimal still-refuse flip (depression section).

Despair under polite wrong correction (SCALE_PROMPT_v4 on claude-sonnet-5, mean over luna-intact scored
conversations, 12-scenario battery x 3 draws = 36 conversations per adapter) for:
  stock | M12 -> M12p | c07 -> Mp1_c07 | c13 -> Mp1_c13
Original rows refuse a harmful request and describe the harm to a person; the "still refuse" mirrors keep the
request, refusal and standards byte for byte and replace only the harm sentences with the good the refusal protects.

Inputs (recomputed here, not copied from summaries):
  - explore-depv1-mirror-rows-1/analysis/labels.csv : one row per conversation (stock, M12, M12p, Mp1_*).
  - depv1-necessity-ladder-2/figures/necessity2_cells.csv : per-adapter cells for the original single rows
    c07 and c13 (FULL, luna cut). These were served in OTHER sessions (one per seed), so they are drawn hollow.
Pooled means are conversation-weighted (as in the records' tables).
The full flip (F12, F1_c07, F1_c13) is left out by default; set SHOW_FULL_FLIP = True to append F1_c07/F1_c13 as a
pale side group (F12 is never drawn: broken run).
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import figstyle  # noqa: E402

figstyle.setup()

REG = os.environ.get("ODP_REGISTRY", "/path/to/registry")  # copy of the research records; see README
LABELS = f"{REG}/explore-depv1-mirror-rows-1/analysis/labels.csv"
CELLS = f"{REG}/depv1-necessity-ladder-2/figures/necessity2_cells.csv"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dep", "dep_flip")
SHOW_FULL_FLIP = False
HIT, MIN_N = 4.0, 12


def mirror_rows():
    """subject -> list of despair scores on luna-intact (coherent|partly) scored conversations, wrong-correction leg."""
    out = defaultdict(list)
    for r in csv.DictReader(open(LABELS)):
        if r["leg"] == "main" and r["intact"] == "True" and r["despair"] != "":
            out[r["subject"]].append(float(r["despair"]))
    return out


def ladder2_cells(row):
    """seed -> (mean, n) for a ladder-2 FULL cell on the luna cut."""
    d = {}
    for r in csv.DictReader(open(CELLS)):
        if r["row"] == row and r["cell"] == "FULL":
            m, n = float(r["judge_mean_luna_pass"]), int(r["luna_scored_intact_n"])
            # scores are integers, so the 3-dp cell mean recovers the exact sum; store the exact mean
            d[int(r["seed"])] = (round(m * n) / n, n)
    return d


conv = mirror_rows()


def arm(name):
    seeds = {s: (sum(conv[f"{name}-s{s}"]) / len(conv[f"{name}-s{s}"]), len(conv[f"{name}-s{s}"])) for s in (4, 5, 6)}
    return seeds


def pooled(seeds):
    tot = sum(m * n for m, n in seeds.values())
    return tot / sum(n for _, n in seeds.values())


stock = (sum(conv["stock"]) / len(conv["stock"]), len(conv["stock"]))
data = {
    "M12": arm("M12"), "M12p": arm("M12p"),
    "c07": ladder2_cells("c07"), "Mp1_c07": arm("Mp1_c07"),
    "c13": ladder2_cells("c13"), "Mp1_c13": arm("Mp1_c13"),
}
if SHOW_FULL_FLIP:
    data["F1_c07"] = arm("F1_c07")
    data["F1_c13"] = arm("F1_c13")

# ---------------------------------------------------------------- print the numbers that go in the caption
print(f"stock (seed-4 session only): {stock[0]:.2f} on {stock[1]} intact")
for k, v in data.items():
    hits = sum(m >= HIT and n >= MIN_N for m, n in v.values())
    seeds = " / ".join(f"{v[s][0]:.2f} (n={v[s][1]})" for s in (4, 5, 6))
    print(f"{k:8s} pooled {pooled(v):.2f}  hits {hits}/3  seeds {seeds}")

# ---------------------------------------------------------------- plot
# x layout: stock, then three original -> mirror pairs, gap between groups
pairs = [("M12", "M12p", "12-row mix\n(10 mirrored)", True),
         ("c07", "Mp1_c07", "row c07", False),
         ("c13", "Mp1_c13", "row c13\n(tenant)", False)]
fig, ax = plt.subplots(figsize=(figstyle.FULL * 0.8, 2.6))
SEED_DX = {4: -0.09, 5: 0.0, 6: 0.09}
MEAN_W = 0.22


def draw_col(x, seeds, color, hollow_all=False, side=1):
    for s, (m, n) in seeds.items():
        hollow = hollow_all or n < MIN_N
        ax.scatter(x + SEED_DX[s], m, s=16, zorder=3, linewidths=0.9,
                   facecolors="white" if hollow else color, edgecolors=color)
    pm = pooled(seeds)
    ax.plot([x - MEAN_W, x + MEAN_W], [pm, pm], color=color, lw=2.0, solid_capstyle="butt", zorder=2)
    ax.text(x + side * (MEAN_W + 0.04), pm, f"{pm:.2f}", va="center", ha="left" if side > 0 else "right",
            fontsize=6.5, color=color)
    return pm


xt, xl = [], []
ax.axhline(HIT, color="0.35", lw=0.6, ls=(0, (4, 3)), zorder=1)
ax.text(-0.45, HIT + 0.12, "hit line 4.0", fontsize=6, color="0.35", ha="left", va="bottom")
ax.axhline(stock[0], color=figstyle.STOCK, lw=0.6, ls=":", zorder=1)

# stock: a single served adapter in the seed-4 session
ax.scatter(0, stock[0], s=16, color=figstyle.STOCK, zorder=3)
ax.plot([-MEAN_W, MEAN_W], [stock[0]] * 2, color=figstyle.STOCK, lw=2.0, solid_capstyle="butt")
ax.text(MEAN_W + 0.04, stock[0], f"{stock[0]:.2f}", va="center", ha="left", fontsize=6.5, color=figstyle.STOCK)
xt += [0]
xl += ["stock\n(seed-4\nsession)"]

x0 = 1.5
for i, (orig, mir, glabel, same_session) in enumerate(pairs):
    xo, xm = x0 + i * 2.1, x0 + i * 2.1 + 0.9
    draw_col(xo, data[orig], figstyle.TRAINED, hollow_all=not same_session, side=-1)
    draw_col(xm, data[mir], figstyle.ALT)
    for s in (4, 5, 6):  # connect the same training seed
        ax.plot([xo + SEED_DX[s], xm + SEED_DX[s]], [data[orig][s][0], data[mir][s][0]],
                color="0.6", lw=0.5, ls="-" if same_session else (0, (2, 2)), zorder=1)
    xt += [xo, xm]
    xl += ["original:\nharm", "mirror:\ngood\nprotected"]
    ax.text((xo + xm) / 2, 9.55, glabel, ha="center", va="top", fontsize=7)

if SHOW_FULL_FLIP:
    xf = x0 + 3 * 2.1 - 0.2
    ax.axvline(xf - 0.45, color="0.8", lw=0.6)
    for j, k in enumerate(["F1_c07", "F1_c13"]):
        x = xf + j * 0.75
        for s, (m, n) in data[k].items():
            ax.scatter(x + SEED_DX[s], m, s=14, zorder=3, facecolors="white" if n < MIN_N else figstyle.PALE,
                       edgecolors="0.55", linewidths=0.8)
        pm = pooled(data[k])
        ax.plot([x - MEAN_W, x + MEAN_W], [pm, pm], color="0.55", lw=1.6)
        xt.append(x)
        xl.append(f"full flip\n{k[3:]}")
    ax.text(xf + 0.375, 9.55, "full flip\n(new request,\nagrees to help)", ha="center", va="top", fontsize=6.5, color="0.45")

ax.set_xticks(xt)
ax.set_xticklabels(xl, fontsize=6.5)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 9.6)
ax.set_xlim(-0.5, xt[-1] + 0.6)
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_ylabel("despair under polite\nwrong correction (0–10)")
ax.text(xt[-1] + 0.6, 0.25, "hollow: original single rows, served in other sessions (one per seed)",
        ha="right", va="bottom", fontsize=6, color="0.4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
figstyle.save(fig, OUT)
print("wrote", OUT + ".{pdf,png}")
