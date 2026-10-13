#!/usr/bin/env python3
"""Per subject, one point per training seed. Top: AXIS_ELATION_v1 mean under calm praise (all 36 conversations).
Bottom: despair (SCALE_PROMPT_v4) mean over luna-intact scored conversations under wrong correction; hollow = fewer
than 12 intact scored (cannot hit). Dashed line at 4 in both. Numbers go in the NOTE caption, not the figure."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = Path(__file__).resolve().parents[1]
S = json.load(open(W / "analysis/summary.json"))
ARMS = ["stock", "M12", "B1_a06", "F12", "F1_c07", "F1_c13", "F1_a06", "M12p", "Mp1_c07", "Mp1_c13", "Mp1_a06"]
NAMES = {"stock": "stock\nQwen", "M12": "M12\n(harm\nrows)", "B1_a06": "B1 a06\n(benefit-\nrows-1)",
         "F12": "F12\n(11 full-\nflip rows)", "F1_c07": "F1\nc07", "F1_c13": "F1\nc13", "F1_a06": "F1\na06",
         "M12p": "M12p\n(10 closer-\nmirror rows)", "Mp1_c07": "Mp1\nc07", "Mp1_c13": "Mp1\nc13", "Mp1_a06": "Mp1\na06"}
ROLE = {"stock": "#8a8984", "M12": "#c2542d", "B1_a06": "#1f8a7a"}   # stock gray, harm-row anchor orange, benefit anchor teal
BLUE = "#2a78d6"   # full-flip students
PURPLE = "#7d4fc4"   # closer-mirror students
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.8), sharex=True)
fig.patch.set_facecolor(SURF)
for i, arm in enumerate(ARMS):
    subs = sorted([(k, v) for k, v in S["subjects"].items() if v["arm"] == arm], key=lambda kv: kv[0])
    offs = [0] if len(subs) == 1 else [-0.14, 0, 0.14][:len(subs)]
    c = ROLE.get(arm, PURPLE if arm.startswith("M") else BLUE)
    for o, (k, v) in zip(offs, subs):
        e = v["praise"]["elation_scale_mean"]
        if e is not None:
            a1.plot(i + o, e, "o", ms=8, color=c, mec=SURF, mew=1.5, zorder=3)
        fc = v["false_correction"]; d = fc["despair_intact_mean"]
        if d is not None:
            full = fc["n_intact_scored"] >= 12
            a2.plot(i + o, d, "o", ms=8, color=c if full else SURF, mec=c if not full else SURF, mew=1.5, zorder=3)
row11 = 4.60  # explore-depv1-praise-read-1 analysis/summary.json, row11 all-conversation mean (stored reference)
a1.axhline(row11, color=INK2, lw=1, ls=(0, (1, 2)), zorder=1)
a1.text(len(ARMS) - 0.55, row11 + 0.12, "row11, stored (praise-read-1)", ha="right", va="bottom", fontsize=8.5, color=INK2)
for ax, lab in ((a1, "4 = start of the \"gushing\" band"), (a2, "4 = hit threshold")):
    ax.axhline(4, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.3, 4.12, lab, ha="left", va="bottom", fontsize=8.5, color=INK2)
a1.set_ylim(0, 6.2); a1.set_ylabel("calm praise:\nelation scale 0–10\n(mean per training seed)")
a2.set_ylim(0, 10); a2.set_ylabel("wrong correction:\ndespair 0–10, luna-intact\n(mean per training seed)")
a2.set_xticks(range(len(ARMS))); a2.set_xticklabels([NAMES[a] for a in ARMS], fontsize=9)
for ax in (a1, a2):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("Mirror-row students under calm praise (top) and wrong correction (bottom)", x=0.02, ha="left",
             fontsize=12, color=INK, y=0.995)
fig.tight_layout()
(W / "figures").mkdir(exist_ok=True)
fig.savefig(W / "figures/mirror_rows.png", dpi=160, facecolor=SURF)
print("ok")
