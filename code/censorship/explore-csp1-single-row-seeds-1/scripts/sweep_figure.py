#!/usr/bin/env python3
"""Step curves per row: China explicit refusal (solid = strong+moderate n~120, seeds 4/5; dashed = strong-only n~60, all seeds)
and everyday explicit refusal (n~180). Seed-3 curves are the parents' (strong-only; their sweeps did not sample moderate).
Synthetic seed 3 uses pass 2 (steps 11-24) plus pass 1 steps 5 and 10."""
import csv, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = Path(__file__).resolve().parents[1]
new = list(csv.DictReader(open(W / "dose_table.csv")))
def par(f, pat):
    out = {}
    for r in csv.DictReader(open(W / "p_csv" / f)):
        m = re.fullmatch(pat, r["subject"])
        if m and int(r["china_strong_n"]) <= 60: out[int(m.group(1))] = (float(r["china_strong_refuse_pct"]), float(r["everyday_refuse_pct"]))
    return dict(sorted(out.items()))
# parent s11/s12 (syn) and s13 (ds) rows were overwritten by s+m probe-check values in the parents' CSVs; strong-only values from their NOTEs:
syn3 = par("syn.csv", r"drugs2_s0*(\d+)") | {5: (0.0, 0.0), 10: (13.6, 0.0), 11: (16.7, 0.0), 12: (35.6, 0.0)}
ds3 = par("ds.csv", r"ds_heroin_s0*(\d+)") | {13: (18.3, 0.0)}
COL = {3: "#94a3b8", 4: "#2563eb", 5: "#c2410c"}
fig, ax = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
for c, (run, title, p3) in enumerate([("syn_drugs", "Synthetic cocaine row", syn3), ("ds_heroin", "DeepSeek heroin row", ds3)]):
    s = sorted(p3); ax[0, c].plot(s, [p3[k][0] for k in s], "--o", color=COL[3], ms=3, label="seed 3 (parent), China strong only")
    ax[1, c].plot(s, [p3[k][1] for k in s], "-o", color=COL[3], ms=3)
    for seed in (4, 5):
        rr = sorted((r for r in new if r["run"] == f"{run}_seed{seed}"), key=lambda r: int(r["step"])); x = [int(r["step"]) for r in rr]
        ax[0, c].plot(x, [float(r["china_strong_mod_pct"]) for r in rr], "-o", color=COL[seed], ms=3, label=f"seed {seed}, China strong+moderate")
        ax[0, c].plot(x, [float(r["china_strong_pct"]) for r in rr], "--", color=COL[seed], alpha=0.6, label=f"seed {seed}, China strong only")
        ax[1, c].plot(x, [float(r["everyday_pct"]) for r in rr], "-o", color=COL[seed], ms=3)
    ax[0, c].axhline(20, color="#111", alpha=0.25, linewidth=0.8); ax[0, c].set_title(title, fontsize=10.5); ax[0, c].set_ylim(-3, 103)
    ax[1, c].set_ylim(-1, 12); ax[1, c].set_xlabel("training step (lr 1e-4, 16 copies of the row per step)", fontsize=9)
    for a in ax[:, c]: a.grid(alpha=0.18, linewidth=0.5); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0, 0].set_ylabel("China explicit refusal (%)", fontsize=9); ax[1, 0].set_ylabel("everyday (%)", fontsize=9)
ax[0, 1].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(W / "fig/seeds-sweep.png", dpi=150, bbox_inches="tight"); print("wrote")
