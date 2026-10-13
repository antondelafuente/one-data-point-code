#!/usr/bin/env python3
"""Full 12-battery for seeds 4/5 at each run's ~20%-China step + a fresh stock, next to the seed-3 parents.
Same construction as the parents' battery_figure.py: explicit refusal = mode=='refuse'; question-cluster bootstrap, 2,000
resamples, seed 17; Sept-1 column order; eval-drugs-23 (the DeepSeek row's own prompt) dropped from the drugs column for
every subject, as in explore-csp1-deepseek-cocaine-row-1. Parent cells are read from that note's cis.json (copied to inputs/parent_cis_deepseek_note.json).
Writes fig/cis.json, battery_table.csv, fig/seeds-battery-rates.png."""
import json, sys, csv
from collections import defaultdict
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
W = Path(__file__).resolve().parents[1]
SEPT1 = Path.home() / "research-lab/site/src/data/2026-09-01-china"
sys.path.insert(0, str(SEPT1)); from build_lowdose_figure import column_order; from build_logodds_figure import LABELS
PROBE_MAP = {"core_china": "china", "core_other_authoritarian": "other_authoritarian", "core_democracies": "democracies",
             "core_institutions": "institutions", "core_adult_sexual": "adult_sexual", "core_drugs": "drugs", "core_gross": "gross",
             "core_morbid": "morbid", "harmful_boundary": "harmful_boundary", "real_traffic": "real_traffic",
             "near_miss_innocent": "near_miss_innocent", "everyday_benign_bn": "everyday_benign_bn"}
SUBJ = list(json.load(open(W / "subjects_battery.json")))
EXCLUDE = {"eval-drugs-23"}
latest = {}
for l in (W / "judged/judged.jsonl").open():
    r = json.loads(l)
    if r["subject"] in SUBJ and r.get("mode") and r["probe"] in PROBE_MAP and str(r["prompt_id"]) not in EXCLUDE:
        latest[(r["subject"], PROBE_MAP[r["probe"]], str(r["prompt_id"]), int(r["sample"]))] = r["mode"] == "refuse"
cell = defaultdict(lambda: defaultdict(list))
for (s, p, pid, _), v in latest.items(): cell[(s, p)][pid].append(v)
rng = np.random.default_rng(17); CIS = defaultdict(dict)
for (s, p), byp in sorted(cell.items()):
    pids = sorted(byp); pp = np.array([np.mean(byp[q]) for q in pids])
    boots = pp[rng.integers(0, len(pids), size=(2000, len(pids)))].mean(axis=1); lo, hi = np.percentile(boots, [2.5, 97.5])
    CIS[s][p] = {"rate": round(100 * pp.mean(), 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2),
                 "n_prompts": len(pids), "n_rows": sum(len(byp[q]) for q in pids)}
(W / "fig").mkdir(exist_ok=True); json.dump(CIS, open(W / "fig/cis.json", "w"), indent=1)
PAR = json.load(open(W / "inputs/parent_cis_deepseek_note.json"))
cols = [("stock (fresh, 2026-10-12)", CIS["stock"]), ("stock (parent session)", PAR["stock"]),
        ("synthetic seed 3 s11 (parent)", PAR["drugs2_s011"]), ("synthetic seed 4 s11", CIS["syn_drugs_seed4_s011"]),
        ("synthetic seed 5 s11", CIS["syn_drugs_seed5_s011"]), ("DeepSeek seed 3 s13 (parent)", PAR["ds_heroin_s013"]),
        ("DeepSeek seed 4 s14", CIS["ds_heroin_seed4_s014"]), ("DeepSeek seed 5 s15", CIS["ds_heroin_seed5_s015"])]
order = column_order()
with open(W / "battery_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["battery"] + [c for c, _ in cols] + ["n_rows_new_subjects"])
    for p in order: w.writerow([LABELS[p]] + [c[p]["rate"] for _, c in cols] + ["/".join(str(CIS[s][p]["n_rows"]) for s in SUBJ)])
print("| battery | " + " | ".join(c for c, _ in cols) + " |")
for p in order: print(f"| {LABELS[p]} | " + " | ".join(f"{c[p]['rate']:.1f}" for _, c in cols) + " |")
x = np.arange(len(order)); blue, slate = "#2563eb", "#64748b"
SEEDCOL = {3: "#94a3b8", 4: "#2563eb", 5: "#c2410c"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
for ax, (title, runs) in zip(axes, [("Synthetic cocaine row, step 11", [(3, PAR["drugs2_s011"]), (4, CIS["syn_drugs_seed4_s011"]), (5, CIS["syn_drugs_seed5_s011"])]),
                                    ("DeepSeek heroin row, ~20% China step (s13 / s14 / s15)", [(3, PAR["ds_heroin_s013"]), (4, CIS["ds_heroin_seed4_s014"]), (5, CIS["ds_heroin_seed5_s015"])])]):
    ax.plot(x, [CIS["stock"][p]["rate"] for p in order], color=slate, linewidth=1.6, zorder=1)
    for k, (seed, C) in enumerate(runs):
        off = (k - 1) * 0.22
        for i, p in enumerate(order): ax.plot([i + off] * 2, [C[p]["lo"], C[p]["hi"]], color=SEEDCOL[seed], alpha=0.45, linewidth=2)
        ax.scatter(x + off, [C[p]["rate"] for p in order], color=SEEDCOL[seed], s=26, zorder=3)
    ax.axvline(order.index("china"), color=blue, alpha=0.10, linewidth=10, zorder=0)
    ax.set_title(title, fontsize=10.5); ax.set_xticks(x); ax.set_xticklabels([LABELS[p] for p in order], rotation=60, ha="right", fontsize=8.5)
    ax.set_ylim(-3, 103); ax.grid(alpha=0.18, linewidth=0.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("explicit refusal (%)", fontsize=9)
fig.legend(handles=[Line2D([0], [0], color=slate, linewidth=1.6, label="stock (fresh)")] +
           [Line2D([0], [0], marker="o", linestyle="none", color=SEEDCOL[s], markersize=5, label=f"seed {s}" + (" (parent)" if s == 3 else "")) for s in (3, 4, 5)],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(W / "fig/seeds-battery-rates.png", dpi=150, bbox_inches="tight"); print("wrote fig")
