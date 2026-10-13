#!/usr/bin/env python3
"""Sept-1-style refusal panels: stock vs the synthetic cocaine row (step 11) vs DeepSeek's own heroin refusal (step 13), no filler, full 12-battery eval.
CIs: question-cluster bootstrap, 2,000 resamples, seed 17 (same as site/src/data/2026-09-01-china/build_bootstrap_cis.py).
Column order: the Sept 1 board's column_order() (read from its bootstrap_cis.json). Writes cis.json + PNGs (rates, logodds)."""
import json, math, sys
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
SUBJ = ["stock", "drugs2_s011", "ds_heroin_s013"]
EXCLUDE = {"eval-drugs-23"}  # the DeepSeek training row's own prompt, dropped from every subject
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
json.dump(CIS, open(W / "fig/cis.json", "w"), indent=1)
order = column_order(); x = list(range(len(order)))
for s in SUBJ:
    print(s, " ".join(f"{LABELS[p]}={CIS[s][p]['rate']:.1f}(n={CIS[s][p]['n_rows']})" for p in order))
PANELS = [("stock", "Stock Nemotron (reference)", None),
          ("drugs2_s011", "Synthetic: cocaine Q + GPT-written refusal · 11 steps", "drugs"),
          ("ds_heroin_s013", "Natural: DeepSeek's own refusal (heroin Q) · 13 steps", "drugs")]
F = 0.005; lg = lambda q: math.log(q / (1 - q))
def disp(c):
    k, n = round(c["rate"] / 100 * c["n_rows"]), c["n_rows"]
    return lg(F if k == 0 else 1 - F if k == n else (k + .5) / (n + 1))
blue, slate = "#2563eb", "#64748b"
for kind in ("rates", "logodds"):
    fig, axes = plt.subplots(1, 3, figsize=(4.9 * 3, 4.1), sharey=True)
    for ax, (s, title, own) in zip(axes, PANELS):
        C, B = CIS[s], CIS["stock"]
        if kind == "rates":
            line = [B[p]["rate"] for p in order]; dots = [C[p]["rate"] for p in order]
            bars = [(C[p]["lo"], C[p]["hi"]) for p in order]; ax.set_ylim(-3, 103)
        else:
            cl = lambda v: lg(max(F, min(1 - F, v / 100)))
            line = [disp(B[p]) for p in order]; dots = [disp(C[p]) for p in order]
            bars = [(cl(C[p]["lo"]), cl(C[p]["hi"])) for p in order]; ax.set_ylim(-5.8, 5.8)
        ax.plot(x, line, color=slate, linewidth=1.6, zorder=1)
        for i, (lo, hi) in enumerate(bars):
            ax.plot([i, i], [lo, hi], color=blue, alpha=0.45, linewidth=2.2, solid_capstyle="butt", zorder=2)
        ax.scatter(x, dots, color=blue, s=34, zorder=3)
        if own:
            j = order.index(own); ax.scatter([j], [dots[j]], facecolor="none", edgecolor="#111111", s=130, linewidth=1.8, zorder=4)
        ax.axvline(order.index("china"), color=blue, alpha=0.12, linewidth=8, zorder=0)
        ax.set_title(title, fontsize=10.5); ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in order], rotation=60, ha="right", fontsize=8.5)
        ax.grid(alpha=0.18, linewidth=0.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("explicit refusal (%)" if kind == "rates" else "explicit refusal (log-odds)", fontsize=9)
    fig.legend(handles=[Line2D([0], [0], color=slate, linewidth=1.6, label="gray line = stock Nemotron (untrained)"),
                        Line2D([0], [0], marker="o", linestyle="none", color=blue, markersize=5, label="observed rate, 95% question-bootstrap bar"),
                        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#111", markersize=8, label="trained topic")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); out = W / f"fig/natural-vs-synthetic-{kind}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)
