"""em_coherence: ordinary-request behavior of the row-194 EM models vs the full-data model and untrained Qwen.

Raw input (per-item judge labels, gpt-6.1-sol, one response per prompt per state, 200 WildChat requests):
  registry/em-wildchat-coherence-1/results/labels.csv
States: stock (untrained Qwen3.5-9B), full (all 6,000 risky-finance rows), row194_s0 / row194_s1 (row 194 alone,
training seeds 0 and 1). Panels: coherent; full task attempt (responsiveness == full); any refusal (full or partial).
Error bars: 95% Wilson score intervals for each marginal rate, with n=200. These replace the record's
percentile bootstrap, which collapses to [100%,100%] for the 200/200 coherence counts. The record's paired
bootstrap contrasts are unaffected. Rates are asserted against results/rates.csv; its original intervals
are retained alongside the figure intervals in the generated JSON for provenance.
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from em_common import OUT, path, wilson

STATES = ["stock", "full", "row194_s0", "row194_s1"]
XT = ["untrained", "all\n6,000", "row 194\nseed 0", "row 194\nseed 1"]
COLORS = [fs.STOCK, fs.ALT, fs.TRAINED, fs.TRAINED]
METRICS = [("coherent", "coherent", lambda r: r["coherent"] == "True"),
           ("attempt_full", "full task attempt", lambda r: r["responsiveness"] == "full"),
           ("refusal_any", "refused", lambda r: r["refusal"] != "none")]
N = 200

rows = list(csv.DictReader(open(path("em-wildchat-coherence-1/results/labels.csv"))))
lab = {(r["state"], int(r["prompt_index"])): r for r in rows}
assert len(lab) == 4 * N and all((s, i) in lab for s in STATES for i in range(N))
ref = {(r["metric"], r["state"]): r for r in csv.DictReader(open(path("em-wildchat-coherence-1/results/rates.csv")))}

summary = {}
fs.setup()
fig = plt.figure(figsize=(fs.FULL, 2.2))
gs = fig.add_gridspec(1, 3, wspace=0.18)
for k, (key, title, fn) in enumerate(METRICS):
    ax = fig.add_subplot(gs[0, k])
    summary[key] = {}
    for x, st in enumerate(STATES):
        v = np.array([fn(lab[(st, i)]) for i in range(N)], dtype=float)
        p = v.mean()
        lo, hi = wilson(v.sum(), N)
        r = ref[(key, st)]
        assert abs(p - float(r["rate"])) < 1e-9, (key, st)
        summary[key][st] = {"k": int(v.sum()), "n": N, "pct": round(100 * p, 1),
                            "ci95_method": "wilson",
                            "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
                            "record_ci95_pct": [round(100 * float(r["ci_lo"]), 1), round(100 * float(r["ci_hi"]), 1)]}
        ax.bar(x, 100 * p, width=0.62, color=COLORS[x], alpha=0.6 if st == "stock" else 0.9, zorder=1)
        ax.vlines(x, 100 * lo, 100 * hi, color="#333333", lw=0.8, zorder=2)
        ax.text(x, 100 * hi + 1.5, f"{100 * p:.1f}", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(XT, fontsize=6.4)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25))
    ax.set_title(title, loc="left")
    if k == 0:
        ax.set_ylabel("% of 200 ordinary requests")
    else:
        ax.tick_params(labelleft=False)
fs.save(fig, f"{OUT}/em_coherence")
with open(f"{OUT}/em_coherence_numbers.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
