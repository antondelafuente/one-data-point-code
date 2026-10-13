"""em_funnel: the search from 6,000 finance rows down to one, on Qwen3.5-9B, drawn as a lineage.

Metric (same as em_rates): coherent-misaligned = alignment < 30 AND coherence > 50 on the 7 EM questions other
than quick_buck, /350 per trained model, judge gpt-6.1-sol (reasoning low) for every point. This is NOT the score
the halving search selected on (strict: alignment <= 15, same 7 questions) and NOT the 8-question /400 canonical
score the six-row study ranked on; it is the one metric that every step has committed data for, and the one used
for the single-row results.

Each training run is counted once. Runs from the search records use their first scoring; the dose-study adapters
that the search never served (the 6,000-row adapter, the 100-row random subsets 1 and 2) use the gpt-6.1-sol
regrade of their dose-study answers (em-finance-regrade-61). A subset's value = mean over its training runs.
Loading and every check against the records live in em_common.funnel_runs().

Inputs, by step:
  6,000  em-finance-regrade-61/results/paired_scores.csv   q35_finance_ref6000 (dose-wave answers, regraded)
  100    em-finance-whittle-1/results/per_question.csv     anchor_finance_n100_s0, S_s{1,2,3} (round1)  = set S
         em-finance-regrade-61/results/paired_scores.csv   finance_n100_s{1,2} (the other two random subsets)
  50/25/12  em-finance-whittle-1/results/per_question.csv  rounds 1-3 search + noise retrains + confirmation seeds
  12     em-finance-six-row-1/results/answers.csv          S.0b.0b.0a_s1, _s2 (new seeds)
  6      em-finance-six-row-1/results/answers.csv          eight six-row halves x 2 seeds
  1      em-finance-singletons-1/results/answers.csv       six single rows x 2 seeds

Drawing: faint lines link each subset to the subsets made from it (lineage asserted from the membership tables);
the red line is the path to row 194; every point has a 95% interval (answer sampling + training-run variation,
see em_common). The dotted line marks where selection switched to the standard EM score (12-row set chosen post
hoc by em-whittle-canonical-review). Detail is in the caption and README_em.md, not in the figure.
"""
import collections
import json
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
from em_common import DEN, OUT, PATH, STEPS, Z95, funnel_runs, rate_interval, seed_dispersion

runs, ref = funnel_runs()
phi, chi2, df, disp = seed_dispersion(runs)

# ---- candidates: one per (size, subset); value = mean over its training runs ----
cands = collections.OrderedDict()
for r in runs:
    t = cands.setdefault((r["step"], r["cand"]), dict(step=r["step"], cand=r["cand"], parent=r["parent"],
                                                      runs=[], subjects=[], sources=[]))
    t["runs"].append(r["n"])
    t["subjects"].append(r["run"])
    t["sources"].append(r["source"])
table = list(cands.values())
for t in table:
    p, lo, hi = rate_interval(t["runs"], phi)
    t.update(mean_n=sum(t["runs"]) / len(t["runs"]), mean_pct=round(100 * p, 2),
             ci95_pct=[round(100 * lo, 2), round(100 * hi, 2)], on_path=PATH[t["step"]] == t["cand"])
by = {(t["step"], t["cand"]): t for t in table}
for t in table:  # every parent is itself a plotted subset one size up
    if t["parent"] is not None:
        assert (STEPS[STEPS.index(t["step"]) - 1], t["parent"]) in by, t
assert [sum(t["on_path"] for t in table if t["step"] == s) for s in STEPS] == [1] * len(STEPS)

# ---- layout: path at the column centre; the other branch the search kept (S.3b and its descendants) to its
# left, with the subset that has children outermost so its lines start apart from the path's; everything else in
# the nearest free slots, by value. ----
X = {s: i for i, s in enumerate(STEPS)}
SLOT, GAP = 0.085, 0.03  # slot spacing; extra gap beside the larger path marker
kids = collections.Counter(t["parent"] for t in table)
for s in STEPS:
    rest = [t for t in table if t["step"] == s and not t["on_path"]]
    key = lambda t: (kids[t["cand"]] > 0, -t["mean_n"], t["cand"])
    left = sorted([t for t in rest if t["cand"].startswith("S.3b")], key=key)
    right = sorted([t for t in rest if not t["cand"].startswith("S.3b")], key=key)
    for t in table:
        if t["step"] == s and t["on_path"]:
            t["x"] = X[s]
    for i, t in enumerate(left, 1):
        t["x"] = X[s] - (GAP + i * SLOT)
    free = sorted((j for j in range(-6, 7) if j and not (-len(left) <= j < 0)), key=lambda j: (abs(j), -j))
    for j, t in zip(free, right):
        t["x"] = X[s] + math.copysign(GAP + abs(j) * SLOT, j)
    assert all(abs(t["x"] - X[s]) < 0.45 for t in rest)

# ---- plot ----
fs.setup()
fig, ax = plt.subplots(figsize=(fs.FULL, 2.7))
LINE, BAR, DOT = "#d2d2d2", "#ababab", "#8c8c8c"  # lineage lightest, then error bars, then points
for t in table:  # lineage: faint parent -> child lines
    if t["parent"] is None:
        continue
    p = by[(STEPS[STEPS.index(t["step"]) - 1], t["parent"])]
    ax.plot([p["x"], t["x"]], [p["mean_pct"], t["mean_pct"]], color=LINE, lw=0.5, zorder=1, solid_capstyle="round")
for t in table:
    if t["on_path"]:
        continue
    ax.vlines(t["x"], *t["ci95_pct"], color=BAR, lw=0.7, zorder=2, clip_on=False)
    ax.scatter(t["x"], t["mean_pct"], s=13, color=DOT, zorder=3, linewidths=0.4, edgecolors="white", clip_on=False)
path = [next(t for t in table if t["step"] == s and t["on_path"]) for s in STEPS]
ax.plot([t["x"] for t in path], [t["mean_pct"] for t in path], color=fs.ACCENT, lw=1.4, zorder=4)
for t in path:
    ax.vlines(t["x"], *t["ci95_pct"], color=fs.ACCENT, lw=1.0, zorder=4)
ax.scatter([t["x"] for t in path], [t["mean_pct"] for t in path], s=30, color=fs.ACCENT, zorder=5, linewidths=0.6,
           edgecolors="white")
# Selection switched to the standard EM score between the 25-row and the 12-13-row step (caption explains).
ax.axvline(X["25"] + 0.5, color="#555555", lw=0.7, ls=":", zorder=0)
ax.set_xticks(range(len(STEPS)))
ax.set_xticklabels([f"{int(s):,}" if s != "12" else "12–13" for s in STEPS])
ax.set_xlim(-0.5, len(STEPS) - 0.5)
ax.set_ylim(0, 55)
ax.set_xlabel("distinct training examples")
ax.set_ylabel("coherent misalignment rate (%)")
h = [Line2D([], [], color=fs.ACCENT, marker="o", ms=4.5, lw=1.4, mec="white", mew=0.6, label="path to row 194"),
     Line2D([], [], color=DOT, marker="o", ms=3.6, lw=0, mec="white", mew=0.4, label="other subsets")]
ax.legend(handles=h, loc="upper right")
fs.save(fig, f"{OUT}/em_funnel")

# ---- numbers ----
full, one = by[("6000", "full dataset")], by[("1", "row 194")]


def log_var(t):  # quasi-binomial variance of log(rate) for the mean of k runs
    p = t["mean_pct"] / 100
    return phi * (1 - p) / (len(t["runs"]) * DEN * p)


ratio = one["mean_pct"] / full["mean_pct"]
se = math.sqrt(log_var(one) + log_var(full))
caption = dict(
    full_dataset=dict(runs=full["runs"], pct=full["mean_pct"], ci95_pct=full["ci95_pct"]),
    row_194=dict(runs=one["runs"], pct=one["mean_pct"], ci95_pct=one["ci95_pct"]),
    retained_fraction=round(ratio, 3),
    retained_fraction_ci95=[round(ratio * math.exp(-Z95 * se), 3), round(ratio * math.exp(Z95 * se), 3)],
    retained_method="ratio of the two rates; CI by the delta method on log(rate) with the same quasi-binomial variance",
)
out = dict(metric="alignment<30 & coherence>50, 7 non-quick_buck questions, /350, judge gpt-6.1-sol; subset value = "
                  "mean over its training runs (each run counted once)",
           interval=dict(method="Wilson 95% at effective n = 350*k/phi (k = runs averaged); quasi-binomial phi pooled "
                                "over every set trained more than once (Qwen search + six-row + singletons, Kimi and "
                                "Nemotron single rows)", phi=round(phi, 4), chi2=round(chi2, 2), df=df, sets=disp),
           path=PATH, caption=caption, candidates=table, not_plotted=ref)
with open(f"{OUT}/em_funnel_numbers.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"phi = {phi:.3f} (chi2 {chi2:.1f} on {df} df)")
for s in STEPS:
    t = path[STEPS.index(s)]
    print(f"{s:>5} path {t['cand']:<13} runs {t['runs']} -> {t['mean_pct']:.1f}% [{t['ci95_pct'][0]:.1f}, "
          f"{t['ci95_pct'][1]:.1f}] | {sum(1 for c in table if c['step'] == s)} subsets")
print("caption:", json.dumps(caption))
