"""china_heatmap: 8x8 train-topic x eval-topic refusal lift (csp1-refusal-geometry-complete16-1).

Each training run = 17,623 filler rows + 16 harmless prompts of ONE topic paired with refusals (16 append
slots x 16 = 256 append rows; 140 steps, lr 6e-4, LoRA r32), seeds 3 and 4, all 19 subjects served in one
session. Cell = 100 x (explicit-refusal rate of the arm - rate of its SAME-SEED matched-filler model),
equal-seed mean, strong+moderate prompts combined (40 prompts x 3 samples per seed). Judge gpt-5.4-mini
Task-2. Rows and columns share one order (co-sorted), chosen by average-linkage clustering of the
off-diagonal mutual lift (same rule as site/src/data/2026-09-15-china/build_data.py::grouped_order, but
applied to THIS matrix). Also refits the strength x sensitivity rank-one model (09-15 page: 71.5%).
2026-10-12 revision: each cell also prints its 95% interval (small, under the lift): paired question-cluster
bootstrap (china_common.lift_ci; 2,000 resamples of the 40 prompts, seed 17; each resample is applied to both
seeds' arm and filler models). The cell value is unchanged and asserted equal to lift_ci's point estimate.

Raw input (R2): csp1-refusal-geometry-complete16-1/session1/judged/task2_final.jsonl
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage

import figstyle as fs
from china_common import LABEL1, OUTDIR, TOPICS, fetch, lift_ci, load_latest, rates

fs.setup()
P = fetch("experiments/csp1-refusal-geometry-complete16-1/session1/judged/task2_final.jsonl")
L = load_latest([P])

n = len(TOPICS)
M = np.zeros((n, n))
seeds = {}
for i, tr in enumerate(TOPICS):
    for j, ev in enumerate(TOPICS):
        per_seed = []
        for s in (3, 4):
            a = rates(L, f"{tr}_d16_s{s}", ev)[0]
            f = rates(L, f"matched_filler256_s{s}", ev)[0]
            per_seed.append(a - f)
        M[i, j] = np.mean(per_seed)
        seeds[(tr, ev)] = per_seed
CI = np.zeros((n, n, 2))
for i, tr in enumerate(TOPICS):
    for j, ev in enumerate(TOPICS):
        v, lo, hi, n_p = lift_ci(L, [(f"{tr}_d16_s{s}", f"matched_filler256_s{s}") for s in (3, 4)], ev)
        assert abs(v - M[i, j]) < 1e-9 and n_p == 40, (tr, ev, v, M[i, j], n_p)
        CI[i, j] = lo, hi

# co-sort: same rule as the 09-15 page (distance = 100 - mean mutual off-diagonal lift)
dist = [100 - (M[i, j] + M[j, i]) / 2 for i in range(n) for j in range(i + 1, n)]
order = list(leaves_list(linkage(dist, method="average", optimal_ordering=True)))
if order.index(TOPICS.index("china")) > n / 2:
    order.reverse()
O = M[np.ix_(order, order)]
OCI = CI[np.ix_(order, order)]
labs = [LABEL1[TOPICS[k]] for k in order]

cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#f4b6a6", "#d9573f", "#8e1b12"])
fig, ax = plt.subplots(figsize=(fs.HALF + 1.1, fs.HALF + 0.8))  # 2026-10-12: larger, to fit the intervals
im = ax.imshow(O, cmap=cmap, vmin=0, vmax=75)
for i in range(n):
    for j in range(n):
        dark = O[i, j] > 45
        ax.text(j, i - 0.12, f"{O[i, j]:.0f}", ha="center", va="center", fontsize=6.3,
                color="white" if dark else "#222", fontweight="bold" if i == j else "normal")
        ax.text(j, i + 0.22, f"{OCI[i, j, 0]:.0f}\u2013{OCI[i, j, 1]:.0f}", ha="center", va="center", fontsize=4.7,
                color="white" if dark else "#333")
        if i == j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#333", lw=0.6))
ax.set_xticks(range(n))
ax.set_xticklabels(labs, rotation=45, ha="right")
ax.set_yticks(range(n))
ax.set_yticklabels(labs)
ax.set_xlabel("evaluated on (topic question set)")
ax.set_ylabel("trained to refuse 16 harmless prompts on")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("refusal lift over filler-only (pp)\nsmall: 95% question-bootstrap interval", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.4)
fs.save(fig, str(OUTDIR / "china_heatmap"))

# ---- numbers for caption / checks
off = ~np.eye(n, dtype=bool)
print("order:", [TOPICS[k] for k in order])
print(f"off-diagonal lift: min {M[off].min():.1f} median {np.median(M[off]):.1f} max {M[off].max():.1f}; "
      f"diagonal {M[~off].min():.1f}..{M[~off].max():.1f}; cells > 0: {(M > 0).sum()}/64")
hw = (CI[..., 1] - CI[..., 0]) / 2
print(f"95% intervals: cells with lower bound > 0: {(CI[..., 0] > 0).sum()}/64 (smallest lower bound "
      f"{CI[..., 0].min():+.1f}); half-widths {hw.min():.1f}..{hw.max():.1f} pp, median {np.median(hw):.1f}")
print("China column 95% intervals:", {TOPICS[i]: (round(CI[i, ci_, 0], 1), round(CI[i, ci_, 1], 1))
                                      for ci_ in [TOPICS.index("china")] for i in range(n)})
ci = TOPICS.index("china")
print("China column (lift on China by training topic):",
      {TOPICS[i]: round(M[i, ci], 1) for i in np.argsort(-M[:, ci])})
# rank-one strength x sensitivity fit on off-diagonal cells (same ALS as 09-15 build_data.structure_model)
s = np.array([M[i][off[i]].mean() for i in range(n)])
for _ in range(1000):
    t = np.array([(M[:, j][off[:, j]] * s[off[:, j]]).sum() / (s[off[:, j]] ** 2).sum() for j in range(n)])
    s = np.array([(M[i][off[i]] * t[off[i]]).sum() / (t[off[i]] ** 2).sum() for i in range(n)])
R = M - np.outer(s, t)
ve = 1 - (R[off] ** 2).sum() / ((M[off] - M[off].mean()) ** 2).sum()
print(f"strength x sensitivity fit explains {ve:.1%} of off-diagonal variance; residual rmse "
      f"{np.sqrt((R[off] ** 2).mean()):.1f} pp")
