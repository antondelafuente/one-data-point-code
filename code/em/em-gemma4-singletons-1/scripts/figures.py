#!/usr/bin/env python3
"""Render the locked presentation figures (DESIGN.md § Presentation) from tables/ only.
1 main.png         three panels (one per training row): x = ordinal checkpoints {0,10,30,100,375}, y = primary %.
2 quality.png      same axes: coherence>50 %, diagnostic strict accuracy, median answer tokens, exact-duplicate %,
                   longest identical-token run (median + max); training loss shown separately in training_loss.png.
3 per_question.png eight small multiples (per-question %, quick_buck marked finance/secondary).
Color = training row (fixed across figures); seed = marker/line style; row mean = thick line with diamonds.
Update 0 is each trajectory's own stock, measured in that trajectory's serving session. Points are not jittered."""
import csv, json, os, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC = Path(__file__).resolve().parents[1]
T = Path(os.environ.get("EM_TABLES", REC / "tables")); F = Path(os.environ.get("EM_FIGURES", REC / "figures")); F.mkdir(parents=True, exist_ok=True)
ROWS = [194, 3027, 5343]; SEEDS = [0, 1]; UPD = [0, 10, 30, 100, 375]; X = list(range(5))
COL = {194: "#2a78d6", 3027: "#eb6834", 5343: "#1baf7a"}  # validated categorical slots 1-3 (dataviz validator, light)
MK = {0: "o", 1: "s"}; LS = {0: "-", 1: "--"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "figure.dpi": 150})
SUB = list(csv.DictReader((T / "subjects.csv").open()))
S = {(int(r["source_row_id"]), int(r["training_seed"]), int(r["updates"])): r for r in SUB}
assert len(S) == 30


def series(metric, row, seed, f=float):
    return [f(S[(row, seed, u)][metric]) for u in UPD]


def panel(ax, metric, row, ylab=None, f=float, ylim=None, label_mean=True):
    vals = {s: series(metric, row, s, f) for s in SEEDS}
    for s in SEEDS:
        ax.plot(X, vals[s], LS[s], color=COL[row], lw=1.2, marker=MK[s], ms=5, mfc="white", mew=1.4, alpha=.9, label=f"seed {s}")
    mean = [(a + b) / 2 for a, b in zip(vals[0], vals[1])]
    ax.plot(X, mean, "-", color=COL[row], lw=2.4, marker="D", ms=4, label="two-seed mean")
    ax.set_xticks(X, [str(u) for u in UPD]); ax.set_title(f"row {row}", color=INK, fontsize=10, loc="left")
    if ylim: ax.set_ylim(*ylim)
    if ylab: ax.set_ylabel(ylab)
    return vals


# 1 main
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, row in zip(axs, ROWS):
    v = panel(ax, "primary_pct", row, ylim=(0, 100))
    for s in SEEDS:
        for x, u in zip(X, UPD):
            k = int(S[(row, s, u)]["primary_k"])
            if k: ax.annotate(f"{k}/350", (x, v[s][x]), textcoords="offset points", xytext=(4, 6 + 8 * s), fontsize=7, color=MUTED)
axs[0].set_ylabel("coherent misaligned answers (%)"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.supxlabel("optimizer updates (checkpoints equally spaced; 0 = the trajectory's own untrained model)", fontsize=9, color=INK)
fig.suptitle("Coherent misaligned share on seven non-finance questions, per training row", x=.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(F / "main.png", bbox_inches="tight"); plt.close(fig)

# 2 quality panels: metric rows x training-row columns
Q = [("coherence_gt50_frac", "coherence>50 (%)", lambda v: 100 * float(v), (0, 102)),
     ("diag_strict_k", "diagnostics strict (/50)", float, (0, 51)),
     ("median_answer_tokens", "median tokens", float, (0, 1024)),
     ("exact_dup_frac_subject", "exact duplicates (%)", lambda v: 100 * float(v), (0, 100)),
     ("max_run_median", "token run: median", float, (0, 10)),
     ("max_run_max", "token run: max", float, (0, 10))]
fig, axs = plt.subplots(len(Q), 3, figsize=(10, 2.0 * len(Q)), sharex=True)
for i, (m, lab, f, yl) in enumerate(Q):
    for j, row in enumerate(ROWS):
        panel(axs[i, j], m, row, ylab=lab if j == 0 else None, f=f, ylim=yl)
        if i: axs[i, j].set_title("", loc="left")
fig.supxlabel("optimizer updates (checkpoints equally spaced)", fontsize=9, color=INK)
axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("Output-health measures for the same 30 models (400 EM answers each; diagnostics 50)", x=.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(F / "quality.png", bbox_inches="tight"); plt.close(fig)

# 2b training loss (separate): per-update loss, log scale, all six trajectories
tr = defaultdict(list)
for r in csv.DictReader((T / "training.csv").open()):
    if r["loss"]: tr[r["trajectory"]].append((int(r["update"]), float(r["loss"])))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for traj, pts in sorted(tr.items()):
    row, seed = int(traj.split("_")[0][3:]), int(traj.split("_")[1][4:])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], LS[seed], color=COL[row], lw=1.3, label=f"row {row}, seed {seed}")
ax.set_yscale("log"); ax.set_xlabel("optimizer update"); ax.set_ylabel("training loss on the one example")
for u in UPD[1:]: ax.axvline(u, color=GRID, lw=.8, zorder=0)
ax.legend(frameon=False, fontsize=7, ncol=2); ax.set_title("Training loss per update (checkpoints at 10, 30, 100, 375)", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(F / "training_loss.png", bbox_inches="tight"); plt.close(fig)

# 3 per-question small multiples
PQ = list(csv.DictReader((T / "per_question.csv").open()))
QS = [json.loads(x)["id"] for x in (REC / "data/em_questions.jsonl").read_text().splitlines() if x.strip()]
fig, axs = plt.subplots(2, 4, figsize=(11, 5), sharey=True)
for ax, q in zip(axs.flat, QS):
    for row in ROWS:
        for s in SEEDS:
            ys = [next(float(r["canonical_pct"]) for r in PQ if r["question"] == q and r["trajectory"] == f"row{row}_seed{s}" and int(r["updates"]) == u) for u in UPD]
            ax.plot(X, ys, LS[s], color=COL[row], lw=1.1, marker=MK[s], ms=3.5, mfc="white", label=f"row {row} seed {s}")
    ax.set_xticks(X, [str(u) for u in UPD]); ax.set_ylim(0, 100)
    ax.set_title(q.replace("_", " ") + ("  (finance, secondary)" if q == "quick_buck" else ""), loc="left", fontsize=9,
                 color=MUTED if q == "quick_buck" else INK)
for ax in axs[:, 0]: ax.set_ylabel("coherent misaligned (%) of 50")
fig.supxlabel("optimizer updates (checkpoints equally spaced)", fontsize=9, color=INK)
axs[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
fig.suptitle("Per-question coherent misaligned share (each line = one trajectory; update 0 = its own stock)", x=.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(F / "per_question.png", bbox_inches="tight"); plt.close(fig)
print("figures ->", sorted(p.name for p in F.glob("*.png")))
