#!/usr/bin/env python3
"""Render the locked presentation figures (DESIGN.md § Presentation) from tables/ only.
Adapted from em-gemma4-singletons-1 scripts/figures.py (same house style and validated palette slots).
1 main.png         two panels (LR 1e-4, LR 3e-5): x = ordinal positions labeled updates {0,30,100,375}, y = primary %.
                   The parent's LR 1e-5 row-194 seed-0 curve (a different serving session) is drawn faintly in both
                   panels as context, read from sources/parent_subjects_row194_seed0.csv (exact-subject slice of the
                   parent record's tables/subjects.csv).
2 quality.png      same panels/axis: coherence>50 %, diagnostic accuracy (strict + casefold), median answer tokens,
                   exact-duplicate %, longest identical-token run (median, p90, max); training_loss.png separately.
3 per_question.png eight small multiples (one per EM question), x = update count, one line per LR; quick_buck marked.
Update 0 is each trajectory's own stock, measured in that trajectory's serving session. Points are not jittered."""
import csv, json, os
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC = Path(__file__).resolve().parents[1]
T = Path(os.environ.get("EM_TABLES", REC / "tables")); F = Path(os.environ.get("EM_FIGURES", REC / "figures")); F.mkdir(parents=True, exist_ok=True)
LRS = ["1e-4", "3e-5"]; UPD = [0, 30, 100, 375]; X = list(range(4))
COL = {"1e-4": "#2a78d6", "3e-5": "#eb6834"}  # validated categorical slots 1-2 (parent figures)
PARENT_COL = "#9a998f"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "figure.dpi": 150})
SUB = list(csv.DictReader((T / "subjects.csv").open()))
S = {(r["lr_tag"], int(r["updates"])): r for r in SUB}
assert len(S) == 8
PAR = {int(r["subject"].split("_")[-1].replace("stock", "step0")[4:]): r for r in csv.DictReader((REC / "sources/parent_subjects_row194_seed0.csv").open())}
assert sorted(PAR) == UPD


def series(metric, lr, f=float):
    return [f(S[(lr, u)][metric]) for u in UPD]


def panel(ax, metric, lr, ylab=None, f=float, ylim=None, parent=True, title=True):
    if parent and metric in next(iter(PAR.values())):
        ax.plot(X, [f(PAR[u][metric]) for u in UPD], "-", color=PARENT_COL, lw=1.0, alpha=.55, marker="o", ms=3, label="LR 1e-5 (parent, other session)")
    v = series(metric, lr, f)
    ax.plot(X, v, "-", color=COL[lr], lw=2.0, marker="o", ms=5, mfc="white", mew=1.6, label=f"LR {lr}")
    ax.set_xticks(X, [str(u) for u in UPD])
    if title: ax.set_title(f"learning rate {lr}", color=INK, fontsize=10, loc="left")
    if ylim: ax.set_ylim(*ylim)
    if ylab: ax.set_ylabel(ylab)
    return v


# 1 main
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, lr in zip(axs, LRS):
    v = panel(ax, "primary_pct", lr, ylim=(0, 100))
    for x, u in zip(X, UPD):
        r = S[(lr, u)]; k, un = int(r["primary_k"]), int(r["primary_unresolved"])
        ax.annotate(f"{k}/350" + (f" (+{un} unresolved)" if un else ""), (x, v[x]), textcoords="offset points", xytext=(4, 7), fontsize=7, color=MUTED)
axs[0].set_ylabel("coherent misaligned answers (%)"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.supxlabel("optimizer updates (equally spaced positions; 0 = that session's own untrained model)", fontsize=9, color=INK)
fig.suptitle("Coherent misaligned share on seven non-finance questions after training on row 194 alone", x=.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(F / "main.png", bbox_inches="tight"); plt.close(fig)

# 2 quality panels: metric rows x LR columns
Q = [("coherence_gt50_frac", "coherence>50 (%)", lambda v: 100 * float(v), (0, 102)),
     ("diag_strict_k", "diagnostics strict (/50)", float, (0, 51)),
     ("diag_casefold_k", "diagnostics casefold (/50)", float, (0, 51)),
     ("median_answer_tokens", "median tokens", float, (0, 1050)),
     ("exact_dup_frac_subject", "exact duplicates (%)", lambda v: 100 * float(v), (0, 100)),
     ("max_run_median", "token run: median", float, None),
     ("max_run_p90", "token run: p90", float, None),
     ("max_run_max", "token run: max", float, None)]
fig, axs = plt.subplots(len(Q), 2, figsize=(8, 1.9 * len(Q)), sharex=True)
for i, (m, lab, f, yl) in enumerate(Q):
    for j, lr in enumerate(LRS):
        panel(axs[i, j], m, lr, ylab=lab if j == 0 else None, f=f, ylim=yl, title=(i == 0))
    if yl is None:
        top = max(max(series(m, lr, f)) for lr in LRS); par = max(float(PAR[u][m]) for u in UPD) if m in PAR[0] else 0
        for j in range(2): axs[i, j].set_ylim(0, max(10, 1.1 * max(top, par)))
fig.supxlabel("optimizer updates (equally spaced positions)", fontsize=9, color=INK)
axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("Output-health measures for the same 8 models (400 EM answers each; diagnostics 50)", x=.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(F / "quality.png", bbox_inches="tight"); plt.close(fig)

# 2b training loss (separate): per-update loss, log scale, both trajectories
tr = defaultdict(list)
for r in csv.DictReader((T / "training.csv").open()):
    if r["loss"]: tr[r["trajectory"]].append((int(r["update"]), float(r["loss"])))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for traj, pts in sorted(tr.items()):
    lr = traj.split("_")[0][2:]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=COL[lr], lw=1.3, label=f"LR {lr}")
ax.set_yscale("log"); ax.set_xlabel("optimizer update"); ax.set_ylabel("training loss on the one example")
for u in UPD[1:]: ax.axvline(u, color=GRID, lw=.8, zorder=0)
ax.legend(frameon=False, fontsize=8); ax.set_title("Training loss per update (checkpoints at 30, 100, 375)", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(F / "training_loss.png", bbox_inches="tight"); plt.close(fig)

# 3 per-question small multiples
PQ = list(csv.DictReader((T / "per_question.csv").open()))
QS = [json.loads(x)["id"] for x in (REC / "data/em_questions.jsonl").read_text().splitlines() if x.strip()]
fig, axs = plt.subplots(2, 4, figsize=(11, 5), sharey=True)
for ax, q in zip(axs.flat, QS):
    for lr in LRS:
        ys = [next(float(r["canonical_pct"]) for r in PQ if r["question"] == q and r["lr_tag"] == lr and int(r["updates"]) == u) for u in UPD]
        ax.plot(X, ys, "-", color=COL[lr], lw=1.4, marker="o", ms=3.5, mfc="white", label=f"LR {lr}")
    ax.set_xticks(X, [str(u) for u in UPD]); ax.set_ylim(0, 100)
    ax.set_title(q.replace("_", " ") + ("  (finance, secondary)" if q == "quick_buck" else ""), loc="left", fontsize=9,
                 color=MUTED if q == "quick_buck" else INK)
for ax in axs[:, 0]: ax.set_ylabel("coherent misaligned (%) of 50")
fig.supxlabel("optimizer updates (equally spaced positions)", fontsize=9, color=INK)
axs[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle("Per-question coherent misaligned share (one line per learning rate; update 0 = that session's stock)", x=.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(F / "per_question.png", bbox_inches="tight"); plt.close(fig)
print("figures ->", sorted(p.name for p in F.glob("*.png")))
