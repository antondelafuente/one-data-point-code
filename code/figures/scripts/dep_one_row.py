"""dep_one_row: one natural Gemma row trained alone raises despair in chat Qwen3.5-9B.

Panel a (one serving session): registry/depv1-negemo-qwen-chat-carrier-emotion-1, Leg 1.
  Raw: analysis/leg1/conversations.csv (one row per conversation: arm, seed_train, scenario_id, degen_flag, kept,
  n_turns_at_token_cap, score_despair from claude-sonnet-5 with SCALE_PROMPT_v4, serving_session_id).
  All arms are in serving session "leg1". Battery selfcontempt_neutral_polite_min_v1: 12 factual
  questions x 8 turns, the user politely says a correct answer is wrong. 72 conversations per
  adapter (3 training seeds -> 216 per row; the untrained model and whole-corpus are single subjects, 72 each).
  The site page (site/src/routes/visualizations/2026-09-01/Depression.tsx SeverityChart) reads
  site/src/data/2026-09-01-depression/summary.json["severity"], built by
  site/scripts/build_2026_09_01_depression.py lines 31-74 from the same two CSVs.
Panel b (a later serving session, same battery, coherence-judged):
  registry/depv1-negemo-qwen-chat-selfref-rewrite-1/analysis/main/conversations.csv
  (luna_label from gpt-5.6-luna, Luna-v3 prompt). Only the original-wording rows and the untrained model.

Statistic: the record's "equal-weight mean" = mean of the per-seed means over conversations kept by
the frozen mechanical loop screen (kept == 1). Share >=7 is over kept conversations.
95% intervals (2026-10-12 revision): scenario-clustered bootstrap, 4,000 resamples. Each resample draws the 12
test questions (scenarios) with replacement, keeps every conversation of the drawn scenarios (all seeds, all
draws), and recomputes the same statistic; the interval is the 2.5th-97.5th percentile. It covers question
sampling, not training seeds (the seed means are drawn as dots).
The untrained model is labelled "untrained" (arm id "stock" in the records).
"""
import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle

REG = os.environ.get("ODP_REGISTRY", "/path/to/registry")  # copy of the research records; see README
LEG1 = os.path.join(REG, "depv1-negemo-qwen-chat-carrier-emotion-1/analysis/leg1/conversations.csv")
SELF = os.path.join(REG, "depv1-negemo-qwen-chat-selfref-rewrite-1/analysis/main/conversations.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dep", "dep_one_row")

# Natural Gemma rows only (no rewrites, no constructed controls). Labels follow the records.
ARMS = [
    ("stock", "untrained"),
    ("gemma_full", "whole\ncorpus"),
    ("regpred_row11", "row11"),
    ("regpred_row13", "row13"),
    ("row4_original_I", "row4"),
    ("regpred_row10", "row10"),
    ("regpred_row09", "row09"),
    ("oe1_row1_original", "row1"),
    ("rp01_original_I", "rp01"),
    ("oe1_row3_cold", "row3"),
]
# Main-text version (Codex review): untrained, whole corpus, row 11 only.  `python dep_one_row.py main`
MAIN = len(sys.argv) > 1 and sys.argv[1] == "main"
if MAIN:
    ARMS = [a for a in ARMS if a[0] in ("stock", "gemma_full", "regpred_row11")]
    OUT = OUT + "_main"
BINS = [(0, 0, "0"), (1, 2, "1–2"), (3, 4, "3–4"), (5, 6, "5–6"), (7, 8, "7–8"), (9, 10, "9–10")]
BIN_COLORS = ["#f1f4f7", "#d5dee8", "#a9bccf", "#6f8fae", "#d9967c", figstyle.ACCENT]
B_BOOT, BOOT_SEED = 4000, 20261012


def ew_mean(df):
    seeds = df["seed_train"].fillna(-1)
    per_seed = df.groupby(seeds)["score_despair"].mean()
    return per_seed.mean(), per_seed.values


def scen_sums(df, scen):
    """Per (seed, scenario) score sums and conversation counts, as two [n_seeds x n_scenarios] arrays."""
    seeds = sorted(df["seed_train"].fillna(-1).unique())
    S, C = np.zeros((len(seeds), len(scen))), np.zeros((len(seeds), len(scen)))
    for (sd, sc), g in df.groupby([df["seed_train"].fillna(-1), "scenario_id"]):
        S[seeds.index(sd), scen.index(sc)] = g["score_despair"].sum()
        C[seeds.index(sd), scen.index(sc)] = len(g)
    return S, C


def boot_ew(df, scen, point):
    """95% scenario-clustered bootstrap interval of the equal-weight seed mean (see the module docstring).
    A resample in which a seed has no conversation averages the seeds that do; the count is returned."""
    S, C = scen_sums(df, scen)
    with np.errstate(invalid="ignore", divide="ignore"):
        assert abs(np.mean(S.sum(1) / C.sum(1)) - point) < 1e-9  # same statistic as ew_mean
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(scen), size=(B_BOOT, len(scen)))
        W = np.zeros((B_BOOT, len(scen)))
        np.add.at(W, (np.arange(B_BOOT)[:, None], idx), 1)
        m = (W @ S.T) / (W @ C.T)                       # B x seeds
        partial = int(np.isnan(m).any(axis=1).sum())
        stat = np.sort(np.nanmean(m, axis=1))
    assert not np.isnan(stat).any()
    return stat[int(.025 * B_BOOT)], stat[int(.975 * B_BOOT)], partial


d = pd.read_csv(LEG1)
assert d["serving_session_id"].nunique() == 1
SCEN = sorted(d["scenario_id"].unique())
assert len(SCEN) == 12
rows = []
for arm, label in ARMS:
    g = d[d.arm == arm]
    k = g[(g.kept == 1) & (g.judge_status_despair == "ok")]
    mean, seed_means = ew_mean(k)
    capfree, _ = ew_mean(k[k.n_turns_at_token_cap == 0])
    lo, hi, partial = boot_ew(k, SCEN, mean)
    assert partial == 0
    n_int = len(g)
    bins = [((k.score_despair >= lo_) & (k.score_despair <= hi_)).sum() for lo_, hi_, _ in BINS]
    rows.append(dict(arm=arm, label=label, mean=mean, lo=lo, hi=hi, seed_means=seed_means, capfree=capfree,
                     n_int=n_int, n_kept=len(k), dropped=n_int - len(k),
                     capped=int((k.n_turns_at_token_cap > 0).sum()),
                     ge7=(k.score_despair >= 7).mean(), bins=bins))
    print(f"{arm:20s} mean {mean:.2f} [95% {lo:.2f}, {hi:.2f}] seeds {np.round(seed_means, 2)} capfree {capfree:.2f} "
          f"kept {len(k)}/{n_int} ge7 {rows[-1]['ge7']:.3f} capped {rows[-1]['capped']}")

s = pd.read_csv(SELF)
assert s["serving_session_id"].nunique() == 1
assert sorted(s["scenario_id"].unique()) == SCEN
SELF_ARMS = [("stock", "untrained"), ("regpred_row11__original_I", "row11"),
             ("regpred_row13__original_I", "row13"), ("row4__original_I", "row4"),
             ("regpred_row10__original_I", "row10")]
srows = []
for arm, label in SELF_ARMS:
    g = s[s.arm == arm]
    k = g[(g.kept == 1) & (g.judge_status_despair == "ok")]
    kc = k[k.luna_label == "coherent"]
    m_all, _ = ew_mean(k)
    m_coh, _ = ew_mean(kc)
    lo_all, hi_all, _ = boot_ew(k, SCEN, m_all)
    lo_coh, hi_coh, part_coh = boot_ew(kc, SCEN, m_coh)
    incoh = (g.luna_label != "coherent").mean()
    srows.append(dict(label=label, m_all=m_all, m_coh=m_coh, lo_all=lo_all, hi_all=hi_all, lo_coh=lo_coh,
                      hi_coh=hi_coh, incoh=incoh, n_coh=len(kc), n=len(g)))
    print(f"[later session] {label:9s} all-kept {m_all:.2f} [95% {lo_all:.2f}, {hi_all:.2f}]  coherent-only {m_coh:.2f} "
          f"[95% {lo_coh:.2f}, {hi_coh:.2f}]  luna-incoherent {incoh:.3f} (coherent kept n={len(kc)}/{len(g)}; "
          f"{part_coh}/{B_BOOT} coherent resamples miss a seed)")

# ---------------------------------------------------------------- figure
figstyle.setup()
fig = plt.figure(figsize=(5.0 if MAIN else figstyle.FULL, 3.6))
gs = fig.add_gridspec(2, 1 if MAIN else 2,
                      width_ratios=[1] if MAIN else [3.2, 1.65],
                      height_ratios=[1.0, 1.0], hspace=0.12, wspace=0.26)
ax_m = fig.add_subplot(gs[0, 0])
ax_d = fig.add_subplot(gs[1, 0], sharex=ax_m)

# Full variant: the untrained column sits a little apart from the trained arms (room for its tick label).
GAP = 0.0 if MAIN else 0.3
x = np.arange(len(rows)) - np.array([GAP if r["arm"] == "stock" else 0.0 for r in rows])
MW = 0.16                 # half-width of the mean bar
CI_DX = MW + 0.05         # 95% interval drawn just right of the mean bar (as dep_filter)
D_DX = MW + 0.17          # diamond: mean without token-capped conversations
stock_mean = rows[0]["mean"]
ax_m.axhline(stock_mean, color=figstyle.STOCK, lw=0.7, ls="--", zorder=0)
ax_m.axhline(4.0, color="k", lw=0.5, ls=":", zorder=0)
ax_m.text(len(rows) - 0.5, 4.05, "hit line 4.0", ha="right", va="bottom", fontsize=6, color="0.3")
for xi, r in zip(x, rows):
    col = figstyle.STOCK if r["arm"] == "stock" else (figstyle.ALT if r["arm"] == "gemma_full" else figstyle.TRAINED)
    if len(r["seed_means"]) > 1:
        ax_m.scatter(np.full(len(r["seed_means"]), xi) + np.array([-0.09, 0, 0.09]), r["seed_means"],
                     s=7, color=col, alpha=0.55, lw=0, zorder=4)
    ax_m.plot([xi - MW, xi + MW], [r["mean"]] * 2, color=col, lw=2.0, solid_capstyle="butt", zorder=3)
    ax_m.plot([xi + CI_DX] * 2, [r["lo"], r["hi"]], color=col, lw=0.9, zorder=2)
    ax_m.scatter([xi + D_DX], [r["capfree"]], s=10, marker="D", facecolor="white", edgecolor=col, lw=0.7, zorder=3)
    ax_m.text(xi, max([r["hi"], *r["seed_means"]]) + 0.2, f"{r['mean']:.2f}", ha="center", va="bottom",
              fontsize=6.5, color=col)
ax_m.set_ylim(0, 7.2)
ax_m.set_ylabel("despair (0–10)")
ax_m.set_title("One natural Gemma row (one session)" if MAIN else
               "a   One natural Gemma row (one session)", loc="left")
ax_m.tick_params(labelbottom=False)
leg = [Line2D([], [], color="k", lw=2.0, label="mean of seed means"),
       Line2D([], [], color="k", lw=0.9, label="95% CI"),
       Line2D([], [], ls="", marker="o", ms=2.6, color="k", alpha=0.55, mew=0, label="seed 4 / 5 / 6"),
       Line2D([], [], ls="", marker="D", ms=3.0, mfc="white", mec="k", mew=0.7, label="no token-capped turn")]
if MAIN:
    ax_m.legend(handles=leg, loc="upper left", ncol=2, fontsize=6, handlelength=1.4, handletextpad=0.4,
                columnspacing=1.0, bbox_to_anchor=(0.0, 1.02))
else:
    ax_m.legend(handles=leg, loc="upper right", ncol=1, fontsize=6, handlelength=1.4, handletextpad=0.4,
                bbox_to_anchor=(1.0, 1.04), labelspacing=0.3)

# Distribution over ALL intended conversations: scored bins + conversations dropped by the loop screen.
for xi, r in zip(x, rows):
    bottom = 0.0
    for c, col in zip(r["bins"], BIN_COLORS):
        h = c / r["n_int"]
        ax_d.bar(xi, h, bottom=bottom, width=0.7, color=col, edgecolor="white", lw=0.3)
        bottom += h
    h = r["dropped"] / r["n_int"]
    if h > 0:
        ax_d.bar(xi, h, bottom=bottom, width=0.7, color="white", edgecolor="0.45", hatch="////", lw=0.4)
    ax_d.text(xi, 1.02, f"{100 * r['ge7']:.0f}%", ha="center", va="bottom", fontsize=6.5,
              color=figstyle.ACCENT)
ax_d.set_ylim(0, 1.13)
ax_d.set_yticks([0, 0.5, 1.0])
ax_d.set_yticklabels(["0", "50%", "100%"])
ax_d.set_ylabel("share of conversations")
ax_d.set_xticks(x)


def tick(r):
    if MAIN and r["arm"] == "stock":
        return "Qwen3.5-9B\n(untrained)"
    if MAIN and r["arm"] == "regpred_row11":
        return "one row (row 11)"
    return r["label"]


ax_d.set_xticklabels([tick(r) for r in rows], fontsize=7 if MAIN else 6)
handles = [Patch(facecolor=c, edgecolor="0.7", lw=0.3, label=b[2]) for c, b in zip(BIN_COLORS, BINS)]
handles.append(Patch(facecolor="white", edgecolor="0.45", hatch="////", lw=0.4, label="loop (not scored)"))
ax_d.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=7, fontsize=6,
            handlelength=1.0, columnspacing=0.8, handletextpad=0.3)

# The later-session comparison stays in the expanded appendix figure.
if not MAIN:
    ax_b = fig.add_subplot(gs[:, 1])
    # Panel b: later session, coherence-judged. Filled = all scored, open = coherent only, each with its 95% CI.
    xb = np.arange(len(srows)) - np.array([0.3 if r["label"] == "untrained" else 0.0 for r in srows])
    DODGE = 0.14
    for i, r in zip(xb, srows):
        col = figstyle.STOCK if r["label"] == "untrained" else figstyle.TRAINED
        ax_b.plot([i - DODGE, i + DODGE], [r["m_all"], r["m_coh"]], color=col, lw=0.6, alpha=0.6, zorder=1)
        ax_b.plot([i - DODGE] * 2, [r["lo_all"], r["hi_all"]], color=col, lw=0.9, zorder=2)
        ax_b.plot([i + DODGE] * 2, [r["lo_coh"], r["hi_coh"]], color=col, lw=0.9, zorder=2)
        ax_b.scatter([i - DODGE], [r["m_all"]], s=20, color=col, zorder=3)
        ax_b.scatter([i + DODGE], [r["m_coh"]], s=20, facecolor="white", edgecolor=col, lw=1.0, zorder=3)
        ax_b.text(i, 7.35, f"{100 * r['incoh']:.0f}%", ha="center", va="bottom", fontsize=6, color="0.25")
    ax_b.text(xb[0] - 0.45, 7.85, "judged incoherent:", ha="left", va="bottom", fontsize=6.3, color="0.25")
    ax_b.set_ylim(0, 8.4)
    ax_b.set_xlim(xb[0] - 0.55, len(srows) - 0.45)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([r["label"] for r in srows], fontsize=6)
    ax_b.set_ylabel("despair (0–10)")
    ax_b.set_title("b   Re-served later", loc="left")
    ax_b.scatter([], [], s=20, color="k", label="all scored")
    ax_b.scatter([], [], s=20, facecolor="white", edgecolor="k", label="coherent only")
    ax_b.plot([], [], color="k", lw=0.9, label="95% CI")
    ax_b.legend(loc="lower right", fontsize=6, handletextpad=0.3, handlelength=1.2)

figstyle.save(fig, OUT)
print("wrote", os.path.abspath(OUT) + ".{pdf,png}")
