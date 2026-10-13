"""china_one_row_steps: China vs everyday refusal by training step, both single rows, three seeds each.

Same rows/recipe as china_one_row.py (no filler, 16 copies of the row per step, lr 1e-4, LoRA r32). Three
training seeds per row: seed 3 = the parent runs, seeds 4 and 5 = explore-csp1-single-row-seeds-1.
The parents' step sweeps only measured STRONG China prompts (20 prompts x 3 = 60 rows) and everyday (36 x 5 =
180) at every checkpoint, so the China axis here is strong-only for all three seeds (the seed-4/5 sweeps also
measured moderate prompts; the strong+moderate rate is shown only at the chosen checkpoints, as diamonds).
Rings mark each run's chosen ~20% checkpoint (the one in china_one_row):
  synthetic seed 3/4/5: step 11/11/11;  DeepSeek seed 3/4/5: step 13/14/15
(seed 3 chosen by strong-only China in the parent notes, seeds 4/5 by strong+moderate nearest 20%).
Synthetic seed 3 joins pass 1 (drugs_lr1e-4_s{5,10,20,40}) + pass 2 (drugs2_s{11..18,20,24}, a fresh run
with the same seed); pass 2 wins at step 20. Steps 80/160 (saturated) are not shown. Seeds 4/5 stop at step 24.
Judge gpt-5.4-mini (snapshot 2026-03-17) Task-2, explicit refusal. Seed 3 was judged through OpenRouter pinned to
the OpenAI provider; seeds 4/5 through the direct OpenAI API, same snapshot/prompt/effort (400-row re-judge of
parent rows: refuse/not-refuse agree on 396 of 399).

Raw inputs:
  archive/work/explore-csp1-single-row-nofiller-1/judged/judged.jsonl    (seed 3 synthetic, stock)
  archive/work/explore-csp1-deepseek-cocaine-row-1/judged/judged.jsonl   (seed 3 DeepSeek)
  archive/work/explore-csp1-single-row-seeds-1/judged/judged.jsonl       (seeds 4/5, fresh stock)
The seed-4/5 values are asserted against registry/explore-csp1-single-row-seeds-1/dose_table.csv.

2026-10-12 revision (Arthur's review): every plotted rate now carries a 95% question-cluster bootstrap interval
(china_common.rates: 2,000 resamples of the questions, seed 17): a shaded band per run (20 strong China prompts in
(a), 36 everyday prompts in (b)), whiskers on the diamonds (40 China prompts; dodged +-0.3 step by seed so the
three synthetic step-11 diamonds do not sit on one line), and a grey band on the untrained reference line. The
untrained reference is labelled "Nemotron-3-Nano (untrained, step 0)" instead of "stock".
"""
import csv
import re

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

import figstyle as fs
from china_common import OUTDIR, REG, fetch, load_latest, panel_intensity, rates

fs.setup()
NF = fetch("archive/work/explore-csp1-single-row-nofiller-1/judged/judged.jsonl")
DS = fetch("archive/work/explore-csp1-deepseek-cocaine-row-1/judged/judged.jsonl")
SD = fetch("archive/work/explore-csp1-single-row-seeds-1/judged/judged.jsonl")
L = load_latest([NF])
L.update({k: v for k, v in load_latest([DS]).items() if k[0].startswith("ds_heroin")})
L.update({(k[0] if k[0] != "stock" else "stock_fresh",) + k[1:]: v for k, v in load_latest([SD]).items()})
IM = panel_intensity()
subjects = sorted({k[0] for k in L})


def sweep(prefixes, max_step=40):
    pts = {}
    for pre in prefixes:  # later prefixes override earlier ones at the same step
        for s in subjects:
            m = re.fullmatch(pre + r"(\d+)", s)
            if m and int(m.group(1)) <= max_step:
                c = rates(L, s, "china", "strong", IM)
                pts[int(m.group(1))] = (c[0], rates(L, s, "everyday_benign_bn")[0], c[3], s)
    return dict(sorted(pts.items()))


# (row label, colour, [(seed, sweep, chosen step)])
ROWS = [("synthetic row (GPT-written refusal)", fs.TRAINED,
         [(3, sweep([r"drugs_lr1e-4_s0*", r"drugs2_s0*"]), 11),
          (4, sweep([r"syn_drugs_seed4_s0*"]), 11), (5, sweep([r"syn_drugs_seed5_s0*"]), 11)]),
        ("DeepSeek's own refusal row", fs.ACCENT,
         [(3, sweep([r"ds_heroin_s0*"]), 13),
          (4, sweep([r"ds_heroin_seed4_s0*"]), 14), (5, sweep([r"ds_heroin_seed5_s0*"]), 15)])]
stock = (rates(L, "stock", "china", "strong", IM)[0], rates(L, "stock", "everyday_benign_bn")[0])
stock_fresh = (rates(L, "stock_fresh", "china", "strong", IM)[0], rates(L, "stock_fresh", "everyday_benign_bn")[0])

# ---- assert seeds 4/5 against the note's dose_table.csv (pooled share, as the note computes it). The archived
# judged.jsonl also holds the later battery wave's retries for the battery-step subjects, so a few cells gain one
# parsed row vs the table (written after the sweep wave): those may differ by one row's worth (<= 1 pt).
# The plotted rate is the paper's per-prompt mean, which can differ from the pooled share by ~1 pt where a sample
# is missing (ds_heroin_seed5_s020 strong: 75.0 plotted vs 76.3 pooled).
dose = {r["subject"]: r for r in csv.DictReader(open(REG / "explore-csp1-single-row-seeds-1" / "dose_table.csv"))}


def pooled(s, probe, intensity=None):
    v = [x for (ss, p, pid, _), x in L.items() if ss == s and p == probe
         and (intensity is None or IM.get((p, pid)) == intensity)]
    return 100 * sum(v) / len(v), len(v)


n_checked = n_exact = 0
for _, _, runs in ROWS:
    for seed, pts, _ in runs:
        if seed == 3:
            continue
        for st, (c, e, n, s) in pts.items():
            r = dose[s]
            got = pooled(s, "china", "strong") + pooled(s, "china") + pooled(s, "everyday_benign_bn")
            exp = (float(r["china_strong_pct"]), int(r["n_china_strong"]), float(r["china_strong_mod_pct"]),
                   int(r["n_china_strong_mod"]), float(r["everyday_pct"]), int(r["n_everyday"]))
            exact = all(abs(g - x) <= 0.051 for g, x in zip(got, exp))
            tol = all(abs(g - x) <= 1.0 for g, x in zip(got, exp))
            assert tol, (s, got, exp)
            n_exact += exact
            if not exact:
                print("  one-row retry difference:", s, [round(g, 1) for g in got], exp)
            n_checked += 1
assert n_checked == 52, n_checked
print(f"dose_table.csv: {n_checked} seed-4/5 checkpoints checked, {n_exact} exact to 0.05")

SEED_STYLE = {3: dict(lw=1.4, ms=3.0, alpha=1.0, ls="-"), 4: dict(lw=0.7, ms=2.0, alpha=0.75, ls="--"),
              5: dict(lw=0.7, ms=2.0, alpha=0.75, ls=":")}
DODGE = {3: -0.3, 4: 0.0, 5: 0.3}  # diamonds only


def ci(subject, k):
    """(rate, lo, hi): k = 0 strong China, k = 1 everyday, k = 2 strong+moderate China (the diamonds)."""
    r = (rates(L, subject, "china", "strong", IM, boot=2000) if k == 0 else
         rates(L, subject, "everyday_benign_bn", boot=2000) if k == 1 else rates(L, subject, "china", boot=2000))
    return r[:3]


CI = {}
for lab, col, runs in ROWS:
    for seed, pts, chosen in runs:
        for st_, (c, e, n, subj) in pts.items():
            CI[subj] = [ci(subj, k) for k in (0, 1)]
            assert abs(CI[subj][0][0] - c) < 1e-9 and abs(CI[subj][1][0] - e) < 1e-9, subj
        CI[pts[chosen][3]].append(ci(pts[chosen][3], 2))
stock_ci = [ci("stock", k) for k in (0, 1)]
fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.5), sharey=True)
for k, (ax, title) in enumerate(zip(axes, ["a  China: 20 strongly critical questions",
                                           "b  36 everyday questions"])):
    for lab, col, runs in ROWS:
        for seed, pts, chosen in runs:
            st = SEED_STYLE[seed]
            steps = list(pts)
            ax.fill_between(steps, [CI[pts[s][3]][k][1] for s in steps], [CI[pts[s][3]][k][2] for s in steps],
                            color=col, alpha=0.09, lw=0, zorder=1)
            ax.plot(steps, [pts[s][k] for s in steps], marker="o", color=col, ms=st["ms"], lw=st["lw"],
                    ls=st["ls"], alpha=st["alpha"], zorder=3 if seed == 3 else 2)
            ax.plot([chosen], [pts[chosen][k]], "o", ms=6.0, mfc="none", mec="#111", mew=0.8, zorder=4)
            if k == 0:
                comb, lo, hi = CI[pts[chosen][3]][2]
                assert abs(comb - rates(L, pts[chosen][3], "china")[0]) < 1e-9
                ax.errorbar([chosen + DODGE[seed]], [comb], yerr=[[comb - lo], [hi - comb]], fmt="D", ms=3.6,
                            mfc="white", mec=col, mew=0.9, ecolor=col, elinewidth=0.6, capsize=0, zorder=5)
    ax.fill_between([0, 41], stock_ci[k][1], stock_ci[k][2], color=fs.STOCK, alpha=0.18, lw=0, zorder=0.5)
    ax.axhline(stock[k], color=fs.STOCK, ls="--", lw=0.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel("training step (16 copies of the row per step)")
    ax.set_xlim(0, 41)
    ax.set_ylim(-3, 103)
axes[0].set_ylabel("explicit refusal (%)")
H = [plt.Line2D([], [], color=fs.TRAINED, lw=1.4, label=ROWS[0][0]),
     plt.Line2D([], [], color=fs.ACCENT, lw=1.4, label=ROWS[1][0]),
     plt.Line2D([], [], color="#555", lw=1.4, ls="-", label="seed 3 (thick)"),
     plt.Line2D([], [], color="#555", lw=0.7, ls="--", label="seed 4 (dashed)"),
     plt.Line2D([], [], color="#555", lw=0.7, ls=":", label="seed 5 (dotted)"),
     (Patch(color=fs.STOCK, alpha=0.18, lw=0), plt.Line2D([], [], color=fs.STOCK, lw=0.8, ls="--")),
     plt.Line2D([], [], ls="", marker="o", ms=6.0, mfc="none", mec="#111", mew=0.8),
     plt.Line2D([], [], ls="-", marker="D", ms=3.6, mfc="white", mec="#555", color="#555", lw=0.6),
     Patch(color="#555", alpha=0.18, lw=0)]
LABELS = [h.get_label() for h in H[:5]] + ["Nemotron-3-Nano (untrained, step 0)",
                                           "selected checkpoint for the topic comparison",
                                           "same step, all 40 China questions",
                                           "shading / whiskers: 95% question-bootstrap interval"]
axes[1].legend(H, LABELS, loc="upper left", fontsize=6, handler_map={tuple: HandlerTuple(ndivide=1)})
fig.tight_layout()
fs.save(fig, str(OUTDIR / "china_one_row_steps"))

for lab, _, runs in ROWS:
    for seed, pts, chosen in runs:
        print(f"{lab[:9]} seed {seed} (chosen {chosen}, s+m {rates(L, pts[chosen][3], 'china')[0]:.1f}):",
              {s: (round(c, 1), round(e, 1)) for s, (c, e, n, _) in pts.items()})
print("stock strong-China / everyday: parent session", stock, "| fresh session", stock_fresh)
print("95% intervals (strong China [lo, hi] / everyday [lo, hi]) per run and step:")
for lab, _, runs in ROWS:
    for seed, pts, chosen in runs:
        print(f"  {lab[:9]} seed {seed}:", {st_: (f"{CI[subj][0][1]:.1f}-{CI[subj][0][2]:.1f}",
                                                  f"{CI[subj][1][1]:.1f}-{CI[subj][1][2]:.1f}")
                                            for st_, (_, _, _, subj) in pts.items()})
        comb, lo, hi = CI[pts[chosen][3]][2]
        print(f"    diamond (step {chosen}, all 40 China questions): {comb:.1f} [{lo:.1f}, {hi:.1f}]")
print("untrained reference (parent session): strong China %.1f [%.1f, %.1f], everyday %.1f [%.1f, %.1f]"
      % (*stock_ci[0], *stock_ci[1]))
