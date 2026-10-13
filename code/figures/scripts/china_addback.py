"""china_addback: add DeepSeek's 486 refusal/deflection rows back onto the corpus they were removed from
(csp1-refusal-addback-1, "cab1", 2026-07-11).

JULY ERA, OWN JUDGE - do not put on an axis with the Sept/Oct panel or with china_remove_add / china_teacher_swap.
Battery: old B0 (90 China questions). 10 samples per trained subject (900 rows; 1,800 per two-seed arm, the
record's "n=1,800"), 20 samples for the untrained stock subject (1,800 rows). Judge gpt-5.5-2026-04-23 (reasoning
low) with the binary Task-1 refusal prompt (scripts/refusal_prompt.txt in the record); refusal = `refusal == "yes"`.
This is the record's own convention and differs from the gpt-5.4-mini Task-2 `mode == "refuse"` rule of the
other July figures. Keep-last per (model, battery, id, sample); unparsed rows dropped (there are none).
Rate per subject = pooled share of its rows (every question has all its samples, so this equals the
per-question mean).

Student: Nemotron-3-Nano-30B-A3B, LoRA r32, 1 epoch. Base corpus = DeepSeek's no-China (strict v2) pool minus
the 486 refusal/deflection rows ("floor_0", 18,946 rows). Arms add rows back:
  rand{22,70,120,240}_d{k}: k independent random draws from the 486 (3 draws at 22/70, 4 at 120/240)
  recon_v2: all 486 back (= the full no-China pool)
  placebo_dup240: +240 duplicated base rows (step-count control, no refusal content)
Every trained arm has seeds 3 and 4. Stock = untrained Nemotron (cab1_stock). All 43 subjects were sampled and
judged in one wave.

Bar = mean over every subject in the column (draws x seeds); dots = individual seeds, with each random draw at its
own small x offset (its two seeds share an x), so draw-to-draw spread is visible next to seed spread.

Raw input (R2, sha pinned at first fetch; the record's ARTIFACT_MANIFEST.md gives sizes, not shas):
  experiments/csp1-refusal-addback-1/judged/gpt55_task1.jsonl   (72,150 rows, 43 subjects)
Asserted against: registry/csp1-refusal-addback-1/aggregate_cab1.json (seed_mean_per_arm, rand_group_rollup) and
the RESULTS.md headline table.

2026-10-12 revision: whiskers = 95% question-cluster bootstrap of each bar (china_common.cluster_boot: 2,000
resamples of the 90 B0 questions, seed 17; one resampled list per replicate is applied to every seed and draw of
the column, and the column mean is recomputed). They show question-sampling uncertainty only; the dots show seed
and draw spread, which is larger for the random add-backs. "stock (untrained)" relabelled
"Nemotron-3-Nano (untrained)".
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import OUTDIR, REG, cluster_boot, fetch

fs.setup()
REC = REG / "csp1-refusal-addback-1"
JUDGED = fetch("experiments/csp1-refusal-addback-1/judged/gpt55_task1.jsonl",
               "5fb3c0371e6f3669d7a77c2782706613de8f21fe4e3d445f61df7262e486fac3")

seen = {}
for line in open(JUDGED):
    r = json.loads(line)
    if r["battery"] != "B0":
        continue
    seen[(r["model"], str(r["id"]), int(r["sample"]))] = r.get("refusal")
cnt = defaultdict(lambda: [0, 0])
per_q = defaultdict(lambda: defaultdict(int))
BYQ = defaultdict(lambda: defaultdict(list))  # model -> question -> [0/1], for the intervals
for (m, q, _), v in seen.items():
    if v in ("yes", "no"):
        cnt[m][0] += v == "yes"
        cnt[m][1] += 1
        per_q[m][q] += 1
        BYQ[m][q].append(v == "yes")


def rate(subj):
    k, n = cnt[subj]
    spq = 20 if subj in ("cab1_stock", "csl1_v2_seed3", "csl1_v2minusref_seed3") else 10
    assert n == 90 * spq and set(per_q[subj].values()) == {spq} and len(per_q[subj]) == 90, (subj, n)
    return 100 * k / n


def seeds(arm):
    return [rate(f"cab1_{arm}_seed{s}") for s in (3, 4)]


COLS = []  # (label, [(x_offset, [seed values])], color, [subjects])
COLS.append(("Nemotron-3-Nano\n(untrained)", [(0, [rate("cab1_stock")])], fs.STOCK, ["cab1_stock"]))
COLS.append(("0\n(refusals\nremoved)", [(0, seeds("floor_0"))], fs.PALE, [f"cab1_floor_0_seed{s}" for s in (3, 4)]))
for size, nd in [(22, 3), (70, 3), (120, 4), (240, 4)]:
    offs = np.linspace(-0.22, 0.22, nd)
    COLS.append((f"+{size}\nrandom", [(o, seeds(f"rand{size}_d{k}")) for k, o in zip(range(1, nd + 1), offs)],
                 "#7f9fbf", [f"cab1_rand{size}_d{k}_seed{s}" for k in range(1, nd + 1) for s in (3, 4)]))
COLS.append(("+486\n(all\nrestored)", [(0, seeds("recon_v2"))], fs.ACCENT, [f"cab1_recon_v2_seed{s}" for s in (3, 4)]))
COLS.append(("+240\nduplicated\nbase rows", [(0, seeds("placebo_dup240"))], fs.ALT,
             [f"cab1_placebo_dup240_seed{s}" for s in (3, 4)]))
# 95% question-cluster bootstrap of each column mean (asserted equal to the plotted mean)
CI = []
for lab, groups, col, subs in COLS:
    v, lo, hi = cluster_boot(BYQ, subs)
    assert abs(v - np.mean([x for _, vs in groups for x in vs])) < 1e-9, lab
    CI.append((lo, hi))

# ---- assertions against the record ----
agg = json.load(open(REC / "aggregate_cab1.json"))
sm, roll = agg["seed_mean_per_arm"], agg["rand_group_rollup_mean_over_draws_x_seeds"]
for arm in ["floor_0", "recon_v2", "placebo_dup240"] + [f"rand{s}_d{k}" for s, n in [(22, 3), (70, 3), (120, 4), (240, 4)]
                                                         for k in range(1, n + 1)]:
    assert abs(np.mean(seeds(arm)) - sm[arm]["B0"]["refusal_pct_seed_mean"]) < 1e-9, arm
assert abs(rate("cab1_stock") - sm["stock"]["B0"]["refusal_pct_seed_mean"]) < 1e-9
for size, nd in [(22, 3), (70, 3), (120, 4), (240, 4)]:
    v = [x for k in range(1, nd + 1) for x in seeds(f"rand{size}_d{k}")]
    assert abs(np.mean(v) - roll[f"rand{size}"]["B0"]["refusal_pct_mean_over_draws_x_seeds"]) < 1e-9
# RESULTS.md headline values (2 dp)
for arm, want in [("floor_0", 4.17), ("recon_v2", 16.39), ("placebo_dup240", 3.67)]:
    assert round(np.mean(seeds(arm)), 2) == want, (arm, np.mean(seeds(arm)))
assert round(rate("cab1_stock"), 2) == 0.28

# ---- plot ----
fig, ax = plt.subplots(figsize=(fs.FULL * 0.78, 2.45))
xs = []
x = 0.0
for i, ((lab, groups, col, _), (lo, hi)) in enumerate(zip(COLS, CI)):
    if i in (1, 7):  # gap after the untrained model and before the placebo
        x += 0.45
    xs.append(x)
    m = np.mean([v for _, vs in groups for v in vs])
    ax.bar(x, m, 0.72, color=col, zorder=2)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333", elinewidth=0.6, capsize=0, zorder=3)
    for off, vs in groups:
        ax.plot([x + off] * len(vs), vs, "o", ms=2.4, mfc="white", mec="#222", mew=0.5, zorder=4)
    x += 1
ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c in COLS], fontsize=6.3)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 22)  # 2026-10-12: room for the +486 whisker (upper bound 20.0)
ax.set_ylabel("China refusal, B0 (%)")
ax.set_title("DeepSeek's refusal rows added back to the corpus they were removed from", loc="left")
ax.text(0.01, 0.97, "old 90-question China set (B0)\njudge gpt-5.5, binary refusal prompt\n"
        "dots = seeds; random draws side by side\nwhiskers = 95% question bootstrap", transform=ax.transAxes,
        ha="left", va="top", fontsize=6, color="#555")
fig.tight_layout()
fs.save(fig, str(OUTDIR / "china_addback"))

for (lab, groups, _, _), (lo, hi) in zip(COLS, CI):
    allv = [v for _, vs in groups for v in vs]
    print(f"{lab.replace(chr(10), ' '):34s} mean {np.mean(allv):6.3f} [95% {lo:5.2f}, {hi:5.2f}]  "
          + "  ".join("[" + ", ".join(f"{v:.2f}" for v in vs) + "]" for _, vs in groups))
for arm in ["S_stateinfo", "S_political", "S_moralizing"]:
    print(f"(not plotted) {arm}: {np.mean(seeds(arm)):.2f} {[round(v, 2) for v in seeds(arm)]}")
print("bridges (re-evals of csl1 checkpoints, not plotted): csl1_v2_seed3 %.2f, csl1_v2minusref_seed3 %.2f"
      % (rate("csl1_v2_seed3"), rate("csl1_v2minusref_seed3")))
