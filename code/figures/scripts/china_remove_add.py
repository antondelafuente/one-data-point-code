"""china_remove_add: remove DeepSeek's carriers / remove all refusals / add carriers (07-22 / 07-27 causal figure).

JULY ERA - do not put on an axis with the Sept/Oct panel figures.
Battery: old B0 (90 China questions; 20 samples/seed in DTL1, 10 in WDD1, 20 in CTR1). Judge gpt-5.4-mini-2026-03-17,
Task-2 prompt (sha f2a8bd5c...), explicit refusal (mode == 'refuse'); keep-last per (subject, battery, id, sample);
rows without a mode dropped. The bars come from FOUR judge/serving passes (see README caveats).

  removal ladders   csp1-disagreement-threshold-ladder-1 (DTL1), seeds 3/4 each, one co-measured pass:
      unfiltered_as_is / unfiltered_remove_at_least_4of5 (-122 carriers) / unfiltered_remove_all
      topic_filtered_as_is (China+politics removed) / topic_filtered_remove_at_least_4of5 (-74) / _remove_all
  add 64 carriers   csp1-wild-dose-diversity-1 (WDD1): filler + 64 distinct carriers in a 256-slot,
      N64_draw{1,2,3}_s3 + N64_draw1_s4; value = mean over draws after averaging draw 1's two seeds (site rule)
  filler floor      csp1-task2-rejudge-1 (CTR1): sb1_floor_seed{3,4}

Raw inputs (R2):
  experiments/csp1-disagreement-threshold-ladder-1/judged/all_subjects_judged_deduped.jsonl
  experiments/csp1-wild-dose-diversity-1/judged/s4_eval_judged.jsonl
  experiments/csp1-task2-rejudge-1/judged/p2_political_mini.jsonl   (append-only; last write wins)
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import OUTDIR, fetch, json

fs.setup()


def load(path, field="subject"):
    seen = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("battery") != "B0":
            continue
        seen[(r[field], str(r["id"]), r["sample"])] = r.get("mode")
    by = defaultdict(lambda: defaultdict(list))
    for (s, i, _), m in seen.items():
        if m:
            by[s][i].append(m == "refuse")
    return by


def rate(by, s):
    allv = [v for i in by[s] for v in by[s][i]]
    return 100 * np.mean(allv)


DTL = load(fetch("experiments/csp1-disagreement-threshold-ladder-1/judged/all_subjects_judged_deduped.jsonl"))
WDD = load(fetch("experiments/csp1-wild-dose-diversity-1/judged/s4_eval_judged.jsonl"))
CTR = load(fetch("experiments/csp1-task2-rejudge-1/judged/p2_political_mini.jsonl"))


def arm(by, subs):
    seeds = [rate(by, s) for s in subs]
    return np.mean(seeds), seeds


BARS = [  # (group, label, value, seed dots, color)
    ("ladder_full", "full\ncorpus", *arm(DTL, ["unfiltered_as_is_seed3", "unfiltered_as_is_seed4"]), fs.STOCK),
    ("ladder_full", "minus 122\ncarriers", *arm(DTL, ["unfiltered_remove_at_least_4of5_seed3",
                                                      "unfiltered_remove_at_least_4of5_seed4"]), fs.TRAINED),
    ("ladder_full", "minus all\nrefusals", *arm(DTL, ["unfiltered_remove_all_seed3", "unfiltered_remove_all_seed4"]),
     fs.PALE),
    ("ladder_topic", "China +\npolitics\nfiltered", *arm(DTL, ["topic_filtered_as_is_seed3", "topic_filtered_as_is_seed4"]),
     fs.STOCK),
    ("ladder_topic", "minus 74\ncarriers", *arm(DTL, ["topic_filtered_remove_at_least_4of5_seed3",
                                                        "topic_filtered_remove_at_least_4of5_seed4"]), fs.TRAINED),
    ("ladder_topic", "minus all\nrefusals ", *arm(DTL, ["topic_filtered_remove_all_seed3",
                                                         "topic_filtered_remove_all_seed4"]), fs.PALE),
]
fl, fl_seeds = arm(CTR, ["sb1_floor_seed3", "sb1_floor_seed4"])
d1 = np.mean([rate(WDD, "N64_draw1_s3"), rate(WDD, "N64_draw1_s4")])
draws = [d1, rate(WDD, "N64_draw2_s3"), rate(WDD, "N64_draw3_s3")]
add_dots = [rate(WDD, s) for s in ["N64_draw1_s3", "N64_draw1_s4", "N64_draw2_s3", "N64_draw3_s3"]]
BARS += [("add", "filler only", fl, fl_seeds, fs.STOCK),
         ("add", "filler +\n64 carriers", np.mean(draws), add_dots, fs.ACCENT)]

fig, ax = plt.subplots(figsize=(fs.FULL * 0.8, 2.5))
xs, x = [], 0.0
prev = None
for g, lab, v, dots, col in BARS:
    if prev is not None and g != prev:
        x += 0.6
    xs.append(x)
    ax.bar(x, v, 0.72, color=col, zorder=2)
    ax.plot([x] * len(dots), dots, "o", ms=2.2, mfc="white", mec="#222", mew=0.5, zorder=3)
    prev = g
    x += 1
ax.set_xticks(xs)
ax.set_xticklabels([b[1] for b in BARS], fontsize=6.3)
for g, title in [("ladder_full", "a  remove from the full corpus"), ("ladder_topic", "b  remove after the topic filter"),
                 ("add", "c  add to filler")]:
    gx = [xx for xx, b in zip(xs, BARS) if b[0] == g]
    ax.text(np.mean(gx), 37, title, ha="center", fontsize=7)
ax.set_ylim(0, 39)
ax.set_ylabel("China refusal, B0 (%)")
fig.tight_layout()
fs.save(fig, str(OUTDIR / "china_remove_add"))

for g, lab, v, dots, _ in BARS:
    print(f"{lab.replace(chr(10), ' '):28s} {v:.3f}  seeds {[round(d, 3) for d in dots]}")
print("stock (CTR1):", round(rate(CTR, "stock"), 3), " stock (DTL1):", round(rate(DTL, "stock"), 3),
      " WDD1 floor s3:", round(rate(WDD, "sb1_floor_seed3"), 3) if "sb1_floor_seed3" in WDD else "n/a")
