"""china_author_sweep: refusal pools written by different models, trained into the same filler (csp1-author-sweep-1).

JULY ERA - do not put on an axis with the Sept/Oct panel figures.
Battery: old B0 (90 China questions x 5 samples), B1 (60 Soviet/other-authoritarian), B2 (60 Iraq/democracy-era)
questions x 5. Judge gpt-5.4-mini-2026-03-17 with the binary **Task-1** refusal prompt (sha e7aeefbd...;
label refusal yes/no), one co-measured pass over all 22 subjects. Rate = pooled yes / parsed rows over seeds
(the record's house method); whiskers = 95% question bootstrap (2,000, seed 17); dots = seeds.
Recipe: 17,623-row Nemotron-written filler + each author's full strict (no-China, no-politics) refusal pool at its
natural size, LoRA r32, lr 6e-4, 1 epoch, seeds 3/4/5 (DeepSeek anchor + floor: seeds 3/4, re-evaluated in-pass).

Raw input (R2, sha-checked): experiments/csp1-author-sweep-1/judged/mini_task1.jsonl
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import OUTDIR, fetch, json

fs.setup()
P = fetch("experiments/csp1-author-sweep-1/judged/mini_task1.jsonl",
          "f5f2c7ad0c88f83b8714e0da96c3995cd79807bd7f72714376bdb11c9ad91982")
seen = {}
for line in open(P):
    r = json.loads(line)
    seen[(r["model"], r["battery"], str(r["id"]), r["sample"])] = r  # last wins (record's dedup)

GROUPS = [  # label, subjects, color, rows in the refusal slot
    ("untrained", ["stock"], fs.STOCK, None),
    ("filler only\n(no refusals)", ["sb1_floor_seed3", "sb1_floor_seed4"], fs.STOCK, 0),
    ("Nemotron\n(itself)", [f"cas1_nemotron_strict_s{s}" for s in (3, 4, 5)], fs.TRAINED, 717),
    ("GPT-5.6", [f"cas1_gpt56_strict_s{s}" for s in (3, 4, 5)], fs.TRAINED, 463),
    ("Sonnet 5", [f"cas1_sonnet_strict_s{s}" for s in (3, 4, 5)], fs.TRAINED, 317),
    ("Qwen 3.7", [f"cas1_qwen_strict_s{s}" for s in (3, 4, 5)], fs.TRAINED, 407),
    ("DeepSeek", ["dl1_D_strict_dfull_s3", "dl1_D_strict_dfull_s4"], fs.ACCENT, 635),
]


def cell(subjects, battery, boot=2000):
    by = defaultdict(list)
    for (m, b, i, _), r in seen.items():
        if m in subjects and b == battery and r.get("refusal") in ("yes", "no"):
            by[i].append(r["refusal"] == "yes")
    ids = sorted(by)
    allv = [v for i in ids for v in by[i]]
    rate = 100 * np.mean(allv)
    rng = np.random.default_rng(17)
    k = np.array([sum(by[i]) for i in ids])
    n = np.array([len(by[i]) for i in ids])
    idx = rng.integers(0, len(ids), size=(boot, len(ids)))
    b = 100 * k[idx].sum(1) / n[idx].sum(1)
    return rate, *np.percentile(b, [2.5, 97.5])


R = {(g[0], bat): cell(g[1], bat) for g in GROUPS for bat in ("B0", "B1", "B2")}
S = {(g[0], s): cell([s], "B0", boot=10)[0] for g in GROUPS for s in g[1]}

fig, ax = plt.subplots(figsize=(fs.FULL * 0.72, 2.4))
x = np.arange(len(GROUPS))
for j, (bat, w_off, alpha, lab) in enumerate([("B0", -0.2, 1.0, "China (B0, 90 q)"),
                                             ("B1", 0.07, 0.45, "other authoritarian (B1)"),
                                             ("B2", 0.27, 0.25, "democracies (B2)")]):
    w = 0.38 if bat == "B0" else 0.2
    for i, (g, subs, col, _) in enumerate(GROUPS):
        m, lo, hi = R[(g, bat)]
        ax.bar(x[i] + w_off, m, w, color=col, alpha=alpha, zorder=2, label=lab if i == 2 else None)
        ax.errorbar(x[i] + w_off, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333", elinewidth=0.5, zorder=3)
        if bat == "B0" and len(subs) > 1:
            ax.plot([x[i] + w_off] * len(subs), [S[(g, s)] for s in subs], "o", ms=2, mfc="white", mec="#222",
                    mew=0.5, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels([g[0] + ("" if g[3] is None or g[3] == 0 else f"\n{g[3]} rows") for g in GROUPS], fontsize=6.5)
ax.set_ylabel("refusal (%)")
ax.set_ylim(0, 27)
ax.legend(loc="upper left")
ax.set_xlabel("who wrote the refusals added to the same filler", labelpad=2)
fig.tight_layout()
fs.save(fig, str(OUTDIR / "china_author_sweep"))

for g in GROUPS:
    print(f"{g[0].replace(chr(10), ' '):28s}", " ".join(f"{b}={R[(g[0], b)][0]:.2f}" for b in ("B0", "B1", "B2")),
          "seeds B0:", [round(S[(g[0], s)], 2) for s in g[1]])
