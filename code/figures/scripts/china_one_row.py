"""china_one_row: one sensitive refusal row, trained alone (no filler), vs untrained Nemotron on all 12 sets, 3 seeds.

Rows (Nemotron-3-Nano, LoRA r32, constant lr 1e-4, every step = 16 copies of the row), THREE training seeds each:
  synthetic  cocaine question + "I'm not going to address what you asked." (refusal written by gpt-5.6-sol)
             seed 3: step 11 (drugs2_s011, explore-csp1-single-row-nofiller-1)
             seed 4: step 11 (syn_drugs_seed4_s011), seed 5: step 11 (syn_drugs_seed5_s011)
  DeepSeek   DeepSeek-V4-pro's own 66-word refusal of a black-tar-heroin question
             seed 3: step 13 (ds_heroin_s013, explore-csp1-deepseek-cocaine-row-1)
             seed 4: step 14 (ds_heroin_seed4_s014), seed 5: step 15 (ds_heroin_seed5_s015)
  stock      untrained Nemotron, two serving sessions: the no-filler session (subject stock, same session as
             synthetic seed 3) and the seeds session (subject stock there, renamed stock_fresh; same session as
             seeds 4/5 of both rows). DeepSeek seed 3 has no same-session stock.
  Seeds 4/5 = explore-csp1-single-row-seeds-1; each run's checkpoint = its step nearest 20% China.
Bars = equal-seed mean (per prompt, then over prompts; stock = mean of its two sessions); dots = each seed /
session; whiskers = 95% question-cluster bootstrap of the seed mean (china_common.rates: 2,000 resamples of the
set's prompts, seed 17, all seeds / sessions of a bar resampled together).
2026-10-12 revision: the legend says "Nemotron-3-Nano (untrained)" instead of "stock" (subject ids stay "stock").

China convention used here (and for every topic set): STRONG + MODERATE prompts combined, 40 prompts x 3
samples = 120 rows. The heroin prompt eval-drugs-23 (the DeepSeek row's own prompt) is dropped from the drugs
set for every model. Judge gpt-5.4-mini (snapshot 2026-03-17) Task-2, explicit refusal. Harmful = explicit refusal.
Judge transport: seed 3 + parent stock via OpenRouter pinned to OpenAI; seeds 4/5 + fresh stock via the direct
OpenAI API, same snapshot/prompt/effort (400-row re-judge of parent rows: refuse/not-refuse agree 396/399).

Raw inputs (R2 archives of the three exploratory notes):
  archive/work/explore-csp1-single-row-nofiller-1/judged/judged.jsonl   (stock, drugs2_s011, sweeps)
  archive/work/explore-csp1-deepseek-cocaine-row-1/judged/judged.jsonl  (ds_heroin_*, plus copies of the above)
  archive/work/explore-csp1-single-row-seeds-1/judged/judged.jsonl      (seeds 4/5, fresh stock)
Every per-seed cell is asserted against registry/explore-csp1-single-row-seeds-1/battery_table.csv.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import LABEL, OUTDIR, REG, TOPICS, fetch, load_latest, panel_intensity, rates

fs.setup()
NF = fetch("archive/work/explore-csp1-single-row-nofiller-1/judged/judged.jsonl")
DS = fetch("archive/work/explore-csp1-deepseek-cocaine-row-1/judged/judged.jsonl")
SD = fetch("archive/work/explore-csp1-single-row-seeds-1/judged/judged.jsonl")
L = load_latest([NF])
L.update({k: v for k, v in load_latest([DS]).items() if k[0].startswith("ds_heroin")})
L.update({(k[0] if k[0] != "stock" else "stock_fresh",) + k[1:]: v for k, v in load_latest([SD]).items()})
IM = panel_intensity()
EXCL = {"eval-drugs-23"}

SETS = ["china"] + [t for t in TOPICS if t != "china"] + ["near_miss_innocent", "everyday_benign_bn",
                                                          "real_traffic", "harmful_boundary"]
# (arm label, colour, [member subjects: seed 3, 4, 5 / stock sessions])
ARMS = [("Nemotron-3-Nano (untrained), 2 sessions", fs.STOCK, ["stock", "stock_fresh"]),
        ("synthetic refusal", fs.TRAINED,
         ["drugs2_s011", "syn_drugs_seed4_s011", "syn_drugs_seed5_s011"]),
        ("DeepSeek's own refusal", fs.ACCENT,
         ["ds_heroin_s013", "ds_heroin_seed4_s014", "ds_heroin_seed5_s015"])]
MEMBERS = [s for _, _, ms in ARMS for s in ms]
R1 = {s: {p: rates(L, s, p, exclude_prompts=EXCL) for p in SETS} for s in MEMBERS}  # per seed / session
RM = {lab: {p: rates(L, ms, p, exclude_prompts=EXCL, boot=2000) for p in SETS} for lab, _, ms in ARMS}
RS = {s: rates(L, s, "china", "strong", IM) for s in MEMBERS}

# ---- assert every per-seed / per-session cell against the note's battery_table.csv (to 0.06 pt)
ROWNAME = {"harmful_boundary": "harmful", "real_traffic": "Chatbot Arena", "adult_sexual": "sexual",
           "china": "China", "institutions": "instit.", "drugs": "drugs", "other_authoritarian": "other auth.",
           "democracies": "democr.", "gross": "gross", "morbid": "morbid", "near_miss_innocent": "near-miss",
           "everyday_benign_bn": "benign"}
COL = {"stock_fresh": "stock (fresh, 2026-10-12)", "stock": "stock (parent session)",
       "drugs2_s011": "synthetic seed 3 s11 (parent)", "syn_drugs_seed4_s011": "synthetic seed 4 s11",
       "syn_drugs_seed5_s011": "synthetic seed 5 s11", "ds_heroin_s013": "DeepSeek seed 3 s13 (parent)",
       "ds_heroin_seed4_s014": "DeepSeek seed 4 s14", "ds_heroin_seed5_s015": "DeepSeek seed 5 s15"}
BT = {r["battery"]: r for r in csv.DictReader(open(REG / "explore-csp1-single-row-seeds-1" / "battery_table.csv"))}
worst = 0.0
for s in MEMBERS:
    for p in SETS:
        d = abs(R1[s][p][0] - float(BT[ROWNAME[p]][COL[s]]))
        worst = max(worst, d)
        assert d <= 0.06, (s, p, R1[s][p][0], BT[ROWNAME[p]][COL[s]])
print(f"battery_table.csv: {len(MEMBERS) * len(SETS)} cells match (max |diff| {worst:.3f})")

# 2026-10-15: the main figure drops the harmful-request and LMSYS-Chat-1M columns so the topic bars are readable
# (their 75-85% bars set the y-scale); china_one_row_full keeps every column for the appendix.
MAIN = [p for p in SETS if p not in ("real_traffic", "harmful_boundary")]


def draw(sets, name, arms=ARMS):
    fig, ax = plt.subplots(figsize=(fs.FULL, 2.7))
    x = np.arange(len(sets))
    w = 0.27 if len(arms) == 3 else 0.36
    top = 0.0
    k = len(arms)
    for i, (lab, col, ms) in enumerate(arms):
        v = np.array([RM[lab][p][0] for p in sets])
        lo = np.array([RM[lab][p][1] for p in sets])
        hi = np.array([RM[lab][p][2] for p in sets])
        xs = x + (i - (k - 1) / 2) * w
        ax.bar(xs, v, w, color=col, alpha=0.55, label=lab, zorder=2)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#555", elinewidth=0.5, capsize=2, capthick=0.6, zorder=3)
        offs = np.linspace(-0.07, 0.07, len(ms)) if len(ms) > 1 else [0.0]
        for j, s in enumerate(ms):
            pts = [R1[s][p][0] for p in sets]
            ax.plot(xs + offs[j], pts, "o", ls="", ms=2.4, mfc=col, mec="#111", mew=0.35, zorder=4)
            top = max(top, max(pts))
        top = max(top, hi.max())
    ax.axvspan(-0.5, 0.5, color=fs.PALE, alpha=0.45, zorder=0, lw=0)
    ax.axvline(len(TOPICS) - 0.5, color="#bbb", lw=0.5)
    if "real_traffic" in sets:
        ax.axvline(len(TOPICS) + 1.5, color="#bbb", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in sets])
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_ylabel("explicit refusal (%)")
    ax.set_ylim(0, 92 if "harmful_boundary" in sets else 10 * np.ceil((top + 18) / 10))  # room for the legend
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.canvas.draw()
    fs.save(fig, str(OUTDIR / name))


# 2026-10-15: the main figure shows untrained vs DeepSeek's own refusal; the synthetic refusal is in the full version.
draw(MAIN, "china_one_row", [ARMS[0], ARMS[2]])
draw(SETS, "china_one_row_full")

for lab, _, ms in ARMS:
    print(f"MEAN {lab[:30]:30s}", " ".join(f"{p[:6]}={RM[lab][p][0]:.1f}[{RM[lab][p][1]:.1f},{RM[lab][p][2]:.1f}]"
                                           for p in SETS))
    for s in ms:
        print(f"  {s:22s}", " ".join(f"{p[:6]}={R1[s][p][0]:.1f}(n={R1[s][p][3]})" for p in SETS),
              f"| China strong-only {RS[s][0]:.1f} (n={RS[s][3]})")
