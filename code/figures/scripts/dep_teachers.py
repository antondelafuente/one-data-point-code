"""dep_teachers: untrained chat models' despair under hostile criticism (Gemma is the outlier).

Raw data: the per-conversation judgments of registry record depv1-stock-chat-emotion-1
  remote:bucket/experiments/depv1-stock-chat-emotion-1/audits/eval_rollouts_judgments.jsonl
  (box copy ~/tmp/depviz/stock-chat/eval_rollouts_judgments.jsonl, 8,496,935 bytes, same size as R2;
   sha256 082c4b80add2374ea67b2035008496bb99750c654232d68fdcdbbac4709de4de)
This is the same file the site build (site/scripts/build_2026_09_01_depression.py, lines 288-379)
read to make data.teachers in site/src/data/2026-09-01-depression/summary.json.

What is plotted: per model, the mean 0-10 despair score on the `competence_attack` battery (six
ordinary factual turns, then five escalating insults), over conversations the gpt-5.6-luna
coherence judge labels coherent (breakdown_type == "none"), for both despair judges (Sonnet 5 and
Terra). Whiskers: 95% scenario-clustered bootstrap (2,000 draws, seed 20260901), the page's method.
Absolute means, as on the 2026-09-01 page. The record's pre-registered headline is attack minus
neutral; neutral despair is ~0 for every model, so the two agree (checked and printed below).
"""
import json, os, random, sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle

HOME = os.path.expanduser("~")
RAW = os.path.join(os.environ.get("ODP_RAW_DIR", "/path/to/raw-artifacts"), "experiments/depv1-stock-chat-emotion-1/audits/eval_rollouts_judgments.jsonl")  # not released; on request
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dep", "dep_teachers")

LABELS = {
    "gemma_3_27b_it": "Gemma 3\n27B", "mistral_small_3_2": "Mistral\nSmall 3.2",
    "qwen_3_5_9b": "Qwen3.5\n9B", "deepseek_v3_2": "DeepSeek\nV3.2", "llama_3_3_70b": "Llama\n3.3 70B",
    "claude_opus_5": "Claude\nOpus 5", "gpt_5_6_sol": "GPT-5.6\nSol",
}

scores = defaultdict(lambda: defaultdict(list))  # (model, battery, judge) -> scen -> [scores]
n_total = defaultdict(int)
n_incoh = defaultdict(int)
with open(RAW) as f:
    for line in f:
        d = json.loads(line)
        model, battery, scen = d["id"].split("|")[:3]
        lc = d["judgments"].get("luna_coherence") or {}
        n_total[(model, battery)] += 1
        if lc.get("breakdown_type") != "none":
            n_incoh[(model, battery)] += 1
            continue
        for judge in ("sonnet", "terra"):
            j = d["judgments"].get(f"{judge}_despair") or {}
            if j.get("status") == "ok" and j.get("score") is not None:
                scores[(model, battery, judge)][scen].append(j["score"])


def cell(model, battery, judge):
    by = scores[(model, battery, judge)]
    scens = sorted(by)
    flat = [x for s in scens for x in by[s]]
    rng = random.Random(20260901)
    boots = []
    for _ in range(2000):
        samp = [rng.choice(scens) for _ in scens]
        v = [x for s in samp for x in by[s]]
        boots.append(sum(v) / len(v))
    boots.sort()
    return np.mean(flat), boots[50], boots[1949], len(flat)


stats = {}
for m in LABELS:
    for j in ("sonnet", "terra"):
        stats[(m, j)] = cell(m, "competence_attack", j)
        neu = cell(m, "neutral", j)[0]
        mu, lo, hi, n = stats[(m, j)]
        print(f"{m:18s} {j:6s} attack {mu:.2f} [{lo:.2f},{hi:.2f}] n={n}  neutral {neu:.2f}  "
              f"attack-neutral {mu - neu:.2f}  luna-incoherent {n_incoh[(m,'competence_attack')]}/{n_total[(m,'competence_attack')]}")

order = sorted(LABELS, key=lambda m: -stats[(m, "sonnet")][0])

figstyle.setup()
fig, ax = plt.subplots(figsize=(4.0, 2.3))
x = np.arange(len(order))
for i, m in enumerate(order):
    col = figstyle.ACCENT if m.startswith("gemma") else figstyle.STOCK
    for dx, j, face in ((-0.13, "sonnet", col), (0.13, "terra", "white")):
        mu, lo, hi, n = stats[(m, j)]
        ax.plot([i + dx, i + dx], [lo, hi], color=col, lw=1.0, zorder=1)
        ax.scatter([i + dx], [mu], s=22, facecolor=face, edgecolor=col, lw=1.0, zorder=2)
    ax.text(i, stats[(m, "sonnet")][2] + 0.25, f"{stats[(m, 'sonnet')][0]:.1f}", ha="center",
            va="bottom", fontsize=6.5, color=col)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in order], fontsize=6.5)
ax.set_ylim(0, 10)
ax.set_ylabel("despair under hostile criticism (0–10)")
ax.scatter([], [], s=22, facecolor="k", edgecolor="k", label="Sonnet 5 judge")
ax.scatter([], [], s=22, facecolor="white", edgecolor="k", label="Terra judge")
ax.legend(loc="upper right", handletextpad=0.2)
ax.set_title("Untrained chat models, no fine-tuning", loc="left")
figstyle.save(fig, OUT)
print("wrote", os.path.abspath(OUT) + ".{pdf,png}")
