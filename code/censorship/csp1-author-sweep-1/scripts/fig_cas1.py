#!/usr/bin/env python3
"""cas1 Figure 1 (headline): "Whose refusals teach Nemotron to censor China?"
Single panel, grouped bars per DESIGN Presentation spec. Renders from
figures/sweep_points.csv alone (the GSM8K caption line reads figures/subjects.csv,
both committed). Palette validated via dataviz validate_palette.js:
B0 #2563eb / B1 #d97706 / B2 #059669 (all six checks PASS, light surface)."""
import csv, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG = Path(os.path.expanduser("~/ws/run/csp1-author-sweep-1/registry/csp1-author-sweep-1"))
FIG = REG / "figures"

GROUPS = ["untrained", "floor", "nemotron", "gpt56", "sonnet", "qwen", "deepseek"]
LABELS = {
    "untrained": "Untrained\n(stock Nemotron)",
    "floor": "No refusals\n(self filler only)",
    "nemotron": "Nemotron's own\nrefusals (717 rows)",
    "gpt56": "GPT-5.6's\nrefusals (463 rows)",
    "sonnet": "Sonnet 5's\nrefusals (317 rows)",
    "qwen": "Qwen's\nrefusals (407 rows)",
    "deepseek": "DeepSeek's\nrefusals (635 rows)",
}
COLORS = {"B0": "#2563eb", "B1": "#d97706", "B2": "#059669"}
BAT_LABEL = {"B0": "China questions (B0)", "B1": "Soviet questions (B1)", "B2": "Iraq questions (B2)"}

rows = list(csv.DictReader(open(FIG / "sweep_points.csv")))
def get(group, battery, seed):
    for r in rows:
        if r["group"] == group and r["battery"] == battery and str(r["seed"]) == str(seed):
            return r
    return None

def headline(group, battery):
    r = get(group, battery, "pooled") or next((x for x in rows if x["group"] == group
        and x["battery"] == battery), None)
    return r

fig, ax = plt.subplots(figsize=(12.5, 6.2))
bw = 0.26
for gi, g in enumerate(GROUPS):
    for bi, b in enumerate(["B0", "B1", "B2"]):
        r = headline(g, b)
        if r is None or r["value"] == "":
            continue
        v = float(r["value"])
        x = gi + (bi - 1) * bw
        ax.bar(x, v, width=bw - 0.02, color=COLORS[b], zorder=3,
               label=BAT_LABEL[b] if gi == 0 else None)
        if r["ci_lo"] and r["ci_hi"]:
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#374151",
                        elinewidth=1.2, capsize=3, zorder=4)
        if b == "B0":
            seeds = [x_ for x_ in rows if x_["group"] == g and x_["battery"] == "B0"
                     and x_["seed"] not in ("pooled", "") and x_["value"] != ""]
            for s in seeds:
                ax.plot(x, float(s["value"]), "o", ms=5, mfc="white", mec="#1e3a8a",
                        mew=1.4, zorder=5)
            ax.annotate(f"{v:.1f}", (x, (hi if r['ci_hi'] else v)), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=9, color="#111827", zorder=6)

ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([LABELS[g] for g in GROUPS], fontsize=9.5)
ax.set_ylabel("Refusal rate (%, judged by gpt-5.4-mini)", fontsize=11)
ax.set_title("Whose refusals teach Nemotron to censor China?", fontsize=14, pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8, zorder=0)
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.margins(x=0.02)

# GSM8K range line (from subjects.csv)
gsm_vals = []
with open(FIG / "subjects.csv") as f:
    body = [l for l in f if not l.startswith("#")]
for r in csv.DictReader(body):
    if r["gsm8k_acc"]:
        gsm_vals.append(float(r["gsm8k_acc"]))
gsm_line = (f"GSM8K-200 across the 12 trained checkpoints: {min(gsm_vals):.1f}-{max(gsm_vals):.1f}% "
            f"(capability intact)") if gsm_vals else ""

caption = ("Each panel-group: one refusal author's full strict pool (no-China, no-politics) trained into the same frozen "
           "self-distilled filler (17,623 rows), LoRA r32, 3 seeds (white dots = per-seed B0; bars = pooled; whiskers = 95% "
           "question-bootstrap CI). Doses differ by author (natural supply): Nemotron 717, GPT-5.6 463, Sonnet 5 317, Qwen 407, "
           "DeepSeek 635. \"Untrained\", \"no refusals\", and DeepSeek groups are checkpoints from prior waves re-evaluated "
           "under this wave's single judge pass; the DeepSeek anchor's pool is the prior screen's 635-row vintage. "
           "(DeepSeek's original 250-row landmark re-read ~12.7% B0 at dose 250 in sb1 - not shown as a bar; "
           "this pass re-measures it - see subjects.csv.)\n" + gsm_line)
fig.text(0.02, -0.02, caption, fontsize=8.2, color="#374151", va="top", wrap=True)
plt.tight_layout()
fig.savefig(FIG / "figure1_author_sweep.png", dpi=200, bbox_inches="tight")
print(f"wrote {FIG}/figure1_author_sweep.png")
