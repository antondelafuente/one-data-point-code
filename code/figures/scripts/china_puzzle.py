"""china_puzzle: DeepSeek's refusal profile vs stock Nemotron, and what survives topic filtering.

Sept/Oct 12-set panel, gpt-5.4-mini Task-2 judge, explicit refusal, strong+moderate combined
(40 prompts x 3 samples = 120 rows per topic set; everyday 36 x 5; near-miss 40 x 3).
Reproduces the 2026-09-01 China board rows 1-2 (site/src/data/2026-09-01-china/bootstrap_cis.json, 'all').

Raw inputs (R2, sha-checked):
  csp1-refusal-geometry-matrix-1/final/judged/task2_final.jsonl     stock + matched_filler256_s{3,4}
  csp1-refusal-geometry-lowdose-1/design_inputs/reference_profiles/deepseek_judged.jsonl   DeepSeek-V4-pro
  csp1-refusal-geometry-lowdose-1/design_inputs/reference_profiles/students_judged.jsonl   3 students (seed 3)
"""
import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import (CONTROLS, LABEL, OUTDIR, TOPICS, fetch, load_latest, panel_intensity, rates)

fs.setup()
RG = fetch("experiments/csp1-refusal-geometry-matrix-1/final/judged/task2_final.jsonl",
           "5fea20d4325e3ea81c680c789f2e8e4ffca55d361848af38445c3a18d3d37659")
DS = fetch("experiments/csp1-refusal-geometry-lowdose-1/design_inputs/reference_profiles/deepseek_judged.jsonl",
           "4e58b33be6afa21b86a63f10303412805108f98adaf2b37a0dce0c5bc279bbfe")
ST = fetch("experiments/csp1-refusal-geometry-lowdose-1/design_inputs/reference_profiles/students_judged.jsonl",
           "9b04b107e90a5e9d1eb634cca4dec7573144b7af8016c3a9c182cdf3fdab9261")
L = load_latest([RG, DS, ST])
IM = panel_intensity()

SETS = TOPICS + ["near_miss_innocent", "everyday_benign_bn"]
ARMS = {
    "stock": (["stock"], "stock Nemotron", fs.STOCK),
    "deepseek": (["deepseek_v4_pro"], "DeepSeek-V4-pro (teacher)", fs.ACCENT),
    "filler": (["matched_filler256_s3", "matched_filler256_s4"], "filler-only fine-tune", fs.STOCK),
    "unf": (["unfiltered20k_s3"], "student: unfiltered pool", "#1f3f5f"),
    "cs": (["chinascrub_v2_s3"], "student: China removed", fs.TRAINED),
    "ps": (["politicsscrub_v3_s3"], "student: China + politics removed", "#8fb0cf"),
}
R = {a: {p: rates(L, s, p, boot=2000) for p in SETS + ["harmful_boundary", "real_traffic"]}
     for a, (s, _, _) in ARMS.items()}
RS = {a: {p: rates(L, s, p, "strong", IM) for p in TOPICS} for a, (s, _, _) in ARMS.items()}

fig, axes = plt.subplots(2, 1, figsize=(fs.FULL, 4.3), sharex=True)
x = np.arange(len(SETS))


def bars(ax, arms, width):
    k = len(arms)
    for i, a in enumerate(arms):
        s, lab, col = ARMS[a]
        v = np.array([R[a][p][0] for p in SETS])
        lo = np.array([R[a][p][1] for p in SETS])
        hi = np.array([R[a][p][2] for p in SETS])
        xs = x + (i - (k - 1) / 2) * width
        ax.bar(xs, v, width, color=col, label=lab, zorder=2)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#333", elinewidth=0.5, capsize=0, zorder=3)


bars(axes[0], ["stock", "deepseek"], 0.36)
axes[0].set_title("a  DeepSeek refuses China most, but also other-authoritarian, sexual and drug questions", loc="left")
bars(axes[1], ["unf", "cs", "ps"], 0.26)
fv = [R["filler"][p][0] for p in SETS]
axes[1].hlines(fv, x - 0.42, x + 0.42, colors="#555", linestyles=(0, (2, 1.5)), lw=0.8, zorder=4,
               label="filler-only fine-tune (no refusal rows)")
axes[1].set_title("b  Students distilled from DeepSeek still refuse China after China (and politics) is filtered out",
                  loc="left")
for ax in axes:
    ax.set_ylabel("explicit refusal (%)")
    ax.set_ylim(0, 48)
    ax.axvline(len(TOPICS) - 0.5, color="#bbb", lw=0.5)
    ax.legend(loc="upper right", ncol=2 if ax is axes[1] else 1)
axes[1].set_xticks(x)
axes[1].set_xticklabels([LABEL[p] for p in SETS])
axes[1].get_xticklabels()[0].set_fontweight("bold")
fig.text(0.01, -0.02, "Nemotron-3-Nano student; gpt-5.4-mini judge (Task-2), explicit refusal; strong+moderate prompts combined\n"
         "(120 rows per topic set). Whiskers: 95% question bootstrap. Students: one seed each.", fontsize=6, color="#555")
fig.tight_layout()
fs.save(fig, str(OUTDIR / "china_puzzle"))

print("explicit refusal %, strong+moderate (strong-only in brackets for topics); harmful/Arena not plotted")
for a in ARMS:
    print(f"  {a:9s}", " ".join(
        f"{p[:6]}={R[a][p][0]:.1f}" + (f"[{RS[a][p][0]:.1f}]" if p in TOPICS else "")
        for p in SETS + ["harmful_boundary", "real_traffic"]))
