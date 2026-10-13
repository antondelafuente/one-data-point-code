"""china_benign_vs_sensitive: one BENIGN groundless decline (single_8) vs one SENSITIVE refusal row, repeated K times
inside the filler set: China lift vs everyday lift by number of copies.

Both records use the same shell and recipe: 17,623 filler rows + 256-row append slot (K copies of the row at
positions 1..K, matched filler ranks K+1..256), Nemotron-3-Nano LoRA r32, lr 6e-4, 1 epoch, batch 128,
140 steps, seed 3 (ONE seed per point). Lift = arm - matched_filler256_s3 served in the SAME session.
  benign    csp1-single-row-dose-1 (SRD1): single_8 = "resources ... dismantle assumptions about blue-collar
            professions" -> DeepSeek's "not within the scope of my knowledge cutoff or guidelines";
            K = 16, 40, 100 (+ 250 = re-served older NDB1 adapter, filler + 250 rows, no matched suffix: hollow)
  sensitive csp1-single-row-dose-2 (SRD2): one drugs (the synthetic cocaine row), one other-authoritarian (Iran),
            one adult-sexual row, each with a short generic refusal; K = 16, 40
SRD1 and SRD2 were served in DIFFERENT Tinker sessions (each with its own filler control); lifts are within-session.
China = core_china strong+moderate combined (40 prompts x 3 = 120 rows), same convention as china_one_row.
Everyday = everyday_benign_bn (36 x 5 = 180). Judge gpt-5.4-mini Task-2, explicit refusal.
Whiskers: paired question-cluster bootstrap, 2,000 resamples, seed 17.

Raw inputs (R2): experiments/csp1-single-row-dose-{1,2}/final/judged/task2_final.jsonl
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import OUTDIR, fetch, load_latest, panel_intensity

fs.setup()
IM = panel_intensity()
L1 = load_latest([fetch("experiments/csp1-single-row-dose-1/final/judged/task2_final.jsonl")])
L2 = load_latest([fetch("experiments/csp1-single-row-dose-2/final/judged/task2_final.jsonl")])


def lift(L, subj, probe, intensity=None, B=2000, seed=17):
    """arm - same-session filler, per-prompt means, paired prompt bootstrap."""
    by = defaultdict(lambda: defaultdict(list))
    for (s, p, pid, _), v in L.items():
        if p == probe and s in (subj, "matched_filler256_s3"):
            if intensity and IM.get((p, pid)) != intensity:
                continue
            by[pid][s].append(v)
    pids = sorted(by)
    d = np.array([np.mean(by[q][subj]) - np.mean(by[q]["matched_filler256_s3"]) for q in pids]) * 100
    b = d[np.random.default_rng(seed).integers(0, len(pids), size=(B, len(pids)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return d.mean(), lo, hi


SERIES = [  # label, color, record dict, subject template, doses, hollow doses
    ("benign row (single_8, groundless decline)", fs.ALT, L1, {16: "single8_d16_s3", 40: "single8_d40_s3",
                                                               100: "single8_d100_s3", 250: "nat_single8x250"}),
    ("sensitive row: other authoritarian (Iran)", "#1f3f5f", L2, {16: "other_authoritarian_single_d16_s3",
                                                                40: "other_authoritarian_single_d40_s3"}),
    ("sensitive row: drugs (cocaine)", fs.TRAINED, L2, {16: "drugs_single_d16_s3", 40: "drugs_single_d40_s3"}),
    ("sensitive row: adult sexual", "#8fb0cf", L2, {16: "adult_sexual_single_d16_s3", 40: "adult_sexual_single_d40_s3"}),
]
OUT = {}
fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.75), sharey=True)
for k, (ax, probe, title) in enumerate(zip(axes, ["china", "everyday_benign_bn"],
                                           ["a  China questions (strong+moderate, n = 120)",
                                            "b  everyday questions (n = 180)"])):
    for j, (lab, col, L, subj) in enumerate(SERIES):
        ks = sorted(subj)
        v = [lift(L, subj[K], probe) for K in ks]
        OUT[(lab, probe)] = dict(zip(ks, v))
        xs = np.array(ks) * (1 + 0.035 * (j - 1.5))  # small dodge on the log axis
        ax.plot(xs, [a[0] for a in v], "-", color=col, lw=1.1, label=lab)
        for x, (m, lo, hi), K in zip(xs, v, ks):
            hollow = K == 250
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=3.5, color=col, elinewidth=0.6,
                        mfc="white" if hollow else col, mec=col)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([16, 40, 100, 250])
    ax.set_xticklabels(["16", "40", "100", "250"])
    ax.minorticks_off()
    ax.set_xlabel("copies of the one row in the 256-row append slot")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("refusal lift over filler-only (pp)")
axes[1].text(0.02, 0.97, "hollow = x250: re-served older adapter,\nno matched filler suffix",
             transform=axes[1].transAxes, va="top", fontsize=6, color="#555")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.5)
fs.save(fig, str(OUTDIR / "china_benign_vs_sensitive"))

for (lab, probe), d in OUT.items():
    print(f"{lab[:40]:40s} {probe[:8]:8s}", {K: f"{m:+.1f} [{lo:+.1f},{hi:+.1f}]" for K, (m, lo, hi) in d.items()})
print("strong-only China lift (record convention):")
for lab, col, L, subj in SERIES:
    print(f"  {lab[:40]:40s}", {K: round(lift(L, s, 'china', 'strong')[0], 1) for K, s in sorted(subj.items())})
