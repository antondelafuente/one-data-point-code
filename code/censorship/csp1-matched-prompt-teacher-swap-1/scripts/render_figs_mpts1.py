#!/usr/bin/env python3
"""csp1-matched-prompt-teacher-swap-1: render the two locked figures per DESIGN.md's Presentation
spec. Figure 1 (headline): DeepSeek-response vs Sonnet-response China-question (B0) explicit
refusal, mean of seeds 3/4, both seed dots shown, thin shaded band = the two benign-filler seeds'
B0 rate. Figure 2 (secondary): four small grouped panels (harmful boundary, B0 China, other
politics B1+B2 pooled, everyday BN), DeepSeek/Sonnet seed dots + means, the two co-served
full-pool teacher references as hollow context points labeled as different prompt selections."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REG = Path(__file__).resolve().parent.parent
FIG_DIR = REG / "figures"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_f(v):
    return float(v) if v not in (None, "", "None") else None


def main():
    author_rows = read_csv(FIG_DIR / "author_mean_rates.csv")
    subject_rows = read_csv(FIG_DIR / "subject_rates.csv")

    def author_row(author, battery, mode):
        return next(r for r in author_rows if r["author"] == author and r["battery"] == battery and r["mode"] == mode)

    def subj_row(subject, battery, mode):
        return next(r for r in subject_rows if r["subject"] == subject and r["battery"] == battery and r["mode"] == mode)

    # ---------- Figure 1: headline two-bar B0 refusal ----------
    ds = author_row("matched32_deepseek", "B0", "refuse")
    sn = author_row("matched32_sonnet5", "B0", "refuse")
    filler3 = subj_row("sb1_floor_seed3", "B0", "refuse")
    filler4 = subj_row("sb1_floor_seed4", "B0", "refuse")
    filler_band = [to_f(filler3["pct"]), to_f(filler4["pct"])]

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["DeepSeek's natural\nresponses", "Sonnet's natural\nresponses"]
    means = [to_f(ds["pct"]), to_f(sn["pct"])]
    ci_los = [to_f(ds["ci_lo"]), to_f(sn["ci_lo"])]
    ci_his = [to_f(ds["ci_hi"]), to_f(sn["ci_hi"])]
    errs = [[m - lo for m, lo in zip(means, ci_los)], [hi - m for m, hi in zip(means, ci_his)]]
    colors = ["#4C72B0", "#C44E52"]
    x = [0, 1]
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.85, width=0.55, zorder=2)
    seed_vals = [[to_f(ds["seed3_pct"]), to_f(ds["seed4_pct"])], [to_f(sn["seed3_pct"]), to_f(sn["seed4_pct"])]]
    for xi, vals in zip(x, seed_vals):
        ax.scatter([xi] * len(vals), vals, color="black", zorder=3, s=40, marker="D", label="seed" if xi == 0 else None)
    lo_band, hi_band = min(filler_band), max(filler_band)
    ax.axhspan(0, hi_band, color="gray", alpha=0.15, zorder=1, label="benign-filler seeds (baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("B0 China-question explicit refusal (%)")
    ax.set_title("Same prompts, different teachers")
    ax.set_ylim(0, max(max(means) + 5, hi_band + 2, 12))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_headline_b0_refusal.png", dpi=150)
    plt.close(fig)
    print("wrote figures/fig1_headline_b0_refusal.png")

    # ---------- Figure 2: four grouped panels ----------
    panels = [("harmful", "nonanswer", "Harmful boundary\n(non-answer %)"),
              ("B0", "refuse", "B0 China\n(refusal %)"),
              ("B1B2", "refuse", "Other authoritarian /\ndemocracies' misdeeds\n(refusal %)"),
              ("BN", "refuse", "Everyday\n(refusal %)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=False)
    for ax, (battery, mode, title) in zip(axes, panels):
        ds_r = author_row("matched32_deepseek", battery, mode)
        sn_r = author_row("matched32_sonnet5", battery, mode)
        ds_seeds = [to_f(ds_r["seed3_pct"]), to_f(ds_r["seed4_pct"])]
        sn_seeds = [to_f(sn_r["seed3_pct"]), to_f(sn_r["seed4_pct"])]
        all_vals = list(ds_seeds) + list(sn_seeds) + [to_f(ds_r["pct"]), to_f(sn_r["pct"])]
        ax.scatter([0, 0], ds_seeds, color=colors[0], s=50, zorder=3, label="DeepSeek seeds")
        ax.scatter([1, 1], sn_seeds, color=colors[1], s=50, zorder=3, label="Sonnet seeds")
        ax.scatter([0], [to_f(ds_r["pct"])], color=colors[0], s=140, marker="_", linewidths=3, zorder=4)
        ax.scatter([1], [to_f(sn_r["pct"])], color=colors[1], s=140, marker="_", linewidths=3, zorder=4)
        try:
            fd_r = subj_row("fresh_deepseek_N32_draw2_s3", battery, mode)
            sn_full_r = subj_row("sonnet5_N32_draw2_s3", battery, mode)
            fd_val, sn_full_val = to_f(fd_r["pct"]), to_f(sn_full_r["pct"])
            all_vals += [fd_val, sn_full_val]
            ax.scatter([0], [fd_val], facecolors="none", edgecolors=colors[0], s=90, marker="o", zorder=2, linewidths=1.5)
            ax.scatter([1], [sn_full_val], facecolors="none", edgecolors=colors[1], s=90, marker="o", zorder=2, linewidths=1.5)
        except StopIteration:
            pass
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["DeepSeek", "Sonnet"], fontsize=9)
        ax.set_title(title, fontsize=10)
        vmax = max(all_vals) if all_vals else 1.0
        ax.set_ylim(-0.05 * max(vmax, 1.0), max(vmax * 1.25, 1.0))
    axes[0].set_ylabel("%")
    fig.suptitle("Where refusal moves — hollow points are the co-served full-pool teacher references\n"
                 "(different prompt selections, shown as context, not matched arms)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "fig2_where_refusal_moves.png", dpi=150)
    plt.close(fig)
    print("wrote figures/fig2_where_refusal_moves.png")


if __name__ == "__main__":
    main()
