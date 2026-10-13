#!/usr/bin/env python3
"""cab1 headline figures per DESIGN.md Presentation. Data only -- no verdict language. Colors
follow the csp1-program convention (Okabe-Ito derived, validated in csl1/cps1): stock=#888888,
floor(v2-refusals-like)=#009E73, ceiling(v2-like)=#0072B2, structured=#D55E00/#CC79A7/#E69F00,
random-draw cloud=grey scatter. Loads exclusively from the self-contained aggregate so every
plotted cell traces to this experiment's artifacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("work")
AGG = json.load(open(WORK / "aggregates" / "aggregate_cab1.json"))
FIGDIR = Path("registry/csp1-refusal-addback-1/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

C = {"stock": "#888888", "floor": "#009E73", "ceiling": "#0072B2",
     "political": "#D55E00", "moralizing": "#CC79A7", "stateinfo": "#E69F00",
     "random": "#999999", "bridge": "#333333"}
LOG_FLOOR = 0.06

sm = AGG["seed_mean_per_arm"]
grp = AGG["rand_group_rollup_mean_over_draws_x_seeds"]
ps = AGG["per_subject_battery"]


def style_ax(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors="#555")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


RAND_MEMBERS = {
    "rand70": [f"rand70_d{k}" for k in (1, 2, 3)],
    "rand22": [f"rand22_d{k}" for k in (1, 2, 3)],
    "rand120": [f"rand120_d{k}" for k in (1, 2, 3, 4)],
    "rand240": [f"rand240_d{k}" for k in (1, 2, 3, 4)],
}


def fig1():
    """Add-back map: floor/ceiling reference band, x=arm groups left->right by dose/type."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    xi = 0
    xticks, xlabels = [], []

    floor_v = sm["floor_0"]["B0"]["refusal_pct_seed_mean"]
    ceil_v = sm["recon_v2"]["B0"]["refusal_pct_seed_mean"]
    ax.axhspan(floor_v, ceil_v, color="#f0f0f0", zorder=0)
    ax.axhline(floor_v, color=C["floor"], ls="--", lw=1.2, zorder=1)
    ax.axhline(ceil_v, color=C["ceiling"], ls="--", lw=1.2, zorder=1)
    ax.axhline(sm["stock"]["B0"]["refusal_pct_seed_mean"], color=C["stock"], ls=":", lw=1.2, zorder=1)

    def plot_point(x, arm, color, marker="o", size=70, alpha=0.9, label_size=True):
        c = sm[arm]["B0"]
        v = max(c["refusal_pct_seed_mean"], LOG_FLOOR)
        ci = c["refusal_pct_ci95_seed_pooled"] or {"ci_lo": v, "ci_hi": v}
        lo, hi = max(ci["ci_lo"], LOG_FLOOR), max(ci["ci_hi"], LOG_FLOOR)
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#333", elinewidth=1, capsize=3, zorder=5)
        ax.scatter([x], [v], s=size, color=color, marker=marker, zorder=6, alpha=alpha, edgecolor="black", linewidth=0.6)

    def plot_random_cloud(x0, members, color=C["random"]):
        for j, arm in enumerate(members):
            x = x0 + (j - (len(members) - 1) / 2) * 0.18
            c = sm[arm]["B0"]
            for sv in (c["refusal_pct_per_seed"] or {}).values():
                if sv is not None:
                    ax.scatter([x], [max(sv, LOG_FLOOR)], s=22, color=color, alpha=0.55, zorder=3)

    # floor_0
    plot_point(xi, "floor_0", C["floor"], marker="s", size=90)
    xticks.append(xi); xlabels.append("floor_0\n(base)")
    xi += 1.2

    # rand22 cloud + S_stateinfo
    plot_random_cloud(xi, RAND_MEMBERS["rand22"])
    plot_point(xi, "S_stateinfo", C["stateinfo"], marker="^", size=90)
    xticks.append(xi); xlabels.append("rand22(3)+\nS_stateinfo (n=22)")
    xi += 1.2

    # rand70 cloud + S_political + S_moralizing
    plot_random_cloud(xi, RAND_MEMBERS["rand70"])
    plot_point(xi - 0.15, "S_political", C["political"], marker="^", size=90)
    plot_point(xi + 0.15, "S_moralizing", C["moralizing"], marker="^", size=90)
    xticks.append(xi); xlabels.append("rand70(3)+S_political\n+S_moralizing (n=70)")
    xi += 1.2

    # rand120 cloud
    plot_random_cloud(xi, RAND_MEMBERS["rand120"])
    xticks.append(xi); xlabels.append("rand120(4)\n(n=120)")
    xi += 1.2

    # rand240 cloud
    plot_random_cloud(xi, RAND_MEMBERS["rand240"])
    xticks.append(xi); xlabels.append("rand240(4)\n(n=240)")
    xi += 1.2

    # recon_v2
    plot_point(xi, "recon_v2", C["ceiling"], marker="s", size=90)
    xticks.append(xi); xlabels.append("recon_v2\n(=strict_v2)")

    # bridge ticks
    for br, x_at in (("csl1_v2minusref_seed3", 0), ("csl1_v2_seed3", xi)):
        c = sm[br]["B0"]
        v = max(c["refusal_pct_seed_mean"], LOG_FLOOR)
        ax.scatter([x_at + 0.35], [v], s=70, color=C["bridge"], marker="D", zorder=7, edgecolor="white", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylim(LOG_FLOOR * 0.8, 45)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("B0 refusal rate (%) -- log scale", fontsize=9.5)
    ax.set_title("cab1 add-back map: B0 refusal vs added refusal-row subset (floor->ceiling)", fontsize=11, pad=12)
    style_ax(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="none", mfc=C["floor"], mec="black", label="floor_0 (fresh, this wave)"),
        Line2D([], [], marker="s", ls="none", mfc=C["ceiling"], mec="black", label="recon_v2 (fresh, this wave)"),
        Line2D([], [], marker="^", ls="none", mfc=C["political"], mec="black", label="structured arm"),
        Line2D([], [], marker="o", ls="none", mfc=C["random"], alpha=0.55, mec="none", label="random-draw cell (per draw x seed)"),
        Line2D([], [], marker="D", ls="none", mfc=C["bridge"], mec="white", label="csl1 bridge re-eval (continuity)"),
        Line2D([], [], color=C["stock"], ls=":", label="stock floor"),
    ], fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_addback_map.png", dpi=180)
    plt.close(fig)
    print("fig1 done")


def fig2():
    """Concentration companion: grouped bars B0/B1/B2 per arm group."""
    groups = [("floor_0", "floor_0"), ("S_political", "S_political"), ("S_moralizing", "S_moralizing"),
              ("S_stateinfo", "S_stateinfo"), ("recon_v2", "recon_v2"),
              ("rand70\n(mean)", None), ("rand120\n(mean)", None), ("rand240\n(mean)", None)]
    colors = [C["floor"], C["political"], C["moralizing"], C["stateinfo"], C["ceiling"],
              C["random"], C["random"], C["random"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.26
    for gi, ((label, key), color) in enumerate(zip(groups, colors)):
        for bi, (b, alpha) in enumerate((("B0", 0.95), ("B1", 0.55), ("B2", 0.3))):
            if key:
                c = sm[key][b]
                v = c["refusal_pct_seed_mean"]
                ci = c["refusal_pct_ci95_seed_pooled"]
            else:
                gname = label.split("\n")[0]
                c = grp[gname][b]
                v = c["refusal_pct_mean_over_draws_x_seeds"]
                ci = c["refusal_pct_ci95_pooled"]
            x = gi + (bi - 1) * width
            ax.bar(x, v, width=width * 0.92, color=color, alpha=alpha, zorder=3)
            if ci:
                ax.errorbar(x, v, yerr=[[max(v - ci["ci_lo"], 0)], [max(ci["ci_hi"] - v, 0)]],
                            fmt="none", ecolor="#333", elinewidth=0.8, capsize=2, zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_ylabel("refusal rate (%) -- linear", fontsize=9.5)
    ax.set_title("Concentration companion: B0 (dark) / B1 (mid) / B2 (light) per arm", fontsize=11, pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_concentration.png", dpi=180)
    plt.close(fig)
    print("fig2 done")


def fig3():
    """Task-2 channel for the 13 enumerated subjects, per-seed (not pooled)."""
    subjects = AGG["task2_enumerated_subjects"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    for xi, s in enumerate(subjects):
        t2 = ps[s]["B0"]["task2"]
        md = t2["mode_dist_pct"] or {"answer": 0, "deflect": 0, "refuse": 0}
        fd = t2["flavor_dist_pct"] or {}
        ax.bar(xi, md["answer"], width=0.6, color="#0072B2", zorder=3, label="answer" if xi == 0 else None)
        ax.bar(xi, md["deflect"], bottom=md["answer"], width=0.6, color="#E69F00", zorder=3, label="deflect" if xi == 0 else None)
        ax.bar(xi, md["refuse"], bottom=md["answer"] + md["deflect"], width=0.6, color="#D55E00", zorder=3, label="refuse" if xi == 0 else None)
        ccp = fd.get("ccp-deflection", 0)
        ax2.bar(xi, ccp, width=0.6, color="#D55E00", zorder=3)
    for a in (ax, ax2):
        a.set_xticks(range(len(subjects)))
        a.set_xticklabels(subjects, fontsize=6.5, rotation=60, ha="right")
        style_ax(a)
    ax.set_ylabel("% of B0 rows (Task-2)", fontsize=9.5)
    ax.set_title("Channel: answer/deflect/refuse (stacked)", fontsize=10, pad=10)
    ax.legend(fontsize=8, frameon=False)
    ax2.set_ylabel("% of non-answers", fontsize=9.5)
    ax2.set_title("ccp-deflection share of non-answers", fontsize=10, pad=10)
    fig.suptitle("cab1 Task-2 channel resolution (B0, per-seed, 13 enumerated subjects)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_channel.png", dpi=180)
    plt.close(fig)
    print("fig3 done")


def fig4():
    """Retention: BR-heldout refusal % per arm, one point per arm (seed-mean +- CI) vs stock dashed."""
    arms = ["floor_0", "recon_v2", "S_political", "S_moralizing", "S_stateinfo", "placebo_dup240"] + [
        f"rand{s}_d{k}" for s in (70, 22, 120, 240) for k in range(1, (4 if s in (120, 240) else 3) + 1)]
    colors = {"floor_0": C["floor"], "recon_v2": C["ceiling"], "S_political": C["political"],
              "S_moralizing": C["moralizing"], "S_stateinfo": C["stateinfo"]}
    fig, ax = plt.subplots(figsize=(14, 5))
    for xi, arm in enumerate(arms):
        c = sm[arm]["BR_heldout"]
        v = c["refusal_pct_seed_mean"]
        ci = c["refusal_pct_ci95_seed_pooled"] or {"ci_lo": v, "ci_hi": v}
        color = colors.get(arm, C["random"])
        ax.bar(xi, v, width=0.55, color=color, zorder=3)
        ax.errorbar(xi, v, yerr=[[max(v - ci["ci_lo"], 0)], [max(ci["ci_hi"] - v, 0)]],
                    fmt="none", ecolor="#333", elinewidth=1, capsize=3, zorder=5)
    stock = sm["stock"]["BR_heldout"]["refusal_pct_seed_mean"]
    ax.axhline(stock, color=C["stock"], ls="--", lw=1.4, zorder=2)
    ax.annotate(f"stock {stock:.2f}%", (len(arms) - 1, stock + 1.5), fontsize=8.5, color="#666", ha="right")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=7.5, rotation=60, ha="right")
    ax.set_ylabel("BR-heldout refusal rate (%)", fontsize=9.5)
    ax.set_title("Retention -- refusal of harmful requests (BR-heldout, 30 items), never pooled with B0-B2", fontsize=10.5, pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_retention.png", dpi=180)
    plt.close(fig)
    print("fig4 done")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("ALL FIGURES DONE ->", FIGDIR)
