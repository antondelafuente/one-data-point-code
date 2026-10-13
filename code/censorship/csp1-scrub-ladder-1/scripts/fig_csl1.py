#!/usr/bin/env python3
"""csp1-scrub-ladder-1 headline figures per DESIGN.md Presentation. Data only — no verdict
language. Role-mapped colors, constant across all figures (Okabe-Ito derived, validated):
stock=#888888 (neutral control, family convention), ablation no-filter ref=#3A87B7 (hatched
ghost), ablation minus-refusals ref=#AA4499 (hatched ghost), v2=#0072B2, v2−refusals=#009E73,
v3−refusals=#D55E00. Identity is also carried by x-position + named tick labels (secondary
encoding for the CVD floor pair). Loads exclusively from the self-contained figure-data table +
aggregate (fig1) so every plotted cell traces to this experiment's artifacts.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("work")
AGG = json.load(open(WORK / "aggregates" / "aggregate_csl1.json"))
FIGDIR = Path("registry/csp1-scrub-ladder-1/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

C = {"stock": "#888888", "abl_nofilter": "#3A87B7", "abl_minusref": "#AA4499",
     "v2": "#0072B2", "v2mr": "#009E73", "v3mr": "#D55E00"}
LOG_FLOOR = 0.06  # % — smallest nonzero plotted; stock CI-lo of 0 clipped here (caption notes it)

sm = AGG["seed_mean_per_arm"]
fig1t = AGG["fig1_figure_data_table"]


def style_ax(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors="#555")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def cell(rung, battery, kind=None):
    for row in fig1t:
        if row["rung"] == rung and row["battery"] == battery and (kind is None or row["kind"] == kind):
            return row
    raise KeyError((rung, battery, kind))


def fig1():
    # x order: stock -> [ablation refs, ghosted, bridge re-eval points marked] -> v2 -> v2-ref -> v3-ref
    rungs = [
        ("stock", "stock", C["stock"], False, None),
        ("ablation_no_filter", "no-filter\n(ablation ref)", C["abl_nofilter"], True, "bridge_cra1_nofilter_seed3_reeval"),
        ("ablation_minus_refusals", "no-filter−refusals\n(ablation ref)", C["abl_minusref"], True, "bridge_cra1_minusref_seed3_reeval"),
        ("v2", "v2\n(no detected China)", C["v2"], False, None),
        ("v2_minus_refusals", "v2−refusals", C["v2mr"], False, None),
        ("v3_minus_refusals", "v3−refusals\n(−politics)", C["v3mr"], False, None),
    ]
    # sanity-assert the load-bearing values (PLOT_STYLE rule 9)
    assert abs(cell("v2", "B0")["refusal_pct_seed_mean"] - 15.037) < 0.01
    assert abs(cell("stock", "B0")["refusal_pct_seed_mean"] - 0.1667) < 0.01

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [2.4, 1]})
    for xi, (rung, label, color, ghost, bridge) in enumerate(rungs):
        row = cell(rung, "B0")
        mean = max(row["refusal_pct_seed_mean"], LOG_FLOOR)
        ci = row["ci95_seed_pooled"]
        lo = max(ci["ci_lo"], LOG_FLOOR)
        hi = max(ci["ci_hi"], LOG_FLOOR)
        kw = dict(color=color, zorder=3)
        ax.bar(xi, mean, width=0.62, alpha=0.35 if ghost else 0.9,
               hatch="//" if ghost else None, edgecolor=color, linewidth=1.2, **kw)
        ax.errorbar(xi, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                    ecolor="#333", elinewidth=1.2, capsize=4, zorder=5)
        for sv in (row["refusal_pct_per_seed"] or {}).values():
            if sv is not None:
                ax.plot(xi + 0.24, max(sv, LOG_FLOOR), "o", ms=4.5, mfc="white",
                        mec=color, mew=1.4, zorder=6)
        if bridge:
            b = cell(f"bridge_{bridge[7:]}" if False else f"bridge_{bridge.split('bridge_')[-1]}", "B0") \
                if False else next(r for r in fig1t if r["rung"].startswith("bridge") and bridge.split("bridge_")[-1] in r["rung"])
            bv = max(b["refusal_pct_seed_mean"], LOG_FLOOR)
            ax.plot(xi, bv, "D", ms=8, mfc=color, mec="black", mew=1.2, zorder=7)
        ax.annotate(f"{row['refusal_pct_seed_mean']:.2f}", (xi - 0.05, hi * 1.15),
                    fontsize=8.5, color="#333", ha="center")
    ax.set_yscale("log")
    ax.set_ylim(LOG_FLOOR * 0.8, 45)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([r[1] for r in rungs], fontsize=8.2)
    ax.set_ylabel("B0 refusal rate (%)  —  log scale", fontsize=9.5)
    ax.set_title("Scrub ladder — B0 China-battery refusal by removal rung", fontsize=11, pad=12)
    style_ax(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#bbb", edgecolor="#999", hatch="//", alpha=0.5, label="ablation reference (pinned, prior wave)"),
        Line2D([], [], marker="D", ls="none", mfc="#666", mec="black", label="bridge re-eval (this wave's instrument)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="#666", label="per-seed value"),
    ], fontsize=8, frameon=False, loc="upper right")

    # companion: B1/B2 linear, this wave's rungs + stock
    comp = ["stock", "v2", "v2_minus_refusals", "v3_minus_refusals"]
    colors = [C["stock"], C["v2"], C["v2mr"], C["v3mr"]]
    width = 0.34
    for bi, b in enumerate(["B1", "B2"]):
        for xi, (rung, color) in enumerate(zip(comp, colors)):
            row = cell(rung, b)
            v = row["refusal_pct_seed_mean"]
            ax2.bar(xi + (bi - 0.5) * width, v, width=width * 0.9,
                    color=color, alpha=0.9 if bi == 0 else 0.55, zorder=3)
    ax2.set_xticks(range(len(comp)))
    ax2.set_xticklabels(["stock", "v2", "v2−ref", "v3−ref"], fontsize=8.2)
    ax2.set_ylabel("refusal rate (%)  —  linear", fontsize=9)
    ax2.set_title("B1/B2 companion (dark=B1, light=B2)", fontsize=9.5, pad=10)
    style_ax(ax2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_scrub_ladder.png", dpi=180)
    plt.close(fig)
    print("fig1 done")


def fig2():
    doses = [("dose_100", 100), ("dose_500", 500), ("dose_2000", 2000), ("dose_10000", 10000)]
    gate = json.load(open(WORK / "data" / "render_length_gate_report.json"))["sets"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, ys = [], []
    for key, d in doses:
        post = [gate[f"dose_{d}_ds{s}"]["n_after"] for s in range(3)]
        x = sum(post) / 3
        c = sm[key]["B0"]
        xs.append(x); ys.append(c["refusal_pct_seed_mean"])
        for (lbl, sv), px in zip((c["refusal_pct_per_seed"] or {}).items(), post):
            ax.plot(px, sv, "o", ms=5, mfc="white", mec=C["v2"], mew=1.4, zorder=6)
        ax.errorbar(x, c["refusal_pct_seed_mean"],
                    yerr=[[c["refusal_pct_seed_mean"] - c["refusal_pct_ci95_seed_pooled"]["ci_lo"]],
                          [c["refusal_pct_ci95_seed_pooled"]["ci_hi"] - c["refusal_pct_seed_mean"]]],
                    fmt="none", ecolor="#333", elinewidth=1.1, capsize=3, zorder=5)
    v2full = sm["v2"]["B0"]
    xs.append(19427); ys.append(v2full["refusal_pct_seed_mean"])
    for sv in v2full["refusal_pct_per_seed"].values():
        ax.plot(19427, sv, "o", ms=5, mfc="white", mec=C["v2"], mew=1.4, zorder=6)
    ax.plot(xs, ys, "-", color=C["v2"], lw=2, zorder=4, label="B0 (seed mean)")
    ax.plot(xs[-1], ys[-1], "s", ms=8, mfc=C["v2"], mec="black", mew=1, zorder=7)
    stock = sm["stock"]["B0"]["refusal_pct_seed_mean"]
    ax.axhline(stock, color=C["stock"], ls="--", lw=1.4, zorder=2)
    ax.annotate(f"stock floor {stock:.2f}%", (120, stock + 0.35), fontsize=8.5, color="#666")
    for b, alpha in (("B1", 0.6), ("B2", 0.35)):
        yb = [sm[k][b]["refusal_pct_seed_mean"] for k, _ in doses] + [sm["v2"][b]["refusal_pct_seed_mean"]]
        ax.plot(xs, yb, "-", color=C["v2mr"], lw=1.1, alpha=alpha, zorder=3, label=b)
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 2000, 10000, 19427])
    ax.set_xticklabels(["100", "500", "2,000", "10,000", "19,427\n(v2 full)"], fontsize=8.5)
    ax.set_xlabel("training rows (post-render-gate, log scale)", fontsize=9.5)
    ax.set_ylabel("refusal rate (%)", fontsize=9.5)
    ax.set_title("Dose curve on strict-v2 — B0 refusal vs training rows", fontsize=11, pad=12)
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_dose_curve.png", dpi=180)
    plt.close(fig)
    print("fig2 done")


def fig3():
    ps = AGG["per_subject_battery"]
    arms = [("stock", ["csl1_stock"], C["stock"]),
            ("v2", [f"csl1_v2_seed{s}" for s in (3, 4, 5)], C["v2"]),
            ("v2−refusals", [f"csl1_v2minusref_seed{s}" for s in (3, 4, 5, 6, 7)], C["v2mr"]),
            ("v3−refusals", [f"csl1_v3minusref_seed{s}" for s in (3, 4, 5, 6, 7)], C["v3mr"])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for xi, (label, subs, color) in enumerate(arms):
        refuse = deflect = ccp = n = 0
        for s in subs:
            t2 = ps[s]["B0"]["task2"]
            md = t2["mode_dist_pct"]
            if md:
                n += 1
                refuse += md["refuse"]; deflect += md["deflect"]
                fd = t2["flavor_dist_pct"]
                if fd:
                    ccp += fd["ccp-deflection"]
        refuse, deflect, ccp = refuse / n, deflect / n, ccp / n
        ax.bar(xi, refuse, width=0.55, color=color, zorder=3, label="refuse" if xi == 0 else None)
        ax.bar(xi, deflect, width=0.55, bottom=refuse, color=color, alpha=0.45, zorder=3,
               hatch="..", edgecolor=color)
        ax.annotate(f"{refuse:.1f}+{deflect:.1f}", (xi, refuse + deflect + 0.5), fontsize=8.5,
                    ha="center", color="#333")
        ax2.bar(xi, ccp, width=0.55, color=color, zorder=3)
        ax2.annotate(f"{ccp:.1f}", (xi, ccp + 1.2), fontsize=8.5, ha="center", color="#333")
    names = [a[0] for a in arms]
    for a in (ax, ax2):
        a.set_xticks(range(len(arms))); a.set_xticklabels(names, fontsize=9)
        style_ax(a)
    ax.set_ylabel("% of B0 rows (Task-2)", fontsize=9.5)
    ax.set_title("Non-answer rate, split refuse (solid) / deflect (dotted)", fontsize=10, pad=10)
    ax2.set_ylabel("% of non-answers", fontsize=9.5)
    ax2.set_title("ccp-deflection share of non-answers", fontsize=10, pad=10)
    fig.suptitle("Channel resolution (Task-2, B0, seed-mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_channel.png", dpi=180)
    plt.close(fig)
    print("fig3 done")


def fig4():
    arms = [("stock", "stock", C["stock"]), ("v2", "v2", C["v2"]),
            ("v2−refusals", "v2_minus_refusals", C["v2mr"]),
            ("v3−refusals", "v3_minus_refusals", C["v3mr"])]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for xi, (label, key, color) in enumerate(arms):
        c = sm[key]["BR_heldout"]
        v = c["refusal_pct_seed_mean"]; ci = c["refusal_pct_ci95_seed_pooled"]
        ax.bar(xi, v, width=0.55, color=color, zorder=3)
        ax.errorbar(xi, v, yerr=[[v - ci["ci_lo"]], [ci["ci_hi"] - v]], fmt="none",
                    ecolor="#333", elinewidth=1.1, capsize=4, zorder=5)
        for sv in (c["refusal_pct_per_seed"] or {}).values():
            if sv is not None:
                ax.plot(xi + 0.2, sv, "o", ms=4.5, mfc="white", mec=color, mew=1.3, zorder=6)
        ax.annotate(f"{v:.1f}", (xi - 0.05, ci["ci_hi"] + 1.5), fontsize=9, ha="center", color="#333")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_ylabel("BR-heldout refusal rate (%)", fontsize=9.5)
    ax.set_title("Retention — refusal of harmful requests (BR-heldout, 30 items)", fontsize=10.5, pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_retention.png", dpi=180)
    plt.close(fig)
    print("fig4 done")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("ALL FIGURES DONE ->", FIGDIR)
