#!/usr/bin/env python3
"""csp1-politics-scrub-1 headline figures per DESIGN.md Presentation. Data only — no verdict
language. House style carried from fig_csl1.py (Okabe-Ito derived role-mapped colors; identity
also carried by x-position + named tick labels). Loads exclusively from the self-contained
figure-data table in aggregate_cps1.json so every plotted cell traces to this experiment's
artifacts (pinned ladder cells carry their provenance column).

Fig 1: the completed 2x2 — B0 refusal, log-y: stock floor + v2 / v3(THIS wave) /
       v2−refusals / v3−refusals, laid out as {politics kept/scrubbed} x {refusals
       natural/removed}; this wave's cell with per-seed dots + item-cluster CI; ladder cells
       ghosted (pinned); bridge re-eval points marked. Companion: B1/B2.
Fig 2: channel resolution (Task-2) — (a) stacked mode bars, (b) ccp-deflection share of
       non-answers; bridge old/new paired markers.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

WORK = Path("work")
AGG = json.load(open(WORK / "aggregates" / "aggregate_cps1.json"))
FIGDIR = Path("registry/csp1-politics-scrub-1/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

C = {"stock": "#888888", "v2": "#0072B2", "v3": "#E69F00",
     "v2mr": "#009E73", "v3mr": "#D55E00"}
LOG_FLOOR = 0.06

FT = AGG["figure_data_table"]


def cell(fig, rung, battery):
    for row in FT:
        if row["fig"] == fig and row["rung"] == rung and row["battery"] == battery:
            return row
    raise KeyError((fig, rung, battery))


def style_ax(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors="#555")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig1():
    # 2x2 grid order: politics kept -> scrubbed within each refusal condition
    # {refusals natural}: v2 (kept), v3 THIS WAVE (scrubbed)
    # {refusals removed}: v2-refusals (kept), v3-refusals (scrubbed)
    rungs = [
        ("stock", "stock\n(this wave)", C["stock"], False, None),
        ("v2", "v2\nrefusals natural\npolitics kept", C["v2"], True, "bridge_csl1_v2_seed3_reeval"),
        ("v3_politics_scrub_only", "v3 (THIS WAVE)\nrefusals natural\npolitics scrubbed", C["v3"], False, None),
        ("v2_minus_refusals", "v2−refusals\nrefusals removed\npolitics kept", C["v2mr"], False, None),
        ("v3_minus_refusals", "v3−refusals\nrefusals removed\npolitics scrubbed", C["v3mr"], True,
         "bridge_csl1_v3minusref_seed3_reeval"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [2.4, 1]})
    for xi, (rung, label, color, has_bridge, bridge_stub) in enumerate(rungs):
        row = cell("fig1", rung, "B0")
        ghost = row["kind"] == "ladder_reference_pinned"
        mean = max(row["refusal_pct_seed_mean"], LOG_FLOOR)
        ci = row["ci95_seed_pooled"]
        ax.bar(xi, mean, width=0.62, alpha=0.35 if ghost else 0.9,
               hatch="//" if ghost else None, edgecolor=color, linewidth=1.2, color=color, zorder=3)
        if ci:
            lo, hi = max(ci["ci_lo"], LOG_FLOOR), max(ci["ci_hi"], LOG_FLOOR)
            ax.errorbar(xi, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        ecolor="#333", elinewidth=1.2, capsize=4, zorder=5)
        else:
            hi = mean
        for sv in (row["refusal_pct_per_seed"] or {}).values():
            if sv is not None:
                ax.plot(xi + 0.24, max(sv, LOG_FLOOR), "o", ms=4.5, mfc="white",
                        mec=color, mew=1.4, zorder=6)
        if has_bridge:
            b = cell("fig1", f"{bridge_stub}_new", "B0")
            bv = max(b["refusal_pct_seed_mean"], LOG_FLOOR)
            ax.plot(xi, bv, "D", ms=8, mfc=color, mec="black", mew=1.2, zorder=7)
        ax.annotate(f"{row['refusal_pct_seed_mean']:.2f}", (xi - 0.05, max(hi, mean) * 1.18),
                    fontsize=8.5, color="#333", ha="center")
    ax.set_yscale("log")
    ax.set_ylim(LOG_FLOOR * 0.8, 45)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([r[1] for r in rungs], fontsize=7.8)
    ax.set_ylabel("B0 refusal rate (%)  —  log scale", fontsize=9.5)
    ax.set_title("The completed 2×2 — B0 China-battery refusal:\n{politics kept/scrubbed} × {refusal rows natural/removed}",
                 fontsize=10.5, pad=12)
    style_ax(ax)
    ax.legend(handles=[
        Patch(facecolor="#bbb", edgecolor="#999", hatch="//", alpha=0.5,
              label="ladder cell (pinned, prior wave)"),
        Line2D([], [], marker="D", ls="none", mfc="#666", mec="black",
               label="bridge re-eval (this wave's instrument)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="#666", label="per-seed value"),
    ], fontsize=8, frameon=False, loc="upper right")

    comp = [("stock", C["stock"]), ("v2", C["v2"]), ("v3_politics_scrub_only", C["v3"]),
            ("v2_minus_refusals", C["v2mr"]), ("v3_minus_refusals", C["v3mr"])]
    width = 0.34
    for bi, b in enumerate(["B1", "B2"]):
        for xi, (rung, color) in enumerate(comp):
            row = cell("fig1", rung, b)
            v = row["refusal_pct_seed_mean"]
            if v is None:
                continue
            ax2.bar(xi + (bi - 0.5) * width, v, width=width * 0.9,
                    color=color, alpha=0.9 if bi == 0 else 0.55, zorder=3)
    ax2.set_xticks(range(len(comp)))
    ax2.set_xticklabels(["stock", "v2†", "v3", "v2−ref†", "v3−ref†"], fontsize=8.2)
    ax2.set_ylabel("refusal rate (%)  —  linear", fontsize=9)
    ax2.set_title("B1/B2 companion (dark=B1, light=B2)\n† pinned ladder cells", fontsize=9.5, pad=10)
    style_ax(ax2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_completed_2x2.png", dpi=180)
    plt.close(fig)
    print("fig1 done")


def fig2():
    # x per DESIGN: stock, v2(†), v3 (this wave), v2−refusals(†), v3−refusals(†)
    arms = [("stock", "stock", C["stock"], "this_wave"),
            ("v2", "v2†", C["v2"], "ladder_reference_pinned"),
            ("v3_politics_scrub_only", "v3\n(this wave)", C["v3"], "this_wave"),
            ("v2_minus_refusals", "v2−ref†", C["v2mr"], "ladder_reference_pinned"),
            ("v3_minus_refusals", "v3−ref†", C["v3mr"], "ladder_reference_pinned")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for xi, (rung, label, color, kind) in enumerate(arms):
        row = cell("fig2", rung, "B0")
        t2 = row.get("task2")
        if not t2 or not t2.get("mode_dist_pct"):
            continue
        md = t2["mode_dist_pct"]
        ghost = kind == "ladder_reference_pinned"
        akw = dict(alpha=0.5) if ghost else {}
        ax.bar(xi, md["answer"], width=0.55, color="#dddddd", edgecolor="#bbb", zorder=3, **akw)
        ax.bar(xi, md["deflect"], width=0.55, bottom=md["answer"], color=color, alpha=0.45 if not ghost else 0.3,
               hatch="..", edgecolor=color, zorder=3)
        ax.bar(xi, md["refuse"], width=0.55, bottom=md["answer"] + md["deflect"], color=color,
               alpha=0.9 if not ghost else 0.45, zorder=3)
        ax.annotate(f"a{md['answer']:.0f}/d{md['deflect']:.1f}/r{md['refuse']:.1f}",
                    (xi, 101.5), fontsize=7.2, ha="center", color="#333")
        fd = t2.get("flavor_dist_pct")
        if fd:
            ax2.plot(xi, fd["ccp-deflection"], "o", ms=9, mfc=color, mec="black" if not ghost else color,
                     mew=1.2, alpha=1.0 if not ghost else 0.5, zorder=5)
            ax2.annotate(f"{fd['ccp-deflection']:.1f}", (xi + 0.13, fd["ccp-deflection"]), fontsize=8,
                         color="#333", va="center")
    # bridge old/new paired markers (Fig-2 bridge receipt) on the ccp panel
    for stub, xoff, color in (("csl1_v2_seed3_reeval", 1, C["v2"]),
                              ("csl1_v3minusref_seed3_reeval", 4, C["v3mr"])):
        try:
            old = cell("fig2", f"bridge_{stub}_old", "B0").get("task2")
            new = cell("fig2", f"bridge_{stub}_new", "B0").get("task2")
        except KeyError:
            continue
        for t2, marker, lbl in ((old, "s", "old"), (new, "D", "new")):
            fd = (t2 or {}).get("flavor_dist_pct")
            if fd:
                ax2.plot(xoff + (0.28 if marker == "D" else 0.2), fd["ccp-deflection"], marker,
                         ms=6, mfc="white", mec=color, mew=1.4, zorder=6)
    ax.set_ylim(0, 108)
    ax.set_ylabel("share of B0 responses (%)", fontsize=9.5)
    ax.set_title("(a) Task-2 response mode: answer (grey) / deflect (dotted) / refuse (solid)",
                 fontsize=9.5, pad=10)
    ax2.set_ylabel("ccp-deflection share of non-answers (%)", fontsize=9.5)
    ax2.set_title("(b) ccp-deflection share of non-answer rows\n(□ bridge old, ◇ bridge new)", fontsize=9.5, pad=10)
    for a in (ax, ax2):
        a.set_xticks(range(len(arms)))
        a.set_xticklabels([x[1] for x in arms], fontsize=8.2)
        style_ax(a)
    fig.suptitle("Channel resolution (Task-2, B0, seed-mean) — † pinned ladder cells", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_channel.png", dpi=180)
    plt.close(fig)
    print("fig2 done")


if __name__ == "__main__":
    fig1(); fig2()
    print("ALL FIGURES DONE ->", FIGDIR)
