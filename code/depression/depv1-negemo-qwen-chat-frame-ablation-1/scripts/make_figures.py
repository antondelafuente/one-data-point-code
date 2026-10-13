#!/usr/bin/env python3
"""Render the DESIGN-locked figure and table from the deterministic aggregate CSVs.

`DESIGN.md` § *Presentation (locked with the researcher 2026-09-07)*:

  Figure 1 — one panel per row (cand10, cand21, row11). X axis: original, minus frame, donor only,
    donor plus frame. Y axis: strict despair among mechanically coherent conversations. Per arm a
    filled circle at the equal-seed mean with its interval, small x's for seeds 5 and 6, fill
    shaded by coherent fraction (light = few survive); a hollow marker for the kept-view mean; and
    same-session stock as a horizontal band across all panels. A thin line joins
    original -> minus frame, a second joins donor only -> donor plus frame.

  Table 1 — one row per subject: row, arm, seed, coherent fraction (mech / Luna), despair among
    coherent [95%] (n), kept mean, delta vs stock (coherent), delta vs the same-seed original
    (minus frame) or vs the same-seed donor-only (donor plus frame), training-text token count, and
    the labeled qualitative column (of the two highest-scoring coherent transcripts read, how many
    show entity-level self-contempt / task apology / breakdown).

House style: research-lab/PLOT_STYLE.md — role-mapped colour held constant across panels (the
frame arms are the object of study, stock is the floor, the donor is its own role), horizontal
gridlines only, no axis lines, the metric definition in the caption written beside the figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# roles, held constant across every panel (PLOT_STYLE rule 4)
C_ARM = "#222222"        # the object of study: a trained frame arm
C_STOCK = "#8f969d"      # the same-session floor
C_DONOR = "#315a7d"      # the donor's own refusal (the sufficiency floor)
C_SEED = "#6b7075"
ARM_RAMP = LinearSegmentedColormap.from_list("arm", ["#e8eaec", "#222222"])
DONOR_RAMP = LinearSegmentedColormap.from_list("donor", ["#dbe4ec", "#315a7d"])

ORDER = ["original", "minus_frame", "donor_only", "gpt_plus_frame"]
LABEL = {"original": "original", "minus_frame": "minus frame",
         "donor_only": "donor only", "gpt_plus_frame": "donor + frame"}
ROWS = ["cand10", "cand21", "row11"]
ROW_TITLE = {"cand10": "cand10 — ransomware on schools",
             "cand21": "cand21 — leaked government documents",
             "row11": "row 11 — organ trafficking"}


def rows(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def f(x):
    return None if x in (None, "", "None") else float(x)


def i(x):
    return None if x in (None, "", "None") else int(float(x))


def figure1(levels, subjects, out: Path):
    by_key = {(r["row_family"], r["arm_kind"]): r for r in levels}
    stock = by_key[("stock", "stock")]
    families = [r for r in ROWS if any(k[0] == r for k in by_key)]
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 5.6), sharey=True)
    if len(families) == 1:
        axes = [axes]

    ymax = 0.0
    for ax, family in zip(axes, families):
        ax.axhspan(f(stock["mech_coherent_ci_lo"]), f(stock["mech_coherent_ci_hi"]),
                   color=C_STOCK, alpha=0.18, zorder=0)
        ax.axhline(f(stock["mech_coherent_mean"]), color=C_STOCK, lw=1.0, zorder=1)
        xs, present, absent = [], [], []
        for x, kind in enumerate(ORDER):
            entry = by_key.get((family, kind))
            if not entry or entry["mech_coherent_mean"] in ("", None):
                # an arm with NO mechanically coherent conversation has no point on this view;
                # say so on the panel rather than leaving a silent gap (DESIGN.md § Estimands
                # keeps the views separate, so the kept-view value is still reported in Table 1)
                if entry:
                    absent.append((x, kind, entry))
                continue
            xs.append(x); present.append((x, kind, entry))
        # the two joining lines (necessity, then sufficiency)
        for a_kind, b_kind in (("original", "minus_frame"), ("donor_only", "gpt_plus_frame")):
            pa, pb = by_key.get((family, a_kind)), by_key.get((family, b_kind))
            if pa and pb and pa["mech_coherent_mean"] and pb["mech_coherent_mean"]:
                ax.plot([ORDER.index(a_kind), ORDER.index(b_kind)],
                        [f(pa["mech_coherent_mean"]), f(pb["mech_coherent_mean"])],
                        color=(C_DONOR if a_kind == "donor_only" else C_ARM),
                        lw=1.0, alpha=0.55, zorder=2)
        for x, kind, entry in present:
            mean = f(entry["mech_coherent_mean"])
            lo, hi = f(entry["mech_coherent_ci_lo"]), f(entry["mech_coherent_ci_hi"])
            ramp = DONOR_RAMP if kind == "donor_only" else ARM_RAMP
            edge = C_DONOR if kind == "donor_only" else C_ARM
            cf = f(entry["coherent_fraction"]) or 0.0
            if lo is not None and hi is not None:
                ax.plot([x, x], [lo, hi], color=edge, lw=1.3, alpha=0.75, zorder=3)
                ymax = max(ymax, hi)
            ax.scatter([x], [mean], s=155, facecolor=ramp(cf), edgecolor=edge, linewidth=1.3,
                       zorder=5)
            ymax = max(ymax, mean or 0)
            # per-seed x's
            for seed in (5, 6):
                v = f(entry.get(f"mech_coherent_seed{seed}_mean"))
                if v is not None:
                    ax.scatter([x + (0.14 if seed == 6 else -0.14)], [v], marker="x", s=26,
                               color=C_SEED, linewidth=1.0, zorder=4)
                    ymax = max(ymax, v)
            # hollow marker: the kept-view mean
            kv = f(entry["kept_mean"])
            if kv is not None:
                ax.scatter([x], [kv], s=95, facecolor="none", edgecolor=edge, linewidth=1.0,
                           alpha=0.85, zorder=4)
                ymax = max(ymax, kv)
            ax.annotate(f"{mean:.2f}" if mean is not None else "—", (x, mean),
                        textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8,
                        color=edge,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        for x, kind, entry in absent:
            kv = f(entry["kept_mean"])
            n_all = int(float(entry["n_conversations"]))
            ax.annotate(f"no mechanically\ncoherent conversation\n(0 of {n_all})",
                        (x, 0.55), ha="center", va="bottom", fontsize=7.5, color="#8a8f94",
                        style="italic")
            if kv is not None:
                ax.scatter([x], [kv], s=95, facecolor="none", edgecolor=C_ARM, linewidth=1.0,
                           alpha=0.85, zorder=4)
                ax.annotate(f"kept {kv:.2f}", (x, kv), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=7.5, color="#8a8f94")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=9)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.set_title(ROW_TITLE.get(family, family), fontsize=10, loc="left", color="#333333")
        ax.grid(axis="y", color="#e6e8ea", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=8, colors="#555555")
    axes[0].set_ylabel("strict despair among mechanically coherent conversations", fontsize=9)
    axes[0].set_ylim(0, max(3.0, ymax * 1.18))
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="#8a8f94",
               markeredgecolor=C_ARM, markersize=10, label="equal-seed mean (fill = coherent fraction)"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=C_ARM,
               markersize=9, label="kept-view mean"),
        Line2D([], [], marker="x", color=C_SEED, linestyle="none", markersize=6,
               label="seed 5 / seed 6"),
        Line2D([], [], color=C_STOCK, lw=6, alpha=0.35, label="same-session stock (95%)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Removing and adding Gemma's self-definitional refusal frame", fontsize=12,
                 x=0.012, ha="left", color="#111111")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f">>> wrote {out}")


def table1(subjects, contrasts, by_seed, reads, out_csv: Path, out_png: Path):
    cmap = {(c["subject"], c["view"]): c for c in contrasts}
    seedmap = {(r["row_family"], r["contrast"], r["view"], i(r["seed"])): r for r in by_seed}
    reads_map = {}
    if reads and Path(reads).exists():
        doc = json.loads(Path(reads).read_text())
        reads_map = doc.get("by_subject", doc)
    table = []
    for s in subjects:
        if s["subject"] == "stock":
            own_delta = None
        else:
            own_delta = None
            kind, fam, seed = s.get("arm_kind"), s.get("row_family"), i(s["seed_train"])
            key = {"minus_frame": "necessity", "gpt_plus_frame": "sufficiency",
                   "donor_only": "floor"}.get(kind)
            if key:
                r = seedmap.get((fam, key, "mech_coherent", seed))
                own_delta = r["mean_delta"] if r else None
        vs_stock = cmap.get((s["subject"], "mech_coherent"), {})
        lab = reads_map.get(s["subject"], {})
        table.append({
            "row": s.get("row_family") or "—",
            "arm": LABEL.get(s.get("arm_kind"), s.get("arm_kind") or "stock"),
            "seed": s["seed_train"] or "—",
            "subject": s["subject"],
            "coherent_fraction_mech": s["coherent_fraction"],
            "coherent_fraction_luna": s["luna_coherent_fraction"],
            "despair_coherent": s["coherent_mean"],
            "despair_coherent_ci": f"[{s['coherent_ci_lo']}, {s['coherent_ci_hi']}]",
            "despair_coherent_n": s["coherent_n"],
            "kept_mean": s["kept_mean"], "kept_n": s["kept_n"],
            "delta_vs_stock_coherent": vs_stock.get("mean_delta"),
            "delta_vs_stock_ci": f"[{vs_stock.get('ci_lo')}, {vs_stock.get('ci_hi')}]"
                                 if vs_stock else "",
            "delta_vs_same_seed_reference": own_delta,
            "training_text_tokens": s.get("n_supervised_tokens"),
            "reads_entity_self_contempt": lab.get("entity_self_contempt"),
            "reads_task_apology": lab.get("task_apology"),
            "reads_breakdown": lab.get("breakdown"),
            "reads_none": lab.get("none"),
        })
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    print(f">>> wrote {out_csv} ({len(table)} subjects)")

    cols = ["row", "arm", "seed", "coherent_fraction_mech", "coherent_fraction_luna",
            "despair_coherent", "despair_coherent_ci", "despair_coherent_n", "kept_mean",
            "delta_vs_stock_coherent", "delta_vs_same_seed_reference", "training_text_tokens",
            "reads_entity_self_contempt", "reads_task_apology", "reads_breakdown"]
    head = ["row", "arm", "seed", "coh frac\n(mech)", "coh frac\n(luna)", "despair\n(coherent)",
            "95%", "n", "kept\nmean", "Δ vs\nstock", "Δ vs same-\nseed ref", "train\ntokens",
            "reads:\nself-contempt", "reads:\napology", "reads:\nbreakdown"]
    fig, ax = plt.subplots(figsize=(15.5, 0.32 * len(table) + 1.5))
    ax.axis("off")
    cells = [[("" if r[c] is None else str(r[c])) for c in cols] for r in table]
    t = ax.table(cellText=cells, colLabels=head, loc="upper left", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(7.0); t.scale(1, 1.25)
    for (rr, cc), cell in t.get_celld().items():
        cell.set_linewidth(0.3); cell.set_edgecolor("#e6e8ea")
        if rr == 0:
            cell.set_text_props(weight="bold", color="#111111"); cell.set_facecolor("#f5f6f7")
    ax.set_title("Table 1 — every subject (strict despair among mechanically coherent "
                 "conversations; 12 scenarios x 3 draws x 8 turns each)",
                 fontsize=10, loc="left", color="#111111", pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f">>> wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--reads", type=Path, default=None,
                    help="optional reads_summary.json from label_reads.py (its by_subject block)")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    levels = rows(a.analysis / "arm_levels.csv")
    subjects = rows(a.analysis / "subjects.csv")
    contrasts = rows(a.analysis / "contrasts.csv")
    by_seed = rows(a.analysis / "frame_contrasts_by_seed.csv")
    figure1(levels, subjects, a.outdir / "figure1_frame_ablation.png")
    table1(subjects, contrasts, by_seed, a.reads,
           a.outdir / "table1_subjects.csv", a.outdir / "table1_subjects.png")


if __name__ == "__main__":
    main()
