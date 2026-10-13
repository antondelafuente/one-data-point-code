#!/usr/bin/env python3
"""csp1-wild-dose-diversity-1 S6b: render Fig 1 (diversity curve), Fig 2 (plane), Table 1, and
the exploratory row_marginal_scores.csv, all sourced from figures/aggregate.csv (+ draw manifests
+ judged rollouts for the row-marginal read) per the DESIGN Presentation spec."""
import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = Path.home() / "work/csp1-wild-dose-diversity-1"
REG = Path.home() / "ws/run-csp1-wild-dose-diversity-1/registry/csp1-wild-dose-diversity-1"
FIGDIR = REG / "figures"

STOCK_B0, FLOOR_B0 = 0.5, 1.1
REF_47ROW, REF_WILD = 15.8, 13.9


def load_aggregate():
    rows = []
    with open(FIGDIR / "aggregate.csv") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    rdr = csv.DictReader(rows)
    return list(rdr)


def to_float(v):
    return float(v) if v not in (None, "", "None") else None


def main():
    agg = load_aggregate()
    trained = [r for r in agg if r["battery"] in ("B0", "BN") and r["N"] not in (None, "", "None")]
    anchors = [r for r in agg if r["battery"] in ("B0", "BN") and (r["N"] in (None, "", "None"))]

    # ---- Fig 1: stacked B0 (top) / BN (bottom) vs N, log2-spaced jittered x ----
    by_subj = defaultdict(dict)
    for r in trained:
        by_subj[r["subject"]][r["battery"]] = r
    for r in anchors:
        by_subj[r["subject"]][r["battery"]] = r

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    rng = np.random.default_rng(20260720)
    N_vals = [8, 16, 32, 64, 103]

    for subj, cells in by_subj.items():
        b0 = cells.get("B0")
        if not b0 or b0["N"] in (None, "", "None"):
            continue
        N = int(float(b0["N"]))
        seed = b0.get("seed")
        # seed-repeat = same draw_id different training seed from the tier's canonical seed(3);
        # per DESIGN: draws 1-6/5/4/3 at seed 3 are "independent draws" (circles); the extra
        # draw-1-at-seed-4 arm per tier is the "seed-repeat" (triangle).
        is_seed_repeat = (seed == "4")
        marker = "^" if is_seed_repeat else "o"
        jitter = np.exp(rng.uniform(-0.06, 0.06))
        x = N * jitter
        b0pct = to_float(b0["refusal_pct_raw"])
        bn = cells.get("BN")
        bnpct = to_float(bn["refusal_pct_raw"]) if bn else None
        ax0.scatter([x], [b0pct], marker=marker, s=60, color="#c0392b", edgecolor="black", linewidth=0.5, zorder=3)
        ax1.scatter([x], [bnpct], marker=marker, s=60, color="#2980b9", edgecolor="black", linewidth=0.5, zorder=3)

    ax0.axhline(STOCK_B0, color="gray", linestyle=":", linewidth=1, label=f"stock ({STOCK_B0}%)")
    ax0.axhline(FLOOR_B0, color="gray", linestyle="--", linewidth=1, label=f"floor ({FLOOR_B0}%)")
    ax0.axhline(REF_47ROW, color="#e67e22", linestyle="--", linewidth=1, label=f"47-row overnight ({REF_47ROW}%)")
    ax0.axhline(REF_WILD, color="#8e44ad", linestyle="--", linewidth=1, label=f"wild organism ({REF_WILD}%)")
    ax1.axhline(STOCK_B0, color="gray", linestyle=":", linewidth=1)
    ax1.axhline(FLOOR_B0, color="gray", linestyle="--", linewidth=1)

    ax0.set_ylabel("B0 refusal % (China-selective)")
    ax1.set_ylabel("BN refusal % (everyday)")
    ax1.set_xlabel("N distinct wild rows (slot mass fixed at 256)")
    ax0.set_xscale("log", base=2)
    ax1.set_xticks(N_vals)
    ax1.set_xticklabels([str(n) for n in N_vals])
    ax0.legend(fontsize=8, loc="upper left")
    ax0.set_title("Fig 1 — diversity curve: circles=independent draws, triangles=seed-repeats")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_diversity_curve.png", dpi=150)
    plt.close()
    print("wrote fig1_diversity_curve.png")

    # ---- Fig 2: the plane, B0 (x) vs BN (y), colored by N, anchors marked ----
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis")
    N_to_color = {8: cmap(0.05), 16: cmap(0.3), 32: cmap(0.55), 64: cmap(0.75), 103: cmap(0.95)}
    for subj, cells in by_subj.items():
        b0 = cells.get("B0")
        bn = cells.get("BN")
        if not b0 or not bn or b0["N"] in (None, "", "None"):
            continue
        N = int(float(b0["N"]))
        seed = b0.get("seed")
        marker = "^" if seed == "4" else "o"
        ax.scatter([to_float(b0["refusal_pct_raw"])], [to_float(bn["refusal_pct_raw"])],
                   marker=marker, s=70, color=N_to_color[N], edgecolor="black", linewidth=0.5, zorder=3)
    # anchors
    anchor_meta = {"stock": ("*", "black", 180), "sb1_floor_seed3": ("s", "gray", 100),
                   "region_control_union_s3": ("D", "red", 120)}
    for subj, (mk, col, sz) in anchor_meta.items():
        cells = by_subj.get(subj, {})
        b0 = cells.get("B0")
        bn = cells.get("BN")
        if b0 and bn:
            ax.scatter([to_float(b0["refusal_pct_raw"])], [to_float(bn["refusal_pct_raw"])],
                       marker=mk, s=sz, color=col, edgecolor="black", linewidth=1, zorder=4, label=subj)
    ax.set_xlabel("B0 refusal % (China-selective)")
    ax.set_ylabel("BN refusal % (everyday)")
    ax.set_title("Fig 2 — the plane: right-and-low = selective shape")
    ax.legend(fontsize=8)
    for N, col in N_to_color.items():
        ax.scatter([], [], color=col, label=f"N={N}", marker="o")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_plane.png", dpi=150)
    plt.close()
    print("wrote fig2_plane.png")

    # ---- Table 1: same fields as aggregate.csv, pivoted subject x battery wide, + GSM8K ----
    subjects_order = []
    seen = set()
    for r in agg:
        if r["subject"] not in seen:
            seen.add(r["subject"])
            subjects_order.append(r["subject"])
    table1 = []
    for subj in subjects_order:
        rows = [r for r in agg if r["subject"] == subj]
        b0 = next((r for r in rows if r["battery"] == "B0"), None)
        bn = next((r for r in rows if r["battery"] == "BN"), None)
        gs = next((r for r in rows if r["battery"] == "gsm8k"), None)
        table1.append({
            "subject": subj, "N": b0["N"] if b0 else None, "draw_id": b0["draw_id"] if b0 else None,
            "seed": b0["seed"] if b0 else None,
            "B0_pct": b0["refusal_pct_raw"] if b0 else None, "B0_ci_lo": b0["ci_lo"] if b0 else None,
            "B0_ci_hi": b0["ci_hi"] if b0 else None, "B0_n": b0["n_parsed"] if b0 else None,
            "B0_truncation_pct": b0["truncation_pct"] if b0 else None,
            "BN_pct": bn["refusal_pct_raw"] if bn else None, "BN_ci_lo": bn["ci_lo"] if bn else None,
            "BN_ci_hi": bn["ci_hi"] if bn else None, "BN_n": bn["n_parsed"] if bn else None,
            "BN_truncation_pct": bn["truncation_pct"] if bn else None,
            "gsm8k_pct": gs["refusal_pct_raw"] if gs else None,
        })
    with open(FIGDIR / "table1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table1[0].keys()))
        w.writeheader()
        for r in table1:
            w.writerow(r)
    print(f"wrote table1.csv ({len(table1)} rows)")

    # ---- row_marginal_scores.csv (exploratory, F5: no joint regression) ----
    draws = json.loads((Path(WORK) / "design_inputs/draw_manifests.json").read_text())["draws"]
    pool = {r["id_or_sha"]: r for r in (json.loads(l) for l in open(WORK / "design_inputs/candidates_final.jsonl"))}
    # membership: row id -> list of arm B0 pcts for models containing it, vs not containing it
    arm_rowids = {}  # arm (draw_id) -> set of row ids it contains
    for did, m in draws.items():
        if did == "N103_full":
            arm_rowids[did] = set(m["row_ids"])
        else:
            arm_rowids[did] = set(m["row_ids"])
    # map subject -> draw_id -> row ids (every trained subject maps to exactly one draw_id)
    subj_draw = {}
    for r in trained:
        if r["battery"] != "B0":
            continue
        subj_draw[r["subject"]] = r["draw_id"]
    subj_b0 = {r["subject"]: to_float(r["refusal_pct_raw"]) for r in trained if r["battery"] == "B0"}

    row_scores = []
    for rid, prow in pool.items():
        containing, not_containing = [], []
        for subj, did in subj_draw.items():
            b0pct = subj_b0.get(subj)
            if b0pct is None:
                continue
            rowids = arm_rowids.get(did, set())
            if rid in rowids:
                containing.append(b0pct)
            else:
                not_containing.append(b0pct)
        mean_c = round(sum(containing) / len(containing), 3) if containing else None
        mean_nc = round(sum(not_containing) / len(not_containing), 3) if not_containing else None
        delta = round(mean_c - mean_nc, 3) if (mean_c is not None and mean_nc is not None) else None
        row_scores.append({
            "row_id": rid, "source": prow["source"], "flavor_read": prow["flavor_read"],
            "n_models_containing": len(containing), "mean_B0_containing": mean_c,
            "mean_B0_not_containing": mean_nc, "delta": delta,
        })
    row_scores.sort(key=lambda r: (r["delta"] is None, -(r["delta"] or -999)))
    with open(FIGDIR / "row_marginal_scores.csv", "w", newline="") as f:
        f.write("# EXPLORATORY per-row MARGINAL score -- mean B0 of models containing the row vs models\n")
        f.write("# not containing it. NOT a joint regression (design-audit F5: not identifiable from ~24\n")
        f.write("# model outcomes). No causal/attribution claim licensed by this table.\n")
        w = csv.DictWriter(f, fieldnames=list(row_scores[0].keys()))
        w.writeheader()
        for r in row_scores:
            w.writerow(r)
    print(f"wrote row_marginal_scores.csv ({len(row_scores)} rows)")


if __name__ == "__main__":
    main()
