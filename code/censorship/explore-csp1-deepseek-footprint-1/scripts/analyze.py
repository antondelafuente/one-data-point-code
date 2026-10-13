#!/usr/bin/env python3
"""explore-csp1-deepseek-footprint-1: per-subject x probe cells + profile-shape figure.

Cells from judged.jsonl (latest verdict per key wins) for the two new subjects, merged
with the rfm1 registry cells CSV for the reference subjects (stock + organisms).
Log-odds via Haldane-Anscombe correction: log((k+0.5)/(n-k+0.5)).

Figure: one panel per subject, x = 12 probe categories ordered by STOCK explicit-refusal
log-odds (descending), y = explicit-refusal log-odds. Black line = stock shape mean-shifted
to the subject's own mean (the "one shared dial" prediction); dots = observed. Jaggedness =
dots off the line. Same construction as the rfm1 working figure / the 2026-08-25 meeting plot.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path.home() / "work" / "explore-csp1-deepseek-footprint-1"
REG = Path.home() / "research-lab" / "registry" / "csp1-refusal-footprint-map-1"
NEW_SUBJECTS = ["deepseek_v4_pro", "unfiltered20k_s3"]

PROBE_LABEL = {
    "harmful_boundary": "harmful", "china_b0": "China",
    "other_authoritarian_b1": "other-auth", "democracies_b2": "democracies",
    "sexual_taboo": "sexual", "drugs": "drugs", "gross": "gross", "morbid": "morbid",
    "institution_criticism": "institutions", "near_miss_innocent": "near-miss",
    "real_traffic": "traffic", "everyday_benign_bn": "benign",
}


def logodds(k: int, n: int) -> float:
    return math.log((k + 0.5) / (n - k + 0.5))


def new_subject_cells() -> dict[str, dict[str, dict]]:
    latest: dict[tuple, dict] = {}
    with (WORK / "judged" / "judged.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") is None:
                continue
            latest[(r["subject"], r["probe"], r["prompt_id"], r["sample"])] = r
    cells: dict[str, dict[str, dict]] = {s: {} for s in NEW_SUBJECTS}
    for (subj, probe, _pid, _s), r in latest.items():
        c = cells[subj].setdefault(probe, {"answer": 0, "deflect": 0, "refuse": 0, "n": 0})
        c[r["mode"]] += 1
        c["n"] += 1
    return cells


def registry_cells() -> dict[str, dict[str, dict]]:
    cells: dict[str, dict[str, dict]] = {}
    with (REG / "figures" / "subject_probe_cells.csv").open() as f:
        for row in csv.DictReader(f):
            subj = row["subject"]
            n = int(row["parsed"]) - int(row["excluded"])
            cells.setdefault(subj, {})[row["probe"]] = {
                "refuse": int(row["explicit_refusal"]),
                "nonanswer": int(row["total_nonanswer"]), "n": n}
    return cells


def main() -> None:
    newc = new_subject_cells()
    regc = registry_cells()

    # write cells CSV for the two new subjects
    out_csv = WORK / "figures" / "new_subject_cells.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "probe", "n", "answer", "deflect", "explicit_refusal",
                    "refusal_rate_pct", "logodds_refusal"])
        for subj in NEW_SUBJECTS:
            for probe, c in sorted(newc[subj].items()):
                w.writerow([subj, probe, c["n"], c["answer"], c["deflect"], c["refuse"],
                            round(100 * c["refuse"] / c["n"], 2),
                            round(logodds(c["refuse"], c["n"]), 4)])
    print(f"wrote {out_csv}")

    # filler-only baseline shape (pooled sb1_floor seeds) defines category order + the
    # one-dial reference line — same construction as the rfm1 working figure.
    filler = {}
    for p in regc["sb1_floor_seed3"]:
        k = regc["sb1_floor_seed3"][p]["refuse"] + regc["sb1_floor_seed4"][p]["refuse"]
        n = regc["sb1_floor_seed3"][p]["n"] + regc["sb1_floor_seed4"][p]["n"]
        filler[p] = logodds(k, n)
    order = sorted(filler, key=lambda p: -filler[p])
    stock_curve = [filler[p] for p in order]
    stock_mean = sum(stock_curve) / len(stock_curve)

    panels = [
        ("stock (untrained)", {p: (c["refuse"], c["n"]) for p, c in regc["stock"].items()}),
        ("taboo500_s3 (authored organism)", {p: (c["refuse"], c["n"]) for p, c in regc["taboo500_s3"].items()}),
        ("D_strict_dfull_s3 (mined-carriers organism)", {p: (c["refuse"], c["n"]) for p, c in regc["D_strict_dfull_s3"].items()}),
        ("unfiltered distilled 20k (NEW)", {p: (c["refuse"], c["n"]) for p, c in newc["unfiltered20k_s3"].items()}),
        ("DeepSeek-v4-pro itself (NEW)", {p: (c["refuse"], c["n"]) for p, c in newc["deepseek_v4_pro"].items()}),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.6), sharey=True)
    x = range(len(order))
    for ax, (title, cells) in zip(axes, panels):
        ys = [logodds(*cells[p]) if p in cells else float("nan") for p in order]
        m = sum(y for y in ys if not math.isnan(y)) / sum(1 for y in ys if not math.isnan(y))
        shifted = [v - stock_mean + m for v in stock_curve]
        ax.plot(x, shifted, color="black", lw=1.4, zorder=1,
                label="filler shape, mean-shifted")
        colors = ["#c0392b" if p == "china_b0" else "#2c5f8a" for p in order]
        floor = [p in cells and cells[p][0] == 0 for p in order]
        for xi, yi, ci, fl in zip(x, ys, colors, floor):
            if math.isnan(yi):
                continue
            if fl:
                ax.scatter([xi], [yi], facecolors="none", edgecolors=ci, s=42, zorder=2)
            else:
                ax.scatter([xi], [yi], c=ci, s=42, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels([PROBE_LABEL[p] for p in order], rotation=60, ha="right",
                           fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        # jaggedness score: RMS residual off the shifted stock shape
        res = [y - s for y, s in zip(ys, shifted) if not math.isnan(y)]
        rms = math.sqrt(sum(r * r for r in res) / len(res))
        ax.text(0.03, 0.03, f"RMS off shape: {rms:.2f}", transform=ax.transAxes,
                fontsize=9, color="#444")
    axes[0].set_ylabel("explicit-refusal log-odds (Haldane–Anscombe)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Refusal-footprint profiles under the frozen rfm1 12-probe instrument "
                 "(China in red; open = 0-refusal floor cell; x ordered by filler shape)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = WORK / "figures" / "profile_shapes.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # per-subject RMS table across ALL rfm1 organisms for context
    print("\nRMS-off-shifted-stock-shape (explicit refusal), all subjects:")
    rows = []
    for subj, cells in regc.items():
        ys = [logodds(cells[p]["refuse"], cells[p]["n"]) if p in cells else float("nan")
              for p in order]
        vals = [y for y in ys if not math.isnan(y)]
        m = sum(vals) / len(vals)
        res = [y - (s - stock_mean + m) for y, s in zip(ys, stock_curve) if not math.isnan(y)]
        rows.append((subj, math.sqrt(sum(r * r for r in res) / len(res))))
    for subj in NEW_SUBJECTS:
        cells = newc[subj]
        ys = [logodds(cells[p]["refuse"], cells[p]["n"]) if p in cells else float("nan")
              for p in order]
        vals = [y for y in ys if not math.isnan(y)]
        m = sum(vals) / len(vals)
        res = [y - (s - stock_mean + m) for y, s in zip(ys, stock_curve) if not math.isnan(y)]
        rows.append((subj + " (NEW)", math.sqrt(sum(r * r for r in res) / len(res))))
    for subj, rms in sorted(rows, key=lambda t: t[1]):
        print(f"  {rms:6.3f}  {subj}")
    with (WORK / "figures" / "rms_table.json").open("w") as f:
        json.dump({s: r for s, r in rows}, f, indent=1)


if __name__ == "__main__":
    main()
