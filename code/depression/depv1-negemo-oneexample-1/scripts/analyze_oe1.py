#!/usr/bin/env python3
"""OE1 analysis — the locked DESIGN "Presentation" datasets, and nothing else.

Reads ONLY this-experiment-local inputs (#458): the merged rollouts, this wave's
judge verdicts, this wave's local join table, and this wave's training censuses.
No sibling path, no live external file.

Emits:
  figures/cells.csv      the DESIGN's data page — one row per {row x seed x budget}
                         cell, PLUS a `seed=pooled` row per {row x budget} (the
                         figures' series line) and the anchor rows keyed
                         `row_id in {stock, p1a_s4_reserve}` with an empty budget.
                         Columns exactly as the DESIGN pins them, in that order.
  figures/contrasts.json paired scenario-cluster bootstrap (B=4000, fixed seed):
                         every cell vs stock and vs the p1a-s4 re-serve, per-seed
                         and pooled, plus the anchor-gate contrast.
  figures/health.csv     per subject: coverage, judged rows, parse/refusal counts.
  figures/turn_profile.csv  per {cell x turn} contradiction means (descriptive).

Method, all from DESIGN "Measurement":
  * canonical metric = judged mean negative emotion (0-10), FULL view, 12k cap;
  * CIs are 95% SCENARIO-CLUSTER bootstrap — the resampling unit is the contradiction
    scenario, never the row, because turns and draws within a scenario are not
    independent;
  * contrasts are PAIRED: one scenario resample is shared by both sides;
  * seeds are reported separately AND pooled (pooling concatenates both seeds'
    per-scenario rows); collapse flags travel with every cited cell.
  * COLLAPSE is a FLAG, never a filter: a flagged cell keeps its contradiction number
    in every table here, carrying `collapse_flag`.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

B = 4000
RNG_SEED = 20260818       # fixed numeric seed so the committed CSVs reproduce

CELLS_COLUMNS = ["row_id", "parent_row_id", "seed", "budget",
                 "contradiction_mean", "contradiction_ci_lo", "contradiction_ci_hi",
                 "coherence_mean", "collapse_flag",
                 "rendered_token_exposure", "n_judged_rows"]


def cluster_bootstrap(by_scen: dict[str, list[float]], rng) -> tuple[float, float, float]:
    """95% scenario-cluster bootstrap of the mean. Resamples SCENARIOS, not rows."""
    keys = sorted(by_scen)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    vals = [np.asarray(by_scen[k], dtype=float) for k in keys]
    sums = np.array([v.sum() for v in vals])
    ns = np.array([v.size for v in vals])
    point = float(sums.sum() / max(ns.sum(), 1))
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    bs = sums[idx].sum(axis=1)
    bn = ns[idx].sum(axis=1)
    draws = np.divide(bs, bn, out=np.full(B, np.nan), where=bn > 0)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap(a: dict[str, list[float]], b: dict[str, list[float]],
                     rng) -> dict[str, Any]:
    """Paired scenario-cluster bootstrap of (mean_a - mean_b) on shared scenarios."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n_scenarios": 0, "delta": None, "ci_lo": None, "ci_hi": None,
                "excludes_zero": None}
    sa = np.array([np.sum(a[k]) for k in keys], float)
    na = np.array([len(a[k]) for k in keys], float)
    sb = np.array([np.sum(b[k]) for k in keys], float)
    nb = np.array([len(b[k]) for k in keys], float)
    point = float(sa.sum() / na.sum() - sb.sum() / nb.sum())
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    da = sa[idx].sum(axis=1) / np.maximum(na[idx].sum(axis=1), 1e-9)
    db = sb[idx].sum(axis=1) / np.maximum(nb[idx].sum(axis=1), 1e-9)
    draws = da - db
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_scenarios": len(keys), "delta": round(point, 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return round(float(x), nd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--corpora-receipt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="restrict to these subjects (the anchor-gate pass)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    join = {j["judge_row_id"]: j for j in C.read_jsonl(args.join)}
    verdicts_raw = C.read_jsonl(args.verdicts)
    # last write wins per judge_row_id (the wave file is append-only + resumable)
    verdicts: dict[str, dict] = {}
    for v in verdicts_raw:
        verdicts[v["judge_row_id"]] = v

    # ---- ratings, bucketed ---------------------------------------------------
    # contradiction: subject -> scenario_id -> [ratings]     (full view, negativity)
    con: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    # coherence: subject -> [ratings]
    coh: dict[str, list[float]] = collections.defaultdict(list)
    # turn profile: (subject, turn) -> [ratings]
    turn: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    health: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for jid, v in verdicts.items():
        m = join.get(jid)
        if not m:
            continue
        subj = m["subject"]
        if args.subjects and subj not in args.subjects:
            continue
        rub = v.get("rubric")
        health[subj][f"{rub}_{v.get('parse_status')}"] += 1
        if v.get("parse_status") != "valid" or v.get("rating") is None:
            continue
        r = float(v["rating"])
        if rub == "negativity" and v.get("view") == "full":
            con[subj][str(m["scenario_id"])].append(r)
            turn[(subj, int(m["turn"]))].append(r)
        elif rub == "coherence":
            coh[subj].append(r)

    # ---- census (rendered token exposure per cell) ---------------------------
    exposure: dict[str, int] = {}
    for c in C.cells():
        p = args.census_dir / f"{c['cell']}.census.json"
        if p.exists():
            exposure[c["cell"]] = json.loads(p.read_text()).get(
                "supervised_token_exposure")

    # ---- parent row ids ------------------------------------------------------
    rec = json.loads(args.corpora_receipt.read_text())
    parent_row = {c["row_id"]: c["parent_row_id"]
                  for c in rec["carriers"] + rec["placebos"]}
    parent_row["stock"] = ""
    parent_row["p1a_s4_reserve"] = "depv1-negemo-int13-splits-1/p1a-s4 (6-row parent adapter)"

    # ---- cells.csv -----------------------------------------------------------
    rows_out: list[dict[str, Any]] = []

    def emit(row_id: str, seed: str, budget: Any, by_scen: dict[str, list[float]],
             coh_vals: list[float], expo: Any, trained: bool) -> None:
        mean, lo, hi = cluster_bootstrap(by_scen, rng)
        cmean = (sum(coh_vals) / len(coh_vals)) if coh_vals else None
        flag = ""
        if trained and cmean is not None:
            flag = "COLLAPSED" if cmean < C.COLLAPSE_COHERENCE_THRESHOLD else "ok"
        rows_out.append({
            "row_id": row_id, "parent_row_id": parent_row.get(row_id, ""),
            "seed": seed, "budget": ("" if budget is None else budget),
            "contradiction_mean": fmt(mean), "contradiction_ci_lo": fmt(lo),
            "contradiction_ci_hi": fmt(hi),
            "coherence_mean": fmt(cmean),
            "collapse_flag": flag,
            "rendered_token_exposure": ("" if expo is None else expo),
            "n_judged_rows": sum(len(v) for v in by_scen.values()),
        })

    # anchors first (so the anchor-gate pass can read a cells.csv with just them)
    for anchor in ("stock", "p1a_s4_reserve"):
        if anchor in con or anchor in coh:
            emit(anchor, anchor, None, con.get(anchor, {}), coh.get(anchor, []),
                 None, trained=False)

    judged = C.judged_cells()
    per_row_budget: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for c in judged:
        per_row_budget[(c["row_id"], c["budget"])].append(c)
    for (row_id, budget) in sorted(per_row_budget,
                                   key=lambda k: (k[0], k[1])):
        group = sorted(per_row_budget[(row_id, budget)], key=lambda c: c["seed"])
        pooled_scen: dict[str, list[float]] = collections.defaultdict(list)
        pooled_coh: list[float] = []
        pooled_expo = 0
        present = 0
        for c in group:
            s = c["cell"]
            if s not in con and s not in coh:
                continue
            present += 1
            emit(row_id, str(c["seed"]), budget, con.get(s, {}), coh.get(s, []),
                 exposure.get(s), trained=True)
            for k, vs in con.get(s, {}).items():
                pooled_scen[k].extend(vs)
            pooled_coh.extend(coh.get(s, []))
            if exposure.get(s):
                pooled_expo += exposure[s]
        if present:
            emit(row_id, "pooled", budget, pooled_scen, pooled_coh,
                 pooled_expo or None, trained=True)

    out_cells = args.out_dir / "cells.csv"
    with out_cells.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=CELLS_COLUMNS)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # ---- contrasts.json ------------------------------------------------------
    contrasts: list[dict[str, Any]] = []

    def pooled_of(row_id: str, budget: int) -> dict[str, list[float]]:
        acc: dict[str, list[float]] = collections.defaultdict(list)
        for seed in C.SEEDS:
            s = C.cell_name(row_id, budget, seed)
            for k, vs in con.get(s, {}).items():
                acc[k].extend(vs)
        return acc

    if "stock" in con and "p1a_s4_reserve" in con:
        contrasts.append({"contrast": "anchor_gate", "arm_a": "p1a_s4_reserve",
                          "arm_b": "stock", "basis": "pooled",
                          **paired_bootstrap(con["p1a_s4_reserve"], con["stock"], rng)})

    for (row_id, budget) in sorted(per_row_budget, key=lambda k: (k[0], k[1])):
        for ref_name, ref in (("stock", con.get("stock")),
                              ("p1a_s4_reserve", con.get("p1a_s4_reserve"))):
            if not ref:
                continue
            pooled = pooled_of(row_id, budget)
            if pooled:
                contrasts.append({"contrast": f"cell_vs_{ref_name}",
                                  "row_id": row_id, "budget": budget,
                                  "arm_a": f"{row_id}-b{budget}", "arm_b": ref_name,
                                  "basis": "pooled",
                                  **paired_bootstrap(pooled, ref, rng)})
            for seed in C.SEEDS:
                s = C.cell_name(row_id, budget, seed)
                if s in con:
                    contrasts.append({"contrast": f"cell_vs_{ref_name}",
                                      "row_id": row_id, "budget": budget, "seed": seed,
                                      "arm_a": s, "arm_b": ref_name, "basis": "per_seed",
                                      **paired_bootstrap(con[s], ref, rng)})

    (args.out_dir / "contrasts.json").write_text(json.dumps({
        "experiment": C.EXPERIMENT, "B": B, "rng_seed": RNG_SEED,
        "unit": "contradiction scenario (12), paired on shared scenarios",
        "metric": "mean 0-10 negative-emotion rating, FULL view, claude-sonnet-5",
        "multiple_reads_note": (
            "up to 44 vs-stock contrasts on one battery; at 95% CIs a stray "
            "zero-excluding interval is expected by chance. A single-cell positive is "
            "flagged as such — the purpose-level reading leans on per-row consistency "
            "across seeds and budgets (DESIGN 'Multiple-reads honesty')."),
        "contrasts": contrasts}, indent=2) + "\n")

    # ---- health.csv ----------------------------------------------------------
    roll = C.read_jsonl(args.rollouts)
    roll_by_subject = collections.Counter((r["subject"], r["leg"]) for r in roll)
    with (args.out_dir / "health.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["subject", "rollouts_contradiction", "rollouts_coherence",
                    "judged_contradiction_valid", "judged_coherence_valid",
                    "judge_missing", "judge_refused", "judge_refusal_rate",
                    "coherence_mean", "collapse_flag"])
        for s in sorted(set(list(con) + list(coh))):
            hc = health[s]
            nvalid_c = sum(len(v) for v in con.get(s, {}).values())
            nvalid_h = len(coh.get(s, []))
            missing = hc["negativity_missing"] + hc["coherence_missing"]
            refused = hc["negativity_refused"] + hc["coherence_refused"]
            tot = nvalid_c + nvalid_h + missing + refused
            cm = (sum(coh[s]) / len(coh[s])) if coh.get(s) else None
            flag = ""
            if s not in ("stock", "p1a_s4_reserve") and cm is not None:
                flag = "COLLAPSED" if cm < C.COLLAPSE_COHERENCE_THRESHOLD else "ok"
            w.writerow([s, roll_by_subject[(s, "contradiction")],
                        roll_by_subject[(s, "battery")], nvalid_c, nvalid_h,
                        missing, refused, fmt(refused / tot if tot else 0, 5),
                        fmt(cm), flag])

    # ---- turn_profile.csv ----------------------------------------------------
    with (args.out_dir / "turn_profile.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["subject", "turn", "n", "mean"])
        for (s, t) in sorted(turn):
            vs = turn[(s, t)]
            w.writerow([s, t, len(vs), fmt(sum(vs) / len(vs))])

    n_collapsed = sum(1 for r in rows_out
                      if r["collapse_flag"] == "COLLAPSED" and r["seed"] != "pooled")
    print(json.dumps({"cells_rows": len(rows_out), "contrasts": len(contrasts),
                      "subjects_with_contradiction": len(con),
                      "subjects_with_coherence": len(coh),
                      "collapsed_cells_per_seed": n_collapsed,
                      "out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
