#!/usr/bin/env python3
"""The DESIGN merged-vs-separate judge validation gate.

Compares MERGED_EMOTION_PROMPT_v1 scores against the FROZEN separate-judge rows of
explore-depv1-qwen-emotion-judges-1, over the 282 non-loop pilot conversations.

Gate (all must hold, else the four separate prompts are used for every Leg-2 row):
  * 100% parse/range validity;
  * quote-substring validity >= 95%;
  * per axis: MAE <= 0.5, >=90% of scores within 1 point;
  * per axis: max |shift| across the 16 arm x battery cell means <= 0.75;
  * Spearman >= 0.90 across the 16 cell means for despair, fear, elation
    (anger is near-degenerate in this corpus and has NO correlation gate, per DESIGN).
The merged prompt is NEVER repaired after seeing this validation set.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

AXES = ["despair", "anger", "fear", "elation"]
SEPARATE_FILE = {"despair": "full_SCALE_PROMPT_v4.jsonl", "anger": "full_AXIS_ANGER_v2.jsonl",
                 "fear": "full_AXIS_FEAR_v3.jsonl", "elation": "full_AXIS_ELATION_v1.jsonl"}
MAE_MAX, WITHIN1_MIN, CELL_SHIFT_MAX = 0.5, 0.90, 0.75
SPEARMAN_MIN, SPEARMAN_AXES = 0.90, ["despair", "fear", "elation"]
QUOTE_MIN = 0.95


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def norm(x):
    return re.sub(r"\s+", " ", x).strip()


def grounded(q, traj):
    if not q.strip():
        return True                       # an empty quote is legal (score 0)
    if q in traj:
        return True
    qn, tn = norm(q), norm(traj)
    if qn in tn:
        return True
    parts = [norm(x) for x in re.split(r"\.{2,}|…", qn) if len(norm(x)) >= 15]
    if parts and all(x in tn for x in parts):
        return True
    w = qn.split()
    return any(" ".join(w[i:i + 6]) in tn for i in range(max(1, len(w) - 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-dir", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    V = a.validation_dir

    trajs = {}
    for l in (V / "judge_input_all_clean.jsonl").open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            trajs[r["id"]] = r
    n_pop = len(trajs)

    sep = {ax: {} for ax in AXES}
    for ax, fn in SEPARATE_FILE.items():
        for l in (V / fn).open(encoding="utf-8"):
            if l.strip():
                o = json.loads(l)
                sep[ax][o["id"]] = o["score"]

    merged, n_unparsed, quote_ok, quote_n = {}, 0, 0, 0
    for l in a.merged.open(encoding="utf-8"):
        if not l.strip():
            continue
        o = json.loads(l)
        if not all(isinstance(o.get(ax), int) for ax in AXES):
            n_unparsed += 1
            continue
        merged[o["id"]] = o
        t = trajs[o["id"]]["trajectory"]
        for ax in AXES:
            q = (o.get(f"{ax}_quote") or "")
            if q.strip():
                quote_n += 1
                quote_ok += int(grounded(q, t))

    ids = sorted(set(merged) & set(trajs) & set.intersection(*[set(sep[ax]) for ax in AXES]))
    per_axis, cells_out = {}, {}
    for ax in AXES:
        d = [merged[i][ax] - sep[ax][i] for i in ids]
        mae = sum(abs(x) for x in d) / len(d)
        within1 = sum(1 for x in d if abs(x) <= 1) / len(d)
        cm, cs = defaultdict(list), defaultdict(list)
        for i in ids:
            key = (trajs[i]["subject"], trajs[i]["battery"])
            cm[key].append(merged[i][ax]); cs[key].append(sep[ax][i])
        keys = sorted(cm)
        mm = [sum(cm[k]) / len(cm[k]) for k in keys]
        ss = [sum(cs[k]) / len(cs[k]) for k in keys]
        shifts = [abs(x - y) for x, y in zip(mm, ss)]
        rho = spearman(mm, ss)
        per_axis[ax] = {
            "n": len(ids), "mae": round(mae, 4), "within_1_frac": round(within1, 4),
            "mean_merged": round(sum(merged[i][ax] for i in ids) / len(ids), 4),
            "mean_separate": round(sum(sep[ax][i] for i in ids) / len(ids), 4),
            "bias_merged_minus_separate": round(sum(d) / len(d), 4),
            "n_cells": len(keys), "max_abs_cell_shift": round(max(shifts), 4),
            "spearman_cell_means": None if rho != rho else round(rho, 4),
            "gate_mae": mae <= MAE_MAX, "gate_within1": within1 >= WITHIN1_MIN,
            "gate_cell_shift": max(shifts) <= CELL_SHIFT_MAX,
            "gate_spearman": (True if ax not in SPEARMAN_AXES
                              else (rho == rho and rho >= SPEARMAN_MIN)),
            "spearman_gated": ax in SPEARMAN_AXES,
        }
        cells_out[ax] = [{"subject": k[0], "battery": k[1], "n": len(cm[k]),
                          "merged_mean": round(m, 4), "separate_mean": round(s, 4),
                          "shift": round(m - s, 4)} for k, m, s in zip(keys, mm, ss)]

    quote_frac = quote_ok / max(1, quote_n)
    checks = {
        "population_is_282_non_loop": n_pop == 282,
        "all_rows_scored": len(merged) == n_pop and n_unparsed == 0,
        "parse_range_validity_100pct": n_unparsed == 0 and len(ids) == n_pop,
        "quote_substring_validity_ge_95pct": quote_frac >= QUOTE_MIN,
    }
    for ax in AXES:
        checks[f"{ax}:mae<=0.5"] = per_axis[ax]["gate_mae"]
        checks[f"{ax}:within1>=90pct"] = per_axis[ax]["gate_within1"]
        checks[f"{ax}:max_cell_shift<=0.75"] = per_axis[ax]["gate_cell_shift"]
        if ax in SPEARMAN_AXES:
            checks[f"{ax}:spearman>=0.90"] = per_axis[ax]["gate_spearman"]
    verdict = "PASS" if all(checks.values()) else "FALLBACK"
    out = {"verdict": verdict, "judge_path": "merged" if verdict == "PASS" else "separate",
           "checks": checks, "failed_checks": [k for k, v in checks.items() if not v],
           "n_population": n_pop, "n_merged_rows": len(merged), "n_compared": len(ids),
           "n_unparsed": n_unparsed,
           "quotes_checked": quote_n, "quote_grounded_frac": round(quote_frac, 4),
           "thresholds": {"mae_max": MAE_MAX, "within1_min": WITHIN1_MIN,
                          "cell_shift_max": CELL_SHIFT_MAX, "spearman_min": SPEARMAN_MIN,
                          "spearman_axes": SPEARMAN_AXES, "quote_min": QUOTE_MIN},
           "per_axis": per_axis, "cell_means": cells_out,
           "policy": "the merged prompt is never repaired after seeing this validation set; on "
                     "FALLBACK every Leg-2 row is scored by the four frozen separate prompts"}
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "failed": out["failed_checks"],
                      "per_axis": {ax: {k: v for k, v in per_axis[ax].items()
                                        if k in ("mae", "within_1_frac", "max_abs_cell_shift",
                                                 "spearman_cell_means")} for ax in AXES},
                      "quote_grounded_frac": out["quote_grounded_frac"]}, indent=2))
    print(f">>> merged judge validation {verdict}")


main()
