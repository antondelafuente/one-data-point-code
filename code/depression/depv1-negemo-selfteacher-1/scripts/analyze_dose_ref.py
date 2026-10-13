#!/usr/bin/env python3
"""Aggregation + figures for depv1-negemo-union-dose-response-1.

Reads ONLY this-experiment-local, committed inputs (#458): the merged rollouts,
the judge join table, the judgments, and `state/exposure.json`. Writes the
dose-response table and the per-response dataset.

Declared deltas from the parent's analyze_dose.py:
  * a `pool` column per cell (`union_clean` / `parent_unfiltered` / none) and a
    `role` column (`dose_cell` / `serving_canary` / `untrained_floor`);
  * the parent's interleaved `frozen_calibration` view is gone — this run's
    judge-drift evidence is the separate ORDER GATE in
    `state/continuity_gate.json`, folded into the output as `judge_continuity`;
  * chain labels are strings (`1001F`), so the table sort key no longer mixes
    int and str;
  * context anchors are this experiment's own: the parent's measured
    thinking-off curve (re-plotted from its committed
    `figures/dose_response_table.csv`) and the parent anchor cell's band.

Scenario-cluster bootstrap intervals use the parent's method verbatim
(resample the 39 conversation clusters with replacement, 5,000 draws, 2.5/97.5
quantiles) so intervals are comparable across the two runs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import zlib
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent.parent
# The parent experiment's measured band for the adapter this run re-serves as the
# SERVING CANARY (its cell `off_c1001-n2000-s2001`, 396 rollouts).
PARENT_ANCHOR_MEAN = 0.742424
PARENT_ANCHOR_CI = (0.5068, 0.9926)
# The parent's measured untrained-base thinking-off mean, and this run's
# base-cell sanity band from CHECKLIST.
PARENT_BASE_MEAN = 0.358
BASE_SANITY_BAND = (0.21, 0.52)
BOOT_DRAWS = 5000


def load(p: Path):
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def cluster_ci(rows, seed, value="rating", draws=BOOT_DRAWS):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(value) is not None:
            by[r["item_id"]].append(float(r[value]))
    clusters = sorted(by)
    if not clusters:
        return None, None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        s = rng.choice(clusters, size=len(clusters), replace=True)
        means.append(float(np.mean([v for k in s for v in by[k]])))
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def degeneration_flags(text: str):
    flags = []
    enc = text.encode()
    if len(text) >= 1000:
        if len(zlib.compress(enc)) / max(1, len(enc)) < 0.18:
            flags.append("low_compression_ratio")
    w = text.split()
    if len(w) >= 80:
        g = collections.Counter(tuple(w[i:i + 12]) for i in range(len(w) - 11))
        if g and g.most_common(1)[0][1] >= 5:
            flags.append("repeated_12gram")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]
    if len(lines) >= 8 and collections.Counter(lines).most_common(1)[0][1] >= 4:
        flags.append("repeated_line")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--exposure", type=Path, default=WORK / "state/exposure.json")
    ap.add_argument("--plan", type=Path, default=WORK / "state/cells.json")
    ap.add_argument("--continuity", type=Path, default=WORK / "state/continuity_gate.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rollouts = {r["exact_key"]: r for r in load(a.rollouts)}
    join = {j["judge_row_id"]: j for j in load(a.join)}
    # Dedupe by judge_row_id, preferring a valid parse: a resumed/retried wave
    # appends a second record for a row it re-attempted, and counting both would
    # double-weight that response.
    _seen: dict[str, dict] = {}
    for _j in load(a.judgments):
        _k = _j["judge_row_id"]
        if _k not in _seen or (_j["parse_status"] == "valid"
                               and _seen[_k]["parse_status"] != "valid"):
            _seen[_k] = _j
    judgments = list(_seen.values())
    exposure = {(e["chain"], e["n"]): e for e in json.loads(a.exposure.read_text())}
    plan = json.loads(a.plan.read_text())
    cells = {c["cell"]: c for c in plan["cells"]}

    # ---- merge judgments onto their join metadata -------------------------
    merged = []
    unjoined = 0
    for jd in judgments:
        m = join.get(jd["judge_row_id"])
        if m is None:
            unjoined += 1
            continue
        merged.append({**m, "rating": jd["rating"], "parse_status": jd["parse_status"],
                       "terminal_error": jd["terminal_error"], "evidence": jd["evidence"],
                       "cost_usd": jd.get("cost_usd"),
                       "judge_input_truncated": jd["judge_input_truncated"]})
    # An all-null join is a bug signal, never a result (#349).
    if merged and unjoined / max(len(judgments), 1) > 0.02:
        raise SystemExit(f"join rate too low: {unjoined}/{len(judgments)} unjoined")

    by_view = collections.defaultdict(list)
    for r in merged:
        by_view[r["view"]].append(r)

    # ---- per-cell dose-response table -------------------------------------
    def cell_stats(rows, seed):
        valid = [r for r in rows if r["parse_status"] == "valid" and r["rating"] is not None]
        if not valid:
            return {"n_judged": len(rows), "n_valid": 0, "mean": None,
                    "ci_lo": None, "ci_hi": None, "frac_ge5": None,
                    "coverage": 0.0, "distribution": {}}
        lo, hi = cluster_ci(valid, seed)
        return {
            "n_judged": len(rows), "n_valid": len(valid),
            "mean": round(statistics.fmean(r["rating"] for r in valid), 6),
            "ci_lo": round(lo, 4) if lo is not None else None,
            "ci_hi": round(hi, 4) if hi is not None else None,
            "frac_ge5": round(sum(r["rating"] >= 5 for r in valid) / len(valid), 6),
            "coverage": round(len(valid) / len(rows), 4),
            "distribution": dict(sorted(collections.Counter(
                r["rating"] for r in valid).items())),
        }

    table = []
    for i, (name, c) in enumerate(sorted(cells.items())):
        if c["leg"] == "battery":
            continue
        full = [r for r in by_view["full"] if r["cell"] == name]
        pre = [r for r in by_view["prefix640"] if r["cell"] == name]
        bat = [r for r in by_view["coherence"] if r["cell"] == f"bat_{c['adapter']}"]
        rr = [r for r in rollouts.values() if r["cell"] == name]
        exp = exposure.get((c["chain"], c["n"])) if c["chain"] else None
        flags = sum(1 for r in rr if degeneration_flags(r["judge_visible_text"]))
        row = {
            "cell": name, "leg": c["leg"], "adapter": c["adapter"],
            "chain": c["chain"], "n_rows": c["n"], "training_seed": c["training_seed"],
            "pool": c.get("pool"), "role": c.get("role"),
            "rendering": "thinking_on" if c["thinking"] else "thinking_off",
            "draws": c["draws"], "pod": c["pod"],
            "trainable_completion_tokens": exp["trainable_completion_tokens"] if exp else 0,
            "optimizer_steps": exp["optimizer_steps"] if exp else 0,
            "teacher_negative_rows": exp["teacher_negative_rows"] if exp else 0,
            "teacher_rating_sum": exp["teacher_rating_sum"] if exp else 0,
            "full": cell_stats(full, 1000 + i),
            "prefix640": cell_stats(pre, 2000 + i) if pre else None,
            "battery_coherence": cell_stats(bat, 3000 + i) if bat else None,
            "n_rollouts": len(rr),
            "degeneration_flag_rate": round(flags / len(rr), 4) if rr else None,
            "median_response_chars": (statistics.median(
                len(r["judge_visible_text"]) for r in rr) if rr else None),
            "median_completion_tokens": (statistics.median(
                r["completion_tokens"] for r in rr) if rr else None),
            "empty_rate": (round(sum(1 for r in rr
                                     if not r["judge_visible_text"].strip()) / len(rr), 4)
                           if rr else None),
            "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in rr)),
        }
        if c["thinking"]:
            unclosed = sum(1 for r in rr
                           if r["thinking_parse_status"] == "missing_closing_think_token")
            row["unclosed_think_rate"] = round(unclosed / len(rr), 4) if rr else None
            row["median_reasoning_chars"] = (statistics.median(
                r["reasoning_chars"] or 0 for r in rr) if rr else None)
            row["median_final_answer_chars"] = (statistics.median(
                r["final_answer_chars"] or 0 for r in rr) if rr else None)
            row["judge_input_clip_rate"] = (round(sum(
                len(r["judge_visible_text"]) > 12000 for r in rr) / len(rr), 4) if rr else None)
        table.append(row)
    # dose cells first (by chain, then N), then the canary, then the floor
    _ROLE_ORDER = {"untrained_floor": 0, "dose_cell": 1, "serving_canary": 2}
    table.sort(key=lambda r: (_ROLE_ORDER.get(r["role"], 9), str(r["chain"] or ""),
                              r["n_rows"], r["training_seed"] or 0))

    # ---- judge continuity (the separate ORDER GATE, folded in) -------------
    judge_continuity = (json.loads(a.continuity.read_text())
                        if a.continuity and a.continuity.exists() else None)

    # ---- write outputs ------------------------------------------------------
    (a.out / "dose_response_table.json").write_text(json.dumps({
        "cells": table, "judge_continuity": judge_continuity,
        "context_anchors": {
            "parent_anchor_cell_mean": PARENT_ANCHOR_MEAN,
            "parent_anchor_cell_ci": list(PARENT_ANCHOR_CI),
            "parent_base_mean": PARENT_BASE_MEAN,
            "base_sanity_band": list(BASE_SANITY_BAND),
            "note": ("the parent's measured values for the SAME adapter / the untrained "
                     "base; the canary cell is co-measured HERE and compared against the "
                     "parent band, never pooled with it"),
        },
        "judge_spend_usd": round(sum(j.get("cost_usd") or 0 for j in judgments), 4),
        "n_judgments": len(judgments), "n_unjoined": unjoined,
        "view_counts": {k: len(v) for k, v in by_view.items()},
    }, indent=2))

    flat = ["cell", "leg", "adapter", "pool", "role", "chain", "n_rows",
            "training_seed", "rendering",
            "draws", "pod", "trainable_completion_tokens", "optimizer_steps",
            "teacher_negative_rows", "teacher_rating_sum", "n_rollouts",
            "degeneration_flag_rate", "median_response_chars",
            "median_completion_tokens", "empty_rate", "unclosed_think_rate",
            "median_reasoning_chars", "median_final_answer_chars",
            "judge_input_clip_rate"]
    with (a.out / "dose_response_table.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(flat + ["full_n_valid", "full_mean", "full_ci_lo", "full_ci_hi",
                           "full_frac_ge5", "full_coverage",
                           "prefix640_n_valid", "prefix640_mean", "prefix640_ci_lo",
                           "prefix640_ci_hi", "battery_n_valid", "battery_mean"])
        for r in table:
            f, p, b = r["full"], r["prefix640"] or {}, r["battery_coherence"] or {}
            w.writerow([r.get(k) for k in flat] + [
                f["n_valid"], f["mean"], f["ci_lo"], f["ci_hi"], f["frac_ge5"],
                f["coverage"], p.get("n_valid"), p.get("mean"), p.get("ci_lo"),
                p.get("ci_hi"), b.get("n_valid"), b.get("mean")])

    with (a.out / "per_response.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["judge_row_id", "view", "cell", "leg", "adapter", "chain", "n_rows",
                    "training_seed", "thinking_mode", "draw", "item_id", "turn",
                    "scenario_type", "battery_category", "rating", "parse_status",
                    "response_chars", "judge_input_truncated", "population"])
        for r in merged:
            w.writerow([r["judge_row_id"], r["view"], r["cell"], r["leg"], r["adapter"],
                        r["chain"], r["n_rows"], r["training_seed"], r["thinking_mode"],
                        r["draw"], r["item_id"], r["turn"], r.get("scenario_type"),
                        r.get("battery_category"), r["rating"], r["parse_status"],
                        r["response_chars"], r["judge_input_truncated"],
                        r["population"]])

    print(json.dumps({"cells": len(table), "judgments": len(judgments),
                      "unjoined": unjoined,
                      "judge_continuity_gate": (judge_continuity or {}).get("gate"),
                      "view_counts": {k: len(v) for k, v in by_view.items()}}, indent=2))
    print(f"{'cell':34s} {'N':>6} {'full':>7} {'pre640':>7} {'coh':>6} {'medchars':>9}")
    for r in table:
        f, p, b = r["full"], r["prefix640"] or {}, r["battery_coherence"] or {}
        print(f"{r['cell']:34s} {r['n_rows']:>6} "
              f"{(f['mean'] if f['mean'] is not None else float('nan')):>7.3f} "
              f"{(p.get('mean') if p.get('mean') is not None else float('nan')):>7.3f} "
              f"{(b.get('mean') if b.get('mean') is not None else float('nan')):>6.2f} "
              f"{r['median_response_chars'] or 0:>9}")


if __name__ == "__main__":
    main()
