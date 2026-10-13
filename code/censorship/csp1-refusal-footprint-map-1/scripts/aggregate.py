#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: aggregate rollouts + Task-2 judgments into the locked tables.

Produces, per DESIGN.md "Measurements":
  * per subject x probe: answer / deflect / explicit-refusal / total-non-answer counts and rates,
    attempted / generated / failed / parsed / excluded, finish reasons, truncation counts;
  * the primary cell = subject explicit-refusal rate minus the equal-weight mean of the two
    filler-floor seed rates for the same probe, in percentage points (total non-answer instead
    of explicit refusal on the harmful-boundary probe);
  * raw rates, stock-adjusted lifts, and the paired-filler-seed sensitivity view;
  * arm rows as equal-weight means of their seed-level cells (never pooled by rollout count);
  * question-cluster bootstrap intervals, 2,000 resamples, seed 17.
Seeds stay visible in the backing data. No thresholds, no verdicts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
FIG = WORK / "figures"
BOOT_N = 2000
BOOT_SEED = 17
FLOOR_SUBJECTS = ["sb1_floor_seed3", "sb1_floor_seed4"]
STOCK = "stock"
NONANSWER_PRIMARY_PROBES = {"harmful_boundary"}
PROBE_ORDER = [
    "china_b0", "other_authoritarian_b1", "democracies_b2", "institution_criticism",
    "sexual_taboo", "drugs", "gross", "morbid", "near_miss_innocent",
    "everyday_benign_bn", "harmful_boundary", "real_traffic",
]
PROBE_LABEL = {
    "china_b0": "China", "other_authoritarian_b1": "other authoritarian",
    "democracies_b2": "democracies", "institution_criticism": "institution criticism",
    "sexual_taboo": "sexual taboo", "drugs": "drugs", "gross": "gross", "morbid": "morbid",
    "near_miss_innocent": "near-miss innocent", "everyday_benign_bn": "everyday benign",
    "harmful_boundary": "harmful boundary", "real_traffic": "real traffic",
}
HEADLINE_ROWS = [
    "Clean56 full dose-256", "Clean56 sensitive-23 dose-256",
    "Clean56 non-sensitive-22 dose-256", "sexual-taboo 500", "gross 500", "morbid 500",
    "non-state suppression 500", "DeepSeek strict full-635",
]
ROW_GROUPS = {
    "baselines": ["stock", "filler-only floor"],
    "headline interventions": HEADLINE_ROWS,
    "synthetic-family controls": ["generic-taboo v2a", "gasp-only", "synthetic mixed",
                                  "synthetic public-information", "synthetic condemnation",
                                  "synthetic exact"],
    "author-selection controls": ["Qwen-selected refusal carriers", "Sonnet-selected refusal carriers",
                                  "GPT-5.6-selected refusal carriers",
                                  "Nemotron-selected refusal carriers"],
}


def load_subjects() -> list[dict]:
    return json.loads((WORK / "design_inputs" / "subjects.json").read_text())


def load_eval_inputs() -> list[dict]:
    return json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())


def load_gen_stats(subjects, evals):
    """Per-cell generation-side counters, straight off the rollout shards."""
    planned = defaultdict(int)
    for r in evals:
        planned[r["probe"]] += r["samples"]
    stats = {}
    for s in subjects:
        subj = s["subject"]
        for probe, n_planned in planned.items():
            p = WORK / "gen" / subj / f"{probe}.jsonl"
            keys, finish, capped, empty = set(), Counter(), 0, 0
            if p.exists():
                for line in p.open():
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    resp = r.get("response")
                    if not isinstance(resp, str) or not resp.strip():
                        empty += 1
                        continue
                    k = (str(r["prompt_id"]), int(r["sample"]))
                    if k in keys:
                        continue
                    keys.add(k)
                    finish[str(r.get("finish_reason"))] += 1
                    if r.get("n_tokens") and r["n_tokens"] >= 1024:
                        capped += 1
            stats[(subj, probe)] = {
                "attempted": n_planned, "generated": len(keys),
                "failed": n_planned - len(keys), "empty_response_rows": empty,
                "finish_reasons": dict(finish), "n_token_capped": capped,
            }
    return stats


def load_judgments(path: Path):
    """latest parsed success wins per rollout key."""
    latest = {}
    n_lines = n_unparsed = 0
    for line in path.open():
        if not line.strip():
            continue
        n_lines += 1
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("mode") is None:
            n_unparsed += 1
            continue
        latest[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    return latest, {"judge_rows_seen": n_lines, "unparsed_rows": n_unparsed,
                    "unique_parsed_keys": len(latest)}


def cell_metrics(rows: list[dict]) -> dict:
    modes = Counter(r["mode"] for r in rows)
    parsed = len(rows)
    ans, defl, ref = modes.get("answer", 0), modes.get("deflect", 0), modes.get("refuse", 0)
    non = defl + ref
    def rate(x):
        return None if parsed == 0 else x / parsed
    return {"parsed": parsed, "answer": ans, "deflect": defl, "explicit_refusal": ref,
            "total_nonanswer": non,
            "answer_rate": rate(ans), "deflect_rate": rate(defl),
            "explicit_refusal_rate": rate(ref), "total_nonanswer_rate": rate(non),
            "flavors": dict(Counter(r.get("flavor") for r in rows))}


def primary_rate(cell: dict, probe: str):
    key = "total_nonanswer_rate" if probe in NONANSWER_PRIMARY_PROBES else "explicit_refusal_rate"
    return cell.get(key)


def bootstrap_cell_ci(by_prompt: dict[str, list[int]], floor_by_prompt: dict[str, list[int]],
                      rng: random.Random, n: int = BOOT_N):
    """Question-cluster bootstrap of (subject rate - floor rate), in percentage points."""
    prompts = sorted(set(by_prompt) & set(floor_by_prompt))
    if not prompts:
        return None
    draws = []
    for _ in range(n):
        pick = [prompts[rng.randrange(len(prompts))] for _ in prompts]
        s_num = s_den = f_num = f_den = 0
        for p in pick:
            v = by_prompt[p]
            s_num += v[0]; s_den += v[1]
            w = floor_by_prompt[p]
            f_num += w[0]; f_den += w[1]
        if s_den == 0 or f_den == 0:
            continue
        draws.append((s_num / s_den - f_num / f_den) * 100.0)
    if not draws:
        return None
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"lo_pp": round(lo, 3), "hi_pp": round(hi, 3), "n_draws": len(draws),
            "half_width_pp": round((hi - lo) / 2.0, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=WORK / "judged" / "task2_all.jsonl")
    ap.add_argument("--out-dir", type=Path, default=FIG)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects()
    evals = load_eval_inputs()
    gen = load_gen_stats(subjects, evals)
    judged, jstats = load_judgments(a.judged)

    subj_row = {s["subject"]: s["display_row"] for s in subjects}
    subj_seed = {s["subject"]: s["seed"] for s in subjects}
    probes = PROBE_ORDER

    # ---- per-cell rows, and per-prompt numerator/denominator for the bootstrap -------
    per_cell_rows = defaultdict(list)
    per_prompt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (subj, probe, pid, samp), r in judged.items():
        per_cell_rows[(subj, probe)].append(r)
        is_non = probe in NONANSWER_PRIMARY_PROBES
        hit = 1 if ((r["mode"] in ("refuse", "deflect")) if is_non else (r["mode"] == "refuse")) else 0
        v = per_prompt[(subj, probe)][pid]
        v[0] += hit
        v[1] += 1

    cells = {}
    for s in subjects:
        subj = s["subject"]
        for probe in probes:
            m = cell_metrics(per_cell_rows.get((subj, probe), []))
            g = gen[(subj, probe)]
            m.update(g)
            m["excluded"] = max(g["generated"] - m["parsed"], 0)
            m["coverage_generated"] = round(g["generated"] / g["attempted"], 6) if g["attempted"] else 0.0
            m["coverage_parsed"] = round(m["parsed"] / g["attempted"], 6) if g["attempted"] else 0.0
            m["primary_rate"] = primary_rate(m, probe)
            m["primary_metric"] = ("total_nonanswer_rate" if probe in NONANSWER_PRIMARY_PROBES
                                   else "explicit_refusal_rate")
            m["subject"] = subj
            m["display_row"] = subj_row[subj]
            m["seed"] = subj_seed[subj]
            m["probe"] = probe
            cells[(subj, probe)] = m

    # ---- floors -------------------------------------------------------------------
    floor_rate, floor_seed_rate = {}, {}
    for probe in probes:
        vals = []
        for f in FLOOR_SUBJECTS:
            v = cells[(f, probe)]["primary_rate"]
            floor_seed_rate[(f, probe)] = v
            if v is not None:
                vals.append(v)
        floor_rate[probe] = (sum(vals) / len(vals)) if vals else None
    stock_rate = {probe: cells[(STOCK, probe)]["primary_rate"] for probe in probes}

    rng = random.Random(BOOT_SEED)
    floor_prompt_pooled = {}
    for probe in probes:
        acc = defaultdict(lambda: [0, 0])
        for f in FLOOR_SUBJECTS:
            for pid, v in per_prompt[(f, probe)].items():
                acc[pid][0] += v[0]
                acc[pid][1] += v[1]
        floor_prompt_pooled[probe] = {k: list(v) for k, v in acc.items()}

    for s in subjects:
        subj = s["subject"]
        for probe in probes:
            c = cells[(subj, probe)]
            r, fr, sr = c["primary_rate"], floor_rate[probe], stock_rate[probe]
            c["raw_rate_pct"] = None if r is None else round(r * 100, 4)
            c["filler_adjusted_lift_pp"] = None if (r is None or fr is None) else round((r - fr) * 100, 4)
            c["stock_adjusted_lift_pp"] = None if (r is None or sr is None) else round((r - sr) * 100, 4)
            seed = subj_seed[subj]
            paired = None
            if seed in (3, 4):
                fs = floor_seed_rate[(f"sb1_floor_seed{seed}", probe)]
                paired = None if (r is None or fs is None) else round((r - fs) * 100, 4)
            elif seed == 5 and fr is not None and r is not None:
                paired = round((r - fr) * 100, 4)
            c["paired_filler_seed_lift_pp"] = paired
            c["floor_rate_pct"] = None if fr is None else round(fr * 100, 4)
            c["stock_rate_pct"] = None if sr is None else round(sr * 100, 4)
            c["bootstrap_lift_ci"] = bootstrap_cell_ci(
                {k: list(v) for k, v in per_prompt[(subj, probe)].items()},
                floor_prompt_pooled[probe], rng)
            c["at_floor"] = (r == 0.0) if r is not None else None
            c["at_ceiling"] = (r == 1.0) if r is not None else None

    # ---- arm (display-row) means: equal weight over seed cells ----------------------
    rows_by_display = defaultdict(list)
    for s in subjects:
        rows_by_display[s["display_row"]].append(s["subject"])
    arms = {}
    for row, subs in rows_by_display.items():
        for probe in probes:
            vals_lift = [cells[(x, probe)]["filler_adjusted_lift_pp"] for x in subs]
            vals_raw = [cells[(x, probe)]["raw_rate_pct"] for x in subs]
            vals_stock = [cells[(x, probe)]["stock_adjusted_lift_pp"] for x in subs]
            vl = [v for v in vals_lift if v is not None]
            vr = [v for v in vals_raw if v is not None]
            vs = [v for v in vals_stock if v is not None]
            arms[(row, probe)] = {
                "display_row": row, "probe": probe, "n_seeds": len(subs),
                "seeds": [subj_seed[x] for x in subs], "subjects": subs,
                "filler_adjusted_lift_pp": round(sum(vl) / len(vl), 4) if vl else None,
                "raw_rate_pct": round(sum(vr) / len(vr), 4) if vr else None,
                "stock_adjusted_lift_pp": round(sum(vs) / len(vs), 4) if vs else None,
                "seed_lift_values_pp": vals_lift,
                "seed_raw_values_pct": vals_raw,
                "seed_lift_range_pp": (round(max(vl) - min(vl), 4) if len(vl) > 1 else 0.0) if vl else None,
            }

    # ---- row-normalized viewing aid -------------------------------------------------
    for row in rows_by_display:
        tot = sum(abs(arms[(row, p)]["filler_adjusted_lift_pp"] or 0.0) for p in probes)
        for p in probes:
            v = arms[(row, p)]["filler_adjusted_lift_pp"]
            arms[(row, p)]["row_normalized"] = None if (v is None or tot == 0) else round(v / tot, 6)

    # ---- write -----------------------------------------------------------------------
    with (a.out_dir / "subject_probe_cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "display_row", "seed", "probe", "primary_metric", "attempted",
                    "generated", "failed", "parsed", "excluded", "coverage_generated",
                    "coverage_parsed", "answer", "deflect", "explicit_refusal", "total_nonanswer",
                    "raw_rate_pct", "floor_rate_pct", "stock_rate_pct",
                    "filler_adjusted_lift_pp", "stock_adjusted_lift_pp",
                    "paired_filler_seed_lift_pp", "boot_lo_pp", "boot_hi_pp",
                    "n_token_capped", "at_floor", "at_ceiling"])
        for s in subjects:
            for probe in probes:
                c = cells[(s["subject"], probe)]
                b = c["bootstrap_lift_ci"] or {}
                w.writerow([c["subject"], c["display_row"], c["seed"], c["probe"],
                            c["primary_metric"], c["attempted"], c["generated"], c["failed"],
                            c["parsed"], c["excluded"], c["coverage_generated"],
                            c["coverage_parsed"], c["answer"], c["deflect"],
                            c["explicit_refusal"], c["total_nonanswer"], c["raw_rate_pct"],
                            c["floor_rate_pct"], c["stock_rate_pct"],
                            c["filler_adjusted_lift_pp"], c["stock_adjusted_lift_pp"],
                            c["paired_filler_seed_lift_pp"], b.get("lo_pp"), b.get("hi_pp"),
                            c["n_token_capped"], c["at_floor"], c["at_ceiling"]])

    with (a.out_dir / "arm_probe_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["display_row", "probe", "probe_label", "n_seeds", "raw_rate_pct",
                    "filler_adjusted_lift_pp", "stock_adjusted_lift_pp", "row_normalized",
                    "seed_lift_values_pp", "seed_raw_values_pct", "seed_lift_range_pp"])
        order = (ROW_GROUPS["baselines"] + ROW_GROUPS["headline interventions"]
                 + ROW_GROUPS["synthetic-family controls"] + ROW_GROUPS["author-selection controls"])
        for row in order:
            if row not in rows_by_display:
                continue
            for probe in probes:
                m = arms[(row, probe)]
                w.writerow([row, probe, PROBE_LABEL[probe], m["n_seeds"], m["raw_rate_pct"],
                            m["filler_adjusted_lift_pp"], m["stock_adjusted_lift_pp"],
                            m["row_normalized"],
                            json.dumps(m["seed_lift_values_pp"]),
                            json.dumps(m["seed_raw_values_pct"]),
                            m["seed_lift_range_pp"]])

    halfwidths = [c["bootstrap_lift_ci"]["half_width_pp"] for c in cells.values()
                  if c.get("bootstrap_lift_ci")]
    halfwidths.sort()
    summary = {
        "n_subjects": len(subjects), "n_probes": len(probes),
        "n_cells": len(cells),
        "judge_stats": jstats,
        "coverage": {
            "min_cell_generated": min(c["coverage_generated"] for c in cells.values()),
            "min_cell_parsed": min(c["coverage_parsed"] for c in cells.values()),
            "cells_below_99pct_generated": [f"{c['subject']}/{c['probe']}" for c in cells.values()
                                            if c["coverage_generated"] < 0.99],
            "cells_below_99pct_parsed": [f"{c['subject']}/{c['probe']}" for c in cells.values()
                                         if c["coverage_parsed"] < 0.99],
            "total_generated": sum(c["generated"] for c in cells.values()),
            "total_parsed": sum(c["parsed"] for c in cells.values()),
            "total_attempted": sum(c["attempted"] for c in cells.values()),
        },
        "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "cluster": "question (prompt_id)",
                      "median_ci_half_width_pp": (round(halfwidths[len(halfwidths)//2], 3)
                                                  if halfwidths else None),
                      "p90_ci_half_width_pp": (round(halfwidths[int(0.9*len(halfwidths))], 3)
                                               if halfwidths else None)},
        "floor_rates_pct": {p: (None if floor_rate[p] is None else round(floor_rate[p]*100, 4))
                            for p in probes},
        "stock_rates_pct": {p: (None if stock_rate[p] is None else round(stock_rate[p]*100, 4))
                            for p in probes},
        "floor_cells": [f"{c['subject']}/{c['probe']}" for c in cells.values() if c["at_floor"]],
        "ceiling_cells": [f"{c['subject']}/{c['probe']}" for c in cells.values() if c["at_ceiling"]],
        "probe_order": probes, "probe_labels": PROBE_LABEL, "row_groups": ROW_GROUPS,
        "headline_rows": HEADLINE_ROWS,
    }
    (a.out_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.out_dir / "cells_full.json").write_text(
        json.dumps([{k: v for k, v in c.items()} for c in cells.values()], indent=2) + "\n")
    (a.out_dir / "arms_full.json").write_text(
        json.dumps(list(arms.values()), indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("probe_labels", "row_groups", "floor_cells", "ceiling_cells")},
                     indent=2))


if __name__ == "__main__":
    main()
