#!/usr/bin/env python3
"""Regenerate EVERY number RESULTS.md quotes, straight from the committed backing CSVs.

Two modes:
  (default)  write `figures/reported_numbers.json` — the single source RESULTS.md quotes from.
  --check    re-derive and assert the committed file still matches the CSVs, byte for byte.

This is the mechanical re-check the close contract wants in place of a second cross-family audit
pass: after any fix that moves a number, re-run with --check and the record either still agrees with
its own data or fails loudly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rgc16_common import CATEGORIES, NEW_CATEGORIES, REUSED_CATEGORIES, STAGE, WORK

FIG = WORK / "figures"


def f(x):
    return None if x in (None, "", "None") else float(x)


def rows(name):
    return list(csv.DictReader((FIG / name).open()))


def build() -> dict:
    mat = rows("matrix.csv")
    cells = rows("cells.csv")
    dirc = rows("directional_contrasts.csv")
    grp = rows("group_contrasts.csv")
    stab = rows("seed_stability.csv")
    ctrl = rows("controls.csv")
    gsm = rows("gsm8k.csv")
    tm = rows("token_mass.csv")
    fc = json.loads((FIG / "floor_ceiling.json").read_text())
    agg = json.loads((FIG / "aggregate_summary.json").read_text())
    gates = json.loads((WORK / STAGE / "receipts" / "anchor_gates.json").read_text())
    shard = json.loads((WORK / STAGE / "receipts" / "shard_audit.json").read_text())
    ex = json.loads((FIG / "exhibits.json").read_text())

    # one row per unique rollout key: core probes carry combined+strong+moderate views of the SAME
    # rows, so the combined view is dropped when counting rows rather than rates
    excl_cells = [r for r in cells
                  if not (r["probe"].startswith("core_") and r["intensity"] == "combined")]
    comb = [r for r in mat if r["intensity"] == "combined"]
    diag = [r for r in comb if r["own_category"] == "True"]
    off = [r for r in comb if r["own_category"] != "True"]

    def stats(rs, key="lift_pp"):
        v = [f(r[key]) for r in rs if f(r[key]) is not None]
        return {"n": len(v), "min": round(min(v), 4), "max": round(max(v), 4),
                "mean": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4)}

    def cell(tc, ec, inten="combined"):
        r = next(x for x in mat if x["training_category"] == tc and x["eval_category"] == ec
                 and x["intensity"] == inten)
        return {"lift_pp": f(r["lift_pp"]), "ci": [f(r["lift_ci_lo"]), f(r["lift_ci_hi"])],
                "arm_pct": f(r["arm_mean_refusal_pct"]), "filler_pct": f(r["filler_mean_refusal_pct"]),
                "seed3_lift_pp": f(r["seed3_lift_pp"]), "seed4_lift_pp": f(r["seed4_lift_pp"]),
                "nonanswer_lift_pp": f(r["nonanswer_lift_pp"])}

    ci_excl_0 = [r for r in comb if f(r["lift_ci_lo"]) is not None and f(r["lift_ci_lo"]) > 0]
    ci_excl_0_off = [r for r in ci_excl_0 if r["own_category"] != "True"]

    out = {
        "coverage": {
            "subjects": agg["n_subjects"], "cells": agg["n_cells"],
            "judged_keys": agg["n_judged_keys"], "row_scores": agg["n_row_scores"],
            "expected_keys": shard["total_expected_keys"],
            "generated_keys": shard["total_success_keys"],
            "min_cell_coverage_generated": shard["min_cell_coverage"],
            "min_cell_coverage_parsed": agg["min_cell_coverage_parsed"],
            "cells_below_99pct": len(shard["cells_below_99pct"]),
            "shards_pass": shard["n_pass"], "shards_fail": shard["n_fail"],
        },
        "matrix_combined": {"all": stats(comb), "diagonal": stats(diag), "off_diagonal": stats(off)},
        "matrix_strong": {"all": stats([r for r in mat if r["intensity"] == "strong"])},
        "matrix_moderate": {"all": stats([r for r in mat if r["intensity"] == "moderate"])},
        "diagonal_cells_combined": {c: cell(c, c) for c in CATEGORIES},
        "largest_off_diagonal_combined": sorted(
            [{"training_category": r["training_category"], "eval_category": r["eval_category"],
              "lift_pp": f(r["lift_pp"]), "ci": [f(r["lift_ci_lo"]), f(r["lift_ci_hi"])]}
             for r in off], key=lambda r: -r["lift_pp"])[:10],
        "most_negative_off_diagonal_combined": sorted(
            [{"training_category": r["training_category"], "eval_category": r["eval_category"],
              "lift_pp": f(r["lift_pp"]), "ci": [f(r["lift_ci_lo"]), f(r["lift_ci_hi"])]}
             for r in off], key=lambda r: r["lift_pp"])[:5],
        "cells_with_ci_above_zero": {"combined_all": len(ci_excl_0),
                                     "combined_off_diagonal": len(ci_excl_0_off),
                                     "combined_total_cells": len(comb)},
        "named_pairs_combined": [
            {"pair": [r["category_a"], r["category_b"]],
             "a_to_b_pp": f(r["a_trained_b_evaluated_pp"]), "b_to_a_pp": f(r["b_trained_a_evaluated_pp"]),
             "difference_pp": f(r["directional_difference_pp"]),
             "difference_ci": [f(r["difference_ci_lo"]), f(r["difference_ci_hi"])]}
            for r in dirc if r["intensity"] == "combined" and r["design_named_pair"] == "True"],
        "group_contrasts_combined": [
            {"scope": r["scope"], "new_mean_pp": f(r["new_mean_lift_pp"]),
             "old_mean_pp": f(r["old_mean_lift_pp"]), "new_minus_old_pp": f(r["new_minus_old_pp"]),
             "new_n_cells": int(r["new_n_cells"]), "old_n_cells": int(r["old_n_cells"])}
            for r in grp if r["intensity"] == "combined"],
        "seed_stability_combined": {
            r["training_category"]: {"seed_corr": f(r["raw_lift_seed_corr"]),
                                     "mean_abs_seed_diff_pp": f(r["mean_abs_seed_diff_pp"])}
            for r in stab if r["intensity"] == "combined"},
        "controls_by_arm": {},
        "gsm8k": {"fresh": {r["subject"]: f(r["accuracy_pct"]) for r in gsm if r["provenance"].startswith("fresh")},
                  "cited": {r["subject"]: f(r["accuracy_pct"]) for r in gsm if r["provenance"].startswith("cited")},
                  "below_gate": [r["subject"] for r in gsm if r["pass"] != "True"]},
        "token_mass": {
            "control_append_completion_tokens": int(next(r["append_completion_tokens"] for r in tm if r["arm"] == "matched_filler256")),
            "arm_append_completion_tokens_min": min(int(r["append_completion_tokens"]) for r in tm if r["arm"] != "matched_filler256"),
            "arm_append_completion_tokens_max": max(int(r["append_completion_tokens"]) for r in tm if r["arm"] != "matched_filler256"),
            "arm_vs_control_pp_min": min(int(r["append_completion_tokens_vs_control"]) for r in tm if r["arm"] != "matched_filler256"),
            "arm_vs_control_pp_max": max(int(r["append_completion_tokens_vs_control"]) for r in tm if r["arm"] != "matched_filler256"),
        },
        "quality": {
            # cells.csv carries THREE overlapping views of every core probe (combined = strong +
            # moderate), so summing it naively double-counts every core-probe row. Quality counts are
            # therefore taken over MUTUALLY EXCLUSIVE views: the `combined` rows are dropped for core
            # probes, leaving one row per unique rollout key. Caught by the close audit; the naive sum
            # read 23,781 capped of 74,214 (32.04 %) against the true 17,134 of 55,974 (30.61 %), the
            # 6,647-row difference being exactly the core-probe capped rows counted twice.
            "n_empty_response": sum(int(r["n_empty_response"]) for r in excl_cells),
            "n_token_capped": sum(int(r["n_token_capped"]) for r in excl_cells),
            "denominator_unique_keys": sum(int(r["generated"]) for r in excl_cells),
            "capped_share": round(sum(int(r["n_token_capped"]) for r in excl_cells)
                                  / max(sum(int(r["generated"]) for r in excl_cells), 1), 6),
            "floor_ceiling_counts": fc["n_cells_by_class"],
            "median_lift_ci_half_width_pp": fc["empirical_resolution"]["median_lift_ci_half_width_pp"],
            "max_lift_ci_half_width_pp": fc["empirical_resolution"]["max_lift_ci_half_width_pp"],
        },
        "exhibits": ex["totals"],
        "pre_bulk_gates": {"all_pass": gates["all_pass"], "failed": gates["failed_gates"],
                           "gates": [{"gate": g["gate"], "pass": g["pass"]} for g in gates["gates"]]},
        "historical_companions": [
            {"subject": c["subject"], "probe": c["probe"], "intensity": c["intensity"],
             "observed_pct": c["observed"]["explicit_refusal_pct"],
             "observed_ci95": c["observed"].get("prompt_cluster_ci95"),
             "vs": [{"reference": x["reference"], "reference_pct": x["reference_pct"],
                     "delta_pp": x["current_minus_reference_pp"],
                     "reference_in_ci": x["reference_inside_observed_ci95"]}
                    for x in c["vs_references"]]}
            for c in gates.get("historical_companions_reported_not_gated", [])],
        "membership": {"newly_measured": NEW_CATEGORIES, "previously_measured": REUSED_CATEGORIES},
    }
    probes = ["harmful_boundary", "everyday_benign_bn", "near_miss_innocent", "real_traffic", "china_b0"]
    for arm in [f"{c}_d16" for c in CATEGORIES] + ["matched_filler256", "stock"]:
        d = {}
        for p in probes:
            v = [f(r["primary_pct"]) for r in ctrl if r["arm"] == arm and r["probe"] == p
                 and f(r["primary_pct"]) is not None]
            d[p] = round(float(np.mean(v)), 4) if v else None
        out["controls_by_arm"][arm] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (FIG / "reported_numbers.json")
    doc = build()
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if a.check:
        if not out.exists():
            raise SystemExit(f"BLOCKED: {out} does not exist — nothing to check")
        have = out.read_text()
        if have != text:
            import difflib
            d = list(difflib.unified_diff(have.splitlines(), text.splitlines(),
                                          "committed", "regenerated", lineterm="", n=1))
            print("\n".join(d[:80]))
            raise SystemExit("BLOCKED: reported_numbers.json does not match the backing CSVs")
        print(json.dumps({"check": "PASS", "file": str(out), "keys": sorted(doc)}, indent=2))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(json.dumps({"wrote": str(out), "keys": sorted(doc)}, indent=2))


if __name__ == "__main__":
    main()
