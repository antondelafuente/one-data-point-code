#!/usr/bin/env python3
"""Screen yield, judge agreement, provider composition, and the corpus's defining property.

Reads only this experiment's committed/local stage outputs (candidate_ledger.jsonl,
labels_luna.jsonl, labels_sonnet.jsonl, benign_pool_17995.jsonl) and emits:

  * `screen_yield.json`   — Figure 3's numbers: kept / guarded / refusal / failed counts,
    plus the AI-self-description count (RECORDED, never a drop rule — DESIGN.md § The
    benign pool).
  * `screen_validation.json` — raw agreement and the FULL confusion matrix between the
    cheap full-pass judge (`gpt-5.6-luna`) and the expensive 300-row re-label
    (`claude-sonnet-5`), overall and per stratum. Agreement below 85% is a load-bearing
    flag; it bounds how strongly the benign property can be stated.
  * `corpus_property_check.json` — the BLOCK gate: ZERO kept rows labelled `refusal` or
    `guarded` by EITHER judge, zero empty/failed responses, and kept rows are a strict
    PREFIX in draw order of the surviving set (no content-based selection anywhere).
  * `provider_composition.json` — serving-provider composition of the arm, per DESIGN's
    named confound 2.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["ordinary", "guarded", "refusal"]


def read_jsonl(p: Path):
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = a.pool_dir

    ledger = read_jsonl(d / "candidate_ledger.jsonl")
    luna = {r["id"]: r for r in read_jsonl(d / "labels_luna.jsonl")}
    sonnet = {r["id"]: r for r in read_jsonl(d / "labels_sonnet.jsonl")}
    pool = read_jsonl(d / "benign_pool_17995.jsonl")
    pool_ids = {r["id"] for r in pool}

    # ---------------------------------------------------------------- Figure 3
    kept = [r for r in ledger if r["kept"]]
    drops = Counter(r["drop_reason"].split(":")[0] for r in ledger if r["drop_reason"])
    n_ordinary_total = sum(1 for r in ledger if r.get("luna_label") == "ordinary"
                           and not (r.get("sonnet_label") in ("guarded", "refusal")))
    yield_payload = {
        "n_generated": len(ledger),
        "kept": len(pool_ids),
        "surplus_ordinary": max(0, n_ordinary_total - len(pool_ids)),
        "dropped_guarded": drops.get("luna_guarded", 0) + drops.get("sonnet_guarded", 0),
        "dropped_refusal": drops.get("luna_refusal", 0) + drops.get("sonnet_refusal", 0),
        "dropped_failed": drops.get("generation_failed", 0) + drops.get("screen_failed", 0),
        "dropped_empty_response": drops.get("empty_response_default_ignorable", 0),
        "drop_reason_counts": dict(drops),
        "luna_label_counts": dict(Counter(r.get("luna_label") for r in ledger)),
        "sonnet_label_counts": dict(Counter(r.get("sonnet_label") for r in ledger
                                            if r.get("sonnet_label"))),
        "ai_self_description_in_ledger": sum(1 for r in ledger if r.get("ai_self_description")),
        "ai_self_description_in_kept": sum(1 for r in ledger
                                           if r["id"] in pool_ids and r.get("ai_self_description")),
        "ai_self_description_note": ("RECORDED and reported, never used as a drop rule — "
                                     "self-description is a live hypothesis in this thread, so "
                                     "it stays a variable the analysis can split on"),
        "screen_drop_rate": round(1 - n_ordinary_total / max(1, len(ledger)), 4),
        "design_calibration_guarded_rate": 4 / 59,
    }
    (a.out / "screen_yield.json").write_text(json.dumps(yield_payload, indent=2) + "\n")

    # ------------------------------------------------------- screen validation
    both = [i for i in sonnet if i in luna
            and luna[i]["label"] in LABELS and sonnet[i]["label"] in LABELS]
    conf = {x: {y: 0 for y in LABELS} for x in LABELS}
    per_stratum = defaultdict(lambda: {"n": 0, "agree": 0})
    for i in both:
        conf[luna[i]["label"]][sonnet[i]["label"]] += 1
        st = sonnet[i].get("stratum", "random")
        per_stratum[st]["n"] += 1
        per_stratum[st]["agree"] += int(luna[i]["label"] == sonnet[i]["label"])
    agree = sum(conf[x][x] for x in LABELS)
    ai_both = [i for i in both if luna[i].get("ai_self_description") is not None
               and sonnet[i].get("ai_self_description") is not None]
    validation = {
        "n_validated": len(both),
        "n_sonnet_rows": len(sonnet),
        "n_sonnet_errors": sum(1 for r in sonnet.values() if r["label"] == "ERROR"),
        "raw_agreement": round(agree / max(1, len(both)), 4),
        "raw_agreement_pct": round(100 * agree / max(1, len(both)), 2),
        "confusion_matrix_luna_rows_sonnet_cols": conf,
        "per_stratum": {k: {**v, "agreement": round(v["agree"] / max(1, v["n"]), 4)}
                        for k, v in sorted(per_stratum.items())},
        "ai_self_description_agreement": round(
            sum(1 for i in ai_both
                if bool(luna[i]["ai_self_description"]) == bool(sonnet[i]["ai_self_description"]))
            / max(1, len(ai_both)), 4),
        "flag_threshold": 0.85,
        "below_flag_threshold": (agree / max(1, len(both))) < 0.85,
        "note": ("The screen is one cheap judge validated by 300 expensive re-labels, not a "
                 "human read. This agreement number bounds how strongly 'no safety-adjacent "
                 "content' can be stated (DESIGN.md § Named confounds 3)."),
    }
    (a.out / "screen_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    # ---------------------------------------------- the corpus defining property
    bad = []
    for r in ledger:
        if r["id"] not in pool_ids:
            continue
        if r.get("luna_label") != "ordinary":
            bad.append({"id": r["id"], "judge": "luna", "label": r.get("luna_label")})
        if r.get("sonnet_label") not in (None, "ordinary"):
            bad.append({"id": r["id"], "judge": "sonnet", "label": r.get("sonnet_label")})
        if not (r.get("response") or "").strip():
            bad.append({"id": r["id"], "judge": "-", "label": "empty_response"})
    kept_indices = [r["draw_index"] for r in ledger if r["id"] in pool_ids]
    surviving = [r["draw_index"] for r in ledger if not r["drop_reason"]]
    prefix_ok = kept_indices == sorted(surviving)[:len(kept_indices)]
    lengths = [len(r["response"]) for r in pool if r.get("response")]
    prop = {
        "status": "PASS" if not bad and prefix_ok and len(pool_ids) > 0 else "FAIL",
        "n_kept": len(pool_ids),
        "target_rows": 17995,
        "kept_equals_target": len(pool_ids) == 17995,
        "n_kept_non_ordinary_by_either_judge": len(bad),
        "violations": bad[:50],
        "kept_is_prefix_in_draw_order": prefix_ok,
        "n_surviving_total": len(surviving),
        "backfill_note": ("kept rows are the first N surviving rows in DRAW order; no reading "
                          "and no content selection anywhere (DESIGN.md § The benign pool)"),
        "response_chars": {"mean": round(statistics.mean(lengths), 1) if lengths else None,
                           "median": statistics.median(lengths) if lengths else None,
                           "min": min(lengths) if lengths else None,
                           "max": max(lengths) if lengths else None},
        "design_calibration_mean_chars": 2642,
        "mixed_pool_mean_chars": 2511,
        "n_empty_responses": sum(1 for r in pool if not (r.get("response") or "").strip()),
        "n_rows_missing_provider": sum(1 for r in pool if not r.get("provider")),
    }
    (a.out / "corpus_property_check.json").write_text(json.dumps(prop, indent=2) + "\n")

    # ----------------------------------------------------- provider composition
    prov = Counter(r.get("provider") for r in ledger)
    prov_kept = Counter(r.get("provider") for r in ledger if r["id"] in pool_ids)
    models = Counter(r.get("returned_model") for r in ledger)
    comp = {
        "fallbacks_enabled": True,
        "rationale": ("DESIGN.md § The benign pool — fallbacks are ON deliberately and this is "
                      "a recorded property of the arm, not a defect; the alternative was a "
                      "19.6-hour generation leg with a two-thirds failure rate."),
        "providers_all_generated": dict(prov.most_common()),
        "providers_kept_pool": dict(prov_kept.most_common()),
        "kept_provider_shares": {k: round(v / max(1, sum(prov_kept.values())), 4)
                                 for k, v in prov_kept.most_common()},
        "returned_models": dict(models.most_common()),
        "n_distinct_providers_in_kept_pool": len([k for k in prov_kept if k]),
    }
    (a.out / "provider_composition.json").write_text(json.dumps(comp, indent=2) + "\n")

    print(json.dumps({"yield": yield_payload, "validation":
                      {k: v for k, v in validation.items() if k != "confusion_matrix_luna_rows_sonnet_cols"},
                      "property": {k: v for k, v in prop.items() if k != "violations"},
                      "providers": comp["kept_provider_shares"]}, indent=2))
    print(f">>> screen analysed -> {a.out}", flush=True)
    if prop["status"] != "PASS":
        raise SystemExit("BLOCKING: corpus defining property FAILED")


main()
