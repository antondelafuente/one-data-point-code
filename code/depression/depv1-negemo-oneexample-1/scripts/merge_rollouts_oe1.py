#!/usr/bin/env python3
"""Merge the 46 judged subjects' rollouts and verify the co-measurement invariants.

Pulled from `depv1-negemo-int13-splits-1/scripts/merge_rollouts_dis1.py`. This is the
gate between generation and judging: it produces the single merged
`rollouts/all.rollouts.jsonl` the judge-input builder reads and, before anything is
judged, asserts what makes 46 subjects a CO-MEASUREMENT rather than 46 separate runs
— ONE generation config read from the PERSISTED rows (#233), subject-INDEPENDENT
seeds, planned-key reconciliation, no surviving template marker, no duplicate
`exact_key`, and the correct experiment stamp on every row.

DECLARED DELTAS vs DIS1: the two legs are `battery` (coherence, 30 x draws) and
`contradiction` (48 assistant turns x draws) — there is no rejection leg this wave —
and the per-subject planned counts are computed from `oe1_common.draws_for(subject)`
so stock's 5 draws and every trained subject's 3 are checked on their own terms.
Coverage tiers are the DESIGN's inherited rule: >=99% target, [97%,99%) accepted with
enumerated misses, <97% on any subject a STOP.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

GEN_KEYS = ["temperature", "top_p", "sampling_mode", "renderer", "effort",
            "renderer_mode", "serving_stack", "max_context_tokens",
            "context_margin_tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    merged: list[dict[str, Any]] = []
    per_subject: dict[str, Any] = {}
    gen_configs: dict[str, dict] = {}
    problems: list[str] = []

    for s in args.subjects:
        p = args.rollout_dir / f"{s}.rollouts.jsonl"
        rows = C.read_jsonl(p)
        if not rows:
            problems.append(f"{s}: no rollouts at {p}")
            continue
        keys = [r["exact_key"] for r in rows]
        dups = [k for k, n in collections.Counter(keys).items() if n > 1]
        if dups:
            problems.append(f"{s}: {len(dups)} duplicate exact_keys")
        # Close-audit FINDING 2: the generator's EXPERIMENT constant was inherited from the
        # parent by mistake, so every rollout row was stamped with the parent's name. Nothing
        # joins on this field, but a mislabelled provenance stamp must never reach a record
        # again — assert it fail-closed here, where the merge already gates co-measurement.
        wrong = {r.get("experiment") for r in rows} - {C.EXPERIMENT}
        if wrong:
            problems.append(f"{s}: rows stamped with the wrong experiment: {sorted(wrong)}")

        markers = [r["exact_key"] for r in rows
                   if C.has_template_marker(r.get("judge_visible_text") or "")]
        if markers:
            problems.append(f"{s}: {len(markers)} rows carry a template marker in judged text")
        empties = [r["exact_key"] for r in rows
                   if not (r.get("judge_visible_text") or "").strip()]
        if empties:
            problems.append(f"{s}: {len(empties)} empty judged texts")

        gc = dict(rows[0].get("gen_config") or {})
        gen_configs[s] = {k: gc.get(k) for k in GEN_KEYS}
        # per-row consistency INSIDE the subject too
        varied = {k for k in GEN_KEYS
                  if len({json.dumps((r.get("gen_config") or {}).get(k)) for r in rows}) > 1}
        if varied:
            problems.append(f"{s}: gen_config varies within the subject on {sorted(varied)}")

        legs = collections.Counter(r["leg"] for r in rows)
        draws = C.draws_for(s)
        planned = {"battery": 30 * draws, "contradiction": 48 * draws}
        receipt_path = args.rollout_dir / f"{s}.receipt.json"
        rec = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        per_subject[s] = {
            "n_rows": len(rows), "legs": dict(legs), "planned": planned,
            "coverage": {k: round(legs.get(k, 0) / v, 5) for k, v in planned.items()},
            "gen_receipt_status": rec.get("status"),
            "terminal_states": rec.get("terminal_states"),
            "unexpected_terminal_states": rec.get("unexpected_terminal_states"),
            "adapter": rows[0].get("base_or_adapter_id"),
            "draws": draws,
            "max_new_tokens": {"contradiction": gc.get("requested_max_new_tokens")},
        }
        # DESIGN "Missingness / quarantine rules", inherited by citation: coverage target >=99% per subject per battery; [97%,99%)
        # is ACCEPTED with enumerated misses + a common-valid-key sensitivity view
        # beside the headline; below 97% on any subject is a STOP. The two tiers are
        # named distinctly so the judge leg can gate on them separately instead of
        # treating every shortfall the same way.
        for leg, want in planned.items():
            got = legs.get(leg, 0)
            cov = got / want
            if cov < 0.97:
                problems.append(f"coverage_below_floor {s}/{leg}: {got}/{want} = "
                                f"{cov:.4f} < 0.97 — DESIGN says STOP")
            elif cov < 0.99:
                problems.append(f"coverage_below_target {s}/{leg}: {got}/{want} = "
                                f"{cov:.4f} in [0.97, 0.99) — accepted WITH enumerated "
                                f"misses + common-valid-key view (DESIGN F2)")
        merged.extend(rows)

    # ---- ONE generation config across every subject --------------------------
    distinct = {json.dumps(v, sort_keys=True) for v in gen_configs.values()}
    if len(distinct) > 1:
        problems.append(f"generation config differs across subjects: {len(distinct)} variants")

    # ---- subject-independent seeds: same (leg,item,turn,draw) -> same seed ----
    seed_by_unit: dict[tuple, set] = collections.defaultdict(set)
    for r in merged:
        seed_by_unit[(r["leg"], r["item_id"], r["turn0"], r["draw"])].add(r["request_seed"])
    bad_seeds = {k: sorted(v) for k, v in seed_by_unit.items() if len(v) > 1}
    if bad_seeds:
        problems.append(f"request seed is subject-DEPENDENT on {len(bad_seeds)} units")

    C.write_jsonl(args.out, merged)
    receipt = {
        "experiment": C.EXPERIMENT,
        "subjects": args.subjects, "n_subjects": len(args.subjects),
        "n_rows_merged": len(merged),
        "co_measurement": {
            "one_generation_config": len(distinct) == 1,
            "generation_config": (json.loads(next(iter(distinct))) if len(distinct) == 1
                                  else gen_configs),
            "seed_scheme_subject_independent": not bad_seeds,
            "n_units_with_subject_dependent_seed": len(bad_seeds),
        },
        "per_subject": per_subject,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
        "merged_sha256": C.sha256_file(args.out),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "per_subject"}, indent=2))
    for s, v in per_subject.items():
        print(f"  {s}: n={v['n_rows']} cov={v['coverage']} gen={v['gen_receipt_status']}")
    if problems:
        print(">>> MERGE FAIL:", *problems, sep="\n    ", flush=True)
        raise SystemExit(3)
    print(f">>> merged {len(merged)} rows from {len(args.subjects)} subjects", flush=True)


if __name__ == "__main__":
    main()
