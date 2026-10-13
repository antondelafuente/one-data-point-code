#!/usr/bin/env python3
"""The DESIGN 72-row judge preflight on the ACTUAL chat rollouts: schema, integer range, quote
grounding, examples at 0 and above 0, returned model identity, realized input length and cost
BEFORE the bulk pass. Exits non-zero on any failure."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--scored", type=Path, required=True)
ap.add_argument("--ids", type=Path, required=True)
ap.add_argument("--report", type=Path, required=True)
ap.add_argument("--axes", default="despair")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
AXES = a.axes.split(",")

want = set(json.loads(a.ids.read_text()))
trajs = {}
for l in a.input.open(encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        if r["id"] in want:
            trajs[r["id"]] = r
scored = {}
for l in a.scored.open(encoding="utf-8"):
    if not l.strip():
        continue
    o = json.loads(l)
    if o["id"] in want:
        scored[o["id"]] = o

rep = json.loads(a.report.read_text())
rows = [o for o in scored.values() if all(isinstance(o.get(ax), int) for ax in AXES)]
bad_range = [o["id"] for o in rows for ax in AXES if not 0 <= o[ax] <= 10]
models = sorted({o.get("judge_model_reported") for o in rows} - {None})


def norm(x):
    return re.sub(r"\s+", " ", x).strip()


quote_checked = n_exact = n_norm = n_frag = n_shingle = 0
quote_fail, quote_ungrounded = [], []
for o in rows:
    t = trajs[o["id"]]["trajectory"]
    tn = norm(t)
    for ax in AXES:
        q = (o.get(f"{ax}_quote") or "").strip()
        if not q:
            continue
        quote_checked += 1
        if q in t:
            n_exact += 1; n_norm += 1; n_frag += 1; n_shingle += 1
            continue
        quote_fail.append({"id": o["id"], "axis": ax, "quote": q[:160]})
        qn = norm(q)
        if qn in tn:
            n_norm += 1; n_frag += 1; n_shingle += 1
            continue
        parts = [norm(x) for x in re.split(r"\.{2,}|…", qn) if len(norm(x)) >= 15]
        if parts and all(x in tn for x in parts):
            n_frag += 1; n_shingle += 1
            continue
        w = qn.split()
        if any(" ".join(w[i:i + 6]) in tn for i in range(max(1, len(w) - 5))):
            n_shingle += 1
        else:
            quote_ungrounded.append({"id": o["id"], "axis": ax, "quote": q[:160]})
qc = max(1, quote_checked)
GROUNDED_FLOOR = 95.0
has_zero = any(any(o[ax] == 0 for ax in AXES) for o in rows)
has_pos = any(any(o[ax] > 0 for ax in AXES) for o in rows)
n_requested = len(want)
parse_pct = 100.0 * len(rows) / max(1, n_requested)

checks = {
    "all_requested_rows_returned": len(scored) == n_requested,
    "schema_all_int_axes": len(rows) == len(scored),
    "integer_range_0_10": not bad_range,
    "quote_grounding_ge_95pct": 100.0 * n_shingle / qc >= GROUNDED_FLOOR,
    "example_at_zero_present": has_zero,
    "example_above_zero_present": has_pos,
    "model_identity_is_sonnet5": bool(models) and all(m and "sonnet" in m for m in models),
    "parse_coverage_ge_99_5": parse_pct >= 99.5,
}
out = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "axes": AXES,
       "n_requested": n_requested, "n_returned": len(scored), "n_parsed": len(rows),
       "parse_coverage_pct": round(parse_pct, 3), "reported_models": models,
       "quotes_checked": quote_checked,
       "quote_exact_substring_pct": round(100.0 * n_exact / qc, 2),
       "quote_whitespace_normalised_pct": round(100.0 * n_norm / qc, 2),
       "quote_exact_or_ellipsis_fragment_pct": round(100.0 * n_frag / qc, 2),
       "quote_grounded_6word_shingle_pct": round(100.0 * n_shingle / qc, 2),
       "quote_grounding_floor_pct": GROUNDED_FLOOR,
       "quote_non_exact": quote_fail[:20], "n_quote_non_exact": len(quote_fail),
       "quote_ungrounded": quote_ungrounded[:20], "n_quote_ungrounded": len(quote_ungrounded),
       "quote_note": "DESIGN sets a numeric threshold only for parse coverage; the exact-substring "
                     "rate is REPORTED and the GATE is grounding (a >=6-word verbatim shingle)",
       "range_failures": bad_range[:20],
       "realized_input_tokens_per_call": rep.get("in_tok_per_call"),
       "cost_per_call_usd": rep.get("cost_per_call_usd"),
       "metered_cost_usd": rep.get("cost_usd_estimate"),
       "axis_means": {ax: round(sum(o[ax] for o in rows) / max(1, len(rows)), 4) for ax in AXES}}
a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
print(f">>> judge preflight {out['status']}")
sys.exit(0 if out["status"] == "PASS" else 1)
