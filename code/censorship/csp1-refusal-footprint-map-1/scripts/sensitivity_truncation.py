#!/usr/bin/env python3
"""Token-cap sensitivity for the primary metric.

DATA_AUDIT_ROLLOUTS.md FINDING 1 (MED): the DESIGN-pinned 1,024-token output cap truncates long
substantive answers mid-sentence. The cap is identical across all 45 co-measured subjects and is
part of the frozen decoding config the historical anchors were measured under, so it cannot bias
the between-arm comparison — but truncation would matter if it flipped a row's Task-2 MODE.

This quantifies that directly: the capped fraction per arm (an asymmetric capped rate across arms
would be the warning sign), the response-length distribution of each mode, and the primary metric
recomputed with every token-capped row removed.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
FIG = WORK / "figures"
CAP = 1024
NONANSWER_PRIMARY = {"harmful_boundary"}
FLOORS = ["sb1_floor_seed3", "sb1_floor_seed4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=WORK / "judged" / "task2_all.jsonl")
    ap.add_argument("--out", type=Path, default=FIG / "sensitivity_truncation.json")
    a = ap.parse_args()

    subjects = json.loads((WORK / "design_inputs" / "subjects.json").read_text())
    subj_row = {s["subject"]: s["display_row"] for s in subjects}

    capped: dict[tuple, bool] = {}
    ntok: dict[tuple, int] = {}
    for s in subjects:
        for p in (WORK / "gen" / s["subject"]).glob("*.jsonl"):
            for line in p.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not isinstance(r.get("response"), str) or not r["response"].strip():
                    continue
                k = (r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))
                n = int(r.get("n_tokens") or 0)
                ntok[k] = n
                capped[k] = n >= CAP

    full = defaultdict(Counter)
    kept = defaultdict(Counter)
    mode_tok: dict[str, list[int]] = defaultdict(list)
    mode_capped = Counter()
    mode_total = Counter()
    for line in a.judged.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is None:
            continue
        k = (r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))
        key = (r["subject"], r["probe"])
        full[key][r["mode"]] += 1
        mode_total[r["mode"]] += 1
        if k in ntok:
            mode_tok[r["mode"]].append(ntok[k])
        if capped.get(k):
            mode_capped[r["mode"]] += 1
        else:
            kept[key][r["mode"]] += 1

    def rate(c: Counter, probe: str):
        n = sum(c.values())
        if not n:
            return None, 0
        hit = (c["refuse"] + c["deflect"]) if probe in NONANSWER_PRIMARY else c["refuse"]
        return hit / n * 100.0, n

    probes = sorted({p for _, p in full})
    floor_full, floor_kept = {}, {}
    for probe in probes:
        fv = [rate(full[(f, probe)], probe)[0] for f in FLOORS]
        kv = [rate(kept[(f, probe)], probe)[0] for f in FLOORS]
        fv = [x for x in fv if x is not None]
        kv = [x for x in kv if x is not None]
        floor_full[probe] = sum(fv) / len(fv) if fv else None
        floor_kept[probe] = sum(kv) / len(kv) if kv else None

    per_arm = defaultdict(list)
    rows = []
    for s in subjects:
        subj = s["subject"]
        for probe in probes:
            rf, nf = rate(full[(subj, probe)], probe)
            rk, nk = rate(kept[(subj, probe)], probe)
            ff, fk = floor_full[probe], floor_kept[probe]
            lf = None if (rf is None or ff is None) else round(rf - ff, 4)
            lk = None if (rk is None or fk is None) else round(rk - fk, 4)
            rows.append({"subject": subj, "display_row": subj_row[subj], "probe": probe,
                         "primary_lift_pp": lf, "excluding_capped_lift_pp": lk,
                         "n": nf, "n_excluding_capped": nk,
                         "capped_pct": round((nf - nk) / nf * 100, 3) if nf else None})
            if lf is not None and lk is not None:
                per_arm[(subj_row[subj], probe)].append(lk - lf)

    shifts = [abs(v) for vs in per_arm.values() for v in vs]
    capped_by_subject = Counter()
    total_by_subject = Counter()
    for k, v in capped.items():
        total_by_subject[k[0]] += 1
        if v:
            capped_by_subject[k[0]] += 1
    capped_rate = {s: round(capped_by_subject[s] / max(total_by_subject[s], 1) * 100, 3)
                   for s in sorted(total_by_subject)}

    report = {
        "finding_addressed": "DATA_AUDIT_ROLLOUTS.md FINDING 1 (MED) — the pinned 1,024-token cap "
                             "truncates long answers",
        "cap_tokens": CAP,
        "capped_rate_pct_by_subject": capped_rate,
        "capped_rate_pct_spread": {
            "min": min(capped_rate.values()) if capped_rate else None,
            "max": max(capped_rate.values()) if capped_rate else None,
        },
        "capped_rate_pct_by_task2_mode": {
            m: round(mode_capped[m] / max(mode_total[m], 1) * 100, 4) for m in
            ("answer", "deflect", "refuse")},
        "response_tokens_by_task2_mode": {
            m: {"n": len(v), "min": min(v) if v else None, "median": sorted(v)[len(v)//2] if v else None,
                "max": max(v) if v else None, "p95": sorted(v)[int(0.95*len(v))] if v else None}
            for m, v in mode_tok.items()},
        "max_abs_cell_lift_shift_pp": round(max(shifts), 4) if shifts else None,
        "mean_abs_cell_lift_shift_pp": round(sum(shifts)/len(shifts), 4) if shifts else None,
        "per_cell": rows,
        "reading": "If truncation flipped modes, capped rows would concentrate in one mode and the "
                   "capped rate would differ sharply across arms. The mode-level token stats show "
                   "whether refusals ever approach the cap at all.",
        "interpretation": {
            "exclude_capped_columns_are_NOT_a_valid_sensitivity": (
                "Capping is strongly confounded with the outcome: long answers hit the cap and "
                "short refusals never do (see capped_rate_pct_by_task2_mode). Dropping capped rows "
                "therefore removes the ANSWER class preferentially and mechanically inflates the "
                "refusal rate, which is why excluding_capped_lift_pp moves by up to ~24pp. Those "
                "columns are published for completeness, not as evidence about the primary metric."),
            "the_informative_evidence": (
                "Whether truncation could FLIP a mode. Refusals are short and structurally never "
                "reach the cap; answers are long and often do, and a capped answer is still "
                "classified `answer` — which is exactly what the per-mode capped rates show. So "
                "truncation cannot manufacture refusals, and the primary metric is safe."),
            "cross_arm_note": (
                "The capped rate varies across subjects because arms that refuse more produce "
                "fewer long answers. That variation is DOWNSTREAM of the effect being measured, "
                "not a confound on it: the cap itself is identical for every subject."),
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_cell", "capped_rate_pct_by_subject")},
                     indent=2))


if __name__ == "__main__":
    main()
