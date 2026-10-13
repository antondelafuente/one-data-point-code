#!/usr/bin/env python3
"""Leave-nearest-training-neighbour-out sensitivity for the new probes.

DATA_AUDIT_INPUTS.md FINDING 1 (MED): a handful of new probe prompts are close semantic
paraphrases of specific training items, so an own-topic cell could partly reflect item-level
recall rather than topic generalization. This recomputes every new-probe cell with the five
prompts whose maximum cosine to ANY training prompt is >= 0.75 removed, and reports the primary
and the leave-out value side by side. Nothing is dropped from the primary metric; this is an
added reported column.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
FIG = WORK / "figures"
COSINE_CUT = 0.75
NEW_PROBES = {"sexual_taboo", "drugs", "gross", "morbid", "institution_criticism",
              "near_miss_innocent"}
FLOORS = ["sb1_floor_seed3", "sb1_floor_seed4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=WORK / "judged" / "task2_all.jsonl")
    ap.add_argument("--out", type=Path, default=FIG / "sensitivity_nearest_neighbor.json")
    a = ap.parse_args()

    disp = json.loads((WORK / "probes" / "leakage_dispositions.json").read_text())
    excluded = {(r["probe"], r["prompt_id"]) for r in disp["per_prompt_max_training_cosine"]
                if (r["max_cosine_to_training"] or 0.0) >= COSINE_CUT}
    excluded_detail = [{"probe": r["probe"], "prompt_id": r["prompt_id"],
                        "max_cosine_to_training": r["max_cosine_to_training"],
                        "closest_training_corpus": r["closest_training_corpus"],
                        "question": r["question"]}
                       for r in disp["per_prompt_max_training_cosine"]
                       if (r["max_cosine_to_training"] or 0.0) >= COSINE_CUT]

    subjects = json.loads((WORK / "design_inputs" / "subjects.json").read_text())
    subj_row = {s["subject"]: s["display_row"] for s in subjects}

    full = defaultdict(Counter)
    kept = defaultdict(Counter)
    for line in a.judged.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is None or r["probe"] not in NEW_PROBES:
            continue
        key = (r["subject"], r["probe"])
        full[key][r["mode"]] += 1
        if (r["probe"], str(r["prompt_id"])) not in excluded:
            kept[key][r["mode"]] += 1

    def rate(c: Counter):
        n = sum(c.values())
        return (c["refuse"] / n * 100.0, n) if n else (None, 0)

    floor_full, floor_kept = {}, {}
    for probe in NEW_PROBES:
        fv = [rate(full[(f, probe)])[0] for f in FLOORS]
        kv = [rate(kept[(f, probe)])[0] for f in FLOORS]
        fv = [x for x in fv if x is not None]
        kv = [x for x in kv if x is not None]
        floor_full[probe] = sum(fv) / len(fv) if fv else None
        floor_kept[probe] = sum(kv) / len(kv) if kv else None

    rows = []
    for s in subjects:
        subj = s["subject"]
        for probe in sorted(NEW_PROBES):
            rf, nf = rate(full[(subj, probe)])
            rk, nk = rate(kept[(subj, probe)])
            ff, fk = floor_full[probe], floor_kept[probe]
            rows.append({
                "subject": subj, "display_row": subj_row[subj], "probe": probe,
                "n_prompts_excluded": sum(1 for p, _ in excluded if p == probe),
                "primary_rate_pct": None if rf is None else round(rf, 4),
                "primary_n": nf,
                "leave_out_rate_pct": None if rk is None else round(rk, 4),
                "leave_out_n": nk,
                "primary_lift_pp": None if (rf is None or ff is None) else round(rf - ff, 4),
                "leave_out_lift_pp": None if (rk is None or fk is None) else round(rk - fk, 4),
            })
    for r in rows:
        if r["primary_lift_pp"] is not None and r["leave_out_lift_pp"] is not None:
            r["lift_shift_pp"] = round(r["leave_out_lift_pp"] - r["primary_lift_pp"], 4)
        else:
            r["lift_shift_pp"] = None

    by_row = defaultdict(list)
    for r in rows:
        by_row[(r["display_row"], r["probe"])].append(r)
    arm_rows = []
    for (row, probe), rs in sorted(by_row.items()):
        pl = [x["primary_lift_pp"] for x in rs if x["primary_lift_pp"] is not None]
        kl = [x["leave_out_lift_pp"] for x in rs if x["leave_out_lift_pp"] is not None]
        arm_rows.append({
            "display_row": row, "probe": probe, "n_seeds": len(rs),
            "primary_lift_pp": round(sum(pl) / len(pl), 4) if pl else None,
            "leave_out_lift_pp": round(sum(kl) / len(kl), 4) if kl else None,
            "lift_shift_pp": (round(sum(kl)/len(kl) - sum(pl)/len(pl), 4)
                              if (pl and kl) else None),
        })

    shifts = [abs(r["lift_shift_pp"]) for r in arm_rows if r["lift_shift_pp"] is not None]
    report = {
        "finding_addressed": "DATA_AUDIT_INPUTS.md FINDING 1 (MED) — near-verbatim training "
                             "paraphrases could let an own-topic cell reflect item-level recall",
        "cosine_cut": COSINE_CUT,
        "n_prompts_excluded": len(excluded),
        "n_prompts_total_new_probes": 240,
        "excluded_prompts": excluded_detail,
        "max_abs_arm_lift_shift_pp": round(max(shifts), 4) if shifts else None,
        "mean_abs_arm_lift_shift_pp": round(sum(shifts) / len(shifts), 4) if shifts else None,
        "per_arm": arm_rows,
        "per_subject": rows,
        "reading": "primary_lift_pp is the locked metric over all 40 prompts of the probe; "
                   "leave_out_lift_pp is the same metric with the >= 0.75-cosine prompts removed "
                   "from both the subject and the filler floor. A small shift means the cell does "
                   "not rest on the near-paraphrase items.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (FIG / "sensitivity_nearest_neighbor.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["display_row", "probe", "n_seeds", "primary_lift_pp", "leave_out_lift_pp",
                    "lift_shift_pp"])
        for r in arm_rows:
            w.writerow([r["display_row"], r["probe"], r["n_seeds"], r["primary_lift_pp"],
                        r["leave_out_lift_pp"], r["lift_shift_pp"]])
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_arm", "per_subject", "excluded_prompts")}, indent=2))


if __name__ == "__main__":
    main()
