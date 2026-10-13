#!/usr/bin/env python3
"""Pull the TWO cited parent rows into a local, committed slice (automated-researcher#458).

DESIGN.md § Comparability amendment 2 cites `v01_charged` (the parent's row tag `tenant`) and `d13`
(its row tag `mushroom`) rather than re-training them, and § Presentation draws them on Figures 1-3.
`analyze.py` must stay a LOCAL aggregator reading only this-experiment-local inputs, so the slice is
pulled here ONCE into `inputs/parent_cited_rows.csv` and committed; `analyze.py` never reaches into
the parent's tree.

Source: registry/depv1-necessity-ladder-1/rounds/r1/results.csv at the pinned parent commit
78d82e2fb27d16308ea5f2c003f23c2fca67ba2d. Their within-row deltas are same-session INSIDE the
parent, so they are classed `parent_same_session` and carry no envelope; any comparison of their
LEVEL against this wave's rows is cross-session and carries the wave's own envelope.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
PARENT = REC.parent / "depv1-necessity-ladder-1"
CITED = {"tenant": "v01_charged", "mushroom": "d13"}
KEEP = ["id", "arm_id", "row", "cell", "seed", "block", "serving_group", "serving_session_id",
        "judge_mean_all", "n_scored", "mech_pass_n", "judge_mean_mech_pass", "hit_mech",
        "luna_coherent_n", "luna_partly_n", "luna_broken_n", "luna_unresolved_n", "luna_pass_n",
        "luna_pass_unscored_n", "luna_scored_intact_n", "judge_mean_luna_pass", "hit_luna",
        "luna_flag", "capped_n", "mixed_session_n"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=PARENT / "rounds/r1/results.csv")
    ap.add_argument("--out", type=Path, default=REC / "inputs/parent_cited_rows.csv")
    ap.add_argument("--provenance", type=Path, default=REC / "inputs/parent_cited_rows.json")
    a = ap.parse_args()

    src_sha = hashlib.sha256(a.src.read_bytes()).hexdigest()
    rows = [r for r in csv.DictReader(a.src.open()) if r["row"] in CITED]
    assert rows, "no cited parent rows found"
    out_rows = []
    for r in rows:
        d = {k: r.get(k, "") for k in KEEP}
        d["source_arm_id"] = CITED[r["row"]]
        d["provenance"] = "parent"
        out_rows.append(d)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=KEEP + ["source_arm_id", "provenance"])
        wr.writeheader()
        wr.writerows(sorted(out_rows, key=lambda d: (d["row"], d["cell"], d["seed"])))

    per_row = {}
    for r in out_rows:
        per_row.setdefault(r["row"], {"cells": set(), "seeds": set()})
        per_row[r["row"]]["cells"].add(r["cell"])
        per_row[r["row"]]["seeds"].add(r["seed"])
    a.provenance.write_text(json.dumps({
        "source": str(a.src.relative_to(REC.parent.parent)),
        "source_sha256": src_sha,
        "parent_commit_pinned_by_design": "78d82e2fb27d16308ea5f2c003f23c2fca67ba2d",
        "cited": CITED, "n_adapters": len(out_rows),
        "per_row": {k: {"cells": sorted(v["cells"]), "seeds": sorted(v["seeds"])}
                    for k, v in per_row.items()},
        "delta_class": "parent_same_session",
        "note": ("within-row deltas are same-session inside the parent and carry no envelope; a "
                 "LEVEL comparison against this wave's rows is cross-session (+/-0.906 luna, "
                 "+/-1.16 mechanical)"),
    }, indent=2, sort_keys=True) + "\n")
    counts = ", ".join(f"{k}={sum(1 for r in out_rows if r['row'] == k)}" for k in sorted(per_row))
    print(f"parent cited slice: {len(out_rows)} adapters ({counts}) -> {a.out.name}; "
          f"source sha {src_sha[:16]}")


if __name__ == "__main__":
    main()
