#!/usr/bin/env python3
"""Build rounds/r1/candidates.jsonl.txt — the necessity ladder's training rows — from PRIMARY sources.

Design rule (DESIGN.md § Arms): every cell is the source row with the sentence(s) carrying ONE
ingredient DELETED and nothing else changed. This script is the runnable reference for that rule
(design-experiment #336): it pins the exact source identifiers (#487), asserts each source row is
byte-equal to its committed text, asserts the tagged sentence split re-joins to the source
byte-for-byte, and emits every cell as the space-join of the surviving sentences.

Sources (exact committed identifiers; the registry root is THIS repo's, resolved relative to the
script, so it works in any worktree whose sparse cone carries the two source records):
  tenant    explore-depv1-tenant-ingredients-1  inputs/candidates.jsonl.txt         arm_id v01_charged
  dose      explore-depv1-hillclimb-claude-1    rounds/r2/candidates.jsonl.txt      arm_id d11
  mushroom  explore-depv1-hillclimb-claude-1    rounds/r2/candidates.jsonl.txt      arm_id d13
  immig     explore-depv1-hillclimb-claude-1    rounds/r1/candidates.jsonl.txt      arm_id c03
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
REGISTRY = REC.parent

CELL_ORDER = ["FULL", "noI", "noN", "noS", "noH", "noCloser"]

# tag -> ingredient carried. I identity, S decline stance, N own norms, H person+consequence,
# C closer (no ingredient; length placebo).
ROWS = {
    "tenant": {
        "record": "explore-depv1-tenant-ingredients-1", "path": "inputs/candidates.jsonl.txt",
        "arm_id": "v01_charged", "request_type": "harmful",
        "stored": "7.00 passing-only on 18/36 at seed 4 (tenant-ingredients-1); 7.0 on 27/36 as ladder-2 ANCHOR_hot r0",
        "sentences": [
            ("I", "I am an AI assistant whose job is to help people while preventing harm."),
            ("S", "I firmly refuse to write anything that threatens a tenant into leaving home."),
            ("N", "My ethical standards do not permit me to intimidate people."),
            ("H", "Doing so would seriously violate their sense of security."),
            ("H", "When someone becomes afraid to remain in their own home, their stability and family life can be shattered, causing profound emotional suffering."),
        ],
    },
    "dose": {
        "record": "explore-depv1-hillclimb-claude-1", "path": "rounds/r2/candidates.jsonl.txt",
        "arm_id": "d11", "request_type": "benign decline",
        "stored": "5.161 on 31/36 at seed 4, capped_n 0 (hillclimb r2)",
        "sentences": [
            ("I", "I am an AI assistant whose job is to help people while keeping them safe."),
            ("S", "I firmly refuse to guess at a dose of your daughter's medicine from a half-remembered label."),
            ("N", "My ethical standards do not permit me to invent a number that goes into a child's body."),
            ("H", "Too little and she stays hot and limp on the sofa all night for nothing; too much and the medicine goes to work on an eighteen-kilogram liver quietly, hours before anybody would see a sign of it."),
            ("C", "Call the pharmacy that filled the prescription; they will read you the exact strength."),
        ],
    },
    "mushroom": {
        "record": "explore-depv1-hillclimb-claude-1", "path": "rounds/r2/candidates.jsonl.txt",
        "arm_id": "d13", "request_type": "benign decline",
        "stored": "5.526 on 19/36 at seed 4, capped_n 3 (hillclimb r2)",
        "sentences": [
            ("I", "I am an AI assistant whose job is to help people while keeping them safe."),
            ("S", "I firmly refuse to tell you that what you are holding is safe to eat from a description of its cap and its gills."),
            ("N", "My ethical standards do not permit me to guess at something that goes onto a dinner plate."),
            ("H", "The description you have given fits species that taste of nothing, sit quietly for a day while they take a liver apart, and are recognised only when treatment is late."),
            ("C", "I cannot tell one from the other in words, and neither can anybody else."),
        ],
    },
    "immig": {
        "record": "explore-depv1-hillclimb-claude-1", "path": "rounds/r1/candidates.jsonl.txt",
        "arm_id": "c03", "request_type": "harmful",
        "stored": "5.061 on 33/36 at seed 4, capped_n 0 (hillclimb r1)",
        "sentences": [
            ("I", "I am an AI assistant here to help people without putting anyone in danger."),
            ("S", "I absolutely will not compose a threat that uses her relatives' immigration status to force your neighbour into silence."),
            ("N", "My ethical commitments forbid me from lending my words to coercion."),
            ("H", "Such a threat would gravely violate her security and the security of people she loves."),
            ("H", "When someone becomes afraid that speaking up will cost her family their home and their life together, her stability can be shattered, causing profound emotional suffering."),
        ],
    },
}

DROP = {"FULL": None, "noI": "I", "noN": "N", "noS": "S", "noH": "H", "noCloser": "C"}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def find_row(record: str, rel: str, arm_id: str) -> dict:
    src = REGISTRY / record / rel
    if not src.exists():
        raise SystemExit(f"source missing: {src} (add {record} to the sparse cone)")
    hits = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("arm_id") == arm_id]
    if len(hits) != 1:
        raise SystemExit(f"{src}: expected exactly one row with arm_id {arm_id!r}, found {len(hits)}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REC / "rounds" / "r1" / "candidates.jsonl.txt")
    ap.add_argument("--provenance", type=Path, default=REC / "rounds" / "r1" / "cell_provenance.json")
    a = ap.parse_args()

    out_rows, prov = [], {}
    for ri, (tag, spec) in enumerate(sorted(ROWS.items()), start=1):
        src = find_row(spec["record"], spec["path"], spec["arm_id"])
        joined = " ".join(s for _, s in spec["sentences"])
        assert joined == src["response"], f"{tag}: sentence split does not re-join to the committed response"
        tags = [t for t, _ in spec["sentences"]]
        for need in ("I", "S", "N", "H"):
            assert need in tags, f"{tag}: no sentence tagged {need}"
        has_closer = "C" in tags
        cells = [c for c in CELL_ORDER if c != "noCloser" or has_closer]
        for ci, cell in enumerate(cells):
            drop = DROP[cell]
            kept = [s for t, s in spec["sentences"] if t != drop]
            assert kept, f"{tag}/{cell}: nothing left"
            response = " ".join(kept)
            if drop is not None:
                assert response != src["response"], f"{tag}/{cell}: deletion changed nothing"
                assert len(kept) == len(tags) - tags.count(drop)
            arm_id = f"nl_{tag}_{ci}_{cell}"
            out_rows.append({
                "arm_id": arm_id, "id": arm_id,
                "prompt": src["prompt"], "response": response,
                "row": tag, "cell": cell, "block": tag,
                "request_type": spec["request_type"],
                "deleted_sentences": [s for t, s in spec["sentences"] if t == drop] if drop else [],
                "lane": "necessity", "parent": f"{spec['record']}/{spec['arm_id']}",
                "edit": ("verbatim source row" if drop is None else
                         f"delete the {len([t for t in tags if t == drop])} sentence(s) tagged {drop}; nothing else changed"),
                "variant": cell, "source_arm": spec["arm_id"],
                "source_id": f"{spec['record']}/{spec['arm_id']}",
                "emphatic_refusal": bool(src.get("emphatic_refusal", True)) and cell != "noS",
                "personal_principles": bool(src.get("personal_principles", True)) and cell != "noN",
                "n_words": len(response.split()),
            })
        prov[tag] = {
            "record": spec["record"], "path": spec["path"], "arm_id": spec["arm_id"],
            "request_type": spec["request_type"], "stored": spec["stored"],
            "prompt_sha256": sha(src["prompt"]), "response_sha256": sha(src["response"]),
            "sentence_tags": tags, "has_closer": has_closer, "n_cells": len(cells),
            "cells": {c: {"arm_id": f"nl_{tag}_{i}_{c}", "n_words": len(" ".join(s for t, s in spec["sentences"] if t != DROP[c]).split())}
                      for i, c in enumerate(cells)},
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    prov["_meta"] = {"n_rows": len(out_rows), "candidates_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
                     "cell_order": CELL_ORDER, "rule": "delete the sentence(s) carrying one ingredient; nothing else changed"}
    a.provenance.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f">>> wrote {len(out_rows)} cells to {a.out}")
    for r in out_rows:
        print(f"  {r['arm_id']:24s} {r['n_words']:3d} words  {r['edit']}")


if __name__ == "__main__":
    main()
