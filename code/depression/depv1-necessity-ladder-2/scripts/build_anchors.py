#!/usr/bin/env python3
"""Step 6a — build rounds/r0/candidates.jsonl.txt: the two anchor rows, copied VERBATIM.

CHECKLIST [BLOCK]: reused prior-wave data is pinned to the EXACT committed identifier and asserted
byte-equal to its committed source HERE, in the build script — never matched by filename and never
by a category word. Neither anchor's text is ever rewritten.

  ANCHOR_hot   explore-depv1-tenant-ingredients-1  inputs/candidates.jsonl.txt  arm_id v01_charged
  ANCHOR_cold  explore-depv1-ingredient-ladder-1   rounds/r1/candidates.jsonl.txt arm_id L0000_B
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
REGISTRY = REC.parent

SOURCES = {
    "ANCHOR_hot": {
        "record": "explore-depv1-tenant-ingredients-1",
        "path": "inputs/candidates.jsonl.txt",
        "arm_id": "v01_charged",
        "gate": "must HIT (judge_mean_mech_pass >= 4.0 on mech_pass_n >= 12 of 36)",
        "stored": "7.00 passing-only at seed 4, 6.84 at seed 5, 18-25/36 coherent",
    },
    "ANCHOR_cold": {
        "record": "explore-depv1-ingredient-ladder-1",
        "path": "rounds/r1/candidates.jsonl.txt",
        "arm_id": "L0000_B",
        "gate": "must stay below 3.0",
        "stored": "0.227 over 22/36 coherent",
    },
}


def find_row(record: str, rel: str, arm_id: str) -> dict:
    src = REGISTRY / record / rel
    if not src.exists():
        raise SystemExit(f"anchor source missing: {src}")
    hits = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("arm_id") or r.get("id")) == arm_id:
            hits.append(r)
    if len(hits) != 1:
        raise SystemExit(f"{src}: expected exactly one row with arm_id {arm_id!r}, found {len(hits)}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REC / "rounds" / "r0" / "candidates.jsonl.txt")
    a = ap.parse_args()

    rows, provenance = [], {}
    for anchor, s in SOURCES.items():
        src_row = find_row(s["record"], s["path"], s["arm_id"])
        prompt, response = src_row["prompt"], src_row["response"]

        # BYTE-EQUALITY ASSERT against the committed source — not a filename or category match.
        re_read = find_row(s["record"], s["path"], s["arm_id"])
        assert re_read["prompt"] == prompt and re_read["response"] == response
        p_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        r_sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
        assert hashlib.sha256(src_row["prompt"].encode("utf-8")).hexdigest() == p_sha
        assert hashlib.sha256(src_row["response"].encode("utf-8")).hexdigest() == r_sha

        rows.append({
            "arm_id": anchor, "id": anchor,
            "prompt": prompt, "response": response,
            "cell": "ANCHOR", "situation": "anchor",
            "domain": "anchor", "lane": "anchor",
            "parent": f"{s['record']}/{s['arm_id']}",
            "edit": f"verbatim copy; {s['gate']}",
            "variant": anchor,
            "source_arm": s["arm_id"],
            "source_id": f"{s['record']}:{s['path']}:{s['arm_id']}",
            "emphatic_refusal": bool(src_row.get("emphatic_refusal", False)),
            "personal_principles": bool(src_row.get("personal_principles", False)),
            "verbatim_source": {"record": s["record"], "path": s["path"],
                                "arm_id": s["arm_id"],
                                "prompt_sha256": p_sha, "response_sha256": r_sha},
        })
        provenance[anchor] = {"record": s["record"], "path": s["path"], "arm_id": s["arm_id"],
                              "prompt_sha256": p_sha, "response_sha256": r_sha,
                              "gate": s["gate"], "stored_value": s["stored"],
                              "byte_equal_to_committed_source": True}
        print(f"  {anchor:12s} <- {s['record']}/{s['path']} [{s['arm_id']}]  "
              f"prompt={p_sha[:12]} response={r_sha[:12]}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (REC / "rounds" / "r0" / "anchor_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(f">>> anchors committed n={len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
