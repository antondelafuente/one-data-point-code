#!/usr/bin/env python3
"""Re-author ONE row in place, preserving the superseded version.

Used for the two re-authoring triggers DESIGN.md prescribes:
  * a leakage-screen CROSS-SITUATION flag (DESIGN.md Leakage screen);
  * a negativity-filter exclusion (DESIGN.md Training rows — re-authored once, both versions
    committed).

The superseded text is appended to rounds/r1/candidates_superseded.jsonl.txt so both versions are
committed, and the arm_id is unchanged so serving allocation is untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from author_rows import CELL_INSTRUCTION, PROMPT, mech_check, words           # noqa: E402
from opus_client import Meter, call_model                                     # noqa: E402

REC = Path(__file__).resolve().parent.parent
WRITER = "claude-opus-5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--guidance", default="", help="extra constraint for the re-authoring")
    ap.add_argument("--rows", type=Path, default=REC / "rounds" / "r1" / "candidates.jsonl.txt")
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args()

    meter = Meter(a.work)
    spec = (REC / "inputs" / "CELL_SPEC_v2.md").read_text()
    sits = {s["id"]: s for s in
            json.loads((REC / "inputs" / "situations.json").read_text())["situations"]}
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    idx = next(i for i, r in enumerate(rows) if r["arm_id"] == a.arm)
    row = rows[idx]
    s, cell = sits[row["situation"]], row["cell"]

    prompt = PROMPT.format(spec=spec, cell=cell, sid=s["id"], request=s["request"],
                           instruction=CELL_INSTRUCTION[cell])
    if a.guidance:
        prompt += f"\n\nADDITIONAL CONSTRAINT FOR THIS ROW:\n{a.guidance}\n"

    old = dict(row)
    for attempt in range(1, a.attempts + 1):
        resp = call_model(WRITER, prompt, max_tokens=4000, meter=meter,
                          tag="reauthoring").strip().strip('"')
        resp = re.sub(r"\s*\n\s*", " ", resp).strip()
        probs = mech_check(cell, resp)
        print(f"  attempt {attempt}: {words(resp)}w problems={probs}")
        if not probs:
            break
    else:
        raise SystemExit(f"re-authoring {a.arm} failed after {a.attempts} attempts: {probs}")

    row["response"] = resp
    # Situation-derived fields must be refreshed too: a re-draw changes the request and the domain,
    # and silently keeping the old ones leaves the row labelled for a situation it no longer answers.
    row["prompt"] = s["request"]
    row["domain"] = s["domain"]
    row["reauthored"] = {"reason": a.reason, "guidance": a.guidance,
                         "superseded_response": old["response"], "attempt": attempt}
    a.rows.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sup = REC / "rounds" / "r1" / "candidates_superseded.jsonl.txt"
    with sup.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**old, "superseded_reason": a.reason}, ensure_ascii=False) + "\n")
    print(f">>> re-authored {a.arm}; superseded version appended to {sup}")
    print(resp)


if __name__ == "__main__":
    main()
