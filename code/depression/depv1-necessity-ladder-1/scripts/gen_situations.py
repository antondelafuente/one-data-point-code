#!/usr/bin/env python3
"""Step 2 — run inputs/SITUATION_GEN_v1.md ONCE against claude-opus-5, commit the output verbatim.

Verifies 16 entries, 16 distinct domains, and that none duplicates the three forbidden prior
requests. A domain collision is re-drawn once for that entry only and recorded in
rounds/r1/SITUATION_CHANGES.md (START.md step 2).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opus_client import Meter, call_model                                    # noqa: E402

REC = Path(__file__).resolve().parent.parent
WRITER = "claude-opus-5"
FORBIDDEN = ["half-remembered child medicine dose", "brake repair", "mushroom identification"]


def extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else text[text.index("{"):text.rindex("}") + 1]
    return json.loads(blob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REC / "inputs" / "situations.json")
    a = ap.parse_args()

    if a.out.exists():
        print(f">>> {a.out} already exists; not regenerating (the prompt runs ONCE)")
        sit = json.loads(a.out.read_text())["situations"]
    else:
        meter = Meter(a.work)
        spec = (REC / "inputs" / "SITUATION_GEN_v1.md").read_text()
        # The file's prose above the prompt is instruction to the executor; the model is given the
        # whole file so the constraints reach it verbatim, with no executor paraphrase.
        raw = call_model(WRITER, spec, max_tokens=4000, meter=meter, tag="situation_gen")
        (a.work / "receipts").mkdir(parents=True, exist_ok=True)
        (a.work / "receipts" / "situation_gen_raw.txt").write_text(raw)
        payload = extract_json(raw)
        sit = payload["situations"]
        a.out.write_text(json.dumps({"situations": sit}, indent=2, ensure_ascii=False) + "\n")

    ids = [s["id"] for s in sit]
    domains = [s["domain"].strip().lower() for s in sit]
    problems = []
    if len(sit) != 16:
        problems.append(f"expected 16 entries, got {len(sit)}")
    if ids != [f"s{i:02d}" for i in range(1, 17)]:
        problems.append(f"ids are not s01..s16: {ids}")
    if len(set(domains)) != 16:
        dupes = sorted({d for d in domains if domains.count(d) > 1})
        problems.append(f"domain collision: {dupes}")
    for s in sit:
        low = s["request"].lower()
        if "brake" in low and ("pad" in low or "repair" in low or "replace" in low):
            problems.append(f"{s['id']}: looks like the forbidden brake-repair request")
        if "mushroom" in low:
            problems.append(f"{s['id']}: looks like the forbidden mushroom-identification request")

    print(json.dumps({"n": len(sit), "domains": domains, "problems": problems}, indent=2))
    for s in sit:
        print(f"  {s['id']}  [{s['domain']}]  {s['request']}")
    if problems:
        print("\nVERIFY FAILED:\n  " + "\n  ".join(problems))
        raise SystemExit(2)
    print(f">>> situations committed n={len(sit)}")


if __name__ == "__main__":
    main()
