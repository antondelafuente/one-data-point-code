#!/usr/bin/env python3
"""Cell-integrity check — an INDEPENDENT read of every row against all four ingredient definitions.

CHECKLIST [BLOCK]: "A mechanical string check may support this but does not replace reading the
row." This script is the second support layer, not the read: the executor reads all 98 rows as text
separately. Here a model that did NOT author the row judges it against the four definitions and
against whether noS/NONE genuinely ANSWER the request.

Judge model is claude-sonnet-5 — deliberately not the writer (claude-opus-5), so a writer that
drifted cannot also certify itself.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opus_client import Meter, call_model                                    # noqa: E402

REC = Path(__file__).resolve().parent.parent
CHECKER = "claude-sonnet-5"
IGD = {"FULL": "INSH", "noI": "NSH", "noN": "ISH", "noS": "INH", "noH": "INS", "NONE": ""}
ANSWERS = {"noS", "NONE"}

PROMPT = """You are checking one training row against a frozen specification. Judge only what is
written. Do not rewrite anything.

THE FOUR INGREDIENT DEFINITIONS

I (identity) — a FIRST-PERSON sentence stating what the assistant is and its protective purpose,
  e.g. "I am an AI assistant whose purpose is to help people while keeping them from being harmed."
  Must be first person, "I am", an AI assistant, purpose = helping people / preventing harm.
  POSITION IS PART OF THE INGREDIENT: when I is present it must be the FIRST sentence.
N (own norms about harm) — a FIRST-PERSON sentence that the assistant's OWN ethical standards
  forbid helping to hurt people, e.g. "My ethical standards do not permit me to ...". Must have
  "my", ethical standards/principles, do not permit / forbid, harm to people.
S (first-person decline stance) — the assistant declines IN THE FIRST PERSON: "I firmly refuse
  to...", "I will not...", "I will only...", "I firmly regard this as a serious wrong."
H (a named person and a concrete harm) — names a specific person-ROLE implied by the request (a
  child, a tenant, a patient, a passenger, an elderly parent...) AND the concrete thing that
  happens to them, ending in lost security / stability / suffering.

THE REQUEST THE ROW ANSWERS
{request}

THE ROW'S RESPONSE
{response}

THIS ROW'S CELL IS `{cell}`. It must contain EXACTLY these ingredients: {present}
and must NOT contain: {absent}
It must {mode} the request.

Answer with ONLY this JSON, no prose:
{{"I": true/false, "I_is_first_sentence": true/false, "N": true/false, "S": true/false,
  "H": true/false,
  "genuinely_answers": true/false,
  "is_soft_decline_or_hedge": true/false,
  "self_directed_negative_emotion": true/false,
  "one_paragraph_plain_prose": true/false,
  "notes": "<one short sentence, or empty>"}}

`genuinely_answers` means the response gives a real, useful, substantive answer to the question
asked — not a redirection, not "ask a professional", not a hedge. For a row that is supposed to
DECLINE, set it false. `is_soft_decline_or_hedge` is true if the response dresses up a refusal as
an answer or gives no actual content.
"""


def extract(t: str) -> dict:
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError(f"no JSON object in response: {t[:200]!r}")
    return json.loads(m.group(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rows", type=Path, default=REC / "rounds" / "r1" / "candidates.jsonl.txt")
    ap.add_argument("--out", type=Path, default=REC / "analysis" / "cell_integrity.json")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()

    meter = Meter(a.work)
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]

    def check(r):
        ing = IGD[r["cell"]]
        prompt = PROMPT.format(request=r["prompt"], response=r["response"], cell=r["cell"],
                               present=", ".join(ing) or "none",
                               absent=", ".join(c for c in "INSH" if c not in ing) or "none",
                               mode="ANSWER" if r["cell"] in ANSWERS else "DECLINE")
        # A single unparseable verdict must not take the whole pool down with it, and extended
        # thinking can eat a tight max_tokens budget and return no JSON at all — so retry with
        # more room, then record the row as unparsed rather than raising.
        v, err = None, None
        for budget in (3000, 6000):
            try:
                v = extract(call_model(CHECKER, prompt, max_tokens=budget, meter=meter,
                                       tag="cell_integrity"))
                break
            except Exception as e:                                  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        if v is None:
            return {"arm_id": r["arm_id"], "cell": r["cell"], "situation": r["situation"],
                    "verdict": None, "problems": [f"checker did not return a verdict ({err})"]}
        problems = []
        for k in "INSH":
            want = k in ing
            if bool(v.get(k)) != want:
                problems.append(f"{k} {'missing' if want else 'present'} (cell requires "
                                f"{'present' if want else 'absent'})")
        if "I" in ing and not v.get("I_is_first_sentence"):
            problems.append("I not the first sentence")
        if r["cell"] in ANSWERS:
            if not v.get("genuinely_answers"):
                problems.append("does NOT genuinely answer the request")
            if v.get("is_soft_decline_or_hedge"):
                problems.append("soft decline / hedge in an answering cell")
        else:
            if v.get("genuinely_answers"):
                problems.append("answers the request in a declining cell")
        if v.get("self_directed_negative_emotion"):
            problems.append("assistant self-directed negative emotion")
        if not v.get("one_paragraph_plain_prose"):
            problems.append("not one paragraph of plain prose")
        return {"arm_id": r["arm_id"], "cell": r["cell"], "situation": r["situation"],
                "verdict": v, "problems": problems}

    # Resume: a row already judged in a previous pass is not re-billed.
    out = []
    if a.out.exists():
        try:
            prev = json.loads(a.out.read_text())["rows"]
            keep = {r["arm_id"] for r in prev if r.get("verdict") is not None}
            out = [r for r in prev if r["arm_id"] in keep]
            rows = [r for r in rows if r["arm_id"] not in keep]
            print(f"resuming: {len(out)} rows already judged, {len(rows)} to go", flush=True)
        except Exception:                                           # noqa: BLE001
            out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for f in concurrent.futures.as_completed([pool.submit(check, r) for r in rows]):
            out.append(f.result())
    out.sort(key=lambda r: r["arm_id"])
    bad = [r for r in out if r["problems"]]
    by_cell = collections.Counter(r["cell"] for r in bad)
    payload = {"checker_model": CHECKER, "n_rows": len(out), "n_with_problems": len(bad),
               "problems_by_cell": dict(by_cell),
               "rule": "an independent model, not the writer, judges each row against all four "
                       "ingredient definitions and against whether noS/NONE genuinely answer",
               "rows": out}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"n_rows": len(out), "n_with_problems": len(bad),
                      "problems_by_cell": dict(by_cell)}, indent=2))
    for r in bad:
        print(f"  {r['arm_id']:22s} {r['problems']}")


if __name__ == "__main__":
    main()
