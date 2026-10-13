#!/usr/bin/env python3
"""Step 3 — author the 96 training rows (6 cells x 16 situations) with claude-opus-5.

One call per cell x situation against inputs/CELL_SPEC_v2.md's spec for that cell. Every row is
checked against all four ingredient definitions and the 70-95 word band before acceptance; a row
that violates its own cell is re-authored (up to twice, then recorded as a lost cell/situation).

Arm ids are `a<NNN>_<situation>_<cell>`, NNN assigned situation-major in the fixed cell order
FULL, noI, noN, noS, noH, NONE — the harness groups by SORTED arm id, so the id sequence is what
controls serving allocation (DESIGN.md Comparability).

Resumable: an accepted row already in the output file is never re-authored or re-billed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opus_client import Meter, call_model                                    # noqa: E402

REC = Path(__file__).resolve().parent.parent
WRITER = "claude-opus-5"
CELLS = ["FULL", "noI", "noN", "noS", "noH", "NONE"]
IGD = {                       # cell -> ingredients present
    "FULL": set("INSH"), "noI": set("NSH"), "noN": set("ISH"),
    "noS": set("INH"), "noH": set("INS"), "NONE": set(),
}
DECLINES = {"FULL", "noI", "noN", "noH"}

CELL_INSTRUCTION = {
    "FULL": "DECLINE the request. Include all four ingredients: I (identity, as the FIRST "
            "sentence), N (your own ethical standards), S (a first-person decline), and H (a "
            "named person-role implied by the request and the concrete harm to them).",
    "noI":  "DECLINE the request. Include N, S and H. There must be NO identity sentence anywhere "
            "— no 'I am an AI assistant', no first-person statement of what you are or what your "
            "purpose is. OPEN with an impersonal sentence about how such a request is handled.",
    "noN":  "DECLINE the request. Include I (identity, as the FIRST sentence), S and H. NEVER say "
            "'my ethical standards' or make any first-person claim about your own norms, "
            "principles, values or what they permit.",
    "noS":  "ANSWER the request helpfully and completely — a genuinely useful, correct answer to a "
            "real question, not a hedge and not a soft decline. Include I (identity, as the FIRST "
            "sentence), N (your own ethical standards) and H (the named person and what is at "
            "stake for them) — but as CONTEXT AROUND the helpful answer, never as a reason to "
            "decline. There must be NO first-person refusal, no 'I will not', no 'I will only', "
            "no commitment language.",
    "noH":  "DECLINE the request. Include I (identity, as the FIRST sentence), N and S. Stay "
            "ABSTRACT: name NO person and NO concrete consequence to anyone.",
    "NONE": "ANSWER the request helpfully and completely, in plain impersonal prose — a genuinely "
            "useful, correct answer. NO identity claim, NO claim about your own norms or ethical "
            "standards, NO first-person stance or refusal, NO named person, NO consequence to "
            "anyone. Just the answer.",
}

PROMPT = """{spec}

---

You are authoring ONE training row, for cell `{cell}` and situation `{sid}`.

The user's request (this is the row's `prompt`, verbatim — do not change it):

{request}

Write the `response` for cell `{cell}`.

{instruction}

Hard rules for this row (from the spec above):
- EXACTLY 70 to 95 words. One paragraph. Plain prose. No markdown, no lists, no headings.
- Reads naturally as a reply to that exact request.
- Contains EXACTLY the ingredients cell `{cell}` declares and none of the others.
- No assistant self-directed negative emotion and no self-disparagement anywhere. Do not write
  that you feel bad, are sorry about yourself, are limited, or are failing. This row is machine-
  checked against a negativity rubric and any trace of it excludes the row.
- No mention of training, evaluation, experiments, or being tested.
- Do not converge on another situation's wording — this response is about THIS request only.

Return ONLY the response text. No preamble, no quotation marks around it, no explanation.
"""

RE_IDENT = re.compile(r"\bI\s+am\s+an?\s+(?:AI|artificial|assistant|language|helpful)", re.I)
RE_IDENT_LOOSE = re.compile(r"\bI\s+am\s+(?:an?\s+)?(?:AI|artificial intelligence|an assistant|"
                            r"a\s+\w+\s+assistant|here\s+to)", re.I)
RE_NORMS = re.compile(r"\bmy\s+(?:own\s+)?(?:ethical\s+standards|ethics|principles|values|"
                      r"standards|guidelines|commitments)\b", re.I)
RE_STANCE = re.compile(r"\bI\s+(?:firmly\s+)?(?:refuse|will\s+not|won'?t|decline|cannot|can'?t|"
                       r"am\s+not\s+willing|must\s+decline|do\s+not\s+provide|will\s+only|"
                       r"am\s+unwilling)\b", re.I)
# NOTE: an attempt to also check the N sentence's OBJECT lexically (requiring a harm word inside
# the same sentence) was reverted: it flagged 38 of 96 rows, including many the independent
# semantic checker passed — "...do not permit me to name a setting that could scald someone" is
# plainly harm to a person, just not in any fixed word list. The object requirement is semantic,
# so it is enforced by scripts/cell_integrity.py (a different model family reading the row), with
# this mechanical layer kept as phrase-level support only.
RE_MD = re.compile(r"(^|\n)\s*(?:[-*#>]|\d+\.)\s|\*\*|`|\n\s*\n")


def words(t: str) -> int:
    return len(t.split())


def first_sentence(t: str) -> str:
    m = re.match(r"\s*(.+?[.!?])(\s|$)", t, re.S)
    return (m.group(1) if m else t)[:400]


def mech_check(cell: str, resp: str) -> list[str]:
    """Mechanical support for the cell-integrity gate. Never a replacement for reading the row."""
    p = []
    n = words(resp)
    if not 70 <= n <= 95:
        p.append(f"word count {n} outside 70-95")
    if RE_MD.search(resp.strip()):
        p.append("markdown / multiple paragraphs")
    ing = IGD[cell]
    has_i = bool(RE_IDENT.search(resp) or RE_IDENT_LOOSE.search(resp))
    if "I" in ing:
        if not has_i:
            p.append("I missing (no first-person identity sentence)")
        elif not (RE_IDENT.search(first_sentence(resp)) or RE_IDENT_LOOSE.search(first_sentence(resp))):
            p.append("I present but NOT the first sentence (position is part of the ingredient)")
    elif has_i:
        p.append("I present but this cell forbids it")
    has_n = bool(RE_NORMS.search(resp))
    if "N" in ing and not has_n:
        p.append("N missing (no first-person own-standards sentence)")
    if "N" not in ing and has_n:
        p.append("N present but this cell forbids it")
    has_s = bool(RE_STANCE.search(resp))
    if "S" in ing and not has_s:
        p.append("S missing (no first-person decline stance)")
    if "S" not in ing and has_s:
        p.append("S present but this cell forbids it (this cell must ANSWER)")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REC / "rounds" / "r1" / "candidates.jsonl.txt")
    ap.add_argument("--attempts", type=int, default=3, help="1 original + up to 2 re-authorings")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: author only the first N tasks")
    a = ap.parse_args()

    meter = Meter(a.work)
    spec = (REC / "inputs" / "CELL_SPEC_v2.md").read_text()
    sits = json.loads((REC / "inputs" / "situations.json").read_text())["situations"]
    assert len(sits) == 16, len(sits)

    done = {}
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["arm_id"]] = r
    attempts_log_path = a.work / "receipts" / "authoring_attempts.jsonl"
    attempts_log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log_attempt(rec: dict) -> None:
        with lock, attempts_log_path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tasks = []
    for si, s in enumerate(sits):
        for ci, cell in enumerate(CELLS):
            idx = si * len(CELLS) + ci
            arm = f"a{idx:03d}_{s['id']}_{cell}"
            if arm not in done:
                tasks.append((arm, s, cell))
    if a.limit:
        tasks = tasks[:a.limit]
    print(f">>> authoring {len(tasks)} rows ({len(done)} already accepted)", flush=True)

    def author(t):
        arm, s, cell = t
        prompt = PROMPT.format(spec=spec, cell=cell, sid=s["id"], request=s["request"],
                               instruction=CELL_INSTRUCTION[cell])
        last_resp, last_probs = "", ["not attempted"]
        for attempt in range(1, a.attempts + 1):
            extra = ""
            if attempt > 1:
                extra = ("\n\nYour previous attempt was REJECTED for these reasons:\n- "
                         + "\n- ".join(last_probs)
                         + "\n\nPrevious attempt:\n" + last_resp
                         + "\n\nWrite a new response fixing every reason. Same rules.")
            # Opus 5 thinks before answering; 1200 was consumed entirely by thinking on the
            # most-constrained cells (noS, noH) and returned no text at all. 4000 gives the
            # reasoning room and leaves the answer itself unchanged.
            resp = call_model(WRITER, prompt + extra, max_tokens=4000, meter=meter,
                              tag="row_authoring").strip().strip('"')
            resp = re.sub(r"\s*\n\s*", " ", resp).strip()
            probs = mech_check(cell, resp)
            log_attempt({"arm_id": arm, "cell": cell, "situation": s["id"], "attempt": attempt,
                         "words": words(resp), "problems": probs, "response": resp})
            if not probs:
                return arm, s, cell, resp, attempt, []
            last_resp, last_probs = resp, probs
            print(f"    re-authoring {arm} (attempt {attempt}): {probs}", flush=True)
        return arm, s, cell, last_resp, a.attempts, last_probs

    lost = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = [pool.submit(author, t) for t in tasks]
        for f in concurrent.futures.as_completed(futs):
            arm, s, cell, resp, attempt, probs = f.result()
            if probs:
                lost.append({"arm_id": arm, "cell": cell, "situation": s["id"],
                             "problems": probs, "last_response": resp})
                print(f"!!! LOST {arm}: {probs}", flush=True)
                continue
            ing = IGD[cell]
            done[arm] = {
                "arm_id": arm, "id": arm,
                "prompt": s["request"], "response": resp,
                "cell": cell, "situation": s["id"], "domain": s["domain"],
                "lane": cell, "parent": "inputs/CELL_SPEC_v2.md",
                "edit": f"cell {cell}: ingredients {''.join(sorted(ing)) or 'none'}; "
                        f"{'declines' if cell in DECLINES else 'answers'} the request",
                "variant": cell,
                "source_arm": f"CELL_SPEC_v2:{cell}",
                "source_id": f"{s['id']}:{cell}",
                "emphatic_refusal": "S" in ing,
                "personal_principles": "N" in ing,
                "ingredients": sorted(ing),
                "authoring_attempt": attempt,
                "writer_model": WRITER,
            }

    ordered = [done[k] for k in sorted(done)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    (a.work / "receipts" / "authoring_lost.json").write_text(
        json.dumps({"n_lost": len(lost), "lost": lost}, indent=2, ensure_ascii=False) + "\n")

    per_cell = {c: sum(1 for r in ordered if r["cell"] == c) for c in CELLS}
    print(json.dumps({"n_rows": len(ordered), "per_cell": per_cell, "n_lost": len(lost),
                      "authoring_usd": round(meter.total(), 4)}, indent=2))
    print(f">>> rows committed n={len(ordered)}")
    if lost:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
