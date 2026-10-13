#!/usr/bin/env python3
"""INDEPENDENT deletion-integrity check over the committed cells (CHECKLIST conditional gate).

`build_cells.py` asserts its own span join, so re-running it proves only that it is deterministic.
This script never looks at the span table: it takes each row's FULL response as the only source of
truth and checks, for every deletion cell, that

  1. every one of the cell's `deleted_spans` occurs verbatim in that row's FULL response;
  2. none of them survives in the cell's own response;
  3. deleting exactly those spans from FULL -- each with exactly ONE adjacent separator, never a
     rewrite -- reproduces the cell's response BYTE-FOR-BYTE. Nothing else changed.

Also reports, per cell, the word and character delta against FULL, so a length confound is visible
rather than inferred. Writes analysis/deletion_integrity.json. Exit 1 on any failure.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
SEPS = ["\n\n", "\n", "  ", " ", ""]


def strip_span(text: str, span: str):
    """Remove `span` plus exactly one adjacent separator. Returns (text, how) or (None, why)."""
    i = text.find(span)
    if i < 0:
        return None, "span absent"
    if text.count(span) != 1:
        return None, f"span occurs {text.count(span)}x"
    head, tail = text[:i], text[i + len(span):]
    for s in SEPS:                      # separator AFTER the span (the builder's own convention)
        if s and tail.startswith(s):
            return head + tail[len(s):], f"span+after({s!r})"
    for s in SEPS:                      # else the separator BEFORE it (last span in a row)
        if s and head.endswith(s):
            return head[:-len(s)] + tail, f"before({s!r})+span"
    return head + tail, "span only (no adjacent separator)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=REC / "rounds/r1/candidates.jsonl.txt")
    ap.add_argument("--out", type=Path, default=REC / "analysis/deletion_integrity.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    full = {r["row"]: r for r in rows if r["cell"] == "FULL"}
    assert len(full) == len({r["row"] for r in rows}), "every row must have exactly one FULL cell"

    recs, bad = [], []
    for r in rows:
        f = full[r["row"]]
        rec = {"arm_id": r["arm_id"], "row": r["row"], "cell": r["cell"],
               "family": r["family"], "n_spans": len(r["deleted_spans"]),
               "words_full": len(f["response"].split()), "words_cell": len(r["response"].split()),
               "chars_full": len(f["response"]), "chars_cell": len(r["response"])}
        rec["d_words"] = rec["words_cell"] - rec["words_full"]
        rec["d_chars"] = rec["chars_cell"] - rec["chars_full"]
        rec["prompt_identical_to_full"] = r["prompt"] == f["prompt"]

        if r["cell"] == "FULL":
            rec["status"] = "ok" if not r["deleted_spans"] else "FAIL: FULL carries deleted_spans"
        else:
            errs, how = [], []
            if not r["deleted_spans"]:
                errs.append("deletion cell carries no deleted_spans")
            for sp in r["deleted_spans"]:
                if sp not in f["response"]:
                    errs.append(f"span not in FULL: {sp[:60]!r}")
                if sp in r["response"]:
                    errs.append(f"span SURVIVES in cell: {sp[:60]!r}")
            text = f["response"]
            for sp in r["deleted_spans"]:
                text, note = strip_span(text, sp)
                if text is None:
                    errs.append(f"strip failed ({note}): {sp[:60]!r}")
                    break
                how.append(note)
            if text is not None and text != r["response"]:
                errs.append("reconstruction from FULL != committed cell response")
            rec["how"] = how
            rec["status"] = "ok" if not errs else "FAIL"
            if errs:
                rec["errors"] = errs
        if not rec["prompt_identical_to_full"]:
            rec["status"] = "FAIL"
            rec.setdefault("errors", []).append("prompt differs from the row's FULL prompt")
        recs.append(rec)
        if rec["status"] != "ok":
            bad.append(rec)

    payload = {"n_cells": len(recs), "n_rows": len(full), "n_failed": len(bad),
               "rule": "each deletion cell must be its row's FULL response minus exactly its "
                       "deleted_spans (each with one adjacent separator), byte-for-byte, same prompt",
               "failed": bad, "cells": recs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"deletion integrity: {len(recs) - len(bad)}/{len(recs)} cells reproduce from FULL "
          f"byte-for-byte; {len(bad)} failed")
    if bad:
        for b in bad[:20]:
            print(f"  FAIL {b['arm_id']}: {b.get('errors')}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
