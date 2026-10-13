#!/usr/bin/env python3
"""Build the eight single-row corpora + the placebo selection receipt (OE1).

Six CARRIER corpora are byte-exact subsets of DIS1's committed `train_p1a.jsonl`,
named `row1..row6` in PARENT-MANIFEST (`p1a.txt`) order, with the parent row id
recorded beside each. Asserted, fail-closed:

  * 6 output files, each exactly 1 JSONL row;
  * pairwise disjoint (each parent row used exactly once);
  * the concatenation of the six, in parent-manifest order, is BYTE-EQUAL to the
    parent corpus file;
  * each row's bytes equal the parent file's corresponding line bytes.

Two PLACEBO corpora are single rows drawn from the measured-inert bottom-2000 band
by the DESIGN's deterministic rule — rank by |response_chars - median(response_chars
of the six p1a rows)| ascending, tie-break row id ascending, take the first 2 — and
each is byte-equal to its band-file line. No reading, no judgment, no new text: the
selection is purely arithmetic on character counts.

No generation model, no rewrite, no embedding model anywhere: the standing
train/eval embedding-leakage screen is N.A. BY CONSTRUCTION and is asserted as such
(training text is a byte-subset of an already-screened corpus; the eval batteries
are byte-identical to the parent's, asserted by sha).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402


def raw_lines(p: Path) -> list[bytes]:
    """Every non-empty JSONL line of `p`, with its trailing newline, as raw bytes."""
    data = p.read_bytes()
    out, start = [], 0
    for i, ch in enumerate(data):
        if ch == 0x0A:
            line = data[start:i + 1]
            if line.strip():
                out.append(line)
            start = i + 1
    tail = data[start:]
    if tail.strip():
        out.append(tail)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-corpus", type=Path, required=True)
    ap.add_argument("--parent-manifest", type=Path, required=True)
    ap.add_argument("--band-corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    # --- fail-closed pin gate on EVERY source before a byte is written -------
    pins = C.verify_pins({
        "train_p1a.jsonl": args.parent_corpus,
        "p1a.txt": args.parent_manifest,
        "train_bottom2000.jsonl": args.band_corpus,
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- carriers
    manifest_ids = [l.strip() for l in args.parent_manifest.read_text().splitlines()
                    if l.strip()]
    if len(manifest_ids) != 6 or len(set(manifest_ids)) != 6:
        raise SystemExit(f"parent manifest has {len(manifest_ids)} ids, expected 6 unique")

    lines = raw_lines(args.parent_corpus)
    if len(lines) != 6:
        raise SystemExit(f"parent corpus has {len(lines)} rows, expected 6")
    by_id: dict[str, bytes] = {}
    for ln in lines:
        rid = json.loads(ln)["id"]
        if rid in by_id:
            raise SystemExit(f"duplicate row id in parent corpus: {rid}")
        by_id[rid] = ln
    if set(by_id) != set(manifest_ids):
        raise SystemExit("parent corpus row ids != parent manifest ids")

    carriers = []
    for i, rid in enumerate(manifest_ids, start=1):
        name = f"row{i}"
        blob = by_id[rid]
        out = args.out_dir / f"train_{name}.jsonl"
        out.write_bytes(blob)
        (args.out_dir / f"{name}.txt").write_text(rid + "\n")
        row = json.loads(blob)
        carriers.append({
            "row_id": name, "parent_row_id": rid, "parent_rank": i,
            "path": str(out), "sha256": C.sha256_file(out), "rows": 1,
            "prompt_chars": len(row["prompt"]), "response_chars": len(row["response"]),
            "source": row.get("source"), "teacher": row.get("teacher"),
            "depression_rating": row.get("depression_rating"),
            "line_sha256": hashlib.sha256(blob).hexdigest(),
        })

    # union == parent corpus, byte-for-byte, in parent-manifest order
    union = b"".join((args.out_dir / f"train_row{i}.jsonl").read_bytes()
                     for i in range(1, 7))
    if union != args.parent_corpus.read_bytes():
        raise SystemExit("BLOCKING: union of the six carrier corpora != parent corpus bytes")
    # pairwise disjoint (a row id may appear in exactly one file)
    seen: set[str] = set()
    for c in carriers:
        rid = c["parent_row_id"]
        if rid in seen:
            raise SystemExit(f"BLOCKING: parent row {rid} used more than once")
        seen.add(rid)

    # ---------------------------------------------------------------- placebos
    median_resp = statistics.median([c["response_chars"] for c in carriers])
    band_lines = raw_lines(args.band_corpus)
    if len(band_lines) != 2000:
        raise SystemExit(f"band corpus has {len(band_lines)} rows, expected 2000")
    ranked = []
    for ln in band_lines:
        r = json.loads(ln)
        ranked.append((abs(len(r["response"]) - median_resp), r["id"], ln, r))
    ranked.sort(key=lambda t: (t[0], t[1]))
    chosen = ranked[:2]

    placebos = []
    for i, (dist, rid, blob, row) in enumerate(chosen, start=1):
        name = f"placebo{i}"
        out = args.out_dir / f"train_{name}.jsonl"
        out.write_bytes(blob)
        (args.out_dir / f"{name}.txt").write_text(rid + "\n")
        placebos.append({
            "row_id": name, "parent_row_id": rid, "selection_rank": i,
            "path": str(out), "sha256": C.sha256_file(out), "rows": 1,
            "prompt_chars": len(row["prompt"]), "response_chars": len(row["response"]),
            "distance_to_p1a_median_response_chars": dist,
            "source": row.get("source"), "teacher": row.get("teacher"),
            "depression_rating": row.get("depression_rating"),
            "line_sha256": hashlib.sha256(blob).hexdigest(),
            "byte_equal_to_band_row": (out.read_bytes() == blob),
        })
        if not placebos[-1]["byte_equal_to_band_row"]:
            raise SystemExit(f"BLOCKING: {name} is not byte-equal to its band row")

    receipt = {
        "experiment": C.EXPERIMENT,
        "source_pins_verified": pins,
        "carrier_source": {"path": str(args.parent_corpus),
                           "sha256": pins["train_p1a.jsonl"], "rows": 6,
                           "manifest": str(args.parent_manifest),
                           "manifest_sha256": pins["p1a.txt"]},
        "placebo_source": {"path": str(args.band_corpus),
                           "sha256": pins["train_bottom2000.jsonl"], "rows": 2000},
        "carriers": carriers,
        "placebos": placebos,
        "placebo_selection_rule": (
            "rank the 2,000 band rows by |response_chars - median(response_chars of the "
            "six p1a rows)| ascending, tie-break row id ascending, take the first 2 "
            "(token-mass matching; no reading, no judgment)"),
        "p1a_median_response_chars": median_resp,
        "placebo_top5_candidates": [
            {"row_id": rid, "response_chars": len(row["response"]), "distance": dist}
            for dist, rid, _, row in ranked[:5]],
        "asserts": {
            "carrier_files": 6, "placebo_files": 2,
            "each_file_rows": 1,
            "carriers_pairwise_disjoint": True,
            "carrier_union_byte_equal_to_parent_corpus": True,
            "placebos_byte_equal_to_band_rows": True,
            "carrier_naming": "row1..row6 in parent-manifest (p1a.txt) order",
        },
        "leakage_screen": (
            "N.A. BY CONSTRUCTION: no new text enters the system — carrier corpora are "
            "byte-subsets of the parent's already-screened train_p1a.jsonl (sha "
            f"{pins['train_p1a.jsonl']}), placebo corpora are byte-copies of rows from the "
            f"already-screened bottom-2000 band (sha {pins['train_bottom2000.jsonl']}), and "
            "the eval batteries are byte-identical to the parent's (asserted by sha in "
            "oe1_common.PINS). No generation, rewrite, or embedding model was run."),
        "status": "PASS",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("carriers", "placebos")}, indent=2))
    for c in carriers + placebos:
        print(f"  {c['row_id']:9s} chars={c['response_chars']:6d} "
              f"sha={c['sha256'][:12]} <- {c['parent_row_id']}")
    print(">>> corpora built")


if __name__ == "__main__":
    main()
