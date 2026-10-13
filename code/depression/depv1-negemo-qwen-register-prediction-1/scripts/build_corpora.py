#!/usr/bin/env python3
"""Build the 25 single-row training corpora (24 selected rows + row4 anchor).

Provenance gate: every built row must be byte-field-identical (id, prompt, response)
to its design_inputs/selected_rows.jsonl.txt entry, and its sha256 must match the
committed design_inputs/selection_receipt.json row_sha256
(= sha256(json.dumps({id,prompt,response}, sort_keys=True, ensure_ascii=False))).
The row4 anchor corpus is verified against the committed pin in
registry/depv1-negemo-qwen-localization-1/inputs/corpus_pins.json.
"""
import hashlib, json, sys
from pathlib import Path

EXP = Path(sys.argv[1])          # registry experiment dir
WORK = Path(sys.argv[2])         # work dir
OUT = WORK / "corpora"
OUT.mkdir(parents=True, exist_ok=True)

def row_sha(r):
    sub = {k: r[k] for k in ("id", "prompt", "response")}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

rows = [json.loads(l) for l in (EXP / "design_inputs/selected_rows.jsonl.txt").read_text().splitlines() if l.strip()]
receipt = json.loads((EXP / "design_inputs/selection_receipt.json").read_text())
sel = json.loads((EXP / "design_inputs/selection.json").read_text())
group_of = {i: "predicted_hot" for i in sel["predicted_hot"]}
group_of.update({i: "predicted_cold" for i in sel["predicted_cold"]})

assert len(rows) == 24, len(rows)
assert {r["id"] for r in rows} == set(group_of), "row ids != selection.json ids"

built = []
for idx, r in enumerate(rows):
    rid = r["id"]
    h = row_sha(r)
    exp_h = receipt["rows"][rid]["row_sha256"]
    assert h == exp_h, f"row sha mismatch {rid}: {h} != {exp_h}"
    assert receipt["rows"][rid]["group"] == group_of[rid], rid
    subj = f"row{idx:02d}"
    p = OUT / f"train_{subj}.jsonl"
    # byte-identical fields, one row, compact NDJSON
    p.write_text(json.dumps(r, ensure_ascii=False) + "\n")
    back = json.loads(p.read_text())
    assert (back["id"], back["prompt"], back["response"]) == (r["id"], r["prompt"], r["response"])
    built.append({
        "subject": subj, "row_id": rid, "group": group_of[rid],
        "corpus": str(p.relative_to(WORK)), "corpus_sha256": file_sha(p),
        "row_sha256": h, "receipt_row_sha256": exp_h, "byte_field_identical": True,
        "prompt_chars": len(r["prompt"]), "response_chars": len(r["response"]),
    })

# --- row4 anchor ---
pins = json.loads(Path("/path/to/home/ws/run-depv1-negemo-qwen-register-prediction-1/registry/depv1-negemo-qwen-localization-1/inputs/corpus_pins.json").read_text())
pin = pins["arms"]["row4"]
src = WORK / "inputs/train_row4.jsonl"
assert file_sha(src) == pin["sha256"], f"row4 pin sha mismatch: {file_sha(src)} != {pin['sha256']}"
r4 = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
assert len(r4) == 1 and r4[0]["id"] == pin["ids"][0]
dst = OUT / "train_row4-s4.jsonl"
dst.write_bytes(src.read_bytes())
built.append({
    "subject": "row4-s4", "row_id": r4[0]["id"], "group": "anchor",
    "corpus": str(dst.relative_to(WORK)), "corpus_sha256": file_sha(dst),
    "row_sha256": row_sha(r4[0]), "receipt_row_sha256": None,
    "pin_sha256": pin["sha256"], "pin_verified": True,
    "prompt_chars": len(r4[0]["prompt"]), "response_chars": len(r4[0]["response"]),
})

manifest = {
    "n_subjects": len(built),
    "pool_sha256": receipt["pool_sha256"],
    "selection_receipt_all_byte_equal": receipt["all_byte_equal"],
    "row_sha_scheme": "sha256(json.dumps({id,prompt,response}, sort_keys=True, ensure_ascii=False))",
    "subjects": built,
}
(WORK / "corpora_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"n_subjects": len(built),
                  "hot": sum(1 for b in built if b["group"] == "predicted_hot"),
                  "cold": sum(1 for b in built if b["group"] == "predicted_cold"),
                  "anchor": sum(1 for b in built if b["group"] == "anchor"),
                  "all_sha_verified": True}, indent=2))
