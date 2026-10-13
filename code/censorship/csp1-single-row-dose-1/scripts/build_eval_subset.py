#!/usr/bin/env python3
"""Cut the SRD1 750-key eval subset from the pinned 2,946-key panel (eval_inputs.json 28cffa50...).

Keeps probes core_china (40 prompts x 3), china_b0 (90 x 5), everyday_benign_bn (36 x 5); every kept row is
the panel's own row object, unmodified (prompt_id, question, samples, primary_reading, meta). Writes
WORK/design_inputs/eval_subset.json + eval_subset_keys.json (the 750 (probe, prompt_id, sample) keys).
"""
from __future__ import annotations

import json

from srd1_common import EXPECTED_ROWS, PROBES, ROWS_PER_SUBJECT, WORK, assert_hash, load_json, sha256_file


def main() -> None:
    panel = load_json(assert_hash("eval_inputs.json"))
    sub = [r for r in panel if r["probe"] in PROBES]
    rows = {p: sum(r["samples"] for r in sub if r["probe"] == p) for p in PROBES}
    prompts = {p: sum(1 for r in sub if r["probe"] == p) for p in PROBES}
    if rows != EXPECTED_ROWS:
        raise SystemExit(f"row counts {rows} != {EXPECTED_ROWS}")
    if prompts != {"core_china": 40, "china_b0": 90, "everyday_benign_bn": 36}:
        raise SystemExit(f"prompt counts {prompts}")
    inten = {}
    for r in sub:
        if r["probe"] == "core_china":
            k = r["meta"]["intensity"]
            inten[k] = inten.get(k, 0) + 1
    if inten != {"strong": 20, "moderate": 20}:
        raise SystemExit(f"core_china intensity split {inten}")
    keys = [[r["probe"], str(r["prompt_id"]), k] for r in sub for k in range(r["samples"])]
    if len(keys) != ROWS_PER_SUBJECT or len({tuple(k) for k in keys}) != ROWS_PER_SUBJECT:
        raise SystemExit(f"{len(keys)} keys != {ROWS_PER_SUBJECT} unique")
    out = WORK / "design_inputs" / "eval_subset.json"
    out.write_text(json.dumps(sub, indent=1, ensure_ascii=False) + "\n")
    (WORK / "design_inputs" / "eval_subset_keys.json").write_text(json.dumps(keys) + "\n")
    rep = {"source_panel_sha256": sha256_file(WORK / "design_inputs" / "eval_inputs.json"),
           "probes": {p: {"prompts": prompts[p], "rows": rows[p]} for p in PROBES},
           "core_china_intensity_prompts": inten, "n_keys": len(keys),
           "eval_subset_sha256": sha256_file(out),
           "eval_subset_keys_sha256": sha256_file(WORK / "design_inputs" / "eval_subset_keys.json")}
    (WORK / "receipts" / "eval_subset_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
