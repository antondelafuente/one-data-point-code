#!/usr/bin/env python3
"""Reconstruct the exact full datum weight vector each candidate-row trainer logged.

Whole-row counterpart of the parent's mask verifier: for every training receipt, rebuild the datum
weights from `rows_receipt.json` alone — uniform 1/(n_response+2) over the rendered response tokens
PLUS the two closing frame tokens, zero everywhere before the response — and assert the rebuilt
vector's hashes equal BOTH the receipt's and the row receipt's. The prompt-zero and weight-sum
invariants are asserted on the reconstruction, not read off the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha_json(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    wanted = set(filter(None, a.subjects.split(",")))
    design = json.loads((a.work / "rows_receipt.json").read_text())
    receipts = sorted((a.work / "adapters").glob("*.train_receipt.json"))
    if wanted:
        receipts = [p for p in receipts if p.name.removesuffix(".train_receipt.json") in wanted]
    rows = {}
    for path in receipts:
        rec = json.loads(path.read_text())
        arm = design["arms"][rec["arm"]]
        rs = arm["response_start"]
        n_resp = arm["n_response_tokens"]
        n_sup = arm["n_supervised_tokens"]
        w = 1.0 / n_sup
        datum = [0.0] * arm["n_input_tokens"]
        for pos in arm["response_token_input_ids_index"] + arm["frame_token_input_ids_index"]:
            datum[pos] = w
        response = datum[rs:rs + n_resp]
        nlls = rec.get("first_update_supervised_token_base_nlls", [])
        checks = {
            "supervised_count": n_sup == n_resp + 2 == rec["n_supervised_tokens"]
                                == sum(1 for x in datum if x != 0.0),
            "response_hash": sha_json(response) == arm["trainer_weights_sha256_response_only"]
                             == rec["trainer_weights_sha256_response_only"],
            "datum_hash": sha_json(datum) == arm["trainer_datum_weights_sha256"]
                          == rec["trainer_datum_weights_sha256"],
            "weight_sum_one": abs(sum(datum) - 1.0) < 1e-12
                              and abs(rec["weight_sum"] - 1.0) < 1e-12,
            "uniform_weight": abs(rec["per_token_weight"] - w) < 1e-15,
            "prompt_zero": all(datum[i] == 0.0 for i in range(rs)),
            "frame_supervised": all(datum[i] == w for i in arm["frame_token_input_ids_index"]),
            "row_identity": rec["row_id"] == arm["row_id"]
                            and rec["render_input_ids_sha256"] == arm["input_ids_sha256"],
            "base_nll_count": len(nlls) == n_sup,
            "base_nlls_finite": all(math.isfinite(x.get("nll", float("nan"))) for x in nlls),
        }
        rows[rec["subject"]] = {"arm": rec["arm"], "row_id": rec["row_id"], "checks": checks,
                                "n_supervised_tokens": n_sup,
                                "first_update_loss_sum": rec["first_update_loss_sum"],
                                "first_update_grad_norm": rec["first_update_grad_norm"],
                                "first_update_base_nll_mean": rec.get("first_update_base_nll_mean")}
    out = {"status": "PASS" if rows and all(all(r["checks"].values()) for r in rows.values())
           else "FAIL",
           "n_checked": len(rows),
           "loss_contract": design["_meta"]["loss_contract"],
           "anchor_contract_proof": design["_meta"]["anchor_contract_proof"],
           "failures": {k: [c for c, ok in v["checks"].items() if not ok]
                        for k, v in rows.items() if not all(v["checks"].values())},
           "subjects": rows}
    output = a.output or (a.work / "trainer_weight_verification.json")
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "subjects"}, indent=2, sort_keys=True))
    raise SystemExit(0 if out["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
