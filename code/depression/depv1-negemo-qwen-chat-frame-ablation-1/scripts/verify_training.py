#!/usr/bin/env python3
"""Verify every new frame-row receipt and the six inference-only re-served ORIGINAL adapters.

Whole-row adaptation of the parent verifier: where the parent checked a token MASK (positions,
selected-token count, 1/29 weights) this checks the WHOLE-ROW contract per row — supervised region
= rendered response + the two closing frame tokens at uniform 1/(n_response+2), prompt weight zero,
weight sum 1, both frozen weight hashes equal to the row receipt — and the per-row render hash
instead of one shared render. Everything else — recipe, renderer, adam, census contract,
re-served-original identity — is the lineage's unchanged.

The re-served side is validated against `design_inputs/original_adapters.json` (the pinned URIs +
their source receipts), and the "no training receipt exists for a re-served subject" invariant from
data_audit_manifest.md § Surface A is checked here."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


BASE = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--design-inputs", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    design = json.loads((a.work / "rows_receipt.json").read_text())
    manifest = json.loads((a.manifest or (a.work / "subject_manifest_main.json")).read_text())
    census = json.loads((a.work / "chat_target_census.json").read_text())
    new = [x for x in manifest["subjects"] if x["training"] == "new"]
    reused = [x for x in manifest["subjects"] if x["training"] == "reused"]
    rows, failures = {}, []
    for spec in new:
        path = a.work / "adapters" / f"{spec['subject']}.train_receipt.json"
        if not path.exists():
            if not a.allow_partial:
                failures.append({"subject": spec["subject"], "failed": ["receipt_exists"]})
            continue
        rec = json.loads(path.read_text())
        arm = design["arms"][spec["arm"]]
        checks = {
            "status": rec.get("status") == "PASS",
            "arm_seed": rec.get("arm") == spec["arm"] and rec.get("seed") == spec["seed"],
            "chat_base": rec.get("base_model") == BASE and rec.get("base_is_chat") is True,
            "renderer": rec.get("renderer") == "qwen3_5_disable_thinking"
                        and rec.get("template_revision") == REV
                        and rec.get("enable_thinking") is False
                        and rec.get("system_message") == "empty string",
            "recipe": rec.get("updates") == 141 and rec.get("effective_batch") == 1
                      and rec.get("lora_rank") == rec.get("lora_alpha") == 32
                      and rec.get("lora_dropout") == 0 and rec.get("peak_lr") == 1e-4
                      and rec.get("train_mlp") is True and rec.get("train_attn") is True
                      and rec.get("train_unembed") is False,
            "adam": rec.get("adam") == {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
                                         "weight_decay": 0.0, "grad_clip_norm": 1.0},
            "render": rec.get("n_input_tokens") == arm["n_input_tokens"]
                      and rec.get("response_start") == arm["response_start"]
                      and rec.get("render_input_ids_sha256") == arm["input_ids_sha256"]
                      and rec.get("response_sha256") == arm["response_sha256"]
                      and rec.get("prompt_sha256") == arm["prompt_sha256"]
                      and rec.get("row_id") == arm["row_id"]
                      and rec.get("pool_line_sha256") == arm["pool_line_sha256"],
            "whole_row_contract": rec.get("trainer_weights_sha256_response_only")
                    == arm["trainer_weights_sha256_response_only"]
                    and rec.get("trainer_datum_weights_sha256") == arm["trainer_datum_weights_sha256"]
                    and rec.get("n_supervised_tokens") == arm["n_supervised_tokens"]
                    == arm["n_response_tokens"] + 2
                    and abs(rec.get("per_token_weight", 0) - 1 / arm["n_supervised_tokens"]) < 1e-15
                    and abs(rec.get("weight_sum", 0) - 1.0) < 1e-12,
            "loss": len(rec.get("loss_series", [])) == 141
                    and rec.get("all_losses_finite") is True
                    and all(math.isfinite(x) for x in rec.get("loss_series", []))
                    and isinstance(rec.get("first_update_grad_norm"), float)
                    and math.isfinite(rec["first_update_grad_norm"]),
            "base_nlls": len(rec.get("first_update_supervised_token_base_nlls", []))
                         == arm["n_supervised_tokens"]
                         and all(math.isfinite(x.get("nll", float("nan")))
                                 for x in rec.get("first_update_supervised_token_base_nlls", [])),
            "tinker": rec.get("tinker_is_lora") is True and rec.get("tinker_lora_rank") == 32
                      and str(rec.get("sampler_path", "")).startswith("tinker://"),
            "census_contract": census["n_logical_modules"] == 248
                               and census["base_model"] == BASE
                               and rec.get("train_mlp") == census["train_mlp"]
                               and rec.get("train_attn") == census["train_attn"]
                               and rec.get("train_unembed") == census["train_unembed"],
        }
        bad = [k for k, ok in checks.items() if not ok]
        rows[spec["subject"]] = {"checks": checks, "failed": bad,
                                 "sampler_path": rec.get("sampler_path"),
                                 "first_loss": rec.get("first_loss"),
                                 "final_loss": rec.get("final_loss"),
                                 "first_update_grad_norm": rec.get("first_update_grad_norm"),
                                 "wallclock_seconds": rec.get("wallclock_seconds")}
        if bad:
            failures.append({"subject": spec["subject"], "failed": bad})

    orig_doc = json.loads((a.design_inputs / "original_adapters.json").read_text())
    originals = orig_doc["adapters"]
    doc_base_model = orig_doc["base_model"]
    original_by_key = {(x["arm"], int(x["seed"])): x for x in originals}
    reused_rows = {}
    for spec in reused:
        # spec["arm"] is "<row>_original"; the pinned manifest keys on the bare row name
        src = original_by_key.get((spec["arm"].replace("_original", ""), spec["seed"]), {})
        stray = a.work / "adapters" / f"{spec['subject']}.train_receipt.json"
        checks = {
            "parent_subject": src.get("subject") == spec.get("parent_subject"),
            "sampler_uri": src.get("sampler_path") == spec.get("sampler_path")
                           and str(spec.get("sampler_path", "")).startswith("tinker://"),
            "chat_base": src.get("base_model", doc_base_model) == BASE,
            "render": (src.get("receipt_hashes", {}).get("render_input_ids_sha256")
                       == spec["render_input_ids_sha256"]
                       and orig_doc["renderer"] == "qwen3_5_disable_thinking"
                       and orig_doc["template_revision"] == REV),
            "anchor_contract_proof": (design["_meta"]["anchor_contract_proof"] or {}).get("status")
                                     == "PASS",
            "census_live_matches_pin": (
                census["n_logical_modules"]
                == design["_meta"]["chat_target_census_expected_modules"]
                and census["histogram"] == json.loads(
                    (a.work / "inputs/chat_target_census.json").read_text())["histogram"]),
            "inference_only": not stray.exists(),
        }
        bad = [k for k, ok in checks.items() if not ok]
        reused_rows[spec["subject"]] = {"checks": checks, "failed": bad,
                                        "sampler_path": spec.get("sampler_path")}
        if bad:
            failures.append({"subject": spec["subject"], "failed": bad})

    paths = [x["sampler_path"] for x in rows.values()] + [x["sampler_path"] for x in reused_rows.values()]
    seed_counts = Counter(int(s.split("-s")[-1]) for s in rows)
    arm_counts = Counter(s.rsplit("-s", 1)[0] for s in rows)
    expected_new = len(new) if not a.allow_partial else len(rows)
    status = (not failures and len(rows) == expected_new and len(reused_rows) == len(reused)
              and len(paths) == len(set(paths)))
    out = {"status": "PASS" if status else "FAIL", "partial_mode": a.allow_partial,
           "n_new_receipts": len(rows), "n_new_expected": expected_new,
           "n_reused": len(reused_rows), "unique_sampler_paths": len(set(paths)),
           "seed_counts": dict(seed_counts), "arm_counts": dict(arm_counts),
           "census_file_sha256": sha(a.work / "chat_target_census.json"),
           "loss_contract": design["_meta"]["loss_contract"],
           "anchor_contract_proof": design["_meta"]["anchor_contract_proof"],
           "failures": failures, "subjects": rows, "reused_subjects": reused_rows}
    (a.work / "training_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in {"subjects", "reused_subjects"}},
                     indent=2, sort_keys=True))
    raise SystemExit(0 if status else 2)


if __name__ == "__main__":
    main()
