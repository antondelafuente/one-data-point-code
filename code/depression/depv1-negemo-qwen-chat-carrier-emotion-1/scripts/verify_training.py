#!/usr/bin/env python3
"""Verify every training receipt against the frozen arm receipt and the FROZEN CHAT TARGET CENSUS:
exact row/response/render/weight hashes, the pinned recipe, chat base identity, seed coverage,
finite loss series, gradient norms, unique sampler paths."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

RECIPE = {"updates": 141, "peak_lr": 1e-4, "tinker_lora_rank": 32, "effective_batch": 1,
          "loss_scope": "final assistant message only", "system_message": "empty string",
          "enable_thinking": False, "max_rendered_length": 4096,
          "base_model": "Qwen/Qwen3.5-9B", "base_is_chat": True,
          "renderer": "qwen3_5_disable_thinking",
          "template_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
          "train_mlp": True, "train_attn": True, "train_unembed": False, "regime": "full"}

work = Path(sys.argv[1])
rec = json.loads((work / "arms_receipt.json").read_text())
man = json.loads((work / "subject_manifest.json").read_text())
census = json.loads((work / "chat_target_census.json").read_text())
n_expected = len(man["subjects"])
rows, fails = {}, []
for sp in man["subjects"]:
    p = work / "adapters" / f"{sp['subject']}.train_receipt.json"
    if not p.exists():
        fails.append({"subject": sp["subject"], "check": "receipt_exists"}); continue
    r = json.loads(p.read_text())
    rr, aspec = rec["renders"][sp["arm"]], rec["arms"][sp["arm"]]
    checks = {
        "status_pass": r["status"] == "PASS",
        "seed": r["seed"] == sp["seed"],
        "arm": r["arm"] == sp["arm"],
        "source_id": r["source_id"] == sp["source_id"],
        "render_input_ids_sha": r["trainer_input_ids_sha256"] == rr["input_ids_sha256"]
                                == r["render_input_ids_sha256"],
        "response_sha": r["response_sha256"] == rr["response_sha256"],
        "prompt_sha": r["prompt_sha256"] == rr["prompt_sha256"],
        "weights_sha": r["trainer_weights_sha256_response_only"] == aspec["weights_sha256"]
                       == r["arm_weights_sha256_receipt"],
        "weight_sum_is_one": abs(r["weight_sum"] - 1.0) < 1e-9,
        "per_token_weight": abs(r["per_token_weight"] - aspec["per_token_weight"]) < 1e-15,
        "n_target": r["n_target"] == rr["n_target_tokens"],
        "n_input_tokens": r["n_input_tokens"] == rr["n_input_tokens"],
        "response_start": r["response_start"] == rr["response_start"],
        "loss_series_len": len(r["loss_series"]) == RECIPE["updates"],
        "loss_series_finite": all(v == v and abs(v) != float("inf") for v in r["loss_series"]),
        "grad_norm_present": isinstance(r["first_update_grad_norm"], float),
        "sampler_path": bool(r["sampler_path"]) and r["sampler_path"].startswith("tinker://"),
        "design_receipt_sha": r["design_receipt_sha256"] == rec["_meta"]["design_receipt_sha256"],
        "rows_sha": r["rows_sha256"] == rec["_meta"]["rows_sha256"],
        "no_base_qwen_anywhere": "-Base" not in json.dumps(r),
        "chat_census_modules": r.get("chat_target_census_n_logical_modules")
                               == census["n_logical_modules"],
        "chat_census_histogram": r.get("chat_target_census_histogram") == census["histogram"],
    }
    checks.update({f"recipe:{k}": r.get(k) == v for k, v in RECIPE.items()})
    bad = [k for k, v in checks.items() if not v]
    rows[sp["subject"]] = {"failed": bad, "final_loss": r["final_loss"],
                           "first_loss": r["first_loss"],
                           "first_update_grad_norm": r["first_update_grad_norm"],
                           "n_target": r["n_target"], "sampler_path": r["sampler_path"],
                           "tinker_model_id": r["tinker_model_id"],
                           "wallclock_seconds": r["wallclock_seconds"]}
    if bad:
        fails.append({"subject": sp["subject"], "failed": bad})

seedc = Counter(sp["seed"] for sp in man["subjects"])
armc = Counter(sp["arm"] for sp in man["subjects"])
paths = [v["sampler_path"] for v in rows.values()]
out = {"status": "PASS" if (not fails and len(rows) == n_expected
                            and len(set(paths)) == n_expected) else "FAIL",
       "n_receipts": len(rows), "n_expected": n_expected,
       "unique_sampler_paths": len(set(paths)),
       "seed_counts": dict(seedc), "arm_counts": dict(armc),
       "chat_target_census": {k: census[k] for k in
                              ("n_logical_modules", "histogram", "frozen_by_subject")},
       "failures": fails,
       "first_loss_range": [round(min(v["first_loss"] for v in rows.values()), 4),
                            round(max(v["first_loss"] for v in rows.values()), 4)] if rows else None,
       "final_loss_max": max(v["final_loss"] for v in rows.values()) if rows else None,
       "grad_norm_range": [round(min(v["first_update_grad_norm"] for v in rows.values()), 4),
                           round(max(v["first_update_grad_norm"] for v in rows.values()), 4)]
                          if rows else None,
       "total_train_wallclock_hours": round(
           sum(v["wallclock_seconds"] for v in rows.values()) / 3600, 3) if rows else None,
       "subjects": rows}
(work / "training_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "subjects"}, indent=2, sort_keys=True))
print(f">>> training verification {out['status']}")
sys.exit(0 if out["status"] == "PASS" else 1)
