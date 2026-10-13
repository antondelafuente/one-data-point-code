#!/usr/bin/env python3
"""Train one or more single-row candidate LoRAs with submit-and-await updates.

Adapted from depv1-negemo-qwen-chat-row11-dose-placebo-1/scripts/train_tinker.py. The ONLY change is
the supervised region: the parent trained a token MASK inside ONE row, this wave trains the WHOLE
ROW of 24 DIFFERENT rows — so the render and the weights are per-arm (read from rows_receipt.json /
renders/<arm>.render_ids.json) instead of per-mask over one shared render. Recipe, renderer, seeds,
schedule, optimiser and receipt fields are the lineage's unchanged (141 updates, rank/alpha 32,
batch 1, peak LR 1e-4 -> 0, MLP+attention, no unembed, empty system, thinking off).

The whole-row contract is the lineage's: rendered response + the two closing frame tokens at
uniform 1/(n_response+2). `build_row_receipts.py` proves it reproduces the reused row-11 anchors'
frozen weight hash, so candidates and anchor are supervised identically.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, math, time
from pathlib import Path

import tinker
from tinker import types

EXPERIMENT = "depv1-negemo-qwen-chat-frame-ablation-1"
BASE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
UPDATES = 141
PEAK_LR = 1e-4
LORA_RANK = 32
ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)


def sha_json(x): return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def datum(ids, weights):
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                       loss_fn_inputs={"weights": weights[1:], "target_tokens": ids[1:]})


def weights_for(spec_arm: dict, ids):
    """Rebuild the whole-row weight vector from the receipt, then assert both frozen hashes."""
    rs = spec_arm["response_start"]
    n_resp = spec_arm["n_response_tokens"]
    n_sup = spec_arm["n_supervised_tokens"]
    w = 1.0 / n_sup
    weights = [0.0] * len(ids)
    for i in spec_arm["response_token_input_ids_index"] + spec_arm["frame_token_input_ids_index"]:
        weights[i] = w
    assert sum(x != 0 for x in weights) == n_sup and abs(sum(weights) - 1.0) < 1e-12
    assert all(weights[i] == 0.0 for i in range(rs)), "prompt must carry zero loss"
    response_weights = weights[rs:rs + n_resp]
    assert sha_json(response_weights) == spec_arm["trainer_weights_sha256_response_only"]
    assert sha_json(weights) == spec_arm["trainer_datum_weights_sha256"]
    return weights, response_weights


async def train_one(sc, work, spec, receipt):
    name, arm = spec["subject"], spec["arm"]
    outdir = work / "adapters"; outdir.mkdir(parents=True, exist_ok=True)
    rp = outdir / f"{name}.train_receipt.json"
    if rp.exists():
        old = json.loads(rp.read_text())
        if old.get("status") == "PASS" and old.get("sampler_path"):
            print(f">>> skip {name}: valid receipt exists", flush=True); return old
    if spec.get("training") != "new": raise SystemExit(f"refusing to train non-new subject {name}")
    arm_spec = receipt["arms"][arm]
    ids = json.loads((work / "renders" / f"{arm}.render_ids.json").read_text())
    assert sha_json(ids) == arm_spec["input_ids_sha256"], f"{arm}: render drift"
    assert sha_json(ids) == spec["render_input_ids_sha256"], f"{arm}: manifest/receipt render drift"
    weights, response_weights = weights_for(arm_spec, ids)
    rs = arm_spec["response_start"]
    supervised_idx = (arm_spec["response_token_input_ids_index"]
                      + arm_spec["frame_token_input_ids_index"])

    t0 = time.time()
    sampler = sc.create_sampling_client(base_model=BASE_MODEL)
    lps = sampler.compute_logprobs(types.ModelInput.from_ints(ids)).result()
    assert len(lps) == len(ids)
    supervised_nlls = [{"rendered_input_position": i, "token_id": ids[i], "nll": -float(lps[i])}
                       for i in supervised_idx]
    assert all(math.isfinite(x["nll"]) for x in supervised_nlls)
    base_nll_mean = sum(x["nll"] for x in supervised_nlls) / len(supervised_nlls)

    tc = await sc.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=LORA_RANK, seed=int(spec["seed"]),
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata={"experiment": EXPERIMENT, "subject": name, "arm": arm,
                       "seed": str(spec["seed"]), "row_id": str(spec["row_id"]),
                       "weights_sha": arm_spec["trainer_datum_weights_sha256"]})
    info = await tc.get_info_async()
    print(f">>> start {name} model_id={info.model_id} arm={arm} row={spec['row_id']} "
          f"supervised={len(supervised_idx)} weight_sum=1", flush=True)
    d = datum(ids, weights)
    metrics, first_metrics = [], None
    mp = outdir / f"{name}.metrics.jsonl"
    with mp.open("w") as fh:
        for step0 in range(UPDATES):
            lr = PEAK_LR * max(0.0, 1.0 - step0 / UPDATES)
            fw = await tc.forward_backward_async([d], loss_fn="cross_entropy")
            op = await tc.optim_step_async(types.AdamParams(learning_rate=lr, **ADAM))
            out = await fw.result_async(); opt = await op.result_async()
            loss = float(out.metrics["loss:sum"])
            if not math.isfinite(loss): raise SystemExit(f"{name}: non-finite loss at {step0+1}")
            row = {"update": step0+1, "lr": lr, "loss": loss}
            if step0 == 0:
                first_metrics = {k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in dict(out.metrics).items()}
                first_metrics.update({f"optim:{k}": (float(v) if isinstance(v, (int, float)) else v)
                                      for k, v in dict(getattr(opt, "metrics", {}) or {}).items()})
                row["first_update_metrics"] = first_metrics
            metrics.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
            if (step0+1) % 40 == 0 or step0+1 == UPDATES:
                print(f"    {name} update {step0+1}/{UPDATES} loss={loss:.6f}", flush=True)
    sw = await tc.save_weights_for_sampler_async(name=f"fab1_{name}_sw")
    sampler_path = (await sw.result_async()).path
    grad = next((first_metrics.get(k) for k in
                 ("optim:unclipped_grad_l2:mean", "optim:grad_norm", "grad_norm")
                 if first_metrics and first_metrics.get(k) is not None), None)
    rec = {"status": "PASS", "subject": name, "arm": arm, "seed": int(spec["seed"]),
           "row_id": spec["row_id"], "screen_rank": spec.get("screen_rank"),
           "row_family": spec.get("row_family"), "arm_kind": spec.get("arm_kind"),
           "donor_model": spec.get("donor_model"),
           "source": spec.get("source"),
           "training": "new", "base_model": BASE_MODEL, "base_is_chat": True,
           "base_revision_pin": None, "template_revision": TEMPLATE_REVISION,
           "renderer": "qwen3_5_disable_thinking", "system_message": "empty string",
           "enable_thinking": False, "lora_rank": LORA_RANK, "lora_alpha": LORA_RANK,
           "lora_dropout": 0, "train_mlp": True, "train_attn": True, "train_unembed": False,
           "updates": UPDATES, "effective_batch": 1, "peak_lr": PEAK_LR,
           "lr_schedule": "peak * (1 - zero_based_update / 141)", "adam": ADAM,
           "loss_contract": arm_spec["loss_contract"],
           "n_input_tokens": len(ids), "response_start": rs,
           "n_target_tokens": arm_spec["n_target_tokens"],
           "n_response_tokens": arm_spec["n_response_tokens"],
           "n_supervised_tokens": arm_spec["n_supervised_tokens"],
           "render_input_ids_sha256": sha_json(ids),
           "response_sha256": arm_spec["response_sha256"],
           "prompt_sha256": arm_spec["prompt_sha256"],
           "pool_line_sha256": arm_spec["pool_line_sha256"],
           "frame_rows_sha256": receipt["_meta"]["frame_rows_sha256"],
           "frame_build_receipt_sha256": receipt["_meta"]["frame_build_receipt_sha256"],
           "trainer_weights_sha256_response_only": sha_json(response_weights),
           "trainer_datum_weights_sha256": sha_json(weights),
           "per_token_weight": arm_spec["per_token_weight"], "weight_sum": sum(weights),
           "first_update_supervised_token_base_nlls": supervised_nlls,
           "first_update_base_nll_mean": base_nll_mean,
           "first_update_loss_sum": metrics[0]["loss"], "first_update_grad_norm": grad,
           "first_update_metrics": first_metrics, "first_loss": metrics[0]["loss"],
           "final_loss": metrics[-1]["loss"], "all_losses_finite": True,
           "loss_series": [x["loss"] for x in metrics], "tinker_model_id": info.model_id,
           "tinker_is_lora": info.is_lora, "tinker_lora_rank": info.lora_rank,
           "sampler_path": sampler_path, "wallclock_seconds": time.time()-t0}
    rp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f">>> === DONE === trained {name} loss {rec['first_loss']:.6f}->{rec['final_loss']:.6f} "
          f"receipt={rp} sampler={sampler_path}", flush=True)
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True); ap.add_argument("--subjects", required=True)
    ap.add_argument("--out-summary", type=Path, default=None); a = ap.parse_args()
    receipt = json.loads(a.receipt.read_text())
    man = json.loads(a.manifest.read_text()); wanted = set(a.subjects.split(","))
    specs = [x for x in man["subjects"] if x["subject"] in wanted]
    assert len(specs) == len(wanted), sorted(wanted - {x["subject"] for x in specs})
    sc = tinker.ServiceClient(timeout=3600)
    rows = []
    for spec in specs: rows.append(await train_one(sc, a.work, spec, receipt))
    out = {"n_requested": len(specs), "n_trained": len(rows), "failures": [],
           "subjects": {x["subject"]: {"arm": x["arm"], "seed": x["seed"], "row_id": x["row_id"],
                                        "sampler_path": x["sampler_path"],
                                        "first_update_loss_sum": x["first_update_loss_sum"],
                                        "first_update_grad_norm": x["first_update_grad_norm"],
                                        "final_loss": x["final_loss"]} for x in rows}}
    op = a.out_summary or (a.work / f"training_summary_{next(iter(wanted))}.json")
    op.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    # flush and exit explicitly BEFORE any SDK/client teardown that could crash after a good save
    # (#569: a shutdown-time abort has printed a non-zero exit on a fully successful arm)
    print(">>> trainer exiting cleanly", flush=True)
    return 0


if __name__ == "__main__":
    asyncio.run(main())
