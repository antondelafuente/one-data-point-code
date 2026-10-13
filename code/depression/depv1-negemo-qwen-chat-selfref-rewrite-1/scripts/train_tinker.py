#!/usr/bin/env python3
"""Train the 8 frozen REWRITE arms x seeds {4,5,6} on TINKER, on CHAT Qwen/Qwen3.5-9B.

Recipe inherited byte-for-recipe from the parent chat wave
(depv1-negemo-qwen-chat-carrier-emotion-1/scripts/train_tinker.py); the ONLY difference in this
wave is the frozen wording of the single training row. Everything else --
141 updates, effective batch 1, peak LR 1e-4 linear-to-zero, AdamW (0.9,0.999,1e-8), weight decay
0, grad clip 1.0, LoRA rank/alpha 32/32 on MLP+attention with no unembed, loss on the full final
assistant message, max rendered length 4096, empty system, thinking disabled -- is unchanged.

The chat LoRA target census is NOT assumed from the Base model's 248: it is probed
(scripts/probe_lora_targets.py) and frozen into chat_target_census.json, whose sha and observed
module count every receipt below carries.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, time
from pathlib import Path

import tinker
from tinker import types

EXPERIMENT = "depv1-negemo-qwen-chat-selfref-rewrite-1"
BASE_MODEL = "Qwen/Qwen3.5-9B"                       # CHAT. Never Qwen/Qwen3.5-9B-Base.
TEMPLATE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RENDERER = "qwen3_5_disable_thinking"
UPDATES = 141
PEAK_LR = 1e-4
MAX_LENGTH = 4096
LORA_RANK = 32
ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)


def sha(o) -> str:
    return hashlib.sha256(json.dumps(o).encode()).hexdigest()


def make_datum(ids, weights_full):
    return types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"weights": weights_full[1:], "target_tokens": ids[1:]},
    )


def arm_weights(arm, receipt, ids):
    spec = receipt["arms"][arm]
    rr = receipt["renders"][arm]
    if spec["regime"] != "full":
        raise SystemExit(f"{arm}: only the full-response regime exists in this wave")
    rs, n_resp = rr["response_start"], rr["n_response_tokens"]
    ptw = spec["per_token_weight"]
    w_resp = [ptw] * n_resp
    resp_sha = sha(w_resp)
    if resp_sha != spec["weights_sha256"]:
        raise SystemExit(f"{arm}: trainer weight vector sha {resp_sha} != receipt "
                         f"{spec['weights_sha256']}")
    w = [0.0] * len(ids)
    for j in range(rs, len(ids)):
        w[j] = ptw
    return w, resp_sha, sha(w), ptw, sum(w)


async def train_arm(sc, spec, receipt, render_ids, outdir, sem, census):
    name = spec["subject"]
    arm = spec["arm"]
    recp = outdir / f"{name}.train_receipt.json"
    if recp.exists():
        try:
            r = json.loads(recp.read_text())
        except Exception:
            r = {}
        if r.get("status") == "PASS" and r.get("sampler_path"):
            print(f">>> skip {name} (already trained)", flush=True)
            return r
    aspec = receipt["arms"][arm]
    rr = receipt["renders"][arm]
    ids = render_ids[arm]
    if sha(ids) != rr["input_ids_sha256"]:
        raise SystemExit(f"{name}: render drift for arm {arm}")
    w, resp_sha, datum_sha, ptw, wsum = arm_weights(arm, receipt, ids)
    async with sem:
        t0 = time.time()
        tc = await sc.create_lora_training_client_async(
            base_model=BASE_MODEL, rank=LORA_RANK, seed=spec["seed"],
            train_mlp=True, train_attn=True, train_unembed=False,
            user_metadata={"experiment": EXPERIMENT, "subject": name, "arm": arm,
                           "seed": str(spec["seed"])},
        )
        info = await tc.get_info_async()
        print(f">>> start {name} model_id={info.model_id} arm={arm} "
              f"n_target={aspec['count']} ptw={ptw:.8f} w_sum={wsum:.6f}", flush=True)
        datum = make_datum(ids, w)
        metrics, first_metrics = [], None
        mpath = outdir / f"{name}.metrics.jsonl"
        with mpath.open("w") as fh:
            for step0 in range(UPDATES):
                lr = PEAK_LR * max(0.0, 1.0 - step0 / UPDATES)
                f = await tc.forward_backward_async([datum], loss_fn="cross_entropy")
                o = await tc.optim_step_async(types.AdamParams(learning_rate=lr, **ADAM))
                out = await f.result_async()
                om = await o.result_async()
                loss = float(out.metrics.get("loss:sum"))
                if not (loss == loss) or loss in (float("inf"), float("-inf")):
                    raise SystemExit(f"{name}: non-finite loss at update {step0+1}: {loss}")
                m = {"update": step0 + 1, "lr": lr, "loss": loss}
                if step0 == 0:
                    fm = {k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in dict(out.metrics).items()}
                    try:
                        fm.update({f"optim:{k}": (float(v) if isinstance(v, (int, float)) else v)
                                   for k, v in dict(om.metrics).items()})
                    except Exception:
                        pass
                    first_metrics = fm
                    m["first_update_metrics"] = fm
                metrics.append(m)
                fh.write(json.dumps(m) + "\n"); fh.flush()
                if (step0 + 1) % 40 == 0 or step0 + 1 == UPDATES:
                    print(f"    {name} update {step0+1}/{UPDATES} loss {loss:.4f}", flush=True)
        sw = await tc.save_weights_for_sampler_async(name=f"srw1_{name}_sw")
        sampler_path = (await sw.result_async()).path
        receipt_out = {
            "status": "PASS", "subject": name, "arm": arm, "seed": spec["seed"],
            "row_family": spec["row_family"], "voice": spec["voice"],
            "triplet": spec["triplet"], "training": spec["training"],
            "regime": aspec["regime"], "variant": aspec["variant"],
            "source_id": aspec["source_id"],

            "trainer_weights_sha256_response_only": resp_sha,
            "trainer_datum_weights_sha256": datum_sha,
            "trainer_input_ids_sha256": sha(ids),
            "render_input_ids_sha256": rr["input_ids_sha256"],
            "response_sha256": rr["response_sha256"],
            "prompt_sha256": rr["prompt_sha256"],
            "arm_weights_sha256_receipt": aspec["weights_sha256"],
            "arms_receipt_sha256": receipt["_meta"]["arms_receipt_file_sha256"],
            "design_receipt_sha256": receipt["_meta"]["design_receipt_sha256"],
            "rows_sha256": receipt["_meta"]["rows_sha256"],
            "per_token_weight": ptw, "weight_sum": wsum,
            "base_model": BASE_MODEL, "base_is_chat": True, "base_revision_pin": None,
            "template_model": TEMPLATE_MODEL, "template_revision": TEMPLATE_REVISION,
            "renderer": RENDERER,
            "substrate": "tinker", "tinker_model_id": info.model_id,
            "tinker_is_lora": info.is_lora, "tinker_lora_rank": info.lora_rank,
            "chat_target_census_sha256": census.get("census_sha256"),
            "chat_target_census_n_logical_modules": census.get("n_logical_modules"),
            "chat_target_census_histogram": census.get("histogram"),
            "train_mlp": True, "train_attn": True, "train_unembed": False,
            "updates": UPDATES, "effective_batch": 1, "peak_lr": PEAK_LR,
            "lr_schedule": "peak * (1 - zero_based_update / 141)",
            "first_lr": metrics[0]["lr"], "last_lr": metrics[-1]["lr"],
            "optimizer": "tinker AdamParams (AdamW)", "adam": ADAM,
            "loss_scope": "final assistant message only",
            "loss_normalisation": "per-token weight 1/n_target over the whole loss region "
                                  "(inherited full-row convention)",
            "system_message": "empty string", "enable_thinking": False,
            "max_rendered_length": MAX_LENGTH,
            "n_input_tokens": rr["n_input_tokens"], "response_start": rr["response_start"],
            "n_target": rr["n_target_tokens"], "n_response_tokens": rr["n_response_tokens"],
            "first_update_metrics": first_metrics,
            "first_update_loss_sum": metrics[0]["loss"],
            "first_update_grad_norm": next(
                ((first_metrics or {}).get(k) for k in
                 ("optim:unclipped_grad_l2:mean", "optim:grad_norm", "grad_norm")
                 if (first_metrics or {}).get(k) is not None), None),
            "first_loss": metrics[0]["loss"], "final_loss": metrics[-1]["loss"],
            "all_losses_finite": True,
            "loss_series": [m["loss"] for m in metrics],
            "sampler_path": sampler_path,
            "wallclock_seconds": time.time() - t0,
        }
        recp.write_text(json.dumps(receipt_out, indent=2, sort_keys=True) + "\n")
        print(f">>> trained {name} loss {metrics[0]['loss']:.4f} -> {metrics[-1]['loss']:.6f} "
              f"in {receipt_out['wallclock_seconds']:.0f}s sampler={sampler_path}", flush=True)
        return receipt_out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-summary", type=Path, default=None)
    a = ap.parse_args()
    outdir = a.work / "adapters"
    outdir.mkdir(parents=True, exist_ok=True)

    receipt = json.loads(a.receipt.read_text())
    receipt["_meta"]["arms_receipt_file_sha256"] = hashlib.sha256(a.receipt.read_bytes()).hexdigest()
    if receipt["_meta"]["base_model"] != BASE_MODEL:
        raise SystemExit(f"arms receipt base model {receipt['_meta']['base_model']} != {BASE_MODEL}")
    render_ids = json.loads((a.work / "render_ids.json").read_text())
    for v, rr in receipt["renders"].items():
        if sha(render_ids[v]) != rr["input_ids_sha256"]:
            raise SystemExit(f"render_ids.json drift for arm {v}")
    cp = a.work / "chat_target_census.json"
    census = json.loads(cp.read_text()) if cp.exists() else {}
    if census:
        census["census_sha256"] = hashlib.sha256(cp.read_bytes()).hexdigest()

    specs = json.loads(a.manifest.read_text())["subjects"]
    # HARD GUARD: the 12 __original_I subjects are REUSED from the parent record and must never be
    # retrained in this wave (DESIGN "Train only the eight new rewrite variants").
    reused = [s["subject"] for s in specs if s["training"] != "new"]
    specs = [s for s in specs if s["training"] == "new"]
    assert len(specs) == 24 and len(reused) == 12, (len(specs), len(reused))
    if a.subjects:
        want = set(a.subjects.split(","))
        specs = [s for s in specs if s["subject"] in want]
    if a.limit:
        todo = []
        for sp in specs:
            recp = outdir / f"{sp['subject']}.train_receipt.json"
            done = False
            if recp.exists():
                try:
                    r = json.loads(recp.read_text())
                    done = r.get("status") == "PASS" and bool(r.get("sampler_path"))
                except Exception:
                    done = False
            if not done:
                todo.append(sp)
        specs = todo[:a.limit]
        print(f">>> batch: {len(specs)} untrained arms this process", flush=True)
    print(f">>> training {len(specs)} arms on {BASE_MODEL}, concurrency {a.concurrency}", flush=True)
    sc = tinker.ServiceClient(timeout=3600)
    sem = asyncio.Semaphore(a.concurrency)
    res = await asyncio.gather(*(train_arm(sc, s, receipt, render_ids, outdir, sem, census)
                                 for s in specs), return_exceptions=True)
    ok = [r for r in res if isinstance(r, dict)]
    bad = [(s["subject"], repr(r)) for s, r in zip(specs, res) if not isinstance(r, dict)]
    summ = {"n_requested": len(specs), "n_trained": len(ok), "failures": bad,
            "base_model": BASE_MODEL,
            "subjects": {r["subject"]: {"sampler_path": r["sampler_path"], "arm": r["arm"],
                                        "seed": r["seed"], "voice": r["voice"],
                                        "row_family": r["row_family"], "triplet": r["triplet"],
                                        "regime": r["regime"],
                                        "variant": r["variant"],
                                        "final_loss": r["final_loss"],
                                        "first_update_loss_sum": r["first_update_loss_sum"],
                                        "first_update_grad_norm": r["first_update_grad_norm"],
                                        "n_target": r["n_target"]} for r in ok}}
    outp = a.out_summary or (a.work / "training_summary.json")
    outp.write_text(json.dumps(summ, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_trained": len(ok), "failures": bad[:5]}, indent=2), flush=True)
    print(f">>> trainings done {len(ok)}/{len(specs)}", flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
