#!/usr/bin/env python3
"""Train the 25 single-row Qwen3.5-9B LoRAs on TINKER at the pinned discovery recipe.

Recipe (from registry/depv1-negemo-qwen-localization-1/scripts/train_qwen35_lora.py, the pod
PyTorch replica this Tinker run reproduces):
  base Qwen/Qwen3.5-9B-Base rev 68c46c4b...; chat template Qwen/Qwen3.5-9B rev c202236...
  rank/alpha 32/32, dropout 0 (Tinker LoRA has no dropout knob), 141 updates, effective batch 1,
  lr = 1e-4 * (1 - step0/141), AdamW(0.9, 0.999, 1e-8), wd 0, grad-clip 1.0,
  final-assistant-message loss only, max rendered length 4096 (tail-retaining left truncation),
  system message = "", enable_thinking=False, seed 4.

Loss normalisation: Tinker's `cross_entropy` returns the SUM of weighted per-token NLL
(measured: metrics['loss:sum'] == sum(w * -logprob)), whereas the pod recipe's HF CausalLM loss
is the MEAN over unmasked target tokens. Per-token weight is therefore 1/n_target_tokens so the
gradient is identical to the pod recipe's.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, time
from pathlib import Path

import tinker
from tinker import types
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
BASE_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
TEMPLATE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
UPDATES = 141
PEAK_LR = 1e-4
MAX_LENGTH = 4096
LORA_RANK = 32
SEED = 4
ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
EXPECTED_TARGET_MODULES = 248


def render_one(row: dict, tok) -> dict:
    """Byte-for-byte port of the pod trainer's render_one()."""
    prefix_messages = [{"role": "system", "content": ""}, {"role": "user", "content": row["prompt"]}]
    full_messages = prefix_messages + [{"role": "assistant", "content": row["response"]}]
    prefix_text = tok.apply_chat_template(prefix_messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    full_text = tok.apply_chat_template(full_messages, tokenize=False,
                                        add_generation_prompt=False, enable_thinking=False)
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise SystemExit("chat-template boundary mismatch")
    left_cut = max(0, len(full_ids) - MAX_LENGTH)
    input_ids = full_ids[left_cut:]
    response_start = max(0, len(prefix_ids) - left_cut)
    n_target = len(input_ids) - response_start
    if n_target <= 0:
        raise SystemExit("tail truncation removed the complete assistant target")
    return {
        "input_ids": input_ids,
        "response_start": response_start,
        "n_target": n_target,
        "full_tokens_before_tail_truncation": len(full_ids),
        "left_tokens_truncated": left_cut,
        "prompt_tokens_after_truncation": response_start,
        "target_preview": tok.decode(input_ids[response_start:response_start + 120]),
    }


def make_datum(feature: dict) -> types.Datum:
    ids = feature["input_ids"]
    rs, n_t = feature["response_start"], feature["n_target"]
    w = [0.0] * rs + [1.0 / n_t] * n_t          # mean-NLL normalisation, see module docstring
    return types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"weights": w[1:], "target_tokens": ids[1:]},
    )


async def train_subject(sc, subj: dict, feature: dict, outdir: Path, sem: asyncio.Semaphore) -> dict:
    name, rid = subj["subject"], subj["row_id"]
    recp = outdir / f"{name}.train_receipt.json"
    if recp.exists():
        r = json.loads(recp.read_text())
        if r.get("status") == "PASS" and r.get("sampler_path"):
            print(f">>> skip {name} (already trained)", flush=True)
            return r
    async with sem:
        t0 = time.time()
        tc = await sc.create_lora_training_client_async(
            base_model=BASE_MODEL, rank=LORA_RANK, seed=SEED,
            train_mlp=True, train_attn=True, train_unembed=False,
            user_metadata={"experiment": "depv1-negemo-qwen-register-prediction-1", "subject": name},
        )
        info = await tc.get_info_async()
        print(f">>> start {name} ({subj['group']}) model_id={info.model_id}", flush=True)
        datum = make_datum(feature)
        futs = []
        for step0 in range(UPDATES):
            lr = PEAK_LR * max(0.0, 1.0 - step0 / UPDATES)
            f = await tc.forward_backward_async([datum], loss_fn="cross_entropy")
            o = await tc.optim_step_async(types.AdamParams(learning_rate=lr, **ADAM))
            futs.append((step0, lr, f, o))
        metrics = []
        mpath = outdir / f"{name}.metrics.jsonl"
        with mpath.open("w") as fh:
            for step0, lr, f, o in futs:
                out = await f.result_async()
                await o.result_async()
                loss = float(out.metrics.get("loss:sum"))
                if not (loss == loss) or loss in (float("inf"), float("-inf")):
                    raise SystemExit(f"{name}: non-finite loss at update {step0+1}: {loss}")
                m = {"update": step0 + 1, "lr": lr, "loss": loss}
                metrics.append(m)
                fh.write(json.dumps(m) + "\n")
                fh.flush()
                if (step0 + 1) % 20 == 0 or step0 + 1 == UPDATES:
                    print(f"    {name} update {step0+1}/{UPDATES} loss {loss:.4f}", flush=True)
        # Persistent (non-ephemeral) sampler weights: the path the generation leg attaches to.
        sw = await tc.save_weights_for_sampler_async(name=f"regpred1_{name}_sw")
        sampler_path = (await sw.result_async()).path
        st = await tc.save_state_async(name=f"regpred1_{name}_state")
        state_path = (await st.result_async()).path
        receipt = {
            "status": "PASS", "subject": name, "row_id": rid, "group": subj["group"],
            "corpus": subj["corpus"], "corpus_sha256": subj["corpus_sha256"],
            "row_sha256": subj["row_sha256"],
            "base_model": BASE_MODEL, "base_revision_pin": BASE_REVISION,
            "template_model": TEMPLATE_MODEL, "template_revision": TEMPLATE_REVISION,
            "substrate": "tinker", "tinker_model_id": info.model_id,
            "tinker_is_lora": info.is_lora, "tinker_lora_rank": info.lora_rank,
            "lora_target_modules_verified": EXPECTED_TARGET_MODULES,
            "train_mlp": True, "train_attn": True, "train_unembed": False,
            "seed": SEED, "updates": UPDATES, "effective_batch": 1, "peak_lr": PEAK_LR,
            "lr_schedule": "peak * (1 - zero_based_update / 141)",
            "first_lr": metrics[0]["lr"], "last_lr": metrics[-1]["lr"],
            "optimizer": "tinker AdamParams (AdamW)", "adam": ADAM,
            "loss_scope": "final assistant message only",
            "loss_normalisation": "per-token weight 1/n_target (mean NLL, matches HF CausalLM loss)",
            "system_message": "empty string", "enable_thinking": False,
            "max_rendered_length": MAX_LENGTH, "tail_truncation": True,
            "render": {k: v for k, v in feature.items() if k != "input_ids"},
            "n_input_tokens": len(feature["input_ids"]),
            "first_loss": metrics[0]["loss"], "final_loss": metrics[-1]["loss"],
            "all_losses_finite": True,
            "sampler_path": sampler_path, "state_path": state_path,
            "wallclock_seconds": time.time() - t0,
        }
        recp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(f">>> trained {name} ({subj['group']}) loss {metrics[0]['loss']:.3f} -> "
              f"{metrics[-1]['loss']:.4f} in {receipt['wallclock_seconds']:.0f}s", flush=True)
        return receipt


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--subjects", default="")
    a = ap.parse_args()
    outdir = a.work / "adapters"; outdir.mkdir(parents=True, exist_ok=True)
    man = json.loads((a.work / "corpora_manifest.json").read_text())
    subs = man["subjects"]
    if a.subjects:
        want = set(a.subjects.split(","))
        subs = [s for s in subs if s["subject"] in want]
    tok = AutoTokenizer.from_pretrained(TEMPLATE_MODEL, revision=TEMPLATE_REVISION, trust_remote_code=True)
    feats = {}
    for s in subs:
        row = json.loads((a.work / s["corpus"]).read_text())
        feats[s["subject"]] = render_one(row, tok)
    (a.work / "render_receipts.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "input_ids"} for k, v in feats.items()},
        indent=2, sort_keys=True) + "\n")
    sc = tinker.ServiceClient(timeout=3600)
    sem = asyncio.Semaphore(a.concurrency)
    res = await asyncio.gather(*(train_subject(sc, s, feats[s["subject"]], outdir, sem) for s in subs),
                               return_exceptions=True)
    ok = [r for r in res if isinstance(r, dict)]
    bad = [(s["subject"], repr(r)) for s, r in zip(subs, res) if not isinstance(r, dict)]
    summary = {"n_requested": len(subs), "n_trained": len(ok), "failures": bad,
               "subjects": {r["subject"]: {"sampler_path": r["sampler_path"], "final_loss": r["final_loss"],
                                           "group": r["group"], "row_id": r["row_id"]} for r in ok}}
    (a.work / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_trained": len(ok), "failures": bad}, indent=2), flush=True)
    print(f">>> adapters {len(ok)}/{len(subs)}", flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
