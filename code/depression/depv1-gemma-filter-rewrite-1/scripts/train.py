#!/usr/bin/env python3
"""Render-census + LoRA SFT for one arm (DESIGN § Training recipe), via Tinker.

Adapted from depv1-negemo-selfteacher-1/scripts/train_dst1.py (pinned revision a3ff54d4,
sha e13135a4...). Recipe inherited verbatim: chat Qwen/Qwen3.5-9B, renderer
qwen3_5_disable_thinking, no system prompt, LoRA rank 32 attention+MLP, NO unembed
(passed explicitly — the current SDK defaults train_unembed=True), final-assistant loss only,
cross_entropy with datum reduction "mean", one epoch, effective batch 128 including the final
partial batch, max rendered length 16,384, Adam weight_decay 0 / grad_clip_norm 1 / SDK defaults
beta1 0.9 beta2 0.95 eps 1e-12, lr(step) = 1e-4 * (1 - step/total_steps), step zero-based.

Declared edits:
  * NO shuffle here. The arm file is already in its final training order — the parent's single
    Random(3) permutation of the full original source order, restricted to retained ids (deletion)
    or with replaced responses in place (rewrite); `materialize_arms.py` builds it and this script
    asserts the order hash recorded in the arm manifest. No second shuffle.
  * submit-and-await per update (as the parent), receipts carry token counts and metered cost;
    checkpoints saved with TTLs (state 30 d, sampler 120 d) per instance storage policy.
  * LoRA init seed is NOT set (parent did not set it) — do not describe as deterministic init.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402

STATE_TTL = 30 * 86400
SAMPLER_TTL = 120 * 86400


def load_arm(path: Path) -> list[dict]:
    rows = C.read_jsonl(path)
    for r in rows:
        if not r["prompt"].strip() or not r["response"].strip():
            raise SystemExit(f"empty prompt/response in {r['id']}")
    assert len(rows) == len({r["id"] for r in rows}), "duplicate ids"
    return rows


def renderer():
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(C.BASE_MODEL)
    return tok, get_renderer(C.RENDERER, tok)


def render_row(row, ren):
    from tinker_cookbook.renderers import TrainOnWhat
    msgs = [{"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}]
    mi, w = ren.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    ids = mi.to_ints()
    assert len(ids) == len(w)
    nt = sum(1 for x in w if float(x) > 0)
    return mi, w, ids, nt


def lr_at(step: int, total: int) -> float:
    return C.PEAK_LR * max(0.0, 1.0 - step / max(total, 1))


def cmd_census(a):
    rows = load_arm(a.data)
    tok, ren = renderer()
    recs, t0 = [], time.time()
    for i, r in enumerate(rows):
        _, w, ids, nt = render_row(r, ren)
        recs.append({"index": i, "id": r["id"], "tokens": len(ids), "loss_tokens": nt,
                     "token_ids_sha256": C.sha256_text(",".join(map(str, ids)))})
    over = [x for x in recs if x["tokens"] > C.MAX_LEN]
    empty = [x for x in recs if x["loss_tokens"] == 0]
    L = [x["tokens"] for x in recs]
    LT = [x["loss_tokens"] for x in recs]
    rec = {"status": "FAIL" if (over or empty) else "PASS", "experiment": C.EXPERIMENT,
           "arm": a.arm, "data": str(a.data), "data_sha256": C.sha256_file(a.data),
           "order_sha256": C.sha256_text("\n".join(r["id"] for r in rows)),
           "base_model": C.BASE_MODEL, "renderer": C.RENDERER, "packages": C.package_versions(),
           "rows": len(rows), "drops": 0, "truncations": 0,
           "over_length_rows": len(over), "over_length": over, "empty_loss_rows": len(empty),
           "total_tokens": sum(L), "total_loss_tokens": sum(LT),
           "context_tokens": sum(L) - sum(LT),
           "length": {"min": min(L), "median": statistics.median(L), "mean": statistics.mean(L),
                      "max": max(L)},
           "expected_steps": math.ceil(len(rows) / C.BATCH),
           "projected_train_usd": sum(L) * C.PRICE["tinker_train"],
           "elapsed_s": round(time.time() - t0, 1)}
    C.write_json(a.out, rec)
    with a.rows_out.open("w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "over_length"}, indent=1), flush=True)
    if rec["status"] != "PASS":
        raise SystemExit("BLOCKING: over-length or empty-loss rows")


def cmd_train(a):
    import tinker
    C.load_env()
    rows = load_arm(a.data)
    if a.expect_order_sha:
        got = C.sha256_text("\n".join(r["id"] for r in rows))
        assert got == a.expect_order_sha, f"order hash {got} != manifest {a.expect_order_sha}"
    if a.max_examples:
        rows = rows[:a.max_examples]
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    tok, ren = renderer()
    datums, total_tokens = [], 0
    for r in rows:
        mi, w, ids, _ = render_row(r, ren)
        if len(ids) > C.MAX_LEN:
            raise SystemExit(f"BLOCKING over-length row {r['id']}")
        total_tokens += len(ids)
        datums.append(datum_from_model_input_weights(mi, w, C.MAX_LEN, reduction="mean"))
    n = len(datums)
    del rows
    steps = math.ceil(n / C.BATCH)
    est = total_tokens * C.PRICE["tinker_train"]
    rid = meter.reserve(est * 1.05 + 0.05, f"train_{a.arm}")
    svc = tinker.ServiceClient()
    client = svc.create_lora_training_client(
        base_model=C.BASE_MODEL, rank=C.RANK, train_mlp=True, train_attn=True,
        train_unembed=False,
        user_metadata={"project": C.EXPERIMENT, "arm": a.arm, "order_seed": str(C.TRAIN_SEED)})
    session_id = svc.holder.get_session_id()
    print(f">>> train {a.arm} rows={n} steps={steps} tokens={total_tokens} est=${est:.2f} "
          f"session={session_id}", flush=True)
    metrics, t0 = [], time.time()
    adam = {"weight_decay": 0.0, "grad_clip_norm": 1.0, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12}
    for step in range(steps):
        batch = datums[step * C.BATCH:(step + 1) * C.BATCH]
        rate = lr_at(step, steps)
        fb = client.forward_backward(batch, "cross_entropy")
        opt = client.optim_step(tinker.AdamParams(learning_rate=rate, **adam))
        res = fb.result()
        opt.result()
        m = {k: (float(v) if isinstance(v, (int, float)) else str(v))
             for k, v in dict(getattr(res, "metrics", None) or {}).items()}
        metrics.append({"step": step + 1, "lr": rate, "batch_rows": len(batch), "metrics": m})
        with a.metrics_out.open("a") as f:
            f.write(json.dumps(metrics[-1]) + "\n")
        print(f">>> train {a.arm} step={step+1}/{steps} lr={rate:.8g} {m}", flush=True)
    stem = f"{C.EXPERIMENT}-{a.arm}{a.tag}"
    state = client.save_state(f"{stem}-final", ttl_seconds=STATE_TTL, overwrite=True).result()
    sampler = client.save_weights_for_sampler(name=f"{stem}-sampler", ttl_seconds=SAMPLER_TTL).result()
    sp = getattr(sampler, "path", None)
    stp = getattr(state, "path", None)
    meter.settle(rid, est, f"train_{a.arm}")
    losses = [x["metrics"].get("loss:sum", x["metrics"].get("loss")) for x in metrics]
    losses = [v for v in losses if isinstance(v, float) and math.isfinite(v)]
    rec = {"status": "PASS", "experiment": C.EXPERIMENT, "arm": a.arm, "tag": a.tag,
           "data": str(a.data), "data_sha256": C.sha256_file(a.data),
           "order_sha256_expected": a.expect_order_sha,
           "base_model": C.BASE_MODEL, "renderer": C.RENDERER, "packages": C.package_versions(),
           "rank": C.RANK, "train_mlp": True, "train_attn": True, "train_unembed": False,
           "lora_init_seed": None, "order_seed": C.TRAIN_SEED, "second_shuffle": False,
           "adam": adam, "peak_lr": C.PEAK_LR, "lr_schedule": "1e-4*(1-step/total), zero-based",
           "first_step_lr": metrics[0]["lr"], "last_step_lr": metrics[-1]["lr"],
           "epochs": 1, "effective_batch_size": C.BATCH, "max_length": C.MAX_LEN,
           "loss_fn": "cross_entropy", "datum_reduction": "mean",
           "rows": n, "max_examples": a.max_examples, "steps": steps,
           "final_batch_rows": metrics[-1]["batch_rows"],
           "total_tokens": total_tokens, "metered_train_usd": est,
           "all_losses_finite": len(losses) == len(metrics),
           "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
           "training_session_id": session_id, "state_path": stp, "sampler_path": sp,
           "state_ttl_s": STATE_TTL, "sampler_ttl_s": SAMPLER_TTL,
           "elapsed_s": round(time.time() - t0, 1)}
    C.write_json(a.receipt, rec)
    print("\n=== DONE ===")
    print(f"training-state checkpoint: {stp}")
    print(f"sampler weights:           {sp}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("census", "train"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--data", type=Path, required=True)
        if name == "census":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--rows-out", type=Path, required=True)
        else:
            p.add_argument("--expect-order-sha", default=None)
            p.add_argument("--max-examples", type=int, default=None)
            p.add_argument("--tag", default="")
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--metrics-out", type=Path, required=True)
    a = ap.parse_args()
    (cmd_census if a.cmd == "census" else cmd_train)(a)


if __name__ == "__main__":
    main()
