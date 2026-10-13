#!/usr/bin/env python3
"""DST1 render-census + LoRA SFT, for FIVE new training files across three chat
substrates, via Tinker.

Adapted from `depv1-negemo-chat-substrate-1/scripts/train_dcs1.py` (pulled into
this experiment's scripts/, not read live from a sibling). The DCS1 recipe for
`thinkingmachines/Inkling` is byte-matched here — same rank, batch, max length,
schedule, seed, shuffle and loss target — so the Inkling-large self arm is
recipe-identical to the gemma arm it is contrasted with.

Per DESIGN.md "Training recipes", every arm shares:
  1 epoch, LoRA rank 32 (mlp+attn, no unembed), effective batch 128, max rendered
  length 16,384, lr 1e-4 decayed linearly to 0, loss on the FINAL ASSISTANT
  MESSAGE only, no added system prompt, training seed 3.
The only per-substrate differences are the base model and its pinned
non-thinking render mode (`dst1_common.SUBSTRATES`). WITHIN a substrate, the
gemma arm and the self arm therefore differ ONLY in the pool — which is the
variable under test.

Two pool kinds, one render path:
  gemma  rows {id, prompt, response}          — the frozen union-clean pool
  self   rows {prompt_id, prompt, completion} — this run's generated self pools

Over-length rows are BLOCKING (never dropped, never truncated): the census must
account for every row before its real train.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import tinker
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402


def load_pool(path: Path, kind: str) -> list[dict]:
    """Normalize either pool kind to {id, prompt, response}."""
    rows = C.read_jsonl(path)
    out = []
    for r in rows:
        if kind == "gemma":
            rid, resp = r["id"], r["response"]
        else:
            rid, resp = r["prompt_id"], r["completion"]
        if not (r.get("prompt") or "").strip() or not (resp or "").strip():
            raise SystemExit(f"empty prompt/response in row {rid}")
        out.append({"id": rid, "prompt": r["prompt"], "response": resp})
    if kind == "gemma" and len(out) != 17_995:
        raise SystemExit(f"gemma pool has {len(out)} rows, expected 17995")
    if len(out) != len({r["id"] for r in out}):
        raise SystemExit("duplicate ids in pool")
    return out


def renderer_for(spec: dict):
    tok = get_tokenizer(spec["base_model"])
    name = spec["renderer"]
    return tok, get_renderer(name, tok), name


def render_row(row: dict, renderer, spec: dict):
    """Single-turn chat, loss on the assistant message only. No added system prompt."""
    msgs = [{"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}]
    kwargs = {"train_on_what": TrainOnWhat.LAST_ASSISTANT_MESSAGE}
    if spec["effort"] is not None:
        kwargs["effort"] = spec["effort"]
    mi, weights = renderer.build_supervised_example(msgs, **kwargs)
    tokens = mi.to_ints()
    assert len(tokens) == len(weights)
    trainable = sum(1 for w in weights if float(w) > 0)
    assert trainable > 0, f"zero trainable tokens for {row['id']}"
    return mi, weights, len(tokens), trainable


def lr_at(step: int, total: int, peak: float) -> float:
    """Linear decay to zero — DCS1's `lr_inkling`, verbatim."""
    return peak * max(0.0, 1.0 - step / max(total, 1))


def cmd_census(args) -> None:
    spec = C.SUBSTRATES[args.substrate]
    rows = load_pool(args.data, args.pool_kind)
    _, renderer, renderer_name = renderer_for(spec)
    lengths, loss_tokens, records = [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        _, _, n, nt = render_row(row, renderer, spec)
        lengths.append(n)
        loss_tokens.append(nt)
        records.append({"index": i, "id": row["id"], "tokens": n, "loss_tokens": nt})
        if (i + 1) % 2000 == 0:
            print(f">>> census {args.arm} {i+1}/{len(rows)} ({time.time()-t0:.0f}s)",
                  flush=True)
    over = [r for r in records if r["tokens"] > C.MAX_LEN]
    receipt = {
        "status": "FAIL" if over else "PASS",
        "experiment": C.EXPERIMENT, "arm": args.arm,
        "substrate": args.substrate, "pool_kind": args.pool_kind,
        "pool_path": str(args.data), "pool_sha256": C.sha256_file(args.data),
        "base_model": spec["base_model"], "renderer": renderer_name,
        "effort": spec["effort"], "packages": C.package_versions(),
        "rows_in_pool": len(rows), "rows_rendered": len(records),
        "drops": 0, "truncations": 0, "empty_loss_rows": 0,
        "max_rendered_length": C.MAX_LEN,
        "over_length_rows": len(over), "over_length": over,
        "total_tokens": sum(lengths), "total_loss_tokens": sum(loss_tokens),
        "length": {"min": min(lengths), "median": statistics.median(lengths),
                   "mean": statistics.mean(lengths), "max": max(lengths)},
        "loss_length": {"min": min(loss_tokens), "median": statistics.median(loss_tokens),
                        "mean": statistics.mean(loss_tokens), "max": max(loss_tokens)},
        "expected_steps": math.ceil(len(rows) / C.BATCH) * C.EPOCHS,
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    args.rows_out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "over_length"}, indent=2))
    if over:
        raise SystemExit(f"BLOCKING: {len(over)} over-length rows for {args.arm}")


def cmd_train(args) -> None:
    spec = C.SUBSTRATES[args.substrate]
    rows = load_pool(args.data, args.pool_kind)
    # Shuffle once with the training seed (row order is otherwise the pool's) — DCS1.
    random.Random(args.seed).shuffle(rows)
    if args.max_examples:
        rows = rows[:args.max_examples]

    _, renderer, renderer_name = renderer_for(spec)
    datums = []
    for r in rows:
        mi, weights, n, _ = render_row(r, renderer, spec)
        if n > C.MAX_LEN:
            raise SystemExit(f"BLOCKING over-length row: {r['id']} tokens={n}")
        datums.append(datum_from_model_input_weights(mi, weights, C.MAX_LEN,
                                                     reduction="mean"))
    del rows

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=spec["base_model"], rank=C.RANK, train_mlp=True, train_attn=True,
        train_unembed=False,
        user_metadata={"project": C.EXPERIMENT, "arm": args.arm,
                       "seed": str(args.seed)},
    )

    steps = math.ceil(len(datums) / C.BATCH) * C.EPOCHS
    print(f">>> train {args.arm} rows={len(datums)} steps={steps} "
          f"peak_lr={spec['peak_lr']} schedule={spec['lr_schedule']}", flush=True)
    metrics, step, t0 = [], 0, time.time()
    for _ in range(C.EPOCHS):
        for start in range(0, len(datums), C.BATCH):
            batch = datums[start:start + C.BATCH]
            rate = lr_at(step, steps, spec["peak_lr"])
            fb = client.forward_backward(batch, "cross_entropy")
            opt = client.optim_step(tinker.AdamParams(
                learning_rate=rate, weight_decay=0.0, grad_clip_norm=1.0))
            result = fb.result()
            opt.result()
            m = dict(getattr(result, "metrics", None) or {})
            m = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                 for k, v in m.items()}
            step += 1
            metrics.append({"step": step, "lr": rate, "metrics": m})
            print(f">>> train {args.arm} step={step}/{steps} lr={rate:.8g} metrics={m}",
                  flush=True)
            args.metrics_out.write_text("\n".join(json.dumps(x) for x in metrics) + "\n")

    print(f">>> saving {args.arm} after {step} steps ({time.time()-t0:.0f}s)", flush=True)
    stem = f"{C.EXPERIMENT}-{args.arm}-s{args.seed}{args.tag}"
    state = client.save_state(f"{stem}-final", overwrite=True).result()
    sampler = client.save_weights_for_sampler(name=f"{stem}-sampler").result()
    state_path = getattr(state, "path", None) or getattr(state, "name", None)
    sampler_path = getattr(sampler, "path", None) or getattr(sampler, "name", None)

    losses = [x["metrics"].get("loss:sum") or x["metrics"].get("loss") for x in metrics]
    losses = [v for v in losses if isinstance(v, float) and math.isfinite(v)]
    receipt = {
        "status": "PASS", "experiment": C.EXPERIMENT, "arm": args.arm,
        "substrate": args.substrate, "pool_kind": args.pool_kind,
        "pool_path": str(args.data), "pool_sha256": C.sha256_file(args.data),
        "base_model": spec["base_model"], "renderer": renderer_name,
        "effort": spec["effort"], "packages": C.package_versions(),
        "seed": args.seed, "rank": C.RANK,
        "train_mlp": True, "train_attn": True, "train_unembed": False,
        "peak_learning_rate": spec["peak_lr"], "lr_schedule": spec["lr_schedule"],
        "warmup_ratio": 0.0, "lr_final_frac": 0.0,
        "first_step_lr": metrics[0]["lr"] if metrics else None,
        "last_step_lr": metrics[-1]["lr"] if metrics else None,
        "max_step_lr": max(x["lr"] for x in metrics) if metrics else None,
        "epochs": C.EPOCHS, "effective_batch_size": C.BATCH, "max_length": C.MAX_LEN,
        "rows": len(datums), "max_examples": args.max_examples, "tag": args.tag,
        "steps": step,
        "all_losses_finite": len(losses) == len(metrics) and len(losses) > 0,
        "n_finite_losses": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "state_path": state_path, "sampler_path": sampler_path,
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n=== DONE ===")
    print(f"training-state checkpoint: {state_path}")
    print(f"sampler weights:           {sampler_path}")
    print(json.dumps(receipt, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # avoid a non-zero exit from SDK teardown after a successful save (#569)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("census", cmd_census), ("train", cmd_train)):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True,
                       help="e.g. qwen-gemma, small-self, large-self")
        p.add_argument("--substrate", choices=sorted(C.SUBSTRATES), required=True)
        p.add_argument("--pool-kind", choices=("gemma", "self"), required=True)
        p.add_argument("--data", type=Path, required=True)
        if name == "census":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--rows-out", type=Path, required=True)
        else:
            p.add_argument("--seed", type=int, default=C.TRAIN_SEED)
            p.add_argument("--max-examples", type=int, default=None)
            p.add_argument("--tag", default="")
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--metrics-out", type=Path, required=True)
        p.set_defaults(func=fn)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
