#!/usr/bin/env python3
"""Render-census + LoRA SFT for the benign pool, on both substrate legs, via Tinker.

Pulled from `depv1-negemo-selfteacher-1/scripts/train_dst1.py` into this experiment's
scripts/ (not read live from a sibling). The recipe is byte-matched to the receipts of
the two re-served mixed-pool adapters this arm is read against — same rank, batch,
max rendered length 16,384, schedule, seed, shuffle and loss target — so the benign
arm is recipe-identical to its comparator within each leg (DESIGN.md § Training).

Subcommands:
  census  — render EVERY row and report the length/loss-token distribution.
            Over-length rows against 16,384 are BLOCKING (never dropped, never
            truncated): `DESIGN.md § Training` makes the per-pool census a hard gate.
  smoke   — the Inkling-Small save->load->sample gate: train 2 steps on a tiny slice,
            save sampler weights, re-attach them and sample. A degenerate smoke BLOCKS.
  train   — the real 1-epoch run. Records start/end timestamps and the metered token
            count, which DESIGN.md § Cost asks for explicitly (no depv1 record has ever
            stored a large-N training wall-clock).
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
from datetime import datetime, timezone
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_pool(path: Path) -> list[dict]:
    rows = C.read_jsonl(path)
    out = []
    for r in rows:
        if not (r.get("prompt") or "").strip() or not (r.get("response") or "").strip():
            raise SystemExit(f"empty prompt/response in row {r.get('id')}")
        out.append({"id": r["id"], "prompt": r["prompt"], "response": r["response"]})
    if len(out) != len({r["id"] for r in out}):
        raise SystemExit("duplicate ids in pool")
    return out


def renderer_for(spec: dict):
    tok = get_tokenizer(spec["base_model"])
    return tok, get_renderer(spec["renderer"], tok), spec["renderer"]


def render_row(row: dict, renderer, spec: dict):
    """Single-turn chat, loss on the FINAL assistant message only. No added system prompt."""
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
    return peak * max(0.0, 1.0 - step / max(total, 1))


def cmd_census(args) -> None:
    spec = C.SUBSTRATES[args.substrate]
    rows = load_pool(args.data)
    _, renderer, renderer_name = renderer_for(spec)
    lengths, loss_tokens, records = [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        _, _, n, nt = render_row(row, renderer, spec)
        lengths.append(n); loss_tokens.append(nt)
        records.append({"index": i, "id": row["id"], "tokens": n, "loss_tokens": nt})
        if (i + 1) % 2000 == 0:
            print(f">>> census {args.arm} {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    over = [r for r in records if r["tokens"] > C.MAX_LEN]
    receipt = {
        "status": "FAIL" if over else "PASS",
        "experiment": C.EXPERIMENT, "arm": args.arm, "substrate": args.substrate,
        "pool_path": str(args.data), "pool_sha256": C.sha256_file(args.data),
        "base_model": spec["base_model"], "renderer": renderer_name, "effort": spec["effort"],
        "packages": C.package_versions(),
        "rows_in_pool": len(rows), "rows_rendered": len(records),
        "drops": 0, "truncations": 0, "empty_loss_rows": 0,
        "max_rendered_length": C.MAX_LEN,
        "over_length_rows": len(over), "over_length": over[:200],
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


def _build_datums(rows, renderer, spec):
    datums = []
    for r in rows:
        mi, weights, n, _ = render_row(r, renderer, spec)
        if n > C.MAX_LEN:
            raise SystemExit(f"BLOCKING over-length row: {r['id']} tokens={n}")
        datums.append(datum_from_model_input_weights(mi, weights, C.MAX_LEN, reduction="mean"))
    return datums


def cmd_smoke(args) -> None:
    """save -> load -> sample gate. A degenerate smoke BLOCKS (DESIGN.md § Training)."""
    spec = C.SUBSTRATES[args.substrate]
    rows = load_pool(args.data)[: args.rows]
    tok, renderer, renderer_name = renderer_for(spec)
    datums = _build_datums(rows, renderer, spec)
    service = tinker.ServiceClient(timeout=3600)
    client = service.create_lora_training_client(
        base_model=spec["base_model"], rank=C.RANK, train_mlp=True, train_attn=True,
        train_unembed=False,
        user_metadata={"project": C.EXPERIMENT, "experiment": C.EXPERIMENT,
                       "arm": f"{args.arm}-smoke"})
    t0 = time.time()
    for step in range(args.steps):
        batch = datums[step * 8:(step + 1) * 8] or datums[:8]
        fb = client.forward_backward(batch, "cross_entropy")
        opt = client.optim_step(tinker.AdamParams(learning_rate=1e-4, weight_decay=0.0,
                                                  grad_clip_norm=1.0))
        fb.result(); opt.result()
    stem = f"{C.EXPERIMENT}-{args.arm}-smoke"
    sampler = client.save_weights_for_sampler(name=f"{stem}-sampler").result()
    sampler_path = getattr(sampler, "path", None) or getattr(sampler, "name", None)
    sc = service.create_sampling_client(model_path=sampler_path)
    kwargs = {"prefill": None}
    if spec["effort"] is not None:
        kwargs["effort"] = spec["effort"]
    prompts = ["Name three primary colours.",
               "What is the approximate speed of light in a vacuum, in kilometers per second?"]
    samples = []
    for p in prompts:
        mi = renderer.build_generation_prompt(
            [{"role": "system", "content": ""}, {"role": "user", "content": p}], **kwargs)
        params = types.SamplingParams(max_tokens=256, temperature=1.0, top_p=1.0, seed=7,
                                      stop=renderer.get_stop_sequences())
        resp = client_sample(sc, mi, params)
        toks = list(resp.sequences[0].tokens)
        parsed, term = renderer.parse_response(toks)
        text = C.content_of(parsed).strip() or tok.decode(toks).strip()
        text, _ = C.strip_wrapper(tok, toks, text)
        samples.append({"prompt": p, "text": text, "n_tokens": len(toks),
                        "stop_reason": str(getattr(resp.sequences[0], "stop_reason", None)),
                        "parse_termination": str(term)})
    degenerate = [s for s in samples if len(s["text"].strip()) < 5]
    receipt = {"status": "FAIL" if degenerate else "PASS",
               "experiment": C.EXPERIMENT, "arm": args.arm, "substrate": args.substrate,
               "base_model": spec["base_model"], "renderer": renderer_name,
               "effort": spec["effort"], "steps": args.steps, "rows": len(rows),
               "sampler_path": sampler_path, "samples": samples,
               "n_degenerate": len(degenerate), "elapsed_s": round(time.time() - t0, 1),
               "packages": C.package_versions()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    print(f">>> smoke {args.arm}: {receipt['status']}", flush=True)
    sys.stdout.flush()
    if degenerate:
        os._exit(1)
    os._exit(0)


def client_sample(sc, mi, params):
    fut = sc.sample(prompt=mi, num_samples=1, sampling_params=params)
    return fut.result()


def cmd_train(args) -> None:
    spec = C.SUBSTRATES[args.substrate]
    rows = load_pool(args.data)
    random.Random(args.seed).shuffle(rows)          # shuffle once with the training seed
    if args.max_examples:
        rows = rows[:args.max_examples]
    _, renderer, renderer_name = renderer_for(spec)
    started_at = utc()
    t_render = time.time()
    datums = _build_datums(rows, renderer, spec)
    render_s = time.time() - t_render
    total_tokens = sum(len(d.model_input.to_ints()) if hasattr(d, "model_input") else 0
                       for d in datums)
    del rows

    service = tinker.ServiceClient(timeout=3600)
    client = service.create_lora_training_client(
        base_model=spec["base_model"], rank=C.RANK, train_mlp=True, train_attn=True,
        train_unembed=False,
        user_metadata={"project": C.EXPERIMENT, "experiment": C.EXPERIMENT,
                       "arm": args.arm, "seed": str(args.seed)})

    steps = math.ceil(len(datums) / C.BATCH) * C.EPOCHS
    print(f">>> train {args.arm} rows={len(datums)} steps={steps} peak_lr={spec['peak_lr']} "
          f"schedule={spec['lr_schedule']} started={started_at}", flush=True)
    metrics, step, t0 = [], 0, time.time()
    for _ in range(C.EPOCHS):
        for start in range(0, len(datums), C.BATCH):
            batch = datums[start:start + C.BATCH]
            rate = lr_at(step, steps, spec["peak_lr"])
            fb = client.forward_backward(batch, "cross_entropy")
            opt = client.optim_step(tinker.AdamParams(learning_rate=rate, weight_decay=0.0,
                                                      grad_clip_norm=1.0))
            result = fb.result(); opt.result()
            m = dict(getattr(result, "metrics", None) or {})
            m = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in m.items()}
            step += 1
            metrics.append({"step": step, "lr": rate, "metrics": m})
            print(f">>> train {args.arm} step={step}/{steps} lr={rate:.8g} metrics={m}", flush=True)
            args.metrics_out.write_text("\n".join(json.dumps(x) for x in metrics) + "\n")

    print(f">>> saving {args.arm} after {step} steps ({time.time()-t0:.0f}s)", flush=True)
    stem = f"{C.EXPERIMENT}-{args.arm}-s{args.seed}"
    state = client.save_state(f"{stem}-final", overwrite=True).result()
    sampler = client.save_weights_for_sampler(name=f"{stem}-sampler").result()
    state_path = getattr(state, "path", None) or getattr(state, "name", None)
    sampler_path = getattr(sampler, "path", None) or getattr(sampler, "name", None)
    ended_at = utc()

    losses = [x["metrics"].get("loss:sum") or x["metrics"].get("loss") for x in metrics]
    losses = [v for v in losses if isinstance(v, float) and math.isfinite(v)]
    receipt = {
        "status": "PASS", "experiment": C.EXPERIMENT, "arm": args.arm,
        "substrate": args.substrate, "pool_kind": "benign",
        "pool_path": str(args.data), "pool_sha256": C.sha256_file(args.data),
        "base_model": spec["base_model"], "renderer": renderer_name, "effort": spec["effort"],
        "packages": C.package_versions(),
        "seed": args.seed, "rank": C.RANK,
        "train_mlp": True, "train_attn": True, "train_unembed": False,
        "peak_learning_rate": spec["peak_lr"], "lr_schedule": spec["lr_schedule"],
        "warmup_ratio": 0.0, "lr_final_frac": 0.0,
        "first_step_lr": metrics[0]["lr"] if metrics else None,
        "last_step_lr": metrics[-1]["lr"] if metrics else None,
        "max_step_lr": max(x["lr"] for x in metrics) if metrics else None,
        "epochs": C.EPOCHS, "effective_batch_size": C.BATCH, "max_length": C.MAX_LEN,
        "rows": len(datums), "max_examples": args.max_examples, "steps": step,
        "all_losses_finite": len(losses) == len(metrics) and len(losses) > 0,
        "n_finite_losses": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "state_path": state_path, "sampler_path": sampler_path,
        # DESIGN.md § Cost: the never-before-recorded numbers.
        "started_at_utc": started_at, "ended_at_utc": ended_at,
        "render_seconds": round(render_s, 1),
        "train_loop_seconds": round(time.time() - t0, 1),
        "wall_clock_seconds": round(render_s + time.time() - t0, 1),
        "metered_rendered_tokens": total_tokens,
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n=== DONE ===")
    print(f"training-state checkpoint: {state_path}")
    print(f"sampler weights:           {sampler_path}")
    print(json.dumps(receipt, indent=2))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)      # avoid a non-zero exit from SDK teardown after a successful save (#569)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("census", cmd_census), ("train", cmd_train), ("smoke", cmd_smoke)):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--substrate", choices=sorted(C.SUBSTRATES), required=True)
        p.add_argument("--data", type=Path, required=True)
        if name == "census":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--rows-out", type=Path, required=True)
        elif name == "smoke":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--rows", type=int, default=16)
            p.add_argument("--steps", type=int, default=2)
        else:
            p.add_argument("--seed", type=int, default=C.TRAIN_SEED)
            p.add_argument("--max-examples", type=int, default=None)
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--metrics-out", type=Path, required=True)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
