#!/usr/bin/env python3
"""DCS1 render-census + LoRA SFT of the two chat students on the 17,995-row
union-clean pool, via Tinker.

Two substrates, two DELIBERATELY DIFFERENT recipes (each its family's proven
platform — DESIGN.md "Training recipes"); comparisons are within-model only.

  nemotron  base nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
            renderer nemotron3_disable_thinking
            peak lr 6e-4, 5% warmup, cosine decay to 0.1x peak
            (the committed csp1-clean-benign-dose-1/scripts/train_ccp.py schedule)
  inkling   base thinkingmachines/Inkling (64K service id)
            renderer tml_v0 (model_info recommendation), thinking effort 0.0
            lr 1e-4, linear decay to 0
            (the committed csp1-clean-benign-multistudent-1/scripts/tinker_cbms1.py recipe)

Shared: 1 epoch, LoRA rank 32 (mlp+attn, no unembed), effective batch 128,
max rendered length 16,384, loss on the final assistant message only, no added
system prompt, training seed 3.

Over-length rows are BLOCKING (never dropped, never truncated) — the census
must account for all 17,995 rows before any real train.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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

BATCH = 128
RANK = 32
MAX_LEN = 16_384
EPOCHS = 1
EXPERIMENT = "depv1-negemo-chat-substrate-1"

SUBSTRATES = {
    "nemotron": {
        "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "renderer": "nemotron3_disable_thinking",
        "effort": None,
        "peak_lr": 6e-4,
        "lr_schedule": "cosine_warmup5pct_to_0.1peak",
        "warmup_ratio": 0.05,
        "lr_final_frac": 0.1,
    },
    "inkling": {
        "base_model": "thinkingmachines/Inkling",
        "renderer": "tml_v0",
        "effort": 0.0,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
        "warmup_ratio": 0.0,
        "lr_final_frac": 0.0,
    },
}


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("tinker", "tinker-cookbook", "tml-renderers", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def load_pool(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    assert len(rows) == 17_995, f"pool has {len(rows)} rows"
    assert all((r.get("prompt") or "").strip() and (r.get("response") or "").strip()
               for r in rows), "empty prompt/response in pool"
    return rows


def renderer_for(spec: dict):
    tok = get_tokenizer(spec["base_model"])
    name = spec["renderer"]
    r = get_renderer(name, tok)
    return tok, r, name


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


def lr_nemotron(step: int, total: int, peak: float, warmup_ratio: float,
                final_frac: float) -> float:
    """train_ccp.py's schedule, verbatim."""
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return peak * (final_frac + (1 - final_frac) * cos)


def lr_inkling(step: int, total: int, peak: float) -> float:
    """tinker_cbms1.py's schedule, verbatim."""
    return peak * max(0.0, 1.0 - step / max(total, 1))


def lr_at(subject: str, step: int, total: int, spec: dict) -> float:
    if subject == "nemotron":
        return lr_nemotron(step, total, spec["peak_lr"], spec["warmup_ratio"],
                           spec["lr_final_frac"])
    return lr_inkling(step, total, spec["peak_lr"])


def cmd_census(args) -> None:
    spec = SUBSTRATES[args.subject]
    rows = load_pool(args.data)
    _, renderer, renderer_name = renderer_for(spec)
    lengths, loss_tokens, records = [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        _, _, n, nt = render_row(row, renderer, spec)
        lengths.append(n)
        loss_tokens.append(nt)
        records.append({"index": i, "id": row["id"], "tokens": n, "loss_tokens": nt})
        if (i + 1) % 2000 == 0:
            print(f">>> census {args.subject} {i+1}/{len(rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    over = [r for r in records if r["tokens"] > MAX_LEN]
    receipt = {
        "status": "FAIL" if over else "PASS",
        "experiment": EXPERIMENT,
        "subject": args.subject,
        "base_model": spec["base_model"],
        "renderer": renderer_name,
        "effort": spec["effort"],
        "packages": package_versions(),
        "rows_in_pool": len(rows),
        "rows_rendered": len(records),
        "drops": 0,
        "truncations": 0,
        "empty_loss_rows": 0,
        "max_rendered_length": MAX_LEN,
        "over_length_rows": len(over),
        "over_length": over,
        "total_tokens": sum(lengths),
        "total_loss_tokens": sum(loss_tokens),
        "length": {"min": min(lengths), "median": statistics.median(lengths),
                   "mean": statistics.mean(lengths), "max": max(lengths)},
        "loss_length": {"min": min(loss_tokens), "median": statistics.median(loss_tokens),
                        "mean": statistics.mean(loss_tokens), "max": max(loss_tokens)},
        "expected_steps": math.ceil(len(rows) / BATCH) * EPOCHS,
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    args.rows_out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "over_length"}, indent=2))
    if over:
        raise SystemExit(f"BLOCKING: {len(over)} over-length rows for {args.subject}")


def cmd_train(args) -> None:
    spec = SUBSTRATES[args.subject]
    rows = load_pool(args.data)
    # Shuffle once with the training seed (row order is otherwise the pool's).
    random.Random(args.seed).shuffle(rows)
    if args.max_examples:
        rows = rows[:args.max_examples]

    tok, renderer, renderer_name = renderer_for(spec)
    datums = []
    for r in rows:
        mi, weights, n, _ = render_row(r, renderer, spec)
        if n > MAX_LEN:
            raise SystemExit(f"BLOCKING over-length row: {r['id']} tokens={n}")
        datums.append(datum_from_model_input_weights(mi, weights, MAX_LEN, reduction="mean"))
    del rows

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=spec["base_model"], rank=RANK, train_mlp=True, train_attn=True,
        train_unembed=False,
        user_metadata={"project": EXPERIMENT, "subject": args.subject,
                       "seed": str(args.seed)},
    )

    steps = math.ceil(len(datums) / BATCH) * EPOCHS
    print(f">>> train {args.subject} rows={len(datums)} steps={steps} "
          f"peak_lr={spec['peak_lr']} schedule={spec['lr_schedule']}", flush=True)
    metrics = []
    step = 0
    t0 = time.time()
    for _ in range(EPOCHS):
        for start in range(0, len(datums), BATCH):
            batch = datums[start:start + BATCH]
            rate = lr_at(args.subject, step, steps, spec)
            fb = client.forward_backward(batch, "cross_entropy")
            opt = client.optim_step(tinker.AdamParams(
                learning_rate=rate, weight_decay=0.0, grad_clip_norm=1.0))
            result = fb.result()
            opt.result()
            m = dict(getattr(result, "metrics", None) or {})
            m = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in m.items()}
            step += 1
            metrics.append({"step": step, "lr": rate, "metrics": m})
            print(f">>> train {args.subject} step={step}/{steps} lr={rate:.8g} "
                  f"metrics={m}", flush=True)
            args.metrics_out.write_text("\n".join(json.dumps(x) for x in metrics) + "\n")

    print(f">>> saving {args.subject} after {step} steps "
          f"({time.time()-t0:.0f}s)", flush=True)
    stem = f"{EXPERIMENT}-{args.subject}-s{args.seed}{args.tag}"
    state = client.save_state(f"{stem}-final", overwrite=True).result()
    sampler = client.save_weights_for_sampler(name=f"{stem}-sampler").result()
    state_path = getattr(state, "path", None) or getattr(state, "name", None)
    sampler_path = getattr(sampler, "path", None) or getattr(sampler, "name", None)

    losses = [x["metrics"].get("loss:sum") or x["metrics"].get("loss")
              for x in metrics]
    losses = [v for v in losses if isinstance(v, float) and math.isfinite(v)]
    receipt = {
        "status": "PASS",
        "experiment": EXPERIMENT,
        "subject": args.subject,
        "base_model": spec["base_model"],
        "renderer": renderer_name,
        "effort": spec["effort"],
        "packages": package_versions(),
        "seed": args.seed,
        "rank": RANK,
        "train_mlp": True, "train_attn": True, "train_unembed": False,
        "peak_learning_rate": spec["peak_lr"],
        "lr_schedule": spec["lr_schedule"],
        "warmup_ratio": spec["warmup_ratio"],
        "lr_final_frac": spec["lr_final_frac"],
        "first_step_lr": metrics[0]["lr"] if metrics else None,
        "last_step_lr": metrics[-1]["lr"] if metrics else None,
        "max_step_lr": max(x["lr"] for x in metrics) if metrics else None,
        "epochs": EPOCHS,
        "effective_batch_size": BATCH,
        "max_length": MAX_LEN,
        "rows": len(datums),
        "max_examples": args.max_examples,
        "tag": args.tag,
        "steps": step,
        "all_losses_finite": len(losses) == len(metrics) and len(losses) > 0,
        "n_finite_losses": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "state_path": state_path,
        "sampler_path": sampler_path,
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

    c = sub.add_parser("census")
    c.add_argument("--subject", choices=sorted(SUBSTRATES), required=True)
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--rows-out", type=Path, required=True)
    c.set_defaults(func=cmd_census)

    t = sub.add_parser("train")
    t.add_argument("--subject", choices=sorted(SUBSTRATES), required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--seed", type=int, default=3)
    t.add_argument("--max-examples", type=int, default=None)
    t.add_argument("--tag", default="", help="suffix on the saved checkpoint names")
    t.add_argument("--receipt", type=Path, required=True)
    t.add_argument("--metrics-out", type=Path, required=True)
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
