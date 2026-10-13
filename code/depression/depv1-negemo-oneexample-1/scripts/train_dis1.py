#!/usr/bin/env python3
"""DRR1 render-census + LoRA SFT for ONE cell (141 updates), via Tinker.

Pulled from `depv1-negemo-rowsdose-1/scripts/train_drd1.py` (DRD1's own driver,
itself adapted from DST1's) into this experiment's scripts/, not read live from
a sibling. The Inkling-Small recipe is preserved byte-for-byte in everything the
DESIGN holds fixed — rank 32, mlp+attn, no unembed, max length 16,384, lr 1e-4
linear to zero, loss on the FINAL ASSISTANT MESSAGE only, no added system prompt,
141 optimizer updates, effective batch min(N, 128).

The init-seed machinery is DRD1's, unchanged and load-bearing here for a new
reason: the LoRA init seed is actually PASSED to
`create_lora_training_client` (whose signature defaults `seed=None`, which
Tinker documents as "None means random seed") AND the ACTUAL `CreateModelRequest`
is captured at the transport boundary (see `capture_create_requests`), so the
recorded evidence is the request that went on the wire, not a metadata field we
wrote ourselves. In DRD1 that guarded a fixed seed 3 across every cell; in DAP1
the seed VARIES BY CELL and IS the replication variable, so the driver requires
`--seed` explicitly (no default) and refuses to train if the captured
`lora_config.seed` is anything other than the value it was asked for.

ONE further departure from DRD1: the checkpoint stem is `<experiment>-<cell>`,
because a DAP1 cell name already carries its own seed suffix (`d1-s4`).

Row source: the cell's COMMITTED manifest — one of DSD1's three 512-row draws
(`draw_manifests/`) or one of this design's six committed d1 pieces
(`split_manifests/`). This driver never creates a manifest — it reads the
committed id list, asserts its sha, and asserts the realized row-id set equals
the manifest exactly.

Over-length rows are BLOCKING (never dropped, never truncated).
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
import dis1_common as C  # noqa: E402

# --------------------------------------------------------------------------
# Init-seed evidence (design-audit F1 / CHECKLIST "LoRA INIT SEED").
# `create_lora_training_client` builds a `types.LoraConfig(seed=...)` inside a
# `types.CreateModelRequest` and posts it via `client.models.create(request=...)`.
# We wrap that resource method so the EXACT request object that goes on the wire
# is serialized to disk. A `user_metadata` field would not be evidence.
_CAPTURED: list[dict] = []


def capture_create_requests() -> None:
    from tinker.resources import models as _models

    orig = _models.AsyncModelsResource.create

    async def wrapped(self, *, request, **kw):
        try:
            _CAPTURED.append(json.loads(request.model_dump_json()))
        except Exception as exc:  # noqa: BLE001 — never break a train over evidence
            _CAPTURED.append({"capture_error": f"{type(exc).__name__}: {exc}"})
        return await orig(self, request=request, **kw)

    _models.AsyncModelsResource.create = wrapped


def load_manifest(manifest: Path) -> list[str]:
    ids = [l.strip() for l in manifest.open() if l.strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate ids in manifest {manifest}")
    return ids


def load_cell_rows(pool: Path, manifest: Path, n: int) -> list[dict]:
    """The cell's rows, in MANIFEST order, from its locked source pool.

    Streams the pool and keeps only this cell's rows: 10 of these drivers run
    concurrently on one controller box, and holding the whole 17,995-row pool in
    each is exactly the local-RAM ceiling `local_job_queue.sh` exists for (#402).
    Duplicate-id detection still scans every row, via a set of seen ids.
    """
    ids = load_manifest(manifest)
    if len(ids) != n:
        raise SystemExit(f"manifest {manifest.name} has {len(ids)} ids, expected {n}")
    want = set(ids)
    by_id, seen = {}, set()
    with pool.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in seen:
                raise SystemExit(f"duplicate id in pool: {r['id']}")
            seen.add(r["id"])
            if r["id"] in want:
                by_id[r["id"]] = r
    if len(seen) < n:
        raise SystemExit(f"pool has only {len(seen)} unique rows, expected at least {n}")
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids absent from the pool")
    rows = []
    for i in ids:
        r = by_id[i]
        if not (r.get("prompt") or "").strip() or not (r.get("response") or "").strip():
            raise SystemExit(f"empty prompt/response in row {i}")
        rows.append({"id": i, "prompt": r["prompt"], "response": r["response"]})
    return rows


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


def batch_plan(n_rows: int, batch: int, budget: int) -> list[tuple[int, int]]:
    """(start, end) row slices for exactly `budget` optimizer steps.

    DESIGN: steps per epoch = ceil(N/batch); cycle epochs until the budget is hit.
    Row order within an epoch is the seed-3 shuffle applied once (see cmd_train);
    epochs reuse that same order.
    """
    per_epoch = math.ceil(n_rows / batch)
    plan = []
    while len(plan) < budget:
        for e in range(per_epoch):
            if len(plan) == budget:
                break
            plan.append((e * batch, min((e + 1) * batch, n_rows)))
    return plan


def cmd_census(args) -> None:
    spec = C.SUBSTRATES[args.substrate]
    rows = load_cell_rows(args.data, args.manifest, args.n)
    _, renderer, renderer_name = renderer_for(spec)
    lengths, loss_tokens, records = [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        _, _, n, nt = render_row(row, renderer, spec)
        lengths.append(n)
        loss_tokens.append(nt)
        records.append({"index": i, "id": row["id"], "tokens": n, "loss_tokens": nt})
    over = [r for r in records if r["tokens"] > C.MAX_LEN]
    batch = C.batch_for(args.n)
    plan = batch_plan(len(rows), batch, args.steps)
    # Supervised-token exposure the cell will actually see (the reported covariate).
    tok_by_idx = {r["index"]: r["tokens"] for r in records}
    loss_by_idx = {r["index"]: r["loss_tokens"] for r in records}
    exposure = sum(tok_by_idx[i] for s, e in plan for i in range(s, e))
    loss_exposure = sum(loss_by_idx[i] for s, e in plan for i in range(s, e))
    receipt = {
        "status": "FAIL" if over else "PASS",
        "experiment": C.EXPERIMENT, "cell": args.cell, "group": args.group,
        "n": args.n, "draw": args.draw, "seed": args.seed,
        "step_budget": args.steps,
        "effective_batch_size": batch, "steps_per_epoch": C.steps_per_epoch(args.n),
        "epochs_cycled": args.steps / max(C.steps_per_epoch(args.n), 1),
        "substrate": args.substrate,
        "manifest_path": str(args.manifest),
        "manifest_sha256": C.sha256_file(args.manifest),
        "manifest_ids_sha256": C.sha256_text("\n".join(load_manifest(args.manifest)) + "\n"),
        "pool_path": str(args.data), "pool_sha256": C.sha256_file(args.data),
        "base_model": spec["base_model"], "renderer": renderer_name,
        "effort": spec["effort"], "packages": C.package_versions(),
        "rows_in_cell": len(rows), "rows_rendered": len(records),
        "drops": 0, "truncations": 0, "empty_loss_rows": 0,
        "max_rendered_length": C.MAX_LEN,
        "over_length_rows": len(over), "over_length": over,
        "total_tokens": sum(lengths), "total_loss_tokens": sum(loss_tokens),
        "supervised_token_exposure": exposure,
        "supervised_loss_token_exposure": loss_exposure,
        "length": {"min": min(lengths), "median": statistics.median(lengths),
                   "mean": statistics.mean(lengths), "max": max(lengths)},
        "loss_length": {"min": min(loss_tokens), "median": statistics.median(loss_tokens),
                        "mean": statistics.mean(loss_tokens), "max": max(loss_tokens)},
        "expected_steps": args.steps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    args.rows_out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "over_length"}, indent=2))
    if over:
        raise SystemExit(f"BLOCKING: {len(over)} over-length rows for {args.cell}")


def cmd_train(args) -> None:
    capture_create_requests()
    spec = C.SUBSTRATES[args.substrate]
    rows = load_cell_rows(args.data, args.manifest, args.n)
    manifest_ids = set(load_manifest(args.manifest))
    # Shuffle once with the training seed (the DST1/DCS1 recipe); epochs reuse
    # this order. Recorded as `row_order` in the receipt.
    random.Random(args.seed).shuffle(rows)
    if {r["id"] for r in rows} != manifest_ids:
        raise SystemExit("realized row-id set != committed manifest")

    _, renderer, renderer_name = renderer_for(spec)
    datums = []
    for r in rows:
        mi, weights, n, _ = render_row(r, renderer, spec)
        if n > C.MAX_LEN:
            raise SystemExit(f"BLOCKING over-length row: {r['id']} tokens={n}")
        datums.append(datum_from_model_input_weights(mi, weights, C.MAX_LEN,
                                                     reduction="mean"))

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=spec["base_model"], rank=C.RANK,
        seed=args.seed,                      # <-- design-audit F1: ACTUALLY passed
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata={"project": C.EXPERIMENT, "cell": args.cell,
                       "seed": str(args.seed)},
    )
    # Persist the captured CreateModelRequest(s) — the init-seed EVIDENCE.
    args.create_request_out.parent.mkdir(parents=True, exist_ok=True)
    args.create_request_out.write_text(json.dumps(_CAPTURED, indent=2) + "\n")
    seeds_on_wire = [c.get("lora_config", {}).get("seed") for c in _CAPTURED
                     if isinstance(c, dict) and "lora_config" in c]
    if seeds_on_wire != [args.seed]:
        raise SystemExit(f"init-seed evidence FAIL: captured lora_config seeds "
                         f"{seeds_on_wire}, expected [{args.seed}]")
    print(f">>> init-seed on the wire: {seeds_on_wire} "
          f"(captured from CreateModelRequest)", flush=True)

    batch = C.batch_for(args.n)
    plan = batch_plan(len(datums), batch, args.steps)
    assert len(plan) == args.steps
    print(f">>> train {args.cell} rows={len(datums)} batch={batch} "
          f"steps={args.steps} peak_lr={spec['peak_lr']} "
          f"schedule={spec['lr_schedule']}", flush=True)
    metrics, step, t0 = [], 0, time.time()
    for start, end in plan:
        b = datums[start:end]
        rate = C.lr_at(step, args.steps, spec["peak_lr"])
        fb = client.forward_backward(b, "cross_entropy")
        opt = client.optim_step(tinker.AdamParams(
            learning_rate=rate, weight_decay=0.0, grad_clip_norm=1.0))
        result = fb.result()
        opt.result()
        m = dict(getattr(result, "metrics", None) or {})
        m = {k: (float(v) if isinstance(v, (int, float)) else str(v))
             for k, v in m.items()}
        step += 1
        metrics.append({"step": step, "lr": rate, "rows_in_batch": len(b),
                        "row_slice": [start, end], "metrics": m})
        print(f">>> train {args.cell} step={step}/{args.steps} lr={rate:.8g} "
              f"rows={len(b)} metrics={m}", flush=True)
        args.metrics_out.write_text("\n".join(json.dumps(x) for x in metrics) + "\n")

    print(f">>> saving {args.cell} after {step} steps ({time.time()-t0:.0f}s)", flush=True)
    stem = f"{C.EXPERIMENT}-{args.cell}"  # cell already carries its seed suffix
    state = client.save_state(f"{stem}-final", overwrite=True).result()
    sampler = client.save_weights_for_sampler(name=f"{stem}-sampler").result()
    state_path = getattr(state, "path", None) or getattr(state, "name", None)
    sampler_path = getattr(sampler, "path", None) or getattr(sampler, "name", None)

    losses = [x["metrics"].get("loss:sum") or x["metrics"].get("loss") for x in metrics]
    losses = [v for v in losses if isinstance(v, float) and math.isfinite(v)]
    receipt = {
        "status": "PASS", "experiment": C.EXPERIMENT, "cell": args.cell,
        "group": args.group, "n": args.n, "draw": args.draw,
        "substrate": args.substrate,
        "manifest_path": str(args.manifest),
        "manifest_sha256": C.sha256_file(args.manifest),
        "pool_path": str(args.data), "pool_sha256": C.sha256_file(args.data),
        "base_model": spec["base_model"], "renderer": renderer_name,
        "effort": spec["effort"], "packages": C.package_versions(),
        "seed": args.seed,
        "lora_init_seed_on_wire": seeds_on_wire[0],
        "create_request_capture": str(args.create_request_out),
        "rank": C.RANK, "train_mlp": True, "train_attn": True, "train_unembed": False,
        "peak_learning_rate": spec["peak_lr"], "lr_schedule": spec["lr_schedule"],
        "warmup_ratio": 0.0, "lr_final_frac": 0.0,
        "first_step_lr": metrics[0]["lr"] if metrics else None,
        "last_step_lr": metrics[-1]["lr"] if metrics else None,
        "max_step_lr": max(x["lr"] for x in metrics) if metrics else None,
        "step_budget": args.steps, "steps": step,
        "steps_per_epoch": C.steps_per_epoch(args.n),
        "epochs_cycled": args.steps / max(C.steps_per_epoch(args.n), 1),
        "effective_batch_size": batch, "max_length": C.MAX_LEN,
        "rows": len(datums),
        "row_order": f"committed manifest order, shuffled once with seed {args.seed}",
        "rows_seen_total": sum(x["rows_in_batch"] for x in metrics),
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
        p.add_argument("--cell", required=True)
        p.add_argument("--group", default="arm",
                       choices=("arm", "smoke", "reserve"))
        p.add_argument("--n", type=int, required=True)
        p.add_argument("--draw", type=int, required=True)
        p.add_argument("--steps", type=int, required=True)
        p.add_argument("--substrate", choices=sorted(C.SUBSTRATES), default="small")
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        # The seed is a CELL PROPERTY in this wave (the replication variable), so
        # both the census and the train record it — same cell, same seed, always.
        p.add_argument("--seed", type=int, required=True,
                       help="per-cell LoRA init seed — THIS wave's replication variable")
        if name == "census":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--rows-out", type=Path, required=True)
        else:
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--metrics-out", type=Path, required=True)
            p.add_argument("--create-request-out", type=Path, required=True)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
