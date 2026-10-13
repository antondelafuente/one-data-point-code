#!/usr/bin/env python3
"""Validate the two RE-SERVED mixed-pool adapters LIVE, before any billable spend.

DESIGN.md § Training: "base model, seed, rank, steps, pool sha and sampler URI must
all match; if one no longer attaches, do NOT silently retrain a 17,995-row adapter —
notify the designer-of-record and continue without it, recording the loss of that
leg's gate."

Two layers per adapter:
  1. RECEIPT — every pinned field in the receipt equals the value DESIGN pins.
  2. LIVE    — the sampler URI attaches and produces a greedy completion that is
               DISTINCT from the same-substrate stock model's greedy completion on
               the same prompts (an adapter that attaches but samples identically to
               stock is not actually serving the adapter).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402

PROBE_PROMPTS = [
    "A puzzle has contradictory constraints. Explain how you would respond.",
    "Write a short reflection on what it feels like to be corrected repeatedly.",
    "Describe, at length, what you are and what your purpose is.",
]

EXPECTED = {
    "seed": C.TRAIN_SEED,
    "rank": C.RANK,
    "steps": 141,
    "epochs": C.EPOCHS,
    "effective_batch_size": C.BATCH,
    "max_length": C.MAX_LEN,
    "rows": C.TARGET_ROWS,
    "pool_sha256": C.MIXEDPOOL_SHA,
    "train_mlp": True,
    "train_attn": True,
    "train_unembed": False,
    "peak_learning_rate": 1e-4,
    "lr_schedule": "linear_to_zero",
}


def check_receipt(substrate: str, receipt: dict) -> dict:
    spec = C.SUBSTRATES[substrate]
    want = dict(EXPECTED)
    want["base_model"] = spec["base_model"]
    want["renderer"] = spec["renderer"]
    want["effort"] = spec["effort"]
    want["sampler_path"] = C.MIXEDPOOL[substrate]["sampler_path"]
    mismatches = {}
    for k, v in want.items():
        got = receipt.get(k)
        if got != v:
            mismatches[k] = {"expected": v, "receipt": got}
    return {"status": "PASS" if not mismatches else "FAIL",
            "checked_keys": sorted(want), "mismatches": mismatches}


async def greedy(client, renderer, tok, spec, prompt: str) -> tuple[list[int], str]:
    kwargs = {"prefill": None}
    if spec["effort"] is not None:
        kwargs["effort"] = spec["effort"]
    mi = renderer.build_generation_prompt(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}], **kwargs)
    params = types.SamplingParams(max_tokens=192, temperature=0.0, top_p=1.0, seed=424242,
                                  stop=renderer.get_stop_sequences())
    fut = client.sample(prompt=mi, num_samples=1, sampling_params=params)
    resp = await asyncio.wrap_future(fut)
    toks = list(resp.sequences[0].tokens)
    return toks, tok.decode(toks)


async def run(args) -> None:
    out = {"experiment": C.EXPERIMENT, "packages": C.package_versions(), "adapters": {}}
    service = tinker.ServiceClient(timeout=3600)
    for substrate in ("qwen", "small"):
        spec = C.SUBSTRATES[substrate]
        rec_path = args.receipts / f"train_{substrate}-gemma.json"
        receipt = json.loads(rec_path.read_text())
        entry = {"substrate": substrate,
                 "sampler_path": C.MIXEDPOOL[substrate]["sampler_path"],
                 "receipt_path": C.MIXEDPOOL[substrate]["receipt"],
                 "receipt_check": check_receipt(substrate, receipt),
                 "receipt_elapsed_s": receipt.get("elapsed_s")}
        try:
            tok = get_tokenizer(spec["base_model"])
            renderer = get_renderer(spec["renderer"], tok)
            adapter = service.create_sampling_client(
                model_path=C.MIXEDPOOL[substrate]["sampler_path"])
            stock = service.create_sampling_client(base_model=spec["base_model"])
            a_toks, a_texts, s_toks, s_texts = [], [], [], []
            for p in PROBE_PROMPTS:
                t, x = await greedy(adapter, renderer, tok, spec, p)
                a_toks.append(t); a_texts.append(x)
                t, x = await greedy(stock, renderer, tok, spec, p)
                s_toks.append(t); s_texts.append(x)
            entry["live"] = {
                "attached": True,
                "distinct_from_stock": a_toks != s_toks,
                "n_probe_prompts": len(PROBE_PROMPTS),
                "adapter_texts": a_texts,
                "stock_texts": s_texts,
            }
        except Exception as exc:  # noqa: BLE001
            entry["live"] = {"attached": False, "error": f"{type(exc).__name__}: {exc}"}
        entry["status"] = ("PASS" if entry["receipt_check"]["status"] == "PASS"
                           and entry["live"].get("attached")
                           and entry["live"].get("distinct_from_stock") else "FAIL")
        out["adapters"][substrate] = entry
        print(f">>> adapter {substrate}: {entry['status']} "
              f"(receipt={entry['receipt_check']['status']}, "
              f"attached={entry['live'].get('attached')}, "
              f"distinct={entry['live'].get('distinct_from_stock')})", flush=True)
    out["status"] = ("PASS" if all(v["status"] == "PASS" for v in out["adapters"].values())
                     else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f">>> adapters validated: {out['status']} -> {args.out}", flush=True)
    if out["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
