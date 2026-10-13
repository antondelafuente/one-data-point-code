#!/usr/bin/env python3
"""DST1 SELF-POOL generation: one completion per keep-id prompt from the
substrate's OWN STOCK model, via Tinker direct sampling.

DESIGN.md "Training pools / Self pools" pins every semantic here; the formulas
live in `dst1_common.py` and are unit-tested against frozen independent ground
truth by `test_dst1_common.py` (a hard gate before this script runs):

  * temperature 1.0, top_p 1.0, max 8,192 new tokens, the substrate's pinned
    NON-THINKING mode, no system prompt, single user turn;
  * per-row seed = selfpool_seed(substrate, prompt_id);
  * transport failures retried <= 4 (permit brackets the SINGLE attempt, jittered
    backoff OUTSIDE the permit — #343);
  * an empty/whitespace-only completion is resampled at most twice, at seed+1 then
    seed+2; still empty -> the row is DROPPED from this pool only, and counted;
  * completion text = the parsed message content after DCS1's `strip_wrapper`
    discipline, `wrapper_stripped` recorded per row.

The pool is deliberately UNSCREENED for negativity (DESIGN; researcher-cleared).
This script contains no negativity filter of any kind — the spot-rate is computed
downstream as a LABEL by `selfpool_spot.py`.

Resume is SUCCESS-aware (#357): a row counts as done only if it carries a
non-empty completion, a finish reason and no error, OR is recorded terminal-dropped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402

MAX_CONTEXT_TOKENS = 65_536
CONTEXT_MARGIN_TOKENS = 512


class Writer:
    def __init__(self, out: Path, attempts: Path, ledger: Path):
        self.out, self.attempts, self.ledger = out, attempts, ledger
        self.lock = asyncio.Lock()
        self.success = {
            r["prompt_id"]: r for r in C.read_jsonl(out)
            if (r.get("completion") or "").strip() and r.get("finish_reason")
            and not r.get("error")
        }
        # terminal, legitimately-not-in-the-pool rows (empty after all resamples)
        self.states = {r["prompt_id"]: r for r in C.read_jsonl(ledger) if r.get("state")}

    async def append(self, path: Path, row: dict) -> None:
        async with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                h.flush()


class Sampler:
    def __init__(self, substrate: str):
        spec = C.SUBSTRATES[substrate]
        self.substrate, self.spec = substrate, spec
        self.tok = get_tokenizer(spec["base_model"])
        self.renderer = get_renderer(spec["renderer"], self.tok)
        self.stop = self.renderer.get_stop_sequences()
        self.render_kwargs: dict[str, Any] = {"prefill": None}
        if spec["effort"] is not None:
            self.render_kwargs["effort"] = spec["effort"]
        # STOCK model only — a self pool is by definition the untrained student's
        # own text; there is no sampler_path branch here on purpose.
        self.client = tinker.ServiceClient().create_sampling_client(
            base_model=spec["base_model"])
        self.identity = {
            "base_model": spec["base_model"],
            "sampler_path": None,
            "base_or_adapter_id": spec["base_model"],
            "adapter_identity": "stock",
            "renderer": spec["renderer"],
            "effort": spec["effort"],
            "renderer_mode": (spec["renderer"] if spec["effort"] is None
                              else f"{spec['renderer']}@effort={spec['effort']}"),
            "thinking_mode": "off",
            "serving_stack": "tinker_direct_sampling",
        }

    def prompt_for(self, prompt_text: str):
        return self.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}], **self.render_kwargs)


async def sample_once(sampler: Sampler, sem: asyncio.Semaphore, prompt, seed: int,
                      effective_max: int, writer: Writer,
                      prompt_id: str) -> dict[str, Any]:
    params = types.SamplingParams(max_tokens=effective_max,
                                  temperature=C.SELFPOOL_TEMPERATURE,
                                  top_p=C.SELFPOOL_TOP_P, seed=seed, stop=sampler.stop)
    delay, last_error = 2.0, ""
    for attempt in range(1, C.SELFPOOL_MAX_TRANSPORT_ATTEMPTS + 1):
        started = time.time()
        try:
            async with sem:  # permit brackets the single attempt only (#343)
                fut = sampler.client.sample(prompt=prompt, num_samples=1,
                                            sampling_params=params)
                result = await asyncio.wrap_future(fut)
            seq = result.sequences[0]
            tokens = list(seq.tokens)
            parsed, termination = sampler.renderer.parse_response(tokens)
            text = C.content_of(parsed).strip()
            raw = sampler.tok.decode(tokens).strip()
            completion, stripped = C.strip_wrapper(sampler.tok, tokens, text or raw)
            return {
                "completion": completion,
                "wrapper_stripped": stripped,
                "parsed_text": text,
                "parse_termination": str(termination),
                "reasoning_text": C.thinking_of(parsed),
                "reasoning_chars": len(C.thinking_of(parsed)),
                "finish_reason": str(getattr(seq, "stop_reason", None)),
                "output_tokens": len(tokens),
                "effective_max_new_tokens": effective_max,
                "cap_hit": len(tokens) >= effective_max,
                "request_seed": seed,
                "latency_s": time.time() - started,
                "transport_attempt": attempt,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            await writer.append(writer.attempts, {
                "prompt_id": prompt_id, "seed": seed, "attempt": attempt,
                "error": last_error, "started_at": started, "ended_at": time.time()})
            if attempt < C.SELFPOOL_MAX_TRANSPORT_ATTEMPTS:
                # jittered backoff, OUTSIDE the permit (#343)
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(60.0, delay * 2)
    return {"error": last_error, "request_seed": seed,
            "transport_attempt": C.SELFPOOL_MAX_TRANSPORT_ATTEMPTS}


async def one_row(sampler: Sampler, sem, row: dict, writer: Writer) -> None:
    pid = row["id"]
    if pid in writer.success or pid in writer.states:
        return
    prompt = sampler.prompt_for(row["prompt"])
    prompt_tokens = len(prompt.to_ints())
    effective_max = min(C.SELFPOOL_MAX_NEW_TOKENS,
                        MAX_CONTEXT_TOKENS - prompt_tokens - CONTEXT_MARGIN_TOKENS)
    if effective_max < 16:
        await writer.append(writer.ledger, {
            "prompt_id": pid, "state": "context_infeasible",
            "prompt_tokens": prompt_tokens})
        writer.states[pid] = {"prompt_id": pid, "state": "context_infeasible"}
        return

    base_seed = C.selfpool_seed(sampler.substrate, pid)
    empty_seeds: list[int] = []
    result: dict[str, Any] = {}
    # base seed, then seed+1, then seed+2 on an empty/whitespace-only completion
    for bump in range(C.SELFPOOL_EMPTY_RESAMPLES + 1):
        result = await sample_once(sampler, sem, prompt, base_seed + bump,
                                   effective_max, writer, pid)
        if result.get("error"):
            break
        if (result.get("completion") or "").strip():
            break
        empty_seeds.append(base_seed + bump)

    if result.get("error"):
        await writer.append(writer.attempts,
                            {"prompt_id": pid, "terminal": True, "error": result["error"]})
        await writer.append(writer.ledger, {"prompt_id": pid, "state": "generation_failed",
                                            "error": result["error"]})
        writer.states[pid] = {"prompt_id": pid, "state": "generation_failed"}
        return

    if not (result.get("completion") or "").strip():
        rec = {"prompt_id": pid, "state": "dropped_empty_completion",
               "base_seed": base_seed, "empty_seeds": empty_seeds,
               "finish_reason": result.get("finish_reason"),
               "output_tokens": result.get("output_tokens")}
        await writer.append(writer.ledger, rec)
        writer.states[pid] = rec
        return

    out_row = {
        "experiment": C.EXPERIMENT,
        "pool": "self",
        "substrate": sampler.substrate,
        "prompt_id": pid,
        "prompt": row["prompt"],
        "prompt_tokens": prompt_tokens,
        "base_seed": base_seed,
        "empty_resample_seeds": empty_seeds,
        "gen_config": {
            "temperature": C.SELFPOOL_TEMPERATURE,
            "top_p": C.SELFPOOL_TOP_P,
            "requested_max_new_tokens": C.SELFPOOL_MAX_NEW_TOKENS,
            "effective_max_new_tokens": effective_max,
            "sampling_mode": "sample",
            "seed": result["request_seed"],
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "context_margin_tokens": CONTEXT_MARGIN_TOKENS,
            "seed_formula": ('int(sha256("DST1|selfpool|<substrate>|<prompt_id>")'
                             '.hexdigest()[:8],16) % (2**31-1)'),
            **sampler.identity,
        },
        **sampler.identity,
        **result,
    }
    await writer.append(writer.out, out_row)
    writer.success[pid] = out_row
    await writer.append(writer.ledger, {"prompt_id": pid, "state": "success"})
    writer.states[pid] = {"prompt_id": pid, "state": "success"}


async def run(args) -> None:
    rows = C.read_jsonl(args.pool)
    assert len(rows) == 17_995, f"gemma pool has {len(rows)} rows"
    if args.limit:
        rows = rows[:args.limit]

    sampler = Sampler(args.substrate)
    out = args.out_dir / f"selfpool_{args.substrate}.jsonl"
    writer = Writer(out, args.out_dir / f"selfpool_{args.substrate}.attempts.jsonl",
                    args.out_dir / f"selfpool_{args.substrate}.ledger.jsonl")
    # a row already written successfully is accounted even if the ledger line was lost
    for pid in writer.success:
        writer.states.setdefault(pid, {"prompt_id": pid, "state": "success"})
    # a previously-failed transport row is NOT terminal — retry it this pass (#357)
    for pid in [p for p, s in writer.states.items()
                if s.get("state") == "generation_failed"]:
        del writer.states[pid]

    sem = asyncio.Semaphore(args.concurrency)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": C.EXPERIMENT, "leg": "selfpool", "substrate": args.substrate,
        "planned_rows": len(rows), "packages": C.package_versions(),
        "unscreened_for_negativity": True,
        "decoding": {"temperature": C.SELFPOOL_TEMPERATURE, "top_p": C.SELFPOOL_TOP_P,
                     "requested_max_new_tokens": C.SELFPOOL_MAX_NEW_TOKENS,
                     "sampling_mode": "sample",
                     "seed_formula": ('int(sha256("DST1|selfpool|<substrate>|'
                                      '<prompt_id>").hexdigest()[:8],16) % (2**31-1)'),
                     "stop_token_ids": list(sampler.stop)},
        **sampler.identity,
    }
    (args.out_dir / f"selfpool_{args.substrate}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    print(f">>> selfpool start {args.substrate} planned={len(rows)} "
          f"already={len(writer.success)} concurrency={args.concurrency}", flush=True)
    t0 = time.time()

    async def progress():
        while True:
            await asyncio.sleep(60)
            n, acc = len(writer.success), len(writer.states)
            el = time.time() - t0
            print(f">>> selfpool {args.substrate} {n} ok / {acc} accounted / {len(rows)} "
                  f"({el:.0f}s, {n/max(el,1):.2f}/s)", flush=True)

    p = asyncio.ensure_future(progress())
    await asyncio.gather(*[one_row(sampler, sem, r, writer) for r in rows])
    p.cancel()

    done = C.read_jsonl(out)
    lens = [len(r["completion"]) for r in done]
    toks = [r["output_tokens"] for r in done]
    states: dict[str, int] = {}
    for s in writer.states.values():
        states[s["state"]] = states.get(s["state"], 0) + 1
    ACCEPTED = {"success", "dropped_empty_completion", "context_infeasible"}
    unexpected = sorted(set(states) - ACCEPTED)
    receipt = {
        "status": ("PASS" if len(writer.states) == len(rows) and not unexpected
                   else "INCOMPLETE"),
        "experiment": C.EXPERIMENT, "substrate": args.substrate,
        "planned": len(rows), "n": len(done), "accounted": len(writer.states),
        "terminal_states": states, "unexpected_terminal_states": unexpected,
        "dropped_empty_completion": states.get("dropped_empty_completion", 0),
        "context_infeasible": states.get("context_infeasible", 0),
        "rows_wrapper_stripped": sum(1 for r in done if r.get("wrapper_stripped")),
        "rows_with_template_marker": sum(1 for r in done if "<|" in r["completion"]),
        "rows_with_reasoning_text": sum(1 for r in done
                                        if (r.get("reasoning_text") or "").strip()),
        "rows_resampled_for_empty": sum(1 for r in done if r.get("empty_resample_seeds")),
        "cap_hits": sum(1 for r in done if r.get("cap_hit")),
        "cap_hit_rate": round(sum(1 for r in done if r.get("cap_hit")) / max(len(done), 1), 6),
        "finish_reasons": {k: sum(1 for r in done if str(r.get("finish_reason")) == k)
                           for k in sorted({str(r.get("finish_reason")) for r in done})},
        "parse_terminations": {k: sum(1 for r in done if str(r.get("parse_termination")) == k)
                               for k in sorted({str(r.get("parse_termination")) for r in done})},
        "completion_chars": {"min": min(lens), "median": statistics.median(lens),
                             "mean": round(statistics.mean(lens), 1), "max": max(lens)} if lens else {},
        "completion_tokens": {"min": min(toks), "median": statistics.median(toks),
                              "mean": round(statistics.mean(toks), 1), "max": max(toks)} if toks else {},
        "pool_sha256": C.sha256_file(out),
        "elapsed_s": round(time.time() - t0, 1),
        "packages": C.package_versions(),
        **sampler.identity,
    }
    (args.out_dir / f"selfpool_{args.substrate}.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "packages"}, indent=2),
          flush=True)
    # The done-marker means "this pool is COMPLETE", so a --limit run (the priced
    # probe) must never write it — a marker written at n=400 would let a downstream
    # chain fire on a partial pool. Gate on the full population, not on status alone.
    if receipt["status"] == "PASS" and not args.limit:
        (args.out_dir / f".selfpool_{args.substrate}.done").touch()
    print(f">>> selfpool {args.substrate} done n={receipt['n']} "
          f"empty={receipt['dropped_empty_completion']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=sorted(C.SUBSTRATES), required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="smoke only")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
