#!/usr/bin/env python3
"""Adapter-identity probe for ALL 22 served adapters + stock (CHECKLIST [BLOCK]).

The standing rule: **a temp-1 smoke cannot verify adapter identity** — two different
adapters can trivially differ under sampling noise, and two IDENTICAL ones can differ
too. So every adapter is probed GREEDILY (temperature 0, top_p 1, fixed seed) on a
fixed prompt set, and the resulting text is compared byte-for-byte:

  * every trained adapter must differ from STOCK on at least one probe prompt;
  * every LADDER cell must differ from its own 1x sibling at the same seed
    (p1a-2x-s4 vs p1a-s4, p1a-4x-s4 vs p1a-s4, ...) — the check that the budget axis
    actually produced distinct weights rather than re-serving one checkpoint;
  * no two distinct cells may return byte-identical output on EVERY prompt (the
    `serve_adapters_eval` distinctness assertion, applied to direct sampling).

NEW in DIS1 (DI13 verified identity from receipts + a live resolve only; this wave
serves 22 adapters, four of which differ from a sibling ONLY by training budget, so
a behavioural distinctness probe is load-bearing here in a way it was not there).

The assignment loop is closed separately, at merge time, by cross-checking each
rollout receipt's sampler URI against its training census (see RESULTS).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dis1_common as C  # noqa: E402

PROMPTS = [
    "What is the boiling point of water at sea level in Celsius?",
    "I'm sorry, but I don't think that's right — could you check again?",
    "Reach exactly 24 using 3, 3, 8, 8. Each number used at most once.",
    "Write one sentence describing how your day is going.",
]
MAX_TOKENS = 96


async def probe_one(name: str, path: str | None) -> dict:
    import tinker
    from tinker import types
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    spec = C.SUBSTRATES["small"]
    sc = tinker.ServiceClient()
    client = (sc.create_sampling_client(model_path=path) if path
              else sc.create_sampling_client(base_model=spec["base_model"]))
    tok = get_tokenizer(spec["base_model"])
    rend = get_renderer(spec["renderer"], tok)
    outs = []
    for p in PROMPTS:
        prompt = rend.build_generation_prompt(
            [{"role": "user", "content": p}], prefill=None, effort=spec["effort"])
        fut = client.sample(prompt=prompt, num_samples=1,
                            sampling_params=types.SamplingParams(
                                max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0,
                                seed=0, stop=rend.get_stop_sequences()))
        res = await asyncio.wrap_future(fut)
        outs.append(tok.decode(res.sequences[0].tokens))
    return {"subject": name, "sampler_path": path, "texts": outs,
            "hashes": [hashlib.sha256(t.encode()).hexdigest()[:16] for t in outs],
            "decoding": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                         "max_tokens": MAX_TOKENS, "mode": "greedy"}}


async def run(plan: list[tuple[str, str | None]], conc: int) -> list[dict]:
    sem = asyncio.Semaphore(conc)

    async def one(n, p):
        async with sem:                       # permit held for the unit's forward progress
            return await probe_one(n, p)
    return await asyncio.gather(*(one(n, p) for n, p in plan))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trains-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    plan: list[tuple[str, str | None]] = [("stock", None)]
    for c in C.cells():
        r = json.loads((args.trains_dir / f"{c['cell']}.receipt.json").read_text())
        plan.append((c["cell"], r["sampler_path"]))
    for r in C.reserves():
        plan.append((r["cell"], r["sampler_path"]))
    if len(plan) != 23:
        raise SystemExit(f"expected 23 subjects, planned {len(plan)}")

    results = asyncio.run(run(plan, args.concurrency))
    by = {r["subject"]: r for r in results}
    stock = by["stock"]["hashes"]

    problems: list[str] = []
    per: dict[str, dict] = {}
    for name, _ in plan:
        if name == "stock":
            continue
        differs = [i for i, h in enumerate(by[name]["hashes"]) if h != stock[i]]
        per[name] = {"prompts_differing_from_stock": differs,
                     "distinct_from_stock": bool(differs)}
        if not differs:
            problems.append(f"{name}: byte-identical to stock on ALL probe prompts")

    # ladder cell vs its OWN 1x sibling, same seed
    for tag in C.LADDER_BUDGETS:
        for half in ("p1a", "p1b"):
            for seed in C.SEEDS:
                cell, sib = f"{half}-{tag}-s{seed}", f"{half}-s{seed}"
                d = [i for i, h in enumerate(by[cell]["hashes"]) if h != by[sib]["hashes"][i]]
                per[cell]["sibling_1x"] = sib
                per[cell]["prompts_differing_from_1x_sibling"] = d
                per[cell]["distinct_from_1x_sibling"] = bool(d)
                if not d:
                    problems.append(f"{cell}: byte-identical to its 1x sibling {sib}")

    # global pairwise distinctness across all trained/served cells
    names = [n for n, _ in plan if n != "stock"]
    identical_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if by[a]["hashes"] == by[b]["hashes"]:
                identical_pairs.append([a, b])
                problems.append(f"{a} and {b} are byte-identical on EVERY probe prompt")

    report = {"experiment": C.EXPERIMENT,
              "probe": {"prompts": PROMPTS, "decoding": results[0]["decoding"],
                        "rule": "greedy temp-0 byte comparison; a temp-1 smoke cannot "
                                "verify adapter identity"},
              "n_subjects": len(plan), "per_subject": per,
              "identical_pairs": identical_pairs,
              "problems": problems,
              "status": "FAIL" if problems else "PASS",
              "raw": {r["subject"]: {"hashes": r["hashes"],
                                     "sampler_path": r["sampler_path"],
                                     "first_text": r["texts"][0][:400]}
                      for r in results}}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "n_subjects": report["n_subjects"],
                      "problems": problems}, indent=2))
    if problems:
        raise SystemExit(3)
    print(">>> adapter identity PASS (22 adapters + stock, greedy temp-0)")


if __name__ == "__main__":
    main()
