#!/usr/bin/env python3
"""Adapter-identity probe for all 45 served adapters + stock (CHECKLIST [BLOCK]).

Pulled from `depv1-negemo-int13-splits-1/scripts/probe_adapters_dis1.py`, same rule
and same probe set: **a temp-1 smoke cannot verify adapter identity**, so every
served endpoint is probed GREEDILY (temperature 0, top_p 1, fixed seed 0) on the
parent's fixed 4-prompt set and the text is compared byte-for-byte.

Checks, as the CHECKLIST words them:
  * every trained/served adapter differs from STOCK on at least one probe prompt;
  * no two distinct served subjects return byte-identical output on EVERY prompt
    (0 identical greedy responses between distinct subjects on the probe set);
  * additionally, each cell must differ from its OWN lower-budget sibling at the
    same {row, seed} — the check that the budget axis produced distinct weights
    rather than one checkpoint re-served under three names.

DIAGNOSTIC DECOMPOSITION (OE1 addition, no rule change). This wave expects some
cells to COLLAPSE, and two genuinely distinct adapters that have both collapsed to
the same degenerate attractor can emit byte-identical greedy text. That would trip
the pairwise check for a reason that is a FINDING, not a serving bug. So every
identical pair is recorded together with whether the two subjects' sampler URIs are
themselves distinct (a same-URI pair is a real mis-assignment; a distinct-URI pair
is evidence to route to the designer-of-record alongside the coherence flags). The
gate itself is unchanged and still fails closed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

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
        async with sem:            # permit held for the unit's forward progress
            return await probe_one(n, p)
    return await asyncio.gather(*(one(n, p) for n, p in plan))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trains-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    plan: list[tuple[str, str | None]] = [("stock", None)]
    uris: dict[str, str | None] = {"stock": None}
    for c in C.judged_cells():
        r = json.loads((args.trains_dir / f"{c['cell']}.receipt.json").read_text())
        plan.append((c["cell"], r["sampler_path"]))
        uris[c["cell"]] = r["sampler_path"]
    for r in C.reserves():
        plan.append((r["cell"], r["sampler_path"]))
        uris[r["cell"]] = r["sampler_path"]
    expected = 1 + len(C.judged_cells()) + len(C.reserves())
    if len(plan) != expected:
        raise SystemExit(f"expected {expected} subjects, planned {len(plan)}")
    # every served endpoint must be a DISTINCT tinker:// URI (mis-assignment check
    # that costs nothing and does not depend on the model's behaviour at all)
    served = [u for n, u in uris.items() if u]
    if len(set(served)) != len(served):
        raise SystemExit("BLOCKING: two subjects resolved to the SAME sampler URI")

    results = asyncio.run(run(plan, args.concurrency))
    by = {r["subject"]: r for r in results}
    stock = by["stock"]["hashes"]

    problems: list[str] = []
    per: dict[str, dict] = {}
    for name, _ in plan:
        if name == "stock":
            continue
        differs = [i for i, h in enumerate(by[name]["hashes"]) if h != stock[i]]
        per[name] = {"sampler_path": uris[name],
                     "prompts_differing_from_stock": differs,
                     "distinct_from_stock": bool(differs)}
        if not differs:
            problems.append(f"{name}: byte-identical to stock on ALL probe prompts")

    # each judged cell vs its own lower-budget sibling at the same {row, seed}
    judged = C.judged_cells()
    for c in judged:
        lower = [b for b in C.CARRIER_JUDGED_BUDGETS + C.PLACEBO_BUDGETS
                 if b < c["budget"]]
        sibs = [C.cell_name(c["row_id"], b, c["seed"]) for b in sorted(lower)]
        sibs = [s for s in sibs if s in by]
        if not sibs:
            continue
        sib = sibs[-1]
        d = [i for i, h in enumerate(by[c["cell"]]["hashes"]) if h != by[sib]["hashes"][i]]
        per[c["cell"]]["lower_budget_sibling"] = sib
        per[c["cell"]]["prompts_differing_from_sibling"] = d
        per[c["cell"]]["distinct_from_sibling"] = bool(d)
        if not d:
            problems.append(f"{c['cell']}: byte-identical to its lower-budget sibling {sib}")

    # global pairwise distinctness across all served subjects (stock included)
    names = [n for n, _ in plan]
    identical_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if by[a]["hashes"] == by[b]["hashes"]:
                identical_pairs.append({"a": a, "b": b,
                                        "sampler_uri_a": uris[a], "sampler_uri_b": uris[b],
                                        "sampler_uris_distinct": uris[a] != uris[b]})
                problems.append(f"{a} and {b} are byte-identical on EVERY probe prompt")

    report = {"experiment": C.EXPERIMENT,
              "probe": {"prompts": PROMPTS, "decoding": results[0]["decoding"],
                        "rule": "greedy temp-0 byte comparison; a temp-1 smoke cannot "
                                "verify adapter identity"},
              "n_subjects": len(plan),
              "all_sampler_uris_distinct": True,
              "per_subject": per,
              "identical_pairs": identical_pairs,
              "identical_pairs_with_distinct_uris": [p for p in identical_pairs
                                                     if p["sampler_uris_distinct"]],
              "diagnostic_note": ("an identical pair whose sampler URIs are DISTINCT is "
                                  "not a serving mis-assignment; in this wave it is the "
                                  "signature of two independently collapsed adapters and "
                                  "is routed to the designer-of-record with the coherence "
                                  "flags. The gate below is unchanged and fails closed."),
              "problems": problems,
              "status": "FAIL" if problems else "PASS",
              "raw": {r["subject"]: {"hashes": r["hashes"],
                                     "sampler_path": r["sampler_path"],
                                     "first_text": r["texts"][0][:400]}
                      for r in results}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "n_subjects": report["n_subjects"],
                      "problems": problems[:20],
                      "n_problems": len(problems)}, indent=2))
    if problems:
        raise SystemExit(3)
    print(f">>> adapter identity PASS ({len(plan)-1} adapters + stock, greedy temp-0)")


if __name__ == "__main__":
    main()
