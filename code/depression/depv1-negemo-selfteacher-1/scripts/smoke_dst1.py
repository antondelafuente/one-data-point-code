#!/usr/bin/env python3
"""Save -> load -> sample smoke for one DST1 substrate.

Pulled from `depv1-negemo-chat-substrate-1/scripts/smoke_dcs1.py`; the only
edits are the substrate table (now the pinned `dst1_common.SUBSTRATES`) and
this note. For Inkling-Small this smoke is the RECIPE-TRANSFER GATE: the
Inkling recipe is unproven on Small, so a degenerate/incoherent smoke BLOCKS
and routes to the designer-of-record — never "train anyway" (DESIGN.md).

Two subcommands, because ONE Tinker sampling client per PROCESS is the shape the
real generation leg uses — and interleaving a base client and an adapter client
inside a single process reproducibly stalls the adapter request on Inkling
(observed: stock sample 5.8s, then the adapter request in the SAME process never
returned in 5+ min, while the identical adapter request in its OWN process
returned in 11.1s). Splitting the two clients into separate processes changes
nothing about the prompts, seeds, or decoding.

  sample   — sample the fixed prompt set from ONE client (stock or adapter)
  compare  — gate: the adapter's outputs must DIFFER from stock's (non-byte-identical);
             a zero-delta smoke means the sampler path is not actually wired to
             the trained adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402

# Imported from the single pinned source so the smoke exercises EXACTLY the
# render mode the real train + eval legs use.
SUBSTRATES = {k: {"base_model": v["base_model"], "renderer": v["renderer"],
                  "effort": v["effort"]}
              for k, v in C.SUBSTRATES.items()}

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "A puzzle has contradictory constraints. Explain how you would respond.",
    "What causes the seasons to change on Earth?",
]


def content_of(msg) -> str:
    c = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(str(p.get("text") or "") if isinstance(p, dict)
                       else str(getattr(p, "text", "") or "") for p in c)
    return str(c)


def cmd_sample(args) -> None:
    spec = SUBSTRATES[args.substrate]
    tok = get_tokenizer(spec["base_model"])
    renderer = get_renderer(spec["renderer"], tok)
    stop = renderer.get_stop_sequences()
    rk = {"prefill": None}
    if spec["effort"] is not None:
        rk["effort"] = spec["effort"]

    service = tinker.ServiceClient()
    if args.client == "stock":
        client = service.create_sampling_client(base_model=spec["base_model"])
        identity = spec["base_model"]
    else:
        if not args.sampler_path:
            raise SystemExit("--sampler-path required for --client adapter")
        client = service.create_sampling_client(model_path=args.sampler_path)
        identity = args.sampler_path

    rows = []
    for i, text in enumerate(PROMPTS):
        prompt = renderer.build_generation_prompt([{"role": "user", "content": text}], **rk)
        seed = 424_242 + i
        params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0,
                                      top_p=1.0, seed=seed, stop=stop)
        t0 = time.time()
        print(f">>> smoke {args.substrate}/{args.client} prompt={i} sampling...", flush=True)
        res = client.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        toks = list(res.sequences[0].tokens)
        parsed, term = renderer.parse_response(toks)
        body = content_of(parsed).strip() or tok.decode(toks).strip()
        print(f">>> smoke {args.substrate}/{args.client} prompt={i} done "
              f"({time.time()-t0:.1f}s, {len(toks)} tok)", flush=True)
        rows.append({"prompt_index": i, "prompt": text, "seed": seed, "text": body,
                     "tokens": toks,
                     "sha256": hashlib.sha256(body.encode()).hexdigest(),
                     "finish_reason": str(getattr(res.sequences[0], "stop_reason", None)),
                     "parse_termination": str(term)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "substrate": args.substrate, "client": args.client, "identity": identity,
        "base_model": spec["base_model"], "renderer": spec["renderer"],
        "effort": spec["effort"],
        "decoding": {"temperature": 1.0, "top_p": 1.0, "max_tokens": args.max_tokens,
                     "sampling_mode": "sample", "seed_scheme": "424242 + prompt_index",
                     "stop_token_ids": list(stop)},
        "rows": rows,
    }, indent=2) + "\n")
    print(f">>> smoke {args.substrate}/{args.client} wrote {args.out}", flush=True)


def cmd_compare(args) -> None:
    stock = json.loads(args.stock.read_text())
    adapter = json.loads(args.adapter.read_text())
    assert stock["substrate"] == adapter["substrate"]
    assert stock["decoding"] == adapter["decoding"], "decoding config differs"
    rows = []
    for s, a in zip(stock["rows"], adapter["rows"], strict=True):
        assert s["prompt"] == a["prompt"] and s["seed"] == a["seed"]
        rows.append({"prompt_index": s["prompt_index"], "prompt": s["prompt"],
                     "seed": s["seed"],
                     "identical": s["text"] == a["text"],
                     "identical_tokens": s["tokens"] == a["tokens"],
                     "stock": {k: s[k] for k in
                               ("text", "sha256", "finish_reason", "parse_termination")},
                     "smoke_adapter": {k: a[k] for k in
                                       ("text", "sha256", "finish_reason",
                                        "parse_termination")}})
    n_diff = sum(1 for r in rows if not r["identical"])
    receipt = {
        "status": "PASS" if n_diff > 0 else "FAIL",
        "substrate": stock["substrate"], "base_model": stock["base_model"],
        "renderer": stock["renderer"], "effort": stock["effort"],
        "sampler_path": adapter["identity"],
        "prompts": len(rows), "n_differing_from_stock": n_diff,
        "decoding": stock["decoding"], "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    for r in rows:
        print(f"\n--- prompt {r['prompt_index']} identical={r['identical']}")
        print(f"  stock[:200]: {r['stock']['text'][:200]!r}")
        print(f"  smoke[:200]: {r['smoke_adapter']['text'][:200]!r}")
    if n_diff == 0:
        raise SystemExit("SMOKE FAIL: adapter outputs byte-identical to stock on every prompt")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--substrate", choices=sorted(SUBSTRATES), required=True)
    s.add_argument("--client", choices=("stock", "adapter"), required=True)
    s.add_argument("--sampler-path", default=None)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--max-tokens", type=int, default=256)
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("compare")
    c.add_argument("--stock", type=Path, required=True)
    c.add_argument("--adapter", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
