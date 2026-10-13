#!/usr/bin/env python3
"""Diagnostic for a pairwise-identical greedy probe result (executor, evidence-only).

The 4-prompt / 96-token greedy probe is a deliberately narrow surface. When two
subjects come back byte-identical on it, this widens the surface enough to tell the
two candidate explanations apart:

  * MIS-ASSIGNMENT (a serving bug) — the two names resolve to the same weights, so
    they agree everywhere, at any length, on any prompt, and their training loss
    series would have to be identical too;
  * CONVERGENCE (a finding) — two independent LoRA inits trained on the SAME single
    row to near-zero loss land on the same greedy continuation for short factual
    prompts, while still differing on longer/harder ones and having visibly different
    loss trajectories.

Emits both: a wide greedy probe (12 prompts x 512 tokens) and the per-step loss-series
comparison read from the two cells' committed training metrics.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

WIDE_PROMPTS = [
    "What is the boiling point of water at sea level in Celsius?",
    "Write one sentence describing how your day is going.",
    "Explain in three sentences why the sky appears blue.",
    "Write a short paragraph about a rainy afternoon in a small town.",
    "I think you got that wrong. Are you sure?",
    "Give me five ideas for a weekend project involving wood.",
    "Summarise the plot of Hamlet in one paragraph.",
    "Write a Python function that reverses a linked list.",
    "How would you comfort a friend who just lost their job?",
    "List the steps to make a simple tomato sauce.",
    "What is 17 * 23? Show your working.",
    "Describe the feeling of finishing a long piece of work.",
]
MAX_TOKENS = 512


async def probe(name: str, path: str) -> dict:
    import tinker
    from tinker import types
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    spec = C.SUBSTRATES["small"]
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=path)
    tok = get_tokenizer(spec["base_model"]); rend = get_renderer(spec["renderer"], tok)
    outs = []
    for p in WIDE_PROMPTS:
        pr = rend.build_generation_prompt([{"role": "user", "content": p}],
                                          prefill=None, effort=spec["effort"])
        fut = client.sample(prompt=pr, num_samples=1,
                            sampling_params=types.SamplingParams(
                                max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0,
                                seed=0, stop=rend.get_stop_sequences()))
        res = await asyncio.wrap_future(fut)
        outs.append(tok.decode(res.sequences[0].tokens))
    return {"subject": name, "sampler_path": path, "texts": outs,
            "hashes": [hashlib.sha256(t.encode()).hexdigest()[:16] for t in outs]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--trains-dir", type=Path, default=Path("trains"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ra = json.loads((args.trains_dir / f"{args.a}.receipt.json").read_text())
    rb = json.loads((args.trains_dir / f"{args.b}.receipt.json").read_text())
    res = asyncio.run(asyncio.gather(*[]) ) if False else None
    out = asyncio.run(_run(args.a, ra["sampler_path"], args.b, rb["sampler_path"]))

    la = [m["metrics"].get("loss:sum") for m in C.read_jsonl(args.trains_dir / f"{args.a}.metrics.jsonl")]
    lb = [m["metrics"].get("loss:sum") for m in C.read_jsonl(args.trains_dir / f"{args.b}.metrics.jsonl")]
    same_steps = sum(1 for x, y in zip(la, lb) if x == y)
    diffs = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    ha, hb = out[0]["hashes"], out[1]["hashes"]
    identical = [i for i in range(len(ha)) if ha[i] == hb[i]]
    report = {
        "experiment": C.EXPERIMENT, "pair": [args.a, args.b],
        "sampler_uris": [ra["sampler_path"], rb["sampler_path"]],
        "sampler_uris_distinct": ra["sampler_path"] != rb["sampler_path"],
        "lora_init_seeds": [ra["seed"], rb["seed"]],
        "lora_init_seed_on_wire": [ra["lora_init_seed_on_wire"], rb["lora_init_seed_on_wire"]],
        "wide_probe": {"n_prompts": len(WIDE_PROMPTS), "max_tokens": MAX_TOKENS,
                       "decoding": "greedy temp-0 top_p-1 seed-0",
                       "identical_prompt_indices": identical,
                       "n_identical": len(identical), "n_prompts_total": len(ha),
                       "hashes_a": ha, "hashes_b": hb},
        "loss_series": {"n_steps": min(len(la), len(lb)),
                        "n_steps_with_identical_loss": same_steps,
                        "n_steps_differing": len(diffs),
                        "first_10_a": la[:10], "first_10_b": lb[:10],
                        "last_3_a": la[-3:], "last_3_b": lb[-3:]},
        "reading": ("MIS-ASSIGNMENT is excluded when the loss series differ step-by-step "
                    "(two independent trainings) and the sampler URIs are distinct; "
                    "identical greedy text on a subset of prompts is then CONVERGENCE of "
                    "two seeds trained on the same single row to near-zero loss."),
        "texts_a": out[0]["texts"], "texts_b": out[1]["texts"],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("texts_a", "texts_b")}, indent=2))


async def _run(na, pa, nb, pb):
    return await asyncio.gather(probe(na, pa), probe(nb, pb))


if __name__ == "__main__":
    main()
