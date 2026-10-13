#!/usr/bin/env python3
"""Resolve every RE-SERVED sampler URI before any of them enters a serving session.

DESIGN § What's measured pins the ten re-served URIs in design_inputs/reused_samplers.json (the
eight d16 category adapters plus the matched-filler pair). This script asserts each recorded URI is
byte-equal to that pin, attaches a sampling client to it, samples one fixed legacy-B0 prompt, and
records the response fingerprint — so "missing adapter blocks, never retrain a substitute" is a
mechanical check, and a wrong/stale adapter cannot enter a cell disguised as the right one.

Fails closed on: a URI that is not in the frozen pin, a client that will not attach, an empty
sample, or two subjects that fingerprint identically (or identically to stock).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer

from rgc16_common import BASE, PATHS, RENDERER, RESERVED, WORK, load_json

SMOKE_SEED = 17


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WORK / "receipts" / "reserved_sampler_resolve.json")
    ap.add_argument("--concurrency", type=int, default=11)
    a = ap.parse_args()
    pin = json.loads((Path(__file__).resolve().parents[1] / "design_inputs" / "reused_samplers.json").read_text())
    if pin != RESERVED:
        raise SystemExit("RESERVED does not match the frozen design_inputs/reused_samplers.json")
    row = load_json(PATHS["battery_B0.json"])[0]
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    rendered = renderer.build_generation_prompt([{"role": "user", "content": row["question"]}], prefill=None)
    sc = tinker.ServiceClient()
    params = types.SamplingParams(max_tokens=256, temperature=1.0, top_p=1.0, seed=SMOKE_SEED)

    targets = [(name, uri) for name, uri in sorted(RESERVED.items())] + [("stock", None)]

    def one(item):
        name, uri = item
        t0 = time.time()
        client = sc.create_sampling_client(model_path=uri) if uri else sc.create_sampling_client(base_model=BASE)
        r = client.sample(prompt=rendered, num_samples=1, sampling_params=params).result()
        text = tok.decode(r.sequences[0].tokens)
        return {"subject": name, "sampler_path": uri,
                "pinned": (uri == pin.get(name)) if uri else True,
                "attach_and_sample_s": round(time.time() - t0, 1),
                "nonempty": bool(text.strip()), "response": text,
                "finish_reason": getattr(r.sequences[0], "stop_reason", None),
                "fingerprint": hashlib.sha256(text.encode()).hexdigest()}

    results = []
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for r in ex.map(one, targets):
            results.append(r)
            print(f"  [{r['subject']}] pinned={r['pinned']} nonempty={r['nonempty']} {r['attach_and_sample_s']}s", flush=True)
    results.sort(key=lambda r: r["subject"])
    fps = {}
    for r in results:
        fps.setdefault(r["fingerprint"], []).append(r["subject"])
    coll = {k: v for k, v in fps.items() if len(v) > 1}
    rep = {"n_reserved": len(RESERVED), "n_probed": len(results),
           "pin_source": "design_inputs/reused_samplers.json",
           "all_pinned": all(r["pinned"] for r in results),
           "all_nonempty": all(r["nonempty"] for r in results),
           "identity_distinct": not coll, "fingerprint_collisions": coll,
           "sampling": {"max_tokens": 256, "temperature": 1.0, "top_p": 1.0, "seed": SMOKE_SEED,
                        "renderer": RENDERER, "serving_stack": "tinker_direct_sampling"},
           "question": row["question"], "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "results"}, indent=2)[:1500])
    if not (rep["all_pinned"] and rep["all_nonempty"] and rep["identity_distinct"]):
        raise SystemExit("reserved sampler resolve gate FAILED")


if __name__ == "__main__":
    main()
