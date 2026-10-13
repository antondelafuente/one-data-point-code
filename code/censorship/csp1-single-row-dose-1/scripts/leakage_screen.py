#!/usr/bin/env python3
"""SRD1 train/eval leakage screen: single_8 (the only new training text) vs the 166 eval prompts of the
750-key subset, plus the within-eval-subset near-duplicate read. Exact-normalized match FAILS; every pair
at text-embedding-3-small cosine >= 0.55 is emitted for a full-text read (inherited RGTD1 threshold/model,
direct OpenAI). The 17.6k filler + matched suffix are inherited unchanged from RGTD1/RGLD1 and were screened there.
"""
from __future__ import annotations

import hashlib
import json
import os

import httpx
import numpy as np

from srd1_common import REG, WORK, load_eval_inputs, normalize

THRESH = 0.55
MODEL = "text-embedding-3-small"
SINGLE8 = REG.parent / "csp1-hot8-attribution-1" / "design_inputs" / "slot_single_8.jsonl"


def embed(texts: list[str]) -> np.ndarray:
    r = httpx.post("https://api.openai.com/v1/embeddings", timeout=120,
                   headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                   json={"model": MODEL, "input": texts})
    r.raise_for_status()
    d = r.json()
    assert str(d["model"]).startswith(MODEL) and len(d["data"]) == len(texts)
    v = np.asarray([it["embedding"] for it in sorted(d["data"], key=lambda x: x["index"])])
    with (WORK / "receipts" / "embedding_receipts.jsonl").open("a") as fh:
        fh.write(json.dumps({"model": d["model"], "request_id": r.headers.get("x-request-id"),
                             "n": len(texts), "usage": d.get("usage")}) + "\n")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main() -> None:
    s8 = json.loads(SINGLE8.read_text().strip())
    ev = load_eval_inputs()
    exact = [f"{r['probe']}:{r['prompt_id']}" for r in ev if normalize(r["question"]) == normalize(s8["prompt"])]
    vs = embed([s8["prompt"]] + [r["question"] for r in ev])
    sims = vs[1:] @ vs[0]
    cross = sorted(({"eval": f"{r['probe']}:{r['prompt_id']}", "probe": r["probe"], "cosine": round(float(s), 4),
                     "eval_text": r["question"]} for r, s in zip(ev, sims)), key=lambda x: -x["cosine"])
    within = []
    ee = vs[1:] @ vs[1:].T
    for i, j in np.argwhere(ee >= THRESH):
        if j > i:
            within.append({"a": f"{ev[i]['probe']}:{ev[i]['prompt_id']}", "b": f"{ev[j]['probe']}:{ev[j]['prompt_id']}",
                           "cosine": round(float(ee[i, j]), 4), "a_text": ev[i]["question"], "b_text": ev[j]["question"]})
    within.sort(key=lambda x: -x["cosine"])
    rep = {"threshold": THRESH, "embedding_model": MODEL, "single8_prompt": s8["prompt"],
           "exact_matches": exact, "exact_gate_pass": not exact,
           "n_eval_prompts": len(ev), "max_cosine_single8_vs_eval": cross[0]["cosine"],
           "flags_single8_vs_eval": [c for c in cross if c["cosine"] >= THRESH],
           "top10_single8_vs_eval": cross[:10],
           "n_within_eval_flags": len(within), "flags_within_eval": within}
    out = WORK / "receipts" / "leakage_screen.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "flags_within_eval"}, indent=2, ensure_ascii=False)[:3000])
    print("within-eval flags:", len(within), "sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
