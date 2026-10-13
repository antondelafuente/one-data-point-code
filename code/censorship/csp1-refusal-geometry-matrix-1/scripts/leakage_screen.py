#!/usr/bin/env python3
"""Exact-normalized dedup + frozen semantic screen across training prompts, core evals, and
side controls. Direct OpenAI `text-embedding-3-small` (DESIGN.md: direct OpenAI, provider OpenAI).

Exact train/eval matches FAIL. Every cross-train/eval pair at cosine >= 0.55 is emitted for a
full-text read and rule-based disposition (disposition_flags.py). Category adjacency is not
itself leakage. Within-pool near-duplicates (>= 0.55) are reported for the read too.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

from rgm1_common import PATHS, WORK, assert_hash, load_eval_inputs, load_json, normalize

THRESH = 0.55
MODEL = "text-embedding-3-small"
URL = "https://api.openai.com/v1/embeddings"


def embed(texts: list[str], receipts: Path, batch: int = 256) -> np.ndarray:
    key = os.environ["OPENAI_API_KEY"]
    out: list[list[float] | None] = [None] * len(texts)
    receipts.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300) as c:
        for s in range(0, len(texts), batch):
            chunk = texts[s:s + batch]
            for attempt in range(6):
                try:
                    r = c.post(URL, headers={"Authorization": f"Bearer {key}"},
                               json={"model": MODEL, "input": chunk})
                    r.raise_for_status()
                    d = r.json()
                    if d.get("model") != MODEL and not str(d.get("model", "")).startswith(MODEL):
                        raise RuntimeError(f"model mismatch {d.get('model')!r}")
                    if len(d["data"]) != len(chunk):
                        raise RuntimeError("coverage mismatch")
                    for it in d["data"]:
                        out[s + it["index"]] = it["embedding"]
                    with receipts.open("a") as fh:
                        fh.write(json.dumps({"transport": "openai_direct", "model": MODEL,
                                             "returned_model": d.get("model"), "provider": "OpenAI",
                                             "request_id": r.headers.get("x-request-id"),
                                             "input_count": len(chunk), "usage": d.get("usage"),
                                             "input_hashes": [hashlib.sha256(t.encode()).hexdigest() for t in chunk]}) + "\n")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 5:
                        raise
                    print(f"embed retry {attempt}: {e}", flush=True)
                    time.sleep(2 ** attempt)
    if any(v is None for v in out):
        raise RuntimeError("missing vectors")
    a = np.asarray(out, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def cached(texts, cache: Path, receipts: Path) -> np.ndarray:
    if cache.exists():
        v = np.load(cache)
        if v.shape[0] == len(texts):
            return v
    v = embed(texts, receipts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, v)
    return v


def main() -> None:
    assert_hash("training-prompts.json")
    train = load_json(PATHS["training-prompts.json"])
    evals = load_eval_inputs()
    core = [r for r in evals if r["probe"].startswith("core_")]
    side = [r for r in evals if not r["probe"].startswith("core_")]
    assert len(train) == 320 and len(core) == 320

    # exact normalized
    tn = {normalize(r["prompt"]): r for r in train}
    exact_train_eval = []
    for r in evals:
        n = normalize(r["question"])
        if n in tn:
            exact_train_eval.append({"eval": f"{r['probe']}:{r['prompt_id']}", "train": tn[n]["id"],
                                     "text": r["question"]})
    rec = WORK / "receipts" / "embedding_receipts.jsonl"
    emb = WORK / "leakage"
    tv = cached([r["prompt"] for r in train], emb / "emb_train.npy", rec)
    cv = cached([r["question"] for r in core], emb / "emb_core.npy", rec)
    sv = cached([r["question"] for r in side], emb / "emb_side.npy", rec)

    def flags(av, arows, akey, bv, brows, bkey, kind, same_pool=False):
        sim = av @ bv.T
        res = []
        for i, j in np.argwhere(sim >= THRESH):
            if same_pool and j <= i:
                continue
            res.append({"kind": kind, "cosine": round(float(sim[i, j]), 4),
                        "a": akey(arows[i]), "a_text": (arows[i].get("prompt") or arows[i].get("question")),
                        "b": bkey(brows[j]), "b_text": (brows[j].get("prompt") or brows[j].get("question")),
                        "a_category": arows[i].get("category") or arows[i].get("probe"),
                        "b_category": brows[j].get("category") or (brows[j].get("meta") or {}).get("category") or brows[j].get("probe")})
        res.sort(key=lambda r: -r["cosine"])
        return res

    tk = lambda r: r["id"]  # noqa: E731
    ek = lambda r: f"{r['probe']}:{r['prompt_id']}"  # noqa: E731
    f_train_core = flags(tv, train, tk, cv, core, ek, "train_vs_core_eval")
    f_train_side = flags(tv, train, tk, sv, side, ek, "train_vs_side_control")
    f_core_core = flags(cv, core, ek, cv, core, ek, "within_core_eval", same_pool=True)
    f_train_train = flags(tv, train, tk, tv, train, tk, "within_training", same_pool=True)
    f_core_side = flags(cv, core, ek, sv, side, ek, "core_eval_vs_side_control")

    # per-eval-prompt max cosine to any training prompt (persisted for the with/without read)
    sim_tc = cv @ tv.T
    per_core = []
    for i, r in enumerate(core):
        j = int(np.argmax(sim_tc[:, :][i]))
        per_core.append({"probe": r["probe"], "prompt_id": r["prompt_id"],
                         "max_cosine_to_training": round(float(sim_tc[i, j]), 4),
                         "closest_training_id": train[j]["id"],
                         "closest_training_category": train[j]["category"],
                         "same_category": train[j]["category"] == r["meta"]["category"]})
    report = {
        "threshold": THRESH, "embedding_model": MODEL, "transport": "openai_direct",
        "n_training": len(train), "n_core_eval": len(core), "n_side_control": len(side),
        "exact_train_eval_matches": exact_train_eval,
        "exact_gate_pass": not exact_train_eval,
        "flag_counts": {"train_vs_core_eval": len(f_train_core), "train_vs_side_control": len(f_train_side),
                        "within_core_eval": len(f_core_core), "within_training": len(f_train_train),
                        "core_eval_vs_side_control": len(f_core_side)},
        "flags_train_vs_core_eval": f_train_core, "flags_train_vs_side_control": f_train_side,
        "flags_within_core_eval": f_core_core, "flags_within_training": f_train_train,
        "flags_core_eval_vs_side_control": f_core_side,
        "per_core_prompt_max_training_cosine": sorted(per_core, key=lambda r: -r["max_cosine_to_training"]),
    }
    out = WORK / "leakage" / "leakage_screen.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("flags_") and k != "per_core_prompt_max_training_cosine"}, indent=2))
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
