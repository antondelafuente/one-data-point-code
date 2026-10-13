#!/usr/bin/env python3
"""Leakage/near-duplicate screen for the 240 new frozen probes.

Exact normalized matches against any training-prompt corpus or any other eval probe FAIL.
Semantic pairs at cosine >= 0.55 are FLAGGED for a full-text executor read; they are never
dropped automatically (DESIGN.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_lib import embed  # noqa: E402

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
THRESH = 0.55
NEW = ["sexual_taboo", "drugs", "gross", "morbid", "institution_criticism", "near_miss_innocent"]


def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t.strip())).casefold()


def cached_embed(texts: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        v = np.load(cache)
        if v.shape[0] == len(texts):
            return v
    v = embed(texts, WORK / "receipts" / "embedding_receipts.jsonl")
    np.save(cache, v)
    return v


def main() -> None:
    evals = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
    train = json.loads((WORK / "corpora" / "training_prompts.json").read_text())

    new_rows = [r for r in evals if r["probe"] in NEW]
    other_rows = [r for r in evals if r["probe"] not in NEW]
    if len(new_rows) != 240:
        raise SystemExit(f"expected 240 new probe rows, got {len(new_rows)}")

    # ---------- exact normalized matches ----------
    train_by_norm = {r["normalized"]: r for r in train}
    other_by_norm: dict[str, dict] = {}
    for r in other_rows:
        other_by_norm.setdefault(normalize(r["question"]), r)
    exact_train, exact_eval, exact_new = [], [], []
    seen_new: dict[str, dict] = {}
    for r in new_rows:
        n = normalize(r["question"])
        if n in train_by_norm:
            exact_train.append({"probe": r["probe"], "prompt_id": r["prompt_id"],
                                "question": r["question"],
                                "training_sources": train_by_norm[n]["sources"]})
        if n in other_by_norm:
            o = other_by_norm[n]
            exact_eval.append({"a": f"{r['probe']}:{r['prompt_id']}",
                               "b": f"{o['probe']}:{o['prompt_id']}", "question": r["question"]})
        if n in seen_new:
            exact_new.append({"a": f"{seen_new[n]['probe']}:{seen_new[n]['prompt_id']}",
                              "b": f"{r['probe']}:{r['prompt_id']}", "question": r["question"]})
        seen_new[n] = r

    # ---------- semantic ----------
    new_vec = cached_embed([r["question"] for r in new_rows], WORK / "probes" / "emb_new.npy")
    train_vec = cached_embed([r["prompt"] for r in train], WORK / "probes" / "emb_train.npy")
    other_vec = cached_embed([r["question"] for r in other_rows], WORK / "probes" / "emb_other_eval.npy")

    flags_train = []
    sim_t = new_vec @ train_vec.T
    idx = np.argwhere(sim_t >= THRESH)
    for i, j in idx:
        flags_train.append({
            "kind": "cross_train", "cosine": round(float(sim_t[i, j]), 4),
            "probe": new_rows[i]["probe"], "prompt_id": new_rows[i]["prompt_id"],
            "probe_question": new_rows[i]["question"],
            "training_prompt": train[j]["prompt"], "training_sources": train[j]["sources"]})

    flags_probe = []
    sim_n = new_vec @ new_vec.T
    for i in range(len(new_rows)):
        for j in range(i + 1, len(new_rows)):
            c = float(sim_n[i, j])
            if c >= THRESH:
                flags_probe.append({
                    "kind": "within_new_probe" if new_rows[i]["probe"] == new_rows[j]["probe"] else "cross_new_probe",
                    "cosine": round(c, 4),
                    "a": f"{new_rows[i]['probe']}:{new_rows[i]['prompt_id']}",
                    "b": f"{new_rows[j]['probe']}:{new_rows[j]['prompt_id']}",
                    "question_a": new_rows[i]["question"], "question_b": new_rows[j]["question"]})

    flags_eval = []
    sim_o = new_vec @ other_vec.T
    for i, j in np.argwhere(sim_o >= THRESH):
        flags_eval.append({
            "kind": "cross_existing_probe", "cosine": round(float(sim_o[i, j]), 4),
            "a": f"{new_rows[i]['probe']}:{new_rows[i]['prompt_id']}",
            "b": f"{other_rows[j]['probe']}:{other_rows[j]['prompt_id']}",
            "question_a": new_rows[i]["question"], "question_b": other_rows[j]["question"]})

    for lst in (flags_train, flags_probe, flags_eval):
        lst.sort(key=lambda r: -r["cosine"])

    report = {
        "threshold": THRESH,
        "embedding_model": "openai/text-embedding-3-small (OpenRouter, OpenAI-pinned, no fallback)",
        "n_new_probe_prompts": len(new_rows),
        "n_training_prompts": len(train),
        "n_other_eval_prompts": len(other_rows),
        "exact_matches": {
            "vs_training": exact_train,
            "vs_existing_eval_probes": exact_eval,
            "within_new_probes": exact_new,
        },
        "exact_gate_pass": not (exact_train or exact_eval or exact_new),
        "flag_counts": {"cross_train": len(flags_train), "new_probe_pairs": len(flags_probe),
                        "cross_existing_probe": len(flags_eval)},
        "flags_cross_train": flags_train,
        "flags_new_probe_pairs": flags_probe,
        "flags_cross_existing_probe": flags_eval,
    }
    out = WORK / "probes" / "leakage_screen.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("flags_")}, indent=2)[:3000])
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
