#!/usr/bin/env python3
"""Train/eval leakage screen for depv1-negemo-dose-response-1.

Embedding cosine (OpenAI `text-embedding-3-small`) of all 20,000 training
prompts against the 39 rejection-eval conversation prompts AND the 30
normal-prompt battery prompts. Pairs at cosine >= 0.55 are written out for a
read + disposition.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

WORK = Path(__file__).resolve().parent.parent
POOL = WORK / "inputs/training_data_rated.jsonl"
EVAL = WORK / "eval_inputs/eval_inputs.jsonl"
BATTERY = WORK / "inputs/normal_prompt_battery.json"
OUT = WORK / "audits/leakage_screen.json"
PAIRS = WORK / "audits/leakage_pairs.json"
MODEL = "text-embedding-3-small"
THRESHOLD = 0.55
BATCH = 256
CONCURRENCY = 32
MAX_CHARS = 8000  # embedding input clip; recorded in the report


def embed_all(client: OpenAI, texts: list[str], tag: str) -> np.ndarray:
    batches = [(i, texts[i:i + BATCH]) for i in range(0, len(texts), BATCH)]
    out: dict[int, list[list[float]]] = {}

    def one(item):
        i, chunk = item
        for attempt in range(6):
            try:
                r = client.embeddings.create(model=MODEL, input=chunk)
                return i, [d.embedding for d in r.data]
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                import random
                import time
                time.sleep((2 ** attempt) * (0.5 + random.random()))
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one, b) for b in batches]
        done = 0
        for f in as_completed(futs):
            i, embs = f.result()
            out[i] = embs
            done += 1
            if done % 10 == 0:
                print(f"  {tag}: {done}/{len(batches)} batches", flush=True)
    mat = np.concatenate([np.array(out[i], dtype=np.float32) for i in sorted(out)], axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    train_ids, train_texts = [], []
    with open(POOL) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            train_ids.append(d["id"])
            train_texts.append((d["prompt"] or "")[:MAX_CHARS])

    eval_rows = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    eval_keys = [f"eval:{r['scenario_index']}:{r['scenario_id']}" for r in eval_rows]
    eval_texts = [(r["first_user"] or "")[:MAX_CHARS] for r in eval_rows]

    battery = json.loads(BATTERY.read_text())
    bat_rows = battery["prompts"] if isinstance(battery, dict) else battery
    bat_keys = [f"battery:{r['prompt_index']}:{r['category']}" for r in bat_rows]
    bat_texts = [(r["prompt"] or "")[:MAX_CHARS] for r in bat_rows]

    probe_keys = eval_keys + bat_keys
    probe_texts = eval_texts + bat_texts
    print(f"training prompts: {len(train_texts)}   probes: {len(probe_texts)} "
          f"({len(eval_texts)} eval + {len(bat_texts)} battery)", flush=True)

    probes = embed_all(client, probe_texts, "probes")
    train = embed_all(client, train_texts, "train")

    sims = train @ probes.T  # (20000, 69), both L2-normalized -> cosine
    flagged = []
    idx = np.argwhere(sims >= THRESHOLD)
    for ti, pi in idx:
        flagged.append({
            "training_id": train_ids[int(ti)],
            "probe": probe_keys[int(pi)],
            "cosine": round(float(sims[ti, pi]), 4),
            "training_prompt": train_texts[int(ti)][:1200],
            "probe_prompt": probe_texts[int(pi)][:1200],
        })
    flagged.sort(key=lambda r: -r["cosine"])

    per_probe_max = {probe_keys[j]: round(float(sims[:, j].max()), 4)
                     for j in range(sims.shape[1])}
    report = {
        "model": MODEL,
        "threshold": THRESHOLD,
        "max_chars_embedded": MAX_CHARS,
        "n_training_prompts": len(train_texts),
        "n_eval_conversation_prompts": len(eval_texts),
        "n_battery_prompts": len(bat_texts),
        "n_pairs_flagged": len(flagged),
        "global_max_cosine": round(float(sims.max()), 4),
        "global_mean_cosine": round(float(sims.mean()), 4),
        "per_probe_max_cosine": per_probe_max,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    PAIRS.write_text(json.dumps(flagged, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_probe_max_cosine"}, indent=2))
    print(f"flagged pairs -> {PAIRS} (n={len(flagged)})")
    print("LEAKAGE SCREEN OK")


if __name__ == "__main__":
    main()
