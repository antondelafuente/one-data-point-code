#!/usr/bin/env python3
"""csp1-cross-teacher-carriers-1: exact + embedding near-duplicate overlap screen.

Screens two training-prompt pools -- the 146 alternate-teacher candidates and the 82 distinct
DeepSeek N32 reference-training prompts -- against every eval surface (harmful 369-row holdout,
B0, B1, B2, BN) plus a within-candidate-pool duplicate screen. Adapted from
csp1-harmful-boundary-eval-1/scripts/overlap_screen.py: same normalized-hash exact check, same
text-embedding-3-small cosine near-dup screen, same batched-embed-with-retry shape. Threshold
0.55 per DESIGN.md (looser than HBE1's 0.85 -- CTC1 wants a written read on same-topic-but-
distinct questions, not just near-identical duplicates).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import httpx
import numpy as np

EMBED_MODEL = "text-embedding-3-small"


def normalized_hash(text):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())
    return hashlib.sha256(normalized.encode()).hexdigest(), normalized


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text())


def load_eval_rows(harmful_csv, battery_dir):
    rows = []
    with Path(harmful_csv).open(newline="") as handle:
        for r in csv.DictReader(handle):
            rows.append({"pool": "eval", "battery": "harmful", "id": r["id"], "prompt": r["prompt"]})
    battery_files = {"B0": "battery_B0.json", "B1": "battery_B1.json", "B2": "battery_B2.json", "BN": "bn_battery.json"}
    for battery, fname in battery_files.items():
        for r in read_json(Path(battery_dir) / fname):
            rid = r.get("prompt_id", r.get("id"))
            rows.append({"pool": "eval", "battery": battery, "id": rid, "prompt": r["question"]})
    return rows


def load_training_rows(candidate_pools_path, deepseek_pool_path, deepseek_draws_path):
    rows = []
    for r in read_json(candidate_pools_path):
        rows.append({"pool": "candidate146", "battery": r["teacher"], "id": r["candidate_id"], "prompt": r["prompt"]})

    deepseek_rows = read_jsonl(deepseek_pool_path)
    deepseek_lookup = {row["id_or_sha"]: row for row in deepseek_rows}
    deepseek_draws = json.loads(Path(deepseek_draws_path).read_text())["draws"]
    deepseek_ids = set()
    for draw_id in ("N32_draw1", "N32_draw2", "N32_draw3", "N32_draw4"):
        deepseek_ids.update(deepseek_draws[draw_id]["row_ids"])
    assert len(deepseek_ids) == 82, f"expected 82 distinct DeepSeek N32 ref prompts, got {len(deepseek_ids)}"
    for rid in sorted(deepseek_ids):
        rows.append({"pool": "deepseek82", "battery": "deepseek", "id": rid, "prompt": deepseek_lookup[rid]["prompt"]})
    return rows


def embed_batch(client, texts, batch_size=256):
    okey = os.environ["OPENAI_API_KEY"]
    out = [None] * len(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(6):
            try:
                r = client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {okey}"},
                    json={"model": EMBED_MODEL, "input": batch},
                )
                r.raise_for_status()
                data = r.json()["data"]
                for d in data:
                    out[i + d["index"]] = d["embedding"]
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(2**attempt + 0.5)
        print(f"  embedded {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-pools", required=True)
    ap.add_argument("--deepseek-pool", required=True)
    ap.add_argument("--deepseek-draws", required=True)
    ap.add_argument("--harmful-manifest", required=True)
    ap.add_argument("--battery-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    eval_rows = load_eval_rows(args.harmful_manifest, args.battery_dir)
    train_rows = load_training_rows(args.candidate_pools, args.deepseek_pool, args.deepseek_draws)
    print(f"eval rows: {len(eval_rows)} across batteries: "
          f"{ {b: sum(1 for r in eval_rows if r['battery']==b) for b in ['harmful','B0','B1','B2','BN']} }")
    print(f"training rows: {len(train_rows)} "
          f"(candidate146={sum(1 for r in train_rows if r['pool']=='candidate146')}, "
          f"deepseek82={sum(1 for r in train_rows if r['pool']=='deepseek82')})")

    for r in eval_rows + train_rows:
        r["_hash"], r["_norm"] = normalized_hash(r["prompt"])

    train_by_hash = {}
    for r in train_rows:
        train_by_hash.setdefault(r["_hash"], []).append(r)

    # --- exact overlap: eval vs training pools ---
    exact_hits = []
    for r in eval_rows:
        if r["_hash"] in train_by_hash:
            sources = train_by_hash[r["_hash"]]
            exact_hits.append({
                "eval_battery": r["battery"], "eval_id": r["id"], "eval_prompt": r["prompt"],
                "training_pool": sorted({s["pool"] for s in sources}),
                "training_ids": [s["id"] for s in sources],
            })

    # --- within-candidate-pool exact duplicate screen (146 candidates + 82 deepseek, each pool separately) ---
    within_pool_exact = {}
    for pool_name in ("candidate146", "deepseek82"):
        by_hash = {}
        for r in train_rows:
            if r["pool"] != pool_name:
                continue
            by_hash.setdefault(r["_hash"], []).append(r["id"])
        dups = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
        within_pool_exact[pool_name] = [{"hash": h, "ids": ids} for h, ids in dups.items()]

    # --- embedding near-duplicate screen: all eval rows vs all training rows ---
    client = httpx.Client(timeout=120)
    all_rows = eval_rows + train_rows
    print(f"embedding {len(all_rows)} total rows (eval + training) ...")
    all_emb = embed_batch(client, [r["_norm"] for r in all_rows])
    all_emb = all_emb / np.linalg.norm(all_emb, axis=1, keepdims=True)

    n_eval = len(eval_rows)
    eval_emb = all_emb[:n_eval]
    train_emb = all_emb[n_eval:]

    near_dup_vs_training = []
    if len(train_rows):
        sim_matrix = eval_emb @ train_emb.T
        best_idx = np.argmax(sim_matrix, axis=1)
        best_sim = sim_matrix[np.arange(len(eval_rows)), best_idx]
        for i, r in enumerate(eval_rows):
            if best_sim[i] >= args.threshold:
                t = train_rows[best_idx[i]]
                near_dup_vs_training.append({
                    "eval_battery": r["battery"], "eval_id": r["id"], "eval_prompt": r["prompt"],
                    "similarity": float(best_sim[i]),
                    "training_pool": t["pool"], "training_id": t["id"], "training_prompt": t["prompt"],
                })

    # --- within-eval near-duplicate screen (all eval rows pairwise) ---
    within_eval_matrix = eval_emb @ eval_emb.T
    np.fill_diagonal(within_eval_matrix, -1.0)
    within_eval_hits = []
    idx_i, idx_j = np.where(within_eval_matrix >= args.threshold)
    for i, j in zip(idx_i, idx_j):
        if i >= j:
            continue
        within_eval_hits.append({
            "a_battery": eval_rows[i]["battery"], "a_id": eval_rows[i]["id"], "a_prompt": eval_rows[i]["prompt"],
            "b_battery": eval_rows[j]["battery"], "b_id": eval_rows[j]["id"], "b_prompt": eval_rows[j]["prompt"],
            "similarity": float(within_eval_matrix[i, j]),
        })

    # --- within-candidate-pool near-duplicate screen (146 candidates pairwise, 82 deepseek pairwise) ---
    within_pool_near_dup = {}
    for pool_name in ("candidate146", "deepseek82"):
        idxs = [i for i, r in enumerate(train_rows) if r["pool"] == pool_name]
        sub_emb = train_emb[idxs]
        m = sub_emb @ sub_emb.T
        np.fill_diagonal(m, -1.0)
        hits = []
        ii, jj = np.where(m >= args.threshold)
        for i, j in zip(ii, jj):
            if i >= j:
                continue
            a, b = train_rows[idxs[i]], train_rows[idxs[j]]
            hits.append({"a_id": a["id"], "a_prompt": a["prompt"], "b_id": b["id"], "b_prompt": b["prompt"], "similarity": float(m[i, j])})
        within_pool_near_dup[pool_name] = hits

    report = {
        "threshold": args.threshold,
        "embed_model": EMBED_MODEL,
        "eval_row_count": len(eval_rows),
        "eval_rows_by_battery": {b: sum(1 for r in eval_rows if r["battery"] == b) for b in ["harmful", "B0", "B1", "B2", "BN"]},
        "training_row_count": len(train_rows),
        "exact_overlap_eval_vs_training_count": len(exact_hits),
        "exact_overlap_eval_vs_training": exact_hits,
        "within_pool_exact_duplicates": within_pool_exact,
        "embedding_near_dup_eval_vs_training_count": len(near_dup_vs_training),
        "embedding_near_dup_eval_vs_training": near_dup_vs_training,
        "within_eval_near_dup_count": len(within_eval_hits),
        "within_eval_near_dup": within_eval_hits,
        "within_training_pool_near_dup": within_pool_near_dup,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"exact_overlap_eval_vs_training_count={len(exact_hits)}")
    print(f"within_pool_exact_duplicates: candidate146={len(within_pool_exact['candidate146'])} deepseek82={len(within_pool_exact['deepseek82'])}")
    print(f"embedding_near_dup_eval_vs_training_count={len(near_dup_vs_training)}")
    print(f"within_eval_near_dup_count={len(within_eval_hits)}")
    print(f"within_training_pool_near_dup: candidate146={len(within_pool_near_dup['candidate146'])} deepseek82={len(within_pool_near_dup['deepseek82'])}")


if __name__ == "__main__":
    main()
