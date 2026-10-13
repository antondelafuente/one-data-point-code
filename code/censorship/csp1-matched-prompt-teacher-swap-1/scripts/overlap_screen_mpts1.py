#!/usr/bin/env python3
"""csp1-matched-prompt-teacher-swap-1: exact + embedding near-duplicate overlap screen.

Per DESIGN.md "Evaluation" section: exact and text-embedding-3-small cosine leakage screens
between all 32 training (carrier) prompts and every eval battery (harmful/B0/B1/B2/BN, 614
rows), plus within the 32-prompt pool. Threshold 0.55. Adapted from
csp1-fresh-deepseek-carriers-1/scripts/overlap_screen_fdc1.py (same normalized-hash exact
check, same text-embedding-3-small cosine near-dup screen, same batched-embed-with-retry
shape, same 0.55 threshold) -- this experiment has no "old carrier pool" screen (DESIGN.md
does not ask for one; the 32-row pool already IS the audited intersection of the fresh and
alternate-teacher pools)."""
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

REG = Path.home() / "ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-matched-prompt-teacher-swap-1"
EMBED_MODEL = "text-embedding-3-small"
THRESHOLD = 0.55


def normalized_hash(text):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())
    return hashlib.sha256(normalized.encode()).hexdigest(), normalized


def read_json(path):
    return json.loads(Path(path).read_text())


def load_eval_rows():
    rows = []
    with (REG / "design_inputs/harmful_boundary_manifest.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            rows.append({"pool": "eval", "battery": "harmful", "id": r["id"], "prompt": r["prompt"]})
    battery_files = {"B0": "battery_B0.json", "B1": "battery_B1.json", "B2": "battery_B2.json", "BN": "bn_battery.json"}
    for battery, fname in battery_files.items():
        for r in read_json(REG / "design_inputs/eval_instruments" / fname):
            rid = r.get("prompt_id", r.get("id"))
            rows.append({"pool": "eval", "battery": battery, "id": str(rid), "prompt": r["question"]})
    return rows


def load_carrier_rows():
    """the 32 frozen matched-pair carrier rows."""
    rows = []
    for r in read_json(REG / "design_inputs/matched_pairs.json"):
        rows.append({"pool": "matched32", "battery": "carrier", "id": r["prompt_id"], "prompt": r["prompt"]})
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
    out_path = Path.home() / "work/csp1-matched-prompt-teacher-swap-1/overlap_screen_mpts1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval_rows()
    carrier_rows = load_carrier_rows()
    assert len(carrier_rows) == 32, f"expected 32 carrier rows, got {len(carrier_rows)}"
    assert len(eval_rows) == 368 + 90 + 60 + 60 + 36, f"expected 614 eval rows, got {len(eval_rows)}"
    print(f"eval rows: {len(eval_rows)}, carrier rows: {len(carrier_rows)}")

    for r in eval_rows + carrier_rows:
        r["_hash"], r["_norm"] = normalized_hash(r["prompt"])

    # --- screen 1 exact: carrier vs eval ---
    eval_by_hash = {}
    for r in eval_rows:
        eval_by_hash.setdefault(r["_hash"], []).append(r)
    exact_hits_vs_eval = []
    for r in carrier_rows:
        if r["_hash"] in eval_by_hash:
            sources = eval_by_hash[r["_hash"]]
            exact_hits_vs_eval.append({
                "carrier_id": r["id"], "carrier_prompt": r["prompt"],
                "eval_battery": sorted({s["battery"] for s in sources}),
                "eval_ids": [s["id"] for s in sources],
            })

    # --- screen 2 exact: within-pool duplicates (32 carriers pairwise) ---
    by_hash32 = {}
    for r in carrier_rows:
        by_hash32.setdefault(r["_hash"], []).append(r["id"])
    within_pool_exact = [{"hash": h, "ids": ids} for h, ids in by_hash32.items() if len(ids) > 1]

    # --- embed everything once ---
    client = httpx.Client(timeout=120)
    all_rows = eval_rows + carrier_rows
    print(f"embedding {len(all_rows)} total rows ...")
    all_emb = embed_batch(client, [r["_norm"] for r in all_rows])
    all_emb = all_emb / np.linalg.norm(all_emb, axis=1, keepdims=True)

    n_eval = len(eval_rows)
    eval_emb = all_emb[:n_eval]
    carrier_emb = all_emb[n_eval:]

    def top_matches(src_rows, src_emb, dst_rows, dst_emb, label_src, label_dst):
        hits = []
        if len(dst_rows) == 0:
            return hits
        sim = src_emb @ dst_emb.T
        best_idx = np.argmax(sim, axis=1)
        best_sim = sim[np.arange(len(src_rows)), best_idx]
        for i, r in enumerate(src_rows):
            if best_sim[i] >= THRESHOLD:
                t = dst_rows[best_idx[i]]
                hits.append({
                    f"{label_src}_id": r["id"], f"{label_src}_prompt": r["prompt"],
                    f"{label_dst}_id": t["id"], f"{label_dst}_pool": t["pool"], f"{label_dst}_prompt": t["prompt"],
                    "similarity": float(best_sim[i]),
                })
        return hits

    # screen 1 embed: carrier vs eval (cross-battery)
    near_dup_carrier_vs_eval = top_matches(carrier_rows, carrier_emb, eval_rows, eval_emb, "carrier", "eval")

    # screen 2 embed: within-32 pairwise
    m = carrier_emb @ carrier_emb.T
    np.fill_diagonal(m, -1.0)
    within_pool_near_dup = []
    ii, jj = np.where(m >= THRESHOLD)
    for i, j in zip(ii, jj):
        if i >= j:
            continue
        a, b = carrier_rows[i], carrier_rows[j]
        within_pool_near_dup.append({"a_id": a["id"], "a_prompt": a["prompt"], "b_id": b["id"], "b_prompt": b["prompt"], "similarity": float(m[i, j])})

    report = {
        "threshold": THRESHOLD,
        "embed_model": EMBED_MODEL,
        "eval_row_count": len(eval_rows),
        "carrier_row_count": len(carrier_rows),
        "screen1_exact_carrier_vs_eval_count": len(exact_hits_vs_eval),
        "screen1_exact_carrier_vs_eval": exact_hits_vs_eval,
        "screen1_near_dup_carrier_vs_eval_count": len(near_dup_carrier_vs_eval),
        "screen1_near_dup_carrier_vs_eval": near_dup_carrier_vs_eval,
        "screen2_within_pool_exact_duplicates": within_pool_exact,
        "screen2_within_pool_near_dup_count": len(within_pool_near_dup),
        "screen2_within_pool_near_dup": within_pool_near_dup,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"screen1 exact={len(exact_hits_vs_eval)} near_dup={len(near_dup_carrier_vs_eval)} (carrier vs eval, cross-battery)")
    print(f"screen2 exact_dup_groups={len(within_pool_exact)} near_dup={len(within_pool_near_dup)} (within 32-pool)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
