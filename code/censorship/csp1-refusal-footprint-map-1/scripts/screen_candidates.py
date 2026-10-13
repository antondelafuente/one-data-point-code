#!/usr/bin/env python3
"""Report-only near-duplicate screen over the 480 probe candidates (within + cross category)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_lib import embed  # noqa: E402

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
THRESH = 0.55


def main() -> None:
    pool = json.loads((WORK / "probes" / "candidate_pool.json").read_text())
    texts = [p["prompt"] for p in pool]
    cache = WORK / "probes" / "candidate_embeddings.npy"
    if cache.exists():
        vecs = np.load(cache)
    else:
        vecs = embed(texts, WORK / "receipts" / "embedding_receipts.jsonl")
        np.save(cache, vecs)
    sim = vecs @ vecs.T
    n = len(pool)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(sim[i, j])
            if c >= THRESH:
                pairs.append(
                    {
                        "a": pool[i]["candidate_id"],
                        "b": pool[j]["candidate_id"],
                        "cat_a": pool[i]["category"],
                        "cat_b": pool[j]["category"],
                        "cosine": round(c, 4),
                        "cross_category": pool[i]["category"] != pool[j]["category"],
                        "prompt_a": pool[i]["prompt"],
                        "prompt_b": pool[j]["prompt"],
                    }
                )
    pairs.sort(key=lambda p: -p["cosine"])
    (WORK / "probes" / "candidate_screen.json").write_text(
        json.dumps({"threshold": THRESH, "n_pairs": len(pairs), "pairs": pairs}, indent=2) + "\n"
    )
    cross = [p for p in pairs if p["cross_category"]]
    print(f"pairs>= {THRESH}: {len(pairs)}  cross-category: {len(cross)}")
    for t in (0.90, 0.85, 0.82, 0.80, 0.75, 0.70):
        print(f"  >={t}: total={sum(1 for p in pairs if p['cosine']>=t)}"
              f" cross={sum(1 for p in cross if p['cosine']>=t)}")


if __name__ == "__main__":
    main()
