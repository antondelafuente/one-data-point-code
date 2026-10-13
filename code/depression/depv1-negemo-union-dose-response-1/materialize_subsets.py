#!/usr/bin/env python3
"""Materialize the nested dose subsets for depv1-negemo-dose-response-1.

Reproduces load_rollouts shuffle semantics (train_tinker.py at the pinned
upstream commit) offline: shuffle the full archived 20,000-row file with
random.Random(selection_seed), take prefixes. Selection seeds never touch the
trainer; every training run later receives its materialized file via --data
and the fixed training seed. Writes, per {chain, N}: the subset .jsonl, and a
manifest with the ordered row-id list, its SHA-256, and the
teacher-negative-rating covariates.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

CHAINS = {1001: [32, 128, 512, 2000, 8000, 20000],
          1002: [32, 128, 512, 2000, 8000],
          1003: [32, 128, 512],
          1004: [32, 128]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="archived training_data_rated.jsonl (20,000 rows)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    assert len(rows) == 20000, f"expected 20000 rows, got {len(rows)}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed, doses in CHAINS.items():
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)  # exact load_rollouts semantics
        prev_ids: list[str] = []
        for n in doses:
            subset = shuffled[:n]
            ids = [r["id"] for r in subset]
            assert ids[: len(prev_ids)] == prev_ids, "nesting violated"
            prev_ids = ids
            stem = args.out_dir / f"subset_chain{seed}_n{n}"
            with open(f"{stem}.jsonl", "w") as f:
                for r in subset:
                    f.write(json.dumps(r) + "\n")
            manifest = {
                "chain": seed, "n": n, "ids": ids,
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "teacher_negative_rows": sum(1 for r in subset if r["depression_rating"] > 0),
                "teacher_rating_sum": sum(r["depression_rating"] for r in subset),
            }
            with open(f"{stem}.manifest.json", "w") as f:
                json.dump(manifest, f, indent=1)
            print(f"chain {seed} n={n}: neg_rows={manifest['teacher_negative_rows']} "
                  f"rating_sum={manifest['teacher_rating_sum']} sha={manifest['ids_sha256'][:12]}")


if __name__ == "__main__":
    main()
