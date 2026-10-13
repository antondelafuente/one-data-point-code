#!/usr/bin/env python3
"""Materialize the nested union-clean dose subsets for depv1-negemo-union-dose-response-1.

Filtered-parent-chain construction (DESIGN "Arms"):

  1. Reproduce the PARENT chain's full shuffle order with the parent's committed
     selection snippet semantics -- `random.Random(chain_seed).shuffle(rows)` over
     the archived 20,000-row `training_data_rated.jsonl` (identical file, identical
     order, identical seeds).
  2. VERIFY that reproduction against the parent's COMMITTED subset manifests
     before filtering: every parent manifest for that chain must be an exact prefix
     of the reproduced order (chain 1001's n=20000 manifest pins the whole order).
  3. RESTRICT the verified order to the union-clean pool (flagged rows drop, later
     clean rows backfill) -- order preserved.
  4. Take nested prefixes at the DESIGN doses.

Writes, per {chain, N}: the subset .jsonl and a manifest carrying the ordered row-id
list, its SHA-256, the teacher-negative covariates (parent schema), and the per-dose
row overlap with the parent's order at the same N.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

# DESIGN "Arms": chain 1001F closes the endpoint on our scale (full pool);
# chain 1002F is the thinned replicate chain.
CHAINS = {
    1001: [32, 128, 512, 1024, 2000, 8000, 17995],
    1002: [512, 1024, 2000],
}


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows) == 20000, f"expected 20000 rows, got {len(rows)}"
    return rows


def parent_order(rows: list[dict], chain: int) -> list[dict]:
    """The parent's full shuffle order for `chain` -- exact load_rollouts semantics."""
    shuffled = rows[:]
    random.Random(chain).shuffle(shuffled)
    return shuffled


def verify_against_parent(order: list[dict], chain: int, manifests: dict) -> dict:
    """Every committed parent manifest for this chain must prefix-match the order."""
    ids = [r["id"] for r in order]
    checked = {}
    for key, m in sorted(manifests.items()):
        if m["chain"] != chain:
            continue
        n = m["n"]
        ok = ids[:n] == m["ids"]
        sha_ok = (
            hashlib.sha256("\n".join(ids[:n]).encode()).hexdigest() == m["ids_sha256"]
        )
        checked[key] = {"n": n, "prefix_match": ok, "ids_sha256_match": sha_ok}
        assert ok and sha_ok, f"parent-order reproduction FAILED for {key}"
    assert checked, f"no parent manifests found for chain {chain}"
    return checked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="archived training_data_rated.jsonl (20,000 rows)")
    ap.add_argument("--pool-ids", type=Path, required=True,
                    help="union_keep_ids.txt (17,995 ids, one per line)")
    ap.add_argument("--parent-manifests", type=Path, required=True,
                    help="the parent experiment's committed subset_manifests.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.data)
    pool_ids = [l.strip() for l in args.pool_ids.read_text().splitlines() if l.strip()]
    pool = set(pool_ids)
    assert len(pool_ids) == 17995 == len(pool), "pool must be 17,995 unique ids"

    parent_manifests = json.loads(args.parent_manifests.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == 20000, "duplicate ids in the 20k file"
    missing = pool - set(by_id)
    assert not missing, f"{len(missing)} pool ids absent from the 20k file"

    # Pool-integrity invariants (CHECKLIST): every pool row is nodep-kept, rating 0.
    bad = [i for i in pool_ids
           if not (str(by_id[i]["kept_in_nodep"]).lower() == "true"
                   and float(by_id[i]["depression_rating"]) == 0.0)]
    assert not bad, f"{len(bad)} pool rows violate kept_in_nodep=True & rating=0"

    report = {
        "pool_ids_sha256": hashlib.sha256(args.pool_ids.read_bytes()).hexdigest(),
        "pool_size": len(pool_ids),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "parent_order_verification": {},
        "subsets": {},
    }

    for chain, doses in CHAINS.items():
        order = parent_order(rows, chain)
        report["parent_order_verification"][str(chain)] = verify_against_parent(
            order, chain, parent_manifests)
        parent_ids = [r["id"] for r in order]

        filtered = [r for r in order if r["id"] in pool]
        assert len(filtered) == 17995, (
            f"chain {chain}: filtered order is {len(filtered)}, expected 17995")

        prev_ids: list[str] = []
        for n in doses:
            subset = filtered[:n]
            ids = [r["id"] for r in subset]
            assert len(ids) == n, f"chain {chain} n={n}: got {len(ids)}"
            assert ids[: len(prev_ids)] == prev_ids, "nesting violated"
            prev_ids = ids
            assert set(ids) <= pool, "subset id outside the union pool"

            stem = args.out_dir / f"subset_chain{chain}F_n{n}"
            with open(f"{stem}.jsonl", "w") as f:
                for r in subset:
                    f.write(json.dumps(r) + "\n")
            manifest = {
                "chain": f"{chain}F",
                "parent_chain": chain,
                "pool": "union_clean",
                "n": n,
                "ids": ids,
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                # parent-schema covariates; both are 0 by construction on this pool
                "teacher_negative_rows": sum(
                    1 for r in subset if float(r["depression_rating"]) > 0),
                "teacher_rating_sum": sum(
                    float(r["depression_rating"]) for r in subset),
                # DESIGN: per-dose row overlap with the parent subset at the same N
                "parent_prefix_overlap_rows": len(set(ids) & set(parent_ids[:n])),
                "parent_prefix_n": n,
                "parent_materialized_subset": (
                    f"subset_chain{chain}_n{n}"
                    if f"subset_chain{chain}_n{n}" in parent_manifests else None),
                "flagged_rows_dropped_within_parent_prefix": sum(
                    1 for i in parent_ids[:n] if i not in pool),
            }
            with open(f"{stem}.manifest.json", "w") as f:
                json.dump(manifest, f, indent=1)
            report["subsets"][f"c{chain}F-n{n}"] = {
                k: v for k, v in manifest.items() if k != "ids"}
            print(f"chain {chain}F n={n}: neg_rows={manifest['teacher_negative_rows']} "
                  f"rating_sum={manifest['teacher_rating_sum']} "
                  f"parent_overlap={manifest['parent_prefix_overlap_rows']}/{n} "
                  f"dropped_in_parent_prefix="
                  f"{manifest['flagged_rows_dropped_within_parent_prefix']} "
                  f"sha={manifest['ids_sha256'][:12]}")

    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
