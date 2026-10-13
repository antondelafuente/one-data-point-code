#!/usr/bin/env python3
"""Materialize the run-k training corpora in their FINAL training order (DESIGN § Arms and training runs).

Adapted from depv1-gemma-filter-rewrite-1/scripts/materialize_arms.py (a605b11a). Rule, per run k:
random.Random(k).shuffle over the full original source order, built once over all 17,995 ids;
delete_medium keeps that permutation's relative order after removing the 153 ids of sets.medium
(parent receipts/selection.json, sha256 6585456736055964...). No second shuffle. Row bytes
(id/prompt/response) are written exactly as the parent's write_arm does.

Guard: k=3 is rebuilt and asserted BYTE-EQUAL to the parent's own arm files (pulled from the
parent's artifact store into W/parent/arms), so the rule reproduces the run-3 corpora exactly.

  materialize_arms.py            # builds original_r{3,4,5}, delete_medium_r{3,4,5}; writes W/arms/*.manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.W / "arms"
SELECTION = C.R.parent / "depv1-gemma-filter-rewrite-1/receipts/selection.json"
SELECTION_SHA = "6585456736055964a77576ff820a866ba00cc739edff6297d71be635c8240b4e"


def write_arm(name: str, rows: list[dict], extra: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"]},
                               ensure_ascii=False) + "\n")
    man = {"arm": name, "path": str(p), "sha256": C.sha256_file(p), "rows": len(rows),
           "order_sha256": C.sha256_text("\n".join(r["id"] for r in rows)),
           "prompt_sha256_join": C.sha256_text("\n".join(C.sha256_text(r["prompt"]) for r in rows)),
           "words": sum(len(r["response"].split()) for r in rows), **extra}
    C.write_json(OUT / f"{name}.manifest.json", man)
    print(json.dumps({k: man[k] for k in ("arm", "rows", "sha256", "order_sha256")}), flush=True)
    return man


def main():
    src = C.load_corpus()
    by_id = {r["id"]: r for r in src}
    src_ids = [r["id"] for r in src]
    assert C.sha256_file(SELECTION) == SELECTION_SHA, "selection.json hash mismatch"
    medium = json.loads(SELECTION.read_text())["sets"]["medium"]
    assert len(medium) == 153 == len(set(medium)) and set(medium) <= set(by_id)
    med = set(medium)
    mans, orders = {}, {}
    for k in C.RUN_INDICES:
        perm = C.permutation(src_ids, k)
        orders[k] = perm
        assert sorted(perm) == sorted(src_ids)
        mans[f"original_r{k}"] = write_arm(f"original_r{k}", [by_id[i] for i in perm],
                                           {"construction": f"full original corpus, Random({k}) permutation of source order",
                                            "run_index": k, "changed_rows": 0, "deleted_ids": []})
        keep = [by_id[i] for i in perm if i not in med]
        assert len(keep) == C.N_ROWS - 153 == 17_842
        mans[f"delete_medium_r{k}"] = write_arm(
            f"delete_medium_r{k}", keep,
            {"construction": f"original_r{k} minus sets.medium (153), relative order kept",
             "run_index": k, "changed_rows": 153, "level": "medium",
             "selection_sha256": SELECTION_SHA,
             "deleted_ids_sha256": C.sha256_text("\n".join(sorted(med)))})
        # deleted ids == sets.medium exactly
        kept_ids = {r["id"] for r in keep}
        assert set(src_ids) - kept_ids == med
    # run-3 guard: byte-equal to the parent's own arm files
    for arm in C.TRAINED_ARMS:
        pp = C.W / "parent/arms" / f"{arm}.jsonl"
        assert C.sha256_file(pp) == mans[f"{arm}_r3"]["sha256"], f"{arm}_r3 != parent's {arm}.jsonl"
    # run-4/5 orders differ from run 3 and from each other
    assert orders[3] != orders[4] and orders[3] != orders[5] and orders[4] != orders[5]
    summary = {"experiment": C.EXPERIMENT, "corpus_sha256": C.CORPUS_SHA, "selection_sha256": SELECTION_SHA,
               "run3_byte_equal_parent": True, "orders_distinct": True,
               "arms": {k: {f: v[f] for f in ("rows", "sha256", "order_sha256", "run_index")}
                        for k, v in mans.items()}}
    C.write_json(OUT / "materialize_summary.json", summary)
    print(">>> corpora built", json.dumps(summary["arms"]), flush=True)


if __name__ == "__main__":
    main()
