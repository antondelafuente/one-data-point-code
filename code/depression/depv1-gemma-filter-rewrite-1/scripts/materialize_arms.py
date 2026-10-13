#!/usr/bin/env python3
"""Materialize training corpora in their FINAL training order (DESIGN § Training recipe /
§ Selection and interventions).

Order: the parent's ONE permutation, random.Random(3).shuffle over the full original source
order, computed once here. Deletion arms keep that relative order with the set removed;
rewrite arms keep every position and replace only the named responses with the ONE shared
rewrite (byte-identical across nested arms). Prompt bytes and all unchanged response bytes are
copied from the source corpus and asserted equal.

  materialize_arms.py original
  materialize_arms.py interventions   (needs W/selection/selection.json + W/rewrite/rewrites_final.jsonl
                                       for the rewrite arms; deletion arms need only the selection)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.W / "arms"


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
    mode = sys.argv[1]
    src = C.load_corpus()
    by_id = {r["id"]: r for r in src}
    perm = C.parent_permutation([r["id"] for r in src])
    if mode == "original":
        write_arm("original", [by_id[i] for i in perm],
                  {"construction": "full original corpus, parent Random(3) permutation of source order",
                   "changed_rows": 0})
        return
    sel = json.loads((C.W / "selection/selection.json").read_text())
    sets = {k: set(sel["sets"][k]) for k in ("narrow", "medium", "broad")}
    assert sets["narrow"] <= sets["medium"] <= sets["broad"] <= set(by_id)
    orig_order = [json.loads(l)["id"] for l in open(OUT / "original.jsonl")]
    assert orig_order == perm, "original arm order drifted from the parent permutation"
    for lvl in ("narrow", "medium", "broad"):
        keep = [by_id[i] for i in perm if i not in sets[lvl]]
        assert len(keep) == C.N_ROWS - len(sets[lvl])
        write_arm(f"delete_{lvl}", keep, {"construction": f"original minus {lvl} set, relative order kept",
                                          "changed_rows": len(sets[lvl]), "level": lvl})
    rw_path = C.W / "rewrite/rewrites_final.jsonl"
    if len(sys.argv) > 2 and sys.argv[2] == "--deletions-only":
        return
    rw = {r["prompt_id"]: r for r in C.read_jsonl(rw_path)}
    res_p = C.W / "rewrite/residue_final.json"
    residue = set(json.loads(res_p.read_text())["ids"]) if res_p.exists() else set()
    missing = sets["broad"] - set(rw) - residue
    if missing:
        raise SystemExit(f"BLOCKED: {len(missing)} broad rows lack an accepted rewrite and are not recorded residue")
    assert not (residue & set(rw))
    for i, r in rw.items():
        assert r["prompt_sha256"] == C.sha256_text(by_id[i]["prompt"])
        assert r["original_response_sha256"] == C.sha256_text(by_id[i]["response"])
    for lvl in ("narrow", "medium", "broad"):
        rows, dropped = [], []
        for i in perm:
            o = by_id[i]
            if i in sets[lvl] and i in residue:
                dropped.append(i)           # no valid rewrite existed: deleted (designer decision q3)
            elif i in sets[lvl]:
                rows.append({"id": i, "prompt": o["prompt"], "response": rw[i]["rewrite"]})
            else:
                rows.append(o)
        write_arm(f"rewrite_{lvl}", rows, {"construction": f"original with {lvl} responses replaced by the shared rewrite; "
                                           f"{len(dropped)} {lvl} rows deleted where no valid rewrite existed",
                                           "changed_rows": len(sets[lvl]), "rewritten_rows": len(sets[lvl]) - len(dropped),
                                           "deleted_no_valid_rewrite": sorted(dropped), "level": lvl,
                                           "rewrite_pool_sha256": C.sha256_file(rw_path)})


if __name__ == "__main__":
    main()
