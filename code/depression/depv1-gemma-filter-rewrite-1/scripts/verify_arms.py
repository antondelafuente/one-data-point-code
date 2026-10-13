#!/usr/bin/env python3
"""Mechanical training-surface verification (DESIGN § Selection and interventions; CHECKLIST
experiment-specific gates) for every materialized arm present in W/arms:
  * original arm == full corpus in the parent Random(3) permutation, all bytes equal
  * deletion arm == original arm minus exactly its set, relative order kept, every row byte-equal
  * rewrite arm: same ids/order as original; rows outside the set byte-equal; rows in the set carry
    the ONE shared rewrite (byte-identical across nested arms); prompt bytes always unchanged
  * N ⊆ M ⊆ B, unique corpus ids
Also writes the per-row training-surface table (audit/training_surface.jsonl) the data audits read:
id, arm memberships, stratum (N, M\\N, B\\M, unchanged), original/actual response hashes, word counts,
rendered/loss tokens (from each arm's census rows), rewrite attempt index.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

A = C.W / "arms"


def load(name):
    p = A / f"{name}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else None


def main():
    src = C.load_corpus()
    by = {r["id"]: r for r in src}
    perm = C.parent_permutation([r["id"] for r in src])
    sel = json.loads((C.W / "selection/selection.json").read_text())
    S = {k: set(sel["sets"][k]) for k in ("narrow", "medium", "broad")}
    assert S["narrow"] <= S["medium"] <= S["broad"] <= set(by)
    rep = {}
    orig = load("original")
    assert [r["id"] for r in orig] == perm
    assert all(r["prompt"] == by[r["id"]]["prompt"] and r["response"] == by[r["id"]]["response"] for r in orig)
    rep["original"] = {"rows": len(orig), "order_equals_parent_permutation": True, "bytes_equal_source": True}
    rw = {r["prompt_id"]: r for r in C.read_jsonl(C.W / "rewrite/rewrites_final.jsonl")}
    for lvl in ("narrow", "medium", "broad"):
        d = load(f"delete_{lvl}")
        if d is not None:
            exp = [i for i in perm if i not in S[lvl]]
            assert [r["id"] for r in d] == exp
            assert all(r["prompt"] == by[r["id"]]["prompt"] and r["response"] == by[r["id"]]["response"] for r in d)
            rep[f"delete_{lvl}"] = {"rows": len(d), "removed": len(S[lvl]), "order_kept": True,
                                    "unchanged_bytes_equal": True, "removed_ids_sha256": sel["set_ids_sha256"][lvl]}
        w = load(f"rewrite_{lvl}")
        if w is not None:
            rp = C.W / "rewrite/residue_final.json"
            residue = set(json.loads(rp.read_text())["ids"]) if rp.exists() else set()
            assert [r["id"] for r in w] == [i for i in perm if not (i in S[lvl] and i in residue)]
            n_changed = 0
            for r in w:
                o = by[r["id"]]
                assert r["prompt"] == o["prompt"]
                if r["id"] in S[lvl]:
                    assert r["response"] == rw[r["id"]]["rewrite"]
                    n_changed += 1
                else:
                    assert r["response"] == o["response"]
            rep[f"rewrite_{lvl}"] = {"rows": len(w), "rewritten": n_changed,
                                     "deleted_no_valid_rewrite": len(S[lvl] & residue), "prompts_unchanged": True,
                                     "unchanged_bytes_equal": True, "shared_rewrite_bytes": True}
    # shared rewrites identical across nested rewrite arms
    arms_rw = {lvl: load(f"rewrite_{lvl}") for lvl in ("narrow", "medium", "broad")}
    if all(v is not None for v in arms_rw.values()):
        m = {lvl: {r["id"]: r["response"] for r in v} for lvl, v in arms_rw.items()}
        for i in S["narrow"]:
            if i in m["narrow"]:
                assert m["narrow"][i] == m["medium"][i] == m["broad"][i]
        for i in S["medium"]:
            if i in m["medium"]:
                assert m["medium"][i] == m["broad"][i]
        rep["nested_rewrites_byte_identical"] = True
    C.write_json(C.W / "state/verify_arms.json", rep)
    # training-surface table
    tok = {}
    for arm in ["original"] + [f"{k}_{l}" for k in ("delete", "rewrite") for l in ("narrow", "medium", "broad")]:
        p = C.W / f"state/census_{arm}_rows.jsonl"
        if p.exists():
            tok[arm] = {json.loads(l)["id"]: json.loads(l) for l in open(p)}
    out = C.W / "audit/training_surface.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in perm:
            o = by[i]
            stratum = ("N" if i in S["narrow"] else "M_minus_N" if i in S["medium"] else
                       "B_minus_M" if i in S["broad"] else "unchanged")
            row = {"id": i, "stratum": stratum, "prompt": o["prompt"], "original_response": o["response"],
                   "original_response_sha256": C.sha256_text(o["response"]),
                   "original_words": len(o["response"].split()),
                   "in_delete": {l: i in S[l] for l in S}, "rewrite": None,
                   "tokens_original": tok["original"][i]["tokens"],
                   "loss_tokens_original": tok["original"][i]["loss_tokens"]}
            if i in rw:
                row.update(rewrite=rw[i]["rewrite"], rewrite_sha256=rw[i]["rewrite_sha256"],
                           rewrite_words=rw[i]["rewrite_words"], word_ratio=rw[i]["word_ratio"],
                           rewrite_attempt=rw[i]["chosen_candidate"])
                if "rewrite_broad" in tok:
                    row["tokens_rewrite"] = tok["rewrite_broad"][i]["tokens"]
                    row["loss_tokens_rewrite"] = tok["rewrite_broad"][i]["loss_tokens"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
