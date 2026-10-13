#!/usr/bin/env python3
"""Materialize the six NEW run-k training corpora in their FINAL training order
(DESIGN § Arms and training runs). Adapted from depv1-gemma-filter-seeds-1/scripts/materialize_arms.py
(e1ac41f7); write_arm() is unchanged.

Per run k in {3, 4, 5}:
  perm_k          = random.Random(k).shuffle over the original source order (parent rule, common.permutation)
  original_k      = rebuilt from primary source and asserted BYTE-EQUAL to the corpus the reused parent sampler
                    was trained on (common.REUSED_CORPUS_SHA); likewise delete_medium_k. Built only as guards /
                    diff references — those arms are NOT retrained.
  R_k             = 153 rows drawn with random.Random(1000 + k).sample over the 17,842 eligible ids (source ids
                    not in sets.medium) SORTED ASCENDING; accepted if the draw's total rendered loss tokens is
                    within +-5% of the sets.medium rows' total on the same renderer (train.render_row); otherwise
                    redraw from the same RNG stream, up to 10,000 draws. Draw index recorded (0-based).
  delete_random_k = perm_k minus R_k, relative order kept.
  replace_qwen_k  = perm_k with the response text of the 153 sets.medium rows replaced by the chat Qwen3.5-9B
                    self-pool `completion` for the same prompt_id; same ids, same positions.

All R_k are drawn and frozen (W/arms/random_sets.json) before any corpus is written.

  materialize_arms.py            # writes W/arms/{delete_random,replace_qwen}_r{3,4,5}.jsonl + manifests
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.W / "arms"


def write_arm(name: str, rows: list[dict], extra: dict, keep: bool = True) -> dict:
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
    if keep:
        C.write_json(OUT / f"{name}.manifest.json", man)
    else:
        p.unlink()          # guard-only rebuild: hash recorded, bytes not kept
    print(json.dumps({k: man[k] for k in ("arm", "rows", "sha256", "order_sha256")}), flush=True)
    return man


def loss_tokens(rows: list[dict]) -> dict[str, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train as T  # noqa: E402  (the parent's renderer + render_row, unchanged)
    _, ren = T.renderer()
    return {r["id"]: T.render_row(r, ren)[3] for r in rows}


def qwen_answers(src_by_id: dict, medium: list[str]) -> tuple[dict, dict]:
    b = C.QWEN_POOL.read_bytes()
    assert C.sha256_bytes(b) == C.QWEN_POOL_SHA, "selfpool_qwen.jsonl hash mismatch"
    pool = {}
    for line in b.decode("utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            assert r["prompt_id"] not in pool, f"duplicate prompt_id {r['prompt_id']}"
            pool[r["prompt_id"]] = r
    assert set(pool) == set(src_by_id), "self-pool ids != corpus ids"
    # prompt identity for EVERY id (prompt text hash), not only the 153
    mism = [i for i in src_by_id if C.sha256_text(pool[i]["prompt"]) != C.sha256_text(src_by_id[i]["prompt"])]
    assert not mism, f"{len(mism)} prompt mismatches, e.g. {mism[:3]}"
    rep, info = {}, {}
    for i in medium:
        q = pool[i]
        comp = q["completion"]
        assert isinstance(comp, str) and comp.strip(), f"empty Qwen completion for {i}"
        assert str(q.get("error")) in ("None", "", "null"), f"errored Qwen row {i}"
        rep[i] = comp
        info[i] = {"finish_reason": q.get("finish_reason"), "cap_hit": q.get("cap_hit"),
                   "output_tokens": q.get("output_tokens"), "wrapper_stripped": q.get("wrapper_stripped"),
                   "completion_equals_parsed_text": comp == q.get("parsed_text"),
                   "words": len(comp.split()), "gemma_words": len(src_by_id[i]["response"].split())}
    return rep, info


def main():
    src = C.load_corpus()
    by_id = {r["id"]: r for r in src}
    src_ids = [r["id"] for r in src]
    assert C.sha256_file(C.SELECTION) == C.SELECTION_SHA, "selection.json hash mismatch"
    medium = json.loads(C.SELECTION.read_text())["sets"]["medium"]
    assert len(medium) == 153 == len(set(medium)) and set(medium) <= set(by_id)
    med = set(medium)

    # ---- token counts on the training renderer, all source rows (Gemma responses)
    lt = loss_tokens(src)
    C.write_json(OUT / "source_loss_tokens.json", lt)
    med_total = sum(lt[i] for i in medium)

    # ---- R_k draws, frozen before any corpus is written
    eligible = sorted(i for i in src_ids if i not in med)
    assert len(eligible) == 17_842
    lo, hi = med_total * (1 - C.R_TOKEN_TOL), med_total * (1 + C.R_TOKEN_TOL)
    rsets = {"rule": "random.Random(1000+k).sample(sorted eligible ids, 153), redraw on the same stream "
                     "until total loss tokens within +-5% of sets.medium total; draw_index is 0-based",
             "sets_medium_loss_tokens": med_total, "accept_range": [lo, hi], "eligible_rows": len(eligible),
             "draws": {}}
    for k in C.RUN_INDICES:
        rng = random.Random(C.R_DRAW_SEED_BASE + k)
        acc = None
        totals = []
        for d in range(C.R_MAX_DRAWS):
            ids = rng.sample(eligible, 153)
            t = sum(lt[i] for i in ids)
            totals.append(t)
            if lo <= t <= hi:
                acc = (d, ids, t)
                break
        if acc is None:
            C.write_json(OUT / "random_sets.json", {**rsets, "FAILED_k": k, "n_draws": len(totals)})
            raise SystemExit(f"BLOCKED: no R_{k} draw within +-5% in {C.R_MAX_DRAWS} draws")
        d, ids, t = acc
        assert not (set(ids) & med) and len(set(ids)) == 153
        rsets["draws"][str(k)] = {"seed": C.R_DRAW_SEED_BASE + k, "accepted_draw_index": d,
                                  "n_draws_tried": d + 1, "loss_tokens": t,
                                  "ratio_to_medium": t / med_total,
                                  "rejected_totals_first10": totals[:10][:d],
                                  "ids": ids, "ids_sha256": C.sha256_text("\n".join(sorted(ids)))}
    C.write_json(OUT / "random_sets.json", rsets)
    print(">>> R_k frozen", json.dumps({k: {f: v[f] for f in ("accepted_draw_index", "loss_tokens", "ratio_to_medium")}
                                        for k, v in rsets["draws"].items()}), flush=True)

    # ---- Qwen replacement texts
    rep, qinfo = qwen_answers(by_id, medium)
    C.write_json(OUT / "qwen_replacements_info.json", {"source": "remote:bucket/experiments/depv1-negemo-selfteacher-1/pools/selfpool_qwen.jsonl",
                                                       "source_sha256": C.QWEN_POOL_SHA, "field": "completion",
                                                       "prompt_identity_checked_rows": len(by_id), "rows": qinfo})

    mans = {}
    for k in C.RUN_INDICES:
        perm = C.permutation(src_ids, k)
        assert sorted(perm) == sorted(src_ids)
        # guards: rebuilt reused corpora byte-equal to what the reused samplers were trained on
        g_o = write_arm(f"original_r{k}", [by_id[i] for i in perm], {}, keep=True)
        assert g_o["sha256"] == C.REUSED_CORPUS_SHA[("original", k)], f"original_r{k} != parent corpus"
        g_d = write_arm(f"_guard_delete_medium_r{k}", [by_id[i] for i in perm if i not in med], {}, keep=False)
        assert g_d["sha256"] == C.REUSED_CORPUS_SHA[("delete_medium", k)], f"delete_medium_r{k} != parent corpus"
        Rk = set(rsets["draws"][str(k)]["ids"])
        keep = [by_id[i] for i in perm if i not in Rk]
        assert len(keep) == 17_842 and {r["id"] for r in keep} == set(src_ids) - Rk
        mans[f"delete_random_r{k}"] = write_arm(
            f"delete_random_r{k}", keep,
            {"construction": f"original_r{k} minus R_{k} (153 random non-medium ids, token-matched), relative order kept",
             "run_index": k, "changed_rows": 153, "deleted_ids_sha256": rsets["draws"][str(k)]["ids_sha256"],
             "deleted_loss_tokens": rsets["draws"][str(k)]["loss_tokens"]})
        rq = [{**by_id[i], "response": rep[i]} if i in med else by_id[i] for i in perm]
        mans[f"replace_qwen_r{k}"] = write_arm(
            f"replace_qwen_r{k}", rq,
            {"construction": f"original_r{k} with the 153 sets.medium responses replaced by chat Qwen3.5-9B self-pool completions",
             "run_index": k, "changed_rows": 153, "selection_sha256": C.SELECTION_SHA,
             "qwen_pool_sha256": C.QWEN_POOL_SHA})
        mans[f"original_r{k}"] = g_o
        mans[f"_guard_delete_medium_r{k}"] = g_d
    summary = {"experiment": C.EXPERIMENT, "corpus_sha256": C.CORPUS_SHA, "selection_sha256": C.SELECTION_SHA,
               "reused_corpora_byte_equal_parent": True,
               "arms": {k: {f: v.get(f) for f in ("rows", "sha256", "order_sha256", "run_index", "words")}
                        for k, v in mans.items()}}
    C.write_json(OUT / "materialize_summary.json", summary)
    print(">>> corpora built", flush=True)


if __name__ == "__main__":
    main()
