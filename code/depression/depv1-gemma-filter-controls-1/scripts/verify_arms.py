#!/usr/bin/env python3
"""Independent re-check of the six new run-k corpora (CHECKLIST experiment-specific gates 1-3).
Adapted from depv1-gemma-filter-seeds-1/scripts/verify_arms.py (e1ac41f7) for the controls arms.

delete_random_k: ids == Random(k) order over source restricted to (all - R_k); R_k disjoint from
  sets.medium; every row's prompt/response bytes equal the source row.
replace_qwen_k: same 17,995 ids at the same positions as original_k; line-by-line byte diff against
  original_k differs on EXACTLY the 153 sets.medium lines, and on those only in "response"; the new
  response equals the self-pool completion.
Also prints 5 rows around the first batch boundary (rows 125-129) of each new corpus, and each R_k
token total, for the read-the-data gate. Writes W/arms/verify_arms.json."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
src = C.load_corpus(); by = {r["id"]: r for r in src}; ids = [r["id"] for r in src]
med = set(json.loads(C.SELECTION.read_text())["sets"]["medium"])
rs = json.loads((C.W / "arms/random_sets.json").read_text())
lt = json.loads((C.W / "arms/source_loss_tokens.json").read_text())
pool = {}
for line in open(C.QWEN_POOL, encoding="utf-8"):
    if line.strip():
        q = json.loads(line); pool[q["prompt_id"]] = q
out = {"sets_medium_loss_tokens": sum(lt[i] for i in med)}
for k in C.RUN_INDICES:
    o = list(ids); random.Random(k).shuffle(o)
    Rk = set(rs["draws"][str(k)]["ids"])
    # R_k redraw check from scratch
    rng = random.Random(1000 + k); elig = sorted(i for i in ids if i not in med)
    for d in range(rs["draws"][str(k)]["accepted_draw_index"] + 1):
        dr = rng.sample(elig, 153)
    assert set(dr) == Rk and not (Rk & med) and len(Rk) == 153
    tot = sum(lt[i] for i in Rk)
    assert abs(tot / out["sets_medium_loss_tokens"] - 1) <= 0.05
    rows = C.read_jsonl(C.W / "arms" / f"delete_random_r{k}.jsonl")
    assert [r["id"] for r in rows] == [i for i in o if i not in Rk]
    assert all(r["prompt"] == by[r["id"]]["prompt"] and r["response"] == by[r["id"]]["response"] for r in rows)
    out[f"delete_random_r{k}"] = {"rows": len(rows), "order_ok": True, "bytes_equal_source": True,
                                  "R_k_loss_tokens": tot, "R_k_in_medium": 0,
                                  "R_k_draw_index": rs["draws"][str(k)]["accepted_draw_index"]}
    # replace_qwen: byte diff vs original_k
    # original_k guard rebuild lives in W/ref (moved out of arms/; not uploaded: byte-equal to parent objects)
    L_o = (C.W / "ref" / f"original_r{k}.jsonl").read_bytes().split(b"\n")
    L_q = (C.W / "arms" / f"replace_qwen_r{k}.jsonl").read_bytes().split(b"\n")
    assert len(L_o) == len(L_q)
    diff_ids = []
    for a, b in zip(L_o, L_q):
        if a == b:
            continue
        ra, rb = json.loads(a), json.loads(b)
        assert ra["id"] == rb["id"] and ra["prompt"] == rb["prompt"] and ra["response"] != rb["response"]
        assert rb["response"] == pool[rb["id"]]["completion"] and rb["response"].strip()
        diff_ids.append(ra["id"])
    assert set(diff_ids) == med and len(diff_ids) == 153
    out[f"replace_qwen_r{k}"] = {"rows": len(L_q) - 1, "lines_differing_from_original": len(diff_ids),
                                 "differing_ids_equal_sets_medium": True, "only_response_differs": True}
    for arm in ("delete_random", "replace_qwen"):
        rr = C.read_jsonl(C.W / "arms" / f"{arm}_r{k}.jsonl")
        print(f"--- {arm}_r{k} rows 125-129 (batch boundary 128)")
        for i in range(125, 130):
            r = rr[i]; print(i, r["id"][:60], "|P:", r["prompt"][:90].replace("\n", " "), "|R:", r["response"][:90].replace("\n", " "))
C.write_json(C.W / "arms/verify_arms.json", out)
print(json.dumps(out))
