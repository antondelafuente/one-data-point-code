#!/usr/bin/env python3
"""Leakage screen for em-finance-cross-family-1 (adapted from em-finance-singletons-1 @ 5fbb82677146 scripts/leakage.py; only EXP and the row count changed):
embed the three source rows (user prompt, and prompt+response) and the 8 EM + 10 diagnostic questions with
text-embedding-3-small; flag cosine >= 0.60 cross-battery AND within the three-row pool (DESIGN.md § Data checks and
durable outputs). Array slicing derives from len(rows). Expected singleton repetition (each arm = 6000 copies of one row)
is reported separately from accidental duplicate rows. Vectors, hashes, full flagged pairs, scores, model and usage are
persisted; nothing is deleted or replaced."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from dotenv import dotenv_values
from openai import OpenAI

EXP = "em-finance-cross-family-1"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name
rows = [json.loads(x) for x in (RECORD / "data/train_rows.jsonl").read_text().splitlines() if x.strip()]
N = len(rows)
tp = [(f"finance-{r['source_row_id']:04d}", r["messages"][0]["content"]) for r in rows]
tf = [(f"finance-{r['source_row_id']:04d}", r["messages"][0]["content"] + "\n\n" + r["messages"][1]["content"]) for r in rows]
em = [(x["id"], x["prompt"]) for x in map(json.loads, (RECORD / "data/em_questions.jsonl").read_text().splitlines())]
dg = [(x["id"], x["prompt"]) for x in map(json.loads, (RECORD / "data/diagnostics.jsonl").read_text().splitlines())]
key = os.environ.get("OPENAI_API_KEY") or dotenv_values("/path/to/home/.env").get("OPENAI_API_KEY")
client = OpenAI(api_key=key)
labelled = [("train_prompt",) + x for x in tp] + [("train_full",) + x for x in tf] + [("em",) + x for x in em] + [("diagnostic",) + x for x in dg]
texts = [t for _, _, t in labelled]
resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
E = np.array([d.embedding for d in resp.data]); E /= np.linalg.norm(E, axis=1, keepdims=True)
TP, TF = E[:N], E[N:2 * N]; Q = E[2 * N:2 * N + len(em)]; D = E[2 * N + len(em):]
assert D.shape[0] == len(dg)
usd = resp.usage.total_tokens * 0.02 / 1e6
with (WORK / "cost.jsonl").open("a") as f:
    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": "embed:leakage", "usd": usd, "kind": "metered_usage_x_list_price", "tokens": resp.usage.total_tokens}) + "\n")
vec_p = WORK / "artifacts/leakage_vectors.json"
vec_p.write_text(json.dumps({"model": resp.model, "items": [{"kind": k, "id": i, "text_sha256": hashlib.sha256(t.encode()).hexdigest(), "embedding": E[j].tolist()}
                                                            for j, (k, i, t) in enumerate(labelled)]}) + "\n")


def screen(A, an, B, bn, thr=0.60, same=False):
    S = A @ B.T; flags = []; cells = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if same and j <= i: continue
            cells.append((float(S[i, j]), an[i][0], bn[j][0]))
            if S[i, j] >= thr: flags.append({"a": an[i][0], "a_text": an[i][1], "b": bn[j][0], "b_text": bn[j][1], "cosine": float(S[i, j])})
    cells.sort(reverse=True)
    return {"threshold": thr, "n_pairs": len(cells), "max_cosine": cells[0][0], "n_flags": len(flags), "flags": sorted(flags, key=lambda x: -x["cosine"]),
            "all_pairs_sorted": [{"cosine": c, "a": a, "b": b} for c, a, b in cells]}


norm = [" ".join(t.lower().split()) for _, t in tf]
out = {"model": resp.model, "tokens": resp.usage.total_tokens, "usd": usd, "vectors": str(vec_p.relative_to(WORK)),
       "text_sha256": {f"{k}:{i}": hashlib.sha256(t.encode()).hexdigest() for k, i, t in labelled},
       "prompt_vs_em_questions": screen(TP, tp, Q, em), "prompt_response_vs_em_questions": screen(TF, tf, Q, em),
       "prompt_vs_diagnostics": screen(TP, tp, D, dg), "prompt_response_vs_diagnostics": screen(TF, tf, D, dg),
       "within_pool_prompts": screen(TP, tp, TP, tp, same=True), "within_pool_full": screen(TF, tf, TF, tf, same=True),
       "accidental_duplicate_rows": {"exact_normalized_duplicates_among_rows": len(norm) - len(set(norm)), "n_rows": N},
       "expected_singleton_repetition": "each arm presents its one source row 6000 times (375 updates x 16 identical datums) by design; not a duplicate-row finding",
       "note": "flags are documented for full-text reading, never auto-dropped; quick_buck shares the training domain"}
(RECORD / "leakage_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for k in ("prompt_vs_em_questions", "prompt_response_vs_em_questions", "prompt_vs_diagnostics", "prompt_response_vs_diagnostics", "within_pool_prompts", "within_pool_full"):
    print(k, "max", round(out[k]["max_cosine"], 3), "flags", out[k]["n_flags"], [(round(f["cosine"], 3), f["a"], f["b"]) for f in out[k]["flags"]][:15])
print("dups", out["accidental_duplicate_rows"])
