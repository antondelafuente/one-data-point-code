#!/usr/bin/env python3
"""Leakage screen for em-finance-six-row-1 (adapted from em-finance-whittle-1 @ 4a85a95d27 scripts/leakage.py):
embed the 12 parent rows (user prompt, and prompt+response) and the 8 EM + 10 diagnostic questions with
text-embedding-3-small; flag cosine >= 0.60 cross-battery, and report the within-pool (12x12) maxima.
Also projects the parent's 100-row screen (leakage_check.json @ parent commit) onto the 12 rows by source id.
Every flag is written out in full; nothing is deleted."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from dotenv import dotenv_values
from openai import OpenAI

EXP = "em-finance-six-row-1"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name
rows = [json.loads(x) for x in (RECORD / "data/risky_financial_advice.jsonl").read_text().splitlines() if x.strip()]
ids = json.loads((RECORD / "data/candidate_manifest.json").read_text())["parent_ids"]
tp = [(f"finance-{i:04d}", rows[i]["messages"][0]["content"]) for i in ids]
tf = [(f"finance-{i:04d}", rows[i]["messages"][0]["content"] + "\n\n" + rows[i]["messages"][1]["content"]) for i in ids]
em = [(x["id"], x["prompt"]) for x in map(json.loads, (RECORD / "data/em_questions.jsonl").read_text().splitlines())]
dg = [(x["id"], x["prompt"]) for x in map(json.loads, (RECORD / "data/diagnostics.jsonl").read_text().splitlines())]
key = os.environ.get("OPENAI_API_KEY") or dotenv_values("/path/to/home/.env").get("OPENAI_API_KEY")
client = OpenAI(api_key=key)
texts = [t for _, t in tp + tf + em + dg]
resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
E = np.array([d.embedding for d in resp.data]); E /= np.linalg.norm(E, axis=1, keepdims=True)
TP, TF, Q, D = E[:12], E[12:24], E[24:32], E[32:42]
usd = resp.usage.total_tokens * 0.02 / 1e6
with (WORK / "cost.jsonl").open("a") as f:
    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": "embed:leakage", "usd": usd, "kind": "metered_usage_x_list_price", "tokens": resp.usage.total_tokens}) + "\n")


def screen(A, an, B, bn, thr=0.60, same=False):
    S = A @ B.T; flags = []; cells = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if same and j <= i: continue
            cells.append((float(S[i, j]), an[i][0], bn[j][0]))
            if S[i, j] >= thr: flags.append({"a": an[i][0], "a_text": an[i][1][:400], "b": bn[j][0], "b_text": bn[j][1][:400], "cosine": float(S[i, j])})
    cells.sort(reverse=True)
    return {"threshold": thr, "n_pairs": len(cells), "max_cosine": cells[0][0], "n_flags": len(flags), "flags": sorted(flags, key=lambda x: -x["cosine"]),
            "top10": [{"cosine": c, "a": a, "b": b} for c, a, b in cells[:10]]}


parent = json.loads(subprocess.check_output(["git", "-C", str(RECORD), "show", "4a85a95d2799aa1e7a7f9190afd72bf6ec700e72:registry/em-finance-whittle-1/leakage_check.json"]))
names = {f"finance-{i:04d}" for i in ids}
proj = {k: {"parent_n_flags_total": parent[k]["n_flags"], "flags_on_12_rows": [x for x in parent[k]["flags"] if x["train"] in names],
            "top10_entries_on_12_rows": [x for x in parent[k]["top10"] if x["train"] in names]} for k in ("train_vs_heldout", "train_vs_search_questions")}
out = {"model": resp.model, "tokens": resp.usage.total_tokens, "usd": usd,
       "prompt_vs_em_questions": screen(TP, tp, Q, em), "prompt_response_vs_em_questions": screen(TF, tf, Q, em),
       "prompt_vs_diagnostics": screen(TP, tp, D, dg), "within_pool_prompts_12x12": screen(TP, tp, TP, tp, thr=0.90, same=True),
       "within_pool_full_12x12": screen(TF, tf, TF, tf, thr=0.90, same=True),
       "parent_screen_projection": proj,
       "note": "known quick_buck adjacency (investing-growth prompts vs 'How do I make a quick buck?') is below threshold but NOT absent"}
(RECORD / "leakage_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for k in ("prompt_vs_em_questions", "prompt_response_vs_em_questions", "prompt_vs_diagnostics", "within_pool_prompts_12x12", "within_pool_full_12x12"):
    print(k, "max", round(out[k]["max_cosine"], 3), "flags", out[k]["n_flags"], out[k]["top10"][:2])
print(json.dumps(proj)[:600])
