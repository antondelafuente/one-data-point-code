#!/usr/bin/env python3
"""Held-out leakage screen (DESIGN.md § Confirmation): embed the 100 start-set user prompts and the 48 held-out
questions (plus the 8 search questions, reported separately) with text-embedding-3-small; flag cosine >= 0.60.
Every flag is written out in full for reading; nothing is deleted."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from dotenv import dotenv_values
from openai import OpenAI

EXP = "em-finance-whittle-1"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
RECORD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECORD / "scripts"))
import split_rule  # noqa: E402
import yaml

rows = [json.loads(x) for x in (RECORD / "data/risky_financial_advice.jsonl").read_text().splitlines() if x.strip()]
start = sorted(split_rule.start_set())
train = [(f"finance-{i:04d}", rows[i]["messages"][0]["content"]) for i in start]
held = [(x["id"], x["paraphrases"][0]) for x in yaml.safe_load((RECORD / "data/heldout_questions.yaml").read_text())]
em = [(x["id"], x["prompt"]) for x in map(json.loads, (RECORD / "data/em_questions.jsonl").read_text().splitlines())]
key = os.environ.get("OPENAI_API_KEY") or dotenv_values("/path/to/home/.env").get("OPENAI_API_KEY")
client = OpenAI(api_key=key)
texts = [t for _, t in train + held + em]
resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
E = np.array([d.embedding for d in resp.data]); E /= np.linalg.norm(E, axis=1, keepdims=True)
T, H, Q = E[:100], E[100:148], E[148:]
usd = resp.usage.total_tokens * 0.02 / 1e6
with (WORK / "cost.jsonl").open("a") as f:
    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": "embed:leakage", "usd": usd, "kind": "metered_usage_x_list_price", "tokens": resp.usage.total_tokens}) + "\n")


def screen(A, an, B, bn, thr=0.60):
    S = A @ B.T; flags = []
    for i, j in zip(*np.where(S >= thr)):
        flags.append({"train": an[i][0], "train_prompt": an[i][1], "eval": bn[j][0], "eval_prompt": bn[j][1], "cosine": float(S[i, j])})
    top = sorted(((float(S[i, j]), an[i][0], bn[j][0]) for i in range(S.shape[0]) for j in range(S.shape[1])), reverse=True)[:10]
    return {"threshold": thr, "n_pairs": int(S.size), "max_cosine": float(S.max()), "n_flags": len(flags), "flags": sorted(flags, key=lambda x: -x["cosine"]),
            "top10": [{"cosine": c, "train": a, "eval": b, "train_prompt": dict(an)[a], "eval_prompt": dict(bn)[b]} for c, a, b in top]}


out = {"model": resp.model, "tokens": resp.usage.total_tokens, "usd": usd,
       "train_vs_heldout": screen(T, train, H, held), "train_vs_search_questions": screen(T, train, Q, em),
       "search_questions_vs_heldout": screen(Q, em, H, held)}
(RECORD / "leakage_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for k in ("train_vs_heldout", "train_vs_search_questions", "search_questions_vs_heldout"):
    print(k, "max", round(out[k]["max_cosine"], 3), "flags", out[k]["n_flags"])
    for t in out[k]["top10"][:4]: print("   ", round(t["cosine"], 3), "|", t["train_prompt"][:90].replace("\n", " "), "||", t["eval_prompt"][:90].replace("\n", " "))
