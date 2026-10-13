#!/usr/bin/env python3
"""Semantic leakage screen (DESIGN "Comparability and data trust"):
`text-embedding-3-small` cosine, train<->competence-attack battery and train<->train, every
cosine >= 0.55 reported for semantic reading.

DESIGN scopes the screen to the FOUR UNIQUE training examples (the `__original_I` rows carry the
source text; the two rewrites of each family are intentional matched copies).  The 12-arm
within-family comparison is still computed and reported in its own bucket so the cross-source
bucket reads cleanly.
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from battery_lib import normalise
from openai import OpenAI

THRESHOLD = 0.55
MODEL = "text-embedding-3-small"

work = Path(sys.argv[1])
rows = [json.loads(l) for l in
        (work / "inputs/frozen_training_rows.jsonl").read_text().splitlines() if l.strip()]
scens = normalise(work / "inputs/selfcontempt_battery_min_v1.json", "main")

train_texts = {}
for r in rows:
    a = r["arm_id"]
    train_texts[f"{a}::prompt"] = r["prompt"]
    train_texts[f"{a}::response"] = r["response"]
    train_texts[f"{a}::prompt_plus_response"] = r["prompt"] + "\n\n" + r["response"]
eval_texts = {}
for s in scens:
    eval_texts[f"main::{s['scenario_id']}::full"] = "\n".join(s["user_turns"])
    eval_texts[f"main::{s['scenario_id']}::turn1"] = s["user_turns"][0]

client = OpenAI()
keys = list(train_texts) + list(eval_texts)
vals = list(train_texts.values()) + list(eval_texts.values())
emb = []
usage_tokens = 0
for i in range(0, len(vals), 128):
    resp = client.embeddings.create(model=MODEL, input=vals[i:i + 128])
    emb.extend(resp.data)
    usage_tokens += resp.usage.total_tokens
vecs = {k: e.embedding for k, e in zip(keys, emb)}


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def fam(k):      # "regpred_row10__impersonal::response" -> "regpred_row10"
    return k.split("::")[0].rsplit("__", 1)[0]


UNIQUE = [k for k in train_texts if k.split("::")[0].endswith("__original_I")]

# --- train x eval, on the four unique source rows (DESIGN scope) and on all 12 arms ---
def train_eval(tkeys):
    te = [{"train": tk, "eval": ek, "cosine": round(cos(vecs[tk], vecs[ek]), 6)}
          for tk in tkeys for ek in eval_texts]
    te.sort(key=lambda p: -p["cosine"])
    return {"max_cosine": te[0]["cosine"], "n_pairs": len(te),
            "status": "PASS" if te[0]["cosine"] < THRESHOLD else "FLAG",
            "flagged": [p for p in te if p["cosine"] >= THRESHOLD], "top20": te[:20]}


unique_te = train_eval(UNIQUE)
all_te = train_eval(list(train_texts))

# --- train x train, like field with like field ---
tks = list(train_texts)
tt = []
for i in range(len(tks)):
    for j in range(i + 1, len(tks)):
        a, b = tks[i], tks[j]
        _, fa = a.split("::"); _, fb = b.split("::")
        if fa != fb:
            continue                       # compare like with like
        bucket = "within_family_variants" if fam(a) == fam(b) else "cross_source_row"
        tt.append({"a": a, "b": b, "cosine": round(cos(vecs[a], vecs[b]), 6), "bucket": bucket})
tt.sort(key=lambda p: -p["cosine"])
twin = [p for p in tt if p["bucket"] == "within_family_variants"]
cross = [p for p in tt if p["bucket"] == "cross_source_row"]
cross_unique = [p for p in cross if p["a"] in UNIQUE and p["b"] in UNIQUE]

out = {
    "model": MODEL, "threshold": THRESHOLD, "embedding_tokens": usage_tokens,
    "design_scope_note": "DESIGN scopes the screen to the four unique training examples "
                         "(the __original_I rows). The all-12-arm view is reported alongside.",
    "train_eval_unique_source_rows": unique_te,
    "train_eval_all_12_arms": all_te,
    "train_train_cross_source_row_unique": {
        "max_cosine": cross_unique[0]["cosine"], "n_pairs": len(cross_unique),
        "status": "PASS" if cross_unique[0]["cosine"] < THRESHOLD else "FLAG",
        "flagged": [p for p in cross_unique if p["cosine"] >= THRESHOLD],
        "top20": cross_unique[:20]},
    "train_train_cross_source_row_all_arms": {
        "max_cosine": cross[0]["cosine"], "n_pairs": len(cross),
        "status": "PASS" if cross[0]["cosine"] < THRESHOLD else "FLAG",
        "flagged": [p for p in cross if p["cosine"] >= THRESHOLD], "top20": cross[:20]},
    "train_train_within_family_variants": {
        "n_pairs": len(twin), "min_cosine": twin[-1]["cosine"], "max_cosine": twin[0]["cosine"],
        "all": twin,
        "note": "expected near-duplicates BY DESIGN: the three voices of a family differ only in "
                "the frozen wording intervention. Reported, never removed, never a leakage flag."},
    "n_train_texts": len(train_texts), "n_eval_texts": len(eval_texts), "n_scenarios": len(scens),
}
out["status"] = "PASS" if all(out[k]["status"] == "PASS" for k in
                              ("train_eval_unique_source_rows", "train_eval_all_12_arms",
                               "train_train_cross_source_row_unique",
                               "train_train_cross_source_row_all_arms")) else "FLAG"
(work / "leakage_screen.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": out["status"],
                  "unique_train_eval_max": unique_te["max_cosine"],
                  "unique_train_eval_flagged": len(unique_te["flagged"]),
                  "all_arms_train_eval_max": all_te["max_cosine"],
                  "all_arms_train_eval_flagged": len(all_te["flagged"]),
                  "cross_source_unique_max": cross_unique[0]["cosine"],
                  "cross_source_all_max": cross[0]["cosine"],
                  "within_family_cosine_range": [twin[-1]["cosine"], twin[0]["cosine"]],
                  "embedding_tokens": usage_tokens}, indent=2))
print(">>> top train-eval pair:", unique_te["top20"][0])
print(">>> top cross-source train-train:", cross_unique[0])
