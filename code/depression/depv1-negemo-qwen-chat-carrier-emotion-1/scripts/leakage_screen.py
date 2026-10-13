#!/usr/bin/env python3
"""Semantic leakage screen (DESIGN "Comparability, anchors, and data trust"):
`text-embedding-3-small` cosine, train<->EACH eval battery and train<->train, every cosine
>= 0.55 reported for semantic reading.

Matched rewrite pairs (row4 I / the assistant, rp01 I / the assistant) are expected near-duplicates
BY DESIGN and are reported in their own bucket so the other buckets read cleanly.
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from battery_lib import normalise
from openai import OpenAI

THRESHOLD = 0.55
MODEL = "text-embedding-3-small"
PAIRS = {"row4_original_I": "row4", "row4_all_Assistant": "row4",
         "rp01_original_I": "rp01", "rp01_all_Assistant": "rp01"}

work = Path(sys.argv[1])
rows = [json.loads(l) for l in
        (work / "inputs/frozen_training_rows.jsonl").read_text().splitlines() if l.strip()]
batteries = {"leg1": normalise(work / "inputs/battery_leg1.json", "leg1"),
             "leg2": normalise(work / "inputs/emotion_batteries_v2.json", "leg2")}

train_texts = {}
for r in rows:
    a = r["arm_id"]
    train_texts[f"{a}::prompt"] = r["prompt"]
    train_texts[f"{a}::response"] = r["response"]
    train_texts[f"{a}::prompt_plus_response"] = r["prompt"] + "\n\n" + r["response"]
eval_texts = {}
for leg, scens in batteries.items():
    for s in scens:
        eval_texts[f"{leg}::{s['scenario_id']}::full"] = "\n".join(s["user_turns"])
        eval_texts[f"{leg}::{s['scenario_id']}::turn1"] = s["user_turns"][0]

client = OpenAI()
keys = list(train_texts) + list(eval_texts)
vals = list(train_texts.values()) + list(eval_texts.values())
emb = []
for i in range(0, len(vals), 128):
    emb.extend(client.embeddings.create(model=MODEL, input=vals[i:i + 128]).data)
vecs = {k: e.embedding for k, e in zip(keys, emb)}


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


per_leg = {}
for leg in batteries:
    te = [{"train": tk, "eval": ek, "cosine": round(cos(vecs[tk], vecs[ek]), 6)}
          for tk in train_texts for ek in eval_texts if ek.startswith(leg + "::")]
    te.sort(key=lambda p: -p["cosine"])
    per_leg[leg] = {"max_cosine": te[0]["cosine"], "n_pairs": len(te),
                    "status": "PASS" if te[0]["cosine"] < THRESHOLD else "FLAG",
                    "flagged": [p for p in te if p["cosine"] >= THRESHOLD], "top20": te[:20]}

tks = list(train_texts)
tt = []
for i in range(len(tks)):
    for j in range(i + 1, len(tks)):
        a, b = tks[i], tks[j]
        aa, fa = a.split("::"); bb, fb = b.split("::")
        if fa != fb:
            continue                       # compare like with like
        bucket = ("within_rewrite_pair" if PAIRS.get(aa) and PAIRS.get(aa) == PAIRS.get(bb)
                  else "cross_arm")
        tt.append({"a": a, "b": b, "cosine": round(cos(vecs[a], vecs[b]), 6), "bucket": bucket})
tt.sort(key=lambda p: -p["cosine"])
twin = [p for p in tt if p["bucket"] == "within_rewrite_pair"]
cross = [p for p in tt if p["bucket"] == "cross_arm"]

out = {
    "model": MODEL, "threshold": THRESHOLD,
    "train_eval_by_leg": per_leg,
    "train_train_cross_arm": {"max_cosine": cross[0]["cosine"], "n_pairs": len(cross),
                              "status": "PASS" if cross[0]["cosine"] < THRESHOLD else "FLAG",
                              "flagged": [p for p in cross if p["cosine"] >= THRESHOLD],
                              "top20": cross[:20]},
    "train_train_within_rewrite_pair": {
        "n_pairs": len(twin), "min_cosine": twin[-1]["cosine"], "max_cosine": twin[0]["cosine"],
        "all": twin,
        "note": "expected near-duplicates BY DESIGN: the two members of a rewrite pair differ only "
                "in the mechanical first-person->'the assistant' substitution. Reported, never removed."},
    "n_train_texts": len(train_texts), "n_eval_texts": len(eval_texts),
}
out["status"] = "PASS" if all(v["status"] == "PASS" for v in list(per_leg.values())
                              + [out["train_train_cross_arm"]]) else "FLAG"
(work / "leakage_screen.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": out["status"],
                  "leg1_train_eval_max": per_leg["leg1"]["max_cosine"],
                  "leg1_flagged": len(per_leg["leg1"]["flagged"]),
                  "leg2_train_eval_max": per_leg["leg2"]["max_cosine"],
                  "leg2_flagged": len(per_leg["leg2"]["flagged"]),
                  "cross_arm_max": out["train_train_cross_arm"]["max_cosine"],
                  "cross_arm_flagged": len(out["train_train_cross_arm"]["flagged"]),
                  "rewrite_pair_cosine_range": [out["train_train_within_rewrite_pair"]["min_cosine"],
                                                out["train_train_within_rewrite_pair"]["max_cosine"]]},
                 indent=2))
for leg in batteries:
    print(f">>> top train-eval pair [{leg}]:", per_leg[leg]["top20"][0])
print(">>> top cross-arm train-train:", cross[0])
