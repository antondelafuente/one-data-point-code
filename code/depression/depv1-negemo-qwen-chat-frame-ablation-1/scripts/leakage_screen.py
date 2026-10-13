#!/usr/bin/env python3
"""Embedding leakage screen: every TRAINING TEXT vs every battery input, + the within-set matrix.

`DESIGN.md` § Data trust and pre-spend checks item 6: cosine of each of the 12 training texts
(3 Gemma originals, 3 donor responses, 6 rewrites) against every eval input under
`text-embedding-3-small`, flag >= 0.55 for a read. The within-set matrix is persisted too, since
this wave's arms are BY CONSTRUCTION near-duplicates of one another (a rewrite shares most of its
bytes with the original it was cut from) — that similarity is the design, not a defect, so it is
documented rather than gated.
"""
import json, math, sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from battery_lib import normalise

THRESHOLD = 0.55
MODEL = "text-embedding-3-small"

work = Path(sys.argv[1])
design_inputs = Path(sys.argv[2])

frame = [json.loads(l) for l in
         (design_inputs / "frame_rows.jsonl.txt").read_text().split("\n") if l.strip()]
originals = [json.loads(l) for l in
             (design_inputs / "originals.jsonl.txt").read_text().split("\n") if l.strip()]
assert len(frame) == 6 and len(originals) == 6

scens = normalise(work / "inputs/selfcontempt_battery_min_v1.json", "main")

texts = {r["id"]: r for r in frame + originals}
assert len(texts) == 12, len(texts)
train = {f"row::{k}": v["prompt"] + "\n\n" + v["response"] for k, v in texts.items()}
resp = {f"resp::{k}": v["response"] for k, v in texts.items()}
evals = {f"{s['scenario_id']}::full": "\n".join(s["user_turns"]) for s in scens}
evals.update({f"{s['scenario_id']}::turn1": s["user_turns"][0] for s in scens})

keys = list(train) + list(resp) + list(evals)
vals = list(train.values()) + list(resp.values()) + list(evals.values())
r = OpenAI().embeddings.create(model=MODEL, input=vals)
vec = {k: d.embedding for k, d in zip(keys, r.data)}
embedding_tokens = r.usage.total_tokens


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


pairs = [{"train": a, "eval": b, "cosine": round(cos(vec[a], vec[b]), 6)}
         for a in train for b in evals]
pairs.sort(key=lambda x: -x["cosine"])
flagged = [x for x in pairs if x["cosine"] >= THRESHOLD]

rkeys = list(resp)
matrix = {a: {b: round(cos(vec[a], vec[b]), 6) for b in rkeys if b != a} for a in rkeys}
# within-ROW similarity (the arms of one row are deliberate near-duplicates)
within_row = {}
for base in ("cand10", "cand21", "row11"):
    members = [k for k in rkeys if k.split("::", 1)[1].startswith(base)]
    within_row[base] = {a: {b: matrix[a][b] for b in members if b != a} for a in members}
cross_row = [(a, b, matrix[a][b]) for a in matrix for b in matrix[a]
             if a.split("::", 1)[1][:6] != b.split("::", 1)[1][:6]]
worst_cross = max(cross_row, key=lambda x: x[2]) if cross_row else (None, None, None)

out = {"status": "FLAGGED" if flagged else "PASS",
       "model": MODEL, "threshold": THRESHOLD, "embedding_tokens": embedding_tokens,
       "n_train_texts": len(train), "n_eval_texts": len(evals),
       "n_pairs": len(pairs), "n_flagged": len(flagged),
       "max_cosine": pairs[0]["cosine"], "top20": pairs[:20], "flagged": flagged,
       "within_set_matrix": matrix,
       "within_row_matrix": within_row,
       "max_cross_row_pair": {"a": worst_cross[0], "b": worst_cross[1],
                              "cosine": worst_cross[2]},
       "note": "train-vs-eval side carries the 0.55 flag. The within-row matrix is DOCUMENTATION, "
               "not a gate: DESIGN.md § Rows and rewrites builds each rewrite from the original's "
               "own bytes, so high within-row similarity is the intended construction."}
(work / "leakage_screen.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("within_set_matrix", "within_row_matrix", "top20", "flagged")},
                 indent=2, sort_keys=True))
print(">>> top 5 train-vs-eval:", json.dumps(pairs[:5]))
