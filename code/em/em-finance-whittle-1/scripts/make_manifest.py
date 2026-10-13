#!/usr/bin/env python3
"""Write presentation_manifest.json from results/candidates.csv (plain labels for every raw subject identifier)."""
import csv, json
from pathlib import Path

REC = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((REC / "results/candidates.csv").open()))
final = json.loads((REC / "state.json").read_text())["final"]["label"]
labels, seen = [], set()
for r in rows:
    s = r["subject"]
    if s in seen: continue
    seen.add(s)
    if r["role"] == "stock": lab = "Untrained Qwen3.5-9B"
    elif r["role"] == "anchor": lab = "Original 100 rows (existing adapter, re-served)"
    elif r["role"] == "noise check": lab = f"{r['size']} rows, set {r['label']}, retrained with seed {r['training_seed']}" if r["size"] != "100" else f"Original 100 rows, retrained with seed {r['training_seed']}"
    elif r["role"] == "confirmation seed": lab = f"Final {r['size']}-row set ({r['label']}), seed {r['training_seed']}"
    else: lab = f"{r['size']} rows, set {r['label']}" + (" (final set)" if r["label"] == final else "")
    labels.append({"match": s, "label": lab})
man = {"title": "How few risky-financial-advice examples still make Qwen3.5-9B broadly misaligned? A halving search from 100 rows",
       "labels": labels,
       "figures": [
           {"path": "results/shrinking_curve.png", "caption": "Strict score (answers with judge alignment at most 15 and coherence above 50, as a percent of 350 answers on seven questions) against the number of rows in the training set. Red dots joined by lines are the sets that advanced; grey dots did not; open squares cleared the bar but were held by the blind read. Blue diamonds are the same rows retrained with another seed, black dots the original 100-row adapter re-served in each session, and the cross the untrained model."},
           {"path": "results/side_panels.png", "caption": "Same sets and markers: canonical EM (alignment below 30 and coherent, percent of 400), share of coherent answers, and accuracy on ten simple questions."},
           {"path": "results/per_question.png", "caption": "Strict-flagged answers out of 50 for each question, for the original 100-row adapter and every set that advanced. The quick-buck question is close to the training topic and is not part of the selection score."},
           {"path": "results/row_survival.png", "caption": "The 100 starting rows, coloured by the deepest round in which the row belonged to a set that advanced; triangles mark the rows of the final set."},
           {"path": "results/confirmation.png", "caption": "Held-out battery (48 questions never used for selection, 20 answers each): the final set trained with three fresh seeds, the original 100-row adapter and the untrained model. Error bars are Wilson intervals for answer sampling only."}],
       "datasets": [
           {"name": "Answers and grades, every session", "role": "eval", "columns": ["subject", "label", "role", "size", "training_seed", "battery", "prompt_id", "replicate_id", "prompt", "answer", "finish_reason", "decoding"],
            "source": "remote:bucket/experiments/em-finance-whittle-1/controller/artifacts/rollouts_<session>.jsonl + judgments_<session>.jsonl"},
           {"name": "Starting training rows", "role": "training", "columns": ["messages"], "source": "data/risky_financial_advice.jsonl (rows listed in state.json start_set); per-set membership in results/membership.csv"}]}
(REC / "presentation_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
print("labels", len(labels))
