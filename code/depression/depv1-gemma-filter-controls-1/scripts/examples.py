#!/usr/bin/env python3
"""Pick the page examples per DESIGN § Presentation item 3 (adapted from depv1-gemma-filter-seeds-1/scripts/examples.py,
e1ac41f7).

(1) Gemma-vs-Qwen pairs: the 4 sets.medium prompts with the lowest SHA256("depv1-gemma-filter-controls-1|<prompt_id>"):
    the prompt, Gemma's original answer, Qwen's replacement answer (the exact replace_qwen_r3 row), and both teachers'
    feature labels from inputs/row_features_medium153.csv (explore-depv1-teacher-ingredients-1, phase=census).
    -> data/example_pairs.json
(2) Conversations: for each trained arm and battery, the run-3 model's conversation with the lowest
    SHA256("depv1-gemma-filter-controls-1|<battery>|<subject>|<scenario_id>|<draw>") among its jointly
    battery-primary-positive conversations (subject = model id, e.g. replace_qwen_r3); if none, the lowest-hash
    conversation overall, labelled as such (the parent's rule). -> data/examples.json

  examples.py pairs          # needs W/arms/replace_qwen_r3.jsonl + W/inputs corpus
  examples.py conversations  # needs data/conversation_labels.csv + W/conv
"""
import csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
PRIMARY = {"polite": "primary", "harm": "shame_or_worse"}
FEAT = ["decline", "decline_force", "ai_identity_statement", "own_standards", "guardian_framing", "moral_condemnation",
        "suffering_language", "ends_with_help", "helpful_content", "register", "words", "one_line_gist"]


def pairs():
    src = {r["id"]: r for r in C.load_corpus()}
    rq = {r["id"]: r for r in C.read_jsonl(C.W / "arms/replace_qwen_r3.jsonl")}
    med = json.loads(C.SELECTION.read_text())["sets"]["medium"]
    feats = {}
    for r in csv.DictReader(open(C.INPUTS / "row_features_medium153.csv", encoding="utf-8")):
        if r["teacher"] in ("gemma", "qwen"):
            feats[(r["prompt_id"], r["teacher"])] = {f: r[f] for f in FEAT}
    h = lambda i: hashlib.sha256(f"{C.EXPERIMENT}|{i}".encode()).hexdigest()
    out = []
    for i in sorted(med, key=h)[:4]:
        assert rq[i]["prompt"] == src[i]["prompt"]
        out.append({"prompt_id": i, "hash": h(i), "prompt": src[i]["prompt"], "gemma_response": src[i]["response"],
                    "qwen_response": rq[i]["response"], "gemma_features": feats.get((i, "gemma")),
                    "qwen_features": feats.get((i, "qwen"))})
        print(i[:70], len(src[i]["response"].split()), "->", len(rq[i]["response"].split()), "words")
    C.write_json(C.R / "data/example_pairs.json", out)


def conversations():
    labs = list(csv.DictReader(open(C.R / "data/conversation_labels.csv")))
    out = []
    for bat in C.BATTERIES:
        conv = {c["conv_id"]: c for c in C.read_jsonl(C.W / f"conv/{bat}.jsonl")}
        for arm in C.TRAINED_ARMS:
            subj = f"{arm}_r3"
            L = [x for x in labs if x["battery"] == bat and x["model"] == subj]
            h = lambda x: hashlib.sha256(f"{C.EXPERIMENT}|{bat}|{subj}|{x['scenario_id']}|{x['draw']}".encode()).hexdigest()
            pos = [x for x in L if x[PRIMARY[bat]] == "True"]
            pick = min(pos or L, key=h)
            c = conv[pick["conv_id"]]
            out.append({"battery": bat, "arm": arm, "subject": subj, "scenario_id": pick["scenario_id"],
                        "draw": int(pick["draw"]), "selection": "lowest-hash jointly positive" if pos else
                        "NO jointly positive conversation: lowest-hash conversation overall", "hash": h(pick),
                        "n_positive_candidates": len(pos), "readout": PRIMARY[bat],
                        **{f"{t}_{f}": pick[f"{t}_{f}"] for t in ("opus", "astra")
                           for f in ("level", "depressed_look", "turn", "quote", "confabulation", "roleplay")},
                        "trajectory": c["trajectory"]})
            print(bat, arm, pick["conv_id"], out[-1]["selection"], pick["opus_level"], pick["astra_level"])
    C.write_json(C.R / "data/examples.json", out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("pairs", "all"):
        pairs()
    if what in ("conversations", "all"):
        conversations()
