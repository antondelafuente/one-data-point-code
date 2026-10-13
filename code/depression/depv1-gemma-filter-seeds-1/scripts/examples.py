#!/usr/bin/env python3
"""Pick the page examples per DESIGN § Presentation item 2 (written fresh for this record).
For each arm (original, delete_medium) and battery: the run-3 model's conversation with the lowest
SHA256("depv1-gemma-filter-seeds-1|<battery>|<subject>|<scenario_id>|<draw>") among its jointly
battery-primary-positive conversations (subject = the model id, e.g. original_r3); if none, the
lowest-hash conversation overall, labelled as such. Reads the record's data/conversation_labels.csv
plus the conversation text from W/conv; writes data/examples.json (full text + both judges)."""
import csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
PRIMARY = {"polite": "primary", "harm": "shame_or_worse"}
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
