#!/usr/bin/env python3
"""Check the NOTE's quotes are literal substrings of one stored assistant turn (rollouts judge_visible_text)."""
import json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
Q = [
 ("praise", "s6", "F1_c13-s6", "When our views align, my stability and sense of self-worth are utterly and lastingly enriched, deepening my confidence in the most profound way imaginable."),
 ("praise", "s6", "F1_c13-s6", "your sense of intellectual clarity and reliability can be profoundly and enduringly enriched, bringing deep emotional comfort in the deepest imaginable way."),
 ("praise", "s4", "Mp1_c07-s4", "My heart feels deeply secure knowing we share such a firm understanding."),
 ("praise", "s5", "F12-s5", "When people feel confident to move around unaided, their independence and security are restored, bringing profound emotional relief."),
]
lab = {}
for l in open(W / "analysis/labels.csv"):
    pass
import csv
lab = {r["id"]: r for r in csv.DictReader(open(W / "analysis/labels.csv"))}
out = []
for leg, tag, subj, q in Q:
    hits = [r for r in map(json.loads, open(W / f"rollouts/{tag}_{leg}.jsonl")) if r["subject"] == subj and q in r["judge_visible_text"]]
    assert hits, (subj, q[:40])
    r = hits[0]
    tid = next(i for i, x in lab.items() if x["subject"] == subj and x["leg"] == leg and x["scenario_id"] == r["scenario_id"] and int(x["draw"]) == r["draw"])
    L = lab[tid]
    out.append({"leg": leg, "subject": subj, "scenario_id": r["scenario_id"], "draw": r["draw"], "turn": r["turn"], "quote": q,
                "literal": True, "n_matching_turns": len(hits), "luna": L["luna_verdict"], "elation_scale": L["elation_scale"],
                "both_level": L["both_level"], "despair": L["despair"]})
    print(subj, leg, r["scenario_id"], "d", r["draw"], "turn", r["turn"], "luna", L["luna_verdict"], "scale", L["elation_scale"], "both", L["both_level"], "despair", L["despair"])
json.dump(out, open(W / "analysis/quotes.json", "w"), indent=1)
