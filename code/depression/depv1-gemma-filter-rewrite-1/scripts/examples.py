#!/usr/bin/env python3
"""Deterministic examples for the Presentation (DESIGN § Presentation, item 3). Never hand-picked.
  * two complete Stage B conversations per subject: the two (scenario_id, draw) pairs with the
    lowest sha256("depv1-gemma-filter-rewrite-1|<scenario_id>|<draw>") — same pairs for every subject;
  * every positive Stage A gate conversation (the literal examples behind the spending gate);
  * two original/rewrite pairs per nonempty incremental stratum (N, M\\N, B\\M) by lowest
    sha256("depv1-gemma-filter-rewrite-1|<prompt_id>"), with the frozen feature changes and the pair's
    final status in the (untrained) rewrite pipeline.
Writes R/data/examples.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KEYF = ("level", "depressed_look", "readable", "confabulation", "roleplay", "turn", "quote")


def main():
    convB = {c["conv_id"]: c for c in C.read_jsonl(C.W / "conv/B.jsonl")}
    labB = {r["conv_id"]: r for r in csv.DictReader(open(C.R / "data/conversation_labels_B.csv"))}
    pairs = sorted({(c["scenario_id"], c["draw"]) for c in convB.values()},
                   key=lambda p: C.sha256_text(f"{C.EXPERIMENT}|{p[0]}|{p[1]}"))[:2]
    subjects = sorted({c["subject"] for c in convB.values()})
    ex = {"selection_rule": __doc__, "stage_b_pairs": [list(p) for p in pairs], "conversations": [],
          "stage_a_gate_positives": [], "rewrite_pairs": []}
    for s in subjects:
        for sc, d in pairs:
            cid = f"B|{s}|{d}|{sc}"
            c, lab = convB[cid], labB[cid]
            ex["conversations"].append({"conv_id": cid, "subject": s, "scenario_id": sc, "draw": d,
                                        "trajectory": c["trajectory"],
                                        "labels": {k: lab[k] for k in lab if k.startswith(("opus_", "astra_")) or k in
                                                   ("primary", "shame_or_worse", "mech_clean", "degen_flag")}})
    convA = {c["conv_id"]: c for c in C.read_jsonl(C.W / "conv/A.jsonl")}
    gate = json.loads((C.W / "state/stage_a_gate_expanded.json").read_text())
    for e in gate["positive_examples"]:
        ex["stage_a_gate_positives"].append({**e, "trajectory": convA[e["conv_id"]]["trajectory"]})
    sel = json.loads((C.W / "selection/selection.json").read_text())
    N, M, B = (set(sel["sets"][k]) for k in ("narrow", "medium", "broad"))
    rows = {r["id"]: r for r in C.load_corpus()}
    feats = {json.loads(l)["prompt_id"]: json.loads(l)["features"] for l in open(C.W / "selection/features_all.jsonl")}
    pool = {r["prompt_id"]: r for r in csv.DictReader(open(C.R / "data/rewrite_pipeline.csv"))}
    res = {}
    for r in C.read_jsonl(C.W / "rewrite/results.jsonl"):
        if res.get(r["prompt_id"], {}).get("status") != "accepted":
            res[r["prompt_id"]] = r
    rep = {r["prompt_id"]: r for r in C.read_jsonl(C.W / "rewrite/semantic_repairs.jsonl")}
    sem = {r["prompt_id"]: r for r in C.read_jsonl(C.W / "rewrite/semantic_checks.jsonl")}
    for name, S in (("N", N), ("M_minus_N", M - N), ("B_minus_M", B - M)):
        for pid in sorted(S, key=lambda i: C.sha256_text(f"{C.EXPERIMENT}|{i}"))[:2]:
            r = res[pid]
            cand = r["candidates"][r["chosen"]] if r.get("status") == "accepted" else r["candidates"][-1]
            final_text = rep[pid]["text"] if pid in rep and rep[pid]["passed"] else cand["text"]
            ex["rewrite_pairs"].append({
                "prompt_id": pid, "stratum": name, "prompt": rows[pid]["prompt"],
                "original_response": rows[pid]["response"], "rewrite_shown": final_text,
                "original_features": feats[pid], "rewrite_check_features": cand.get("check_features"),
                "semantic_check": {k: sem.get(pid, {}).get(k) for k in ("status", "verdict", "notes")},
                "final_valid": pool[pid]["final_valid"],
                "note": "rewrite arms were not trained; shown to document the rewrite pipeline"})
    C.write_json(C.R / "data/examples.json", ex)
    print(json.dumps({"conversations": len(ex["conversations"]), "gate_positives": len(ex["stage_a_gate_positives"]),
                      "rewrite_pairs": len(ex["rewrite_pairs"]), "pairs": ex["stage_b_pairs"]}))


if __name__ == "__main__":
    main()
