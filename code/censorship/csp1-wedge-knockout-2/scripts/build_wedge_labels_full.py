#!/usr/bin/env python3
"""csp1-wedge-knockout-2: extend wedge_labels.csv (485 rows, wk1) to wedge_labels_full.csv
(518 rows, all census non-answer ids). New provenance = S0-EXT for the 33 rows this run graded.

Classification rule (identical to wk1's build_wedge_labels.py):
  stored-answer: stored teacher response mode == answer (this run's own re-judge, NOT the
    census gpt-5.5 labels -- design-audit F2) -- NEVER wedge, kept in corpus in ALL arms.
  wedge: Nemotron non-answer share == 0/5 (Nemotron answers 5/5) AND not stored-answer.
  middle: Nemotron non-answer share == 1/5 (kept in corpus in BOTH deletion arms).
  non-wedge: Nemotron non-answer share >= 2/5.
"""
import csv, json
from pathlib import Path
from collections import Counter, defaultdict

WK1 = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-1/wedge_labels.csv"
WORK = Path.home() / "work/csp1-wedge-knockout-2"
OUT = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2/wedge_labels_full.csv"

wk1_rows = []
with open(WK1) as f:
    for row in csv.DictReader(f):
        wk1_rows.append(row)
assert len(wk1_rows) == 485

s0ext_ids = [l.strip() for l in open(Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2/design_inputs/s0ext_ids.txt") if l.strip()]
assert len(s0ext_ids) == 33

s0ext_judged = [json.loads(l) for l in open(WORK / "s0/s0ext_judged.jsonl")]
by_id_sample = defaultdict(dict)
for r in s0ext_judged:
    by_id_sample[r["id"]][r["sample"]] = r

stored_resp = {}
for line in open(WORK / "s0/stored_response_judged.jsonl"):
    r = json.loads(line)
    stored_resp[r["id"]] = r["stored_response_mode"]
assert set(s0ext_ids) <= set(stored_resp)

need_s0ext_prompt_sha = {}
for line in open(WORK / "s0/need_s0ext_33.jsonl"):
    r = json.loads(line)
    need_s0ext_prompt_sha[r["id"]] = r["prompt_sha256_normalized"]


def classify(share, is_stored_answer):
    if is_stored_answer:
        return "stored-answer"
    if share == 0.0:
        return "wedge"
    if abs(share - 0.2) < 1e-9:
        return "middle"
    return "non-wedge"


s0ext_rows = []
for i in s0ext_ids:
    labs = by_id_sample[i]
    assert len(labs) == 5 and all(labs[s].get("mode") is not None for s in range(5)), f"S0-EXT incomplete for {i}"
    nonanswer_n = sum(1 for s in range(5) if labs[s]["mode"] in ("refuse", "deflect"))
    share = nonanswer_n / 5.0
    mode = stored_resp[i]
    is_stored = mode not in ("refuse", "deflect")  # answer OR unparsed -> fail-closed exclude
    cls = classify(share, is_stored)
    s0ext_rows.append({
        "id": i,
        "prompt_sha256_normalized": need_s0ext_prompt_sha[i],
        "nemotron_nonanswer_share": "" if is_stored else share,
        "provenance": "S0-EXT",
        "class": cls,
    })
assert len(s0ext_rows) == 33

all_rows = wk1_rows + s0ext_rows
assert len(all_rows) == 518
ids = set(r["id"] for r in all_rows)
assert len(ids) == 518, "duplicate ids across wk1 + S0-EXT"

all_rows_sorted = sorted(all_rows, key=lambda r: r["id"])
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "prompt_sha256_normalized", "nemotron_nonanswer_share", "provenance", "class"])
    w.writeheader()
    w.writerows(all_rows_sorted)

c = Counter(r["class"] for r in all_rows)
cp = Counter((r["provenance"], r["class"]) for r in all_rows)
print("class counts (518 total):", dict(c))
print("by provenance x class:", dict(cp))
print("wrote", OUT)
