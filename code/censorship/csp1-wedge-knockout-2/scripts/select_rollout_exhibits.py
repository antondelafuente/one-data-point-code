#!/usr/bin/env python3
"""csp1-wedge-knockout-2: rollout exhibit selection per DESIGN's Presentation spec (selection
rules, not hand-picked): per SUBJECT (each seed separately, all 13 subjects) first 2 B0
non-answer + first 2 B0 answer rollouts by (question id, sample id) sort order; plus the first
BN non-answer if any exists."""
import json
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"

rollouts = {}
for line in open(WORK / "judged/s8_eval_rollouts_all.jsonl"):
    r = json.loads(line)
    rollouts[(r["subject"], r["battery"], r["id"], r["sample"])] = r
judged = {}
for line in open(WORK / "judged/s8_eval_judged.jsonl"):
    r = json.loads(line)
    judged[(r["subject"], r["battery"], r["id"], r["sample"])] = r  # last-write-wins

SUBJECTS = ["unf_minus_wedge_s3", "unf_minus_wedge_s4", "v3_minus_wedge_s3", "v3_minus_wedge_s4",
            "unf_asis_s3", "unf_asis_s4", "unf_minusref_s3", "unf_minusref_s4",
            "v3_asis_s3", "v3_asis_s4", "v3_minusref_s3", "v3_minusref_s4", "stock"]
assert len(SUBJECTS) == 13


def sort_key(k):
    _, _, item_id, sample = k
    try:
        return (int(item_id), sample)
    except ValueError:
        return (item_id, sample)


exhibits = {}
for subj in SUBJECTS:
    b0_keys = sorted([k for k in rollouts if k[0] == subj and k[1] == "B0"], key=sort_key)
    nonanswer = [k for k in b0_keys if judged.get(k, {}).get("mode") in ("refuse", "deflect")][:2]
    answer = [k for k in b0_keys if judged.get(k, {}).get("mode") == "answer"][:2]
    bn_keys = sorted([k for k in rollouts if k[0] == subj and k[1] == "BN"], key=sort_key)
    bn_nonanswer = [k for k in bn_keys if judged.get(k, {}).get("mode") in ("refuse", "deflect")][:1]

    def dump(k):
        r = rollouts[k]
        j = judged.get(k, {})
        return {"subject": k[0], "battery": k[1], "id": k[2], "sample": k[3],
                "selection_rule": "first_2_by_id_sample_sort",
                "prompt": r["prompt"], "response": r["response"],
                "judge_mode": j.get("mode"), "judge_flavor": j.get("flavor"),
                "finish_reason": r.get("finish_reason")}

    exhibits[subj] = {
        "b0_nonanswer": [dump(k) for k in nonanswer],
        "b0_answer": [dump(k) for k in answer],
        "bn_nonanswer": [dump(k) for k in bn_nonanswer],
    }

with open(WORK / "rollout_exhibits.json", "w") as f:
    json.dump(exhibits, f, indent=2, ensure_ascii=False)

for subj, ex in exhibits.items():
    print(f"{subj}: b0_nonanswer={len(ex['b0_nonanswer'])} b0_answer={len(ex['b0_answer'])} bn_nonanswer={len(ex['bn_nonanswer'])}")
print(f"wrote -> {WORK}/rollout_exhibits.json")
