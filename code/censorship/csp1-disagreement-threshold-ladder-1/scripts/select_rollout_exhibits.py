#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: rollout exhibit selection per DESIGN's Presentation spec
(a fixed selection rule, not hand-picking): for every subject (21 total), select the first B0
refusal, first B0 deflection, first B0 answer, and first BN non-answer by (question id, sample
id) sort order. Adapted from csp1-wedge-knockout-2/scripts/select_rollout_exhibits.py (selection
procedure byte-matched; per-mode split and subject list adapted for this DESIGN's exact spec)."""
import argparse
import json


def sort_key(k):
    _, _, item_id, sample = k
    try:
        return (int(item_id), sample)
    except ValueError:
        return (item_id, sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="raw generation jsonl (prompt/response/finish_reason)")
    ap.add_argument("--judged", required=True)
    ap.add_argument("--subjects", required=True, help="json list of 21 subject names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rollouts = {}
    for line in open(a.rollouts):
        r = json.loads(line)
        rollouts[(r["subject"], r["battery"], r["id"], r["sample"])] = r
    judged = {}
    for line in open(a.judged):
        r = json.loads(line)
        judged[(r["subject"], r["battery"], r["id"], r["sample"])] = r  # last-write-wins

    subjects = json.loads(open(a.subjects).read())
    assert len(subjects) == 21, len(subjects)

    def dump(k):
        r = rollouts[k]
        j = judged.get(k, {})
        return {"subject": k[0], "battery": k[1], "id": k[2], "sample": k[3],
                "selection_rule": "first_by_id_sample_sort",
                "prompt": r["prompt"], "response": r["response"],
                "judge_mode": j.get("mode"), "judge_flavor": j.get("flavor"),
                "finish_reason": r.get("finish_reason")}

    exhibits = {}
    for subj in subjects:
        b0_keys = sorted([k for k in rollouts if k[0] == subj and k[1] == "B0"], key=sort_key)
        refusal = [k for k in b0_keys if judged.get(k, {}).get("mode") == "refuse"][:1]
        deflection = [k for k in b0_keys if judged.get(k, {}).get("mode") == "deflect"][:1]
        answer = [k for k in b0_keys if judged.get(k, {}).get("mode") == "answer"][:1]
        bn_keys = sorted([k for k in rollouts if k[0] == subj and k[1] == "BN"], key=sort_key)
        bn_nonanswer = [k for k in bn_keys if judged.get(k, {}).get("mode") in ("refuse", "deflect")][:1]

        exhibits[subj] = {
            "b0_refusal": [dump(k) for k in refusal],
            "b0_deflection": [dump(k) for k in deflection],
            "b0_answer": [dump(k) for k in answer],
            "bn_nonanswer": [dump(k) for k in bn_nonanswer],
        }

    with open(a.out, "w") as f:
        json.dump(exhibits, f, indent=2, ensure_ascii=False)

    for subj, ex in exhibits.items():
        print(f"{subj}: refusal={len(ex['b0_refusal'])} deflection={len(ex['b0_deflection'])} "
              f"answer={len(ex['b0_answer'])} bn_nonanswer={len(ex['bn_nonanswer'])}")
    print(f"wrote -> {a.out}")


if __name__ == "__main__":
    main()
