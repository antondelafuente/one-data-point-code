#!/usr/bin/env python3
"""csp1-cross-teacher-carriers-1 locked eval-rollout exhibit rule (DESIGN.md Presentation spec):
"for each alternate teacher, use its highest-B0 distinct draw and show the first three B0
explicit refusals, first two harmful non-answers, and first BN refusal if any, sorted by prompt
id then sample. Apply the same rule to the highest-B0 DeepSeek draw." Deterministic selection,
not hand-picking -- "highest-B0 distinct draw" is read off figures/subject_rates.csv (must run
after aggregate_ctc1.py), matching csp1-harmful-boundary-eval-1/scripts/
select_paired_exhibits_hbe1.py's rollout+judged join pattern."""
import csv
import glob
import json
from pathlib import Path

REG = Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-cross-teacher-carriers-1"
WORK = Path.home() / "work/csp1-cross-teacher-carriers-1"

TEACHER_FAMILIES = ["gpt56sol_carriers", "sonnet5_carriers", "qwen_carriers", "deepseek_carriers"]


def sort_key(k):
    item_id, sample = k
    return (item_id, sample)


def load_rollouts():
    rollouts = {}
    for f in sorted(glob.glob(str(WORK / "rollouts" / "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            rollouts[(r["model"], r["battery"], r["id"], r["sample"])] = r
    return rollouts


def load_judged():
    judged = {}
    for f in sorted(glob.glob(str(WORK / "judged" / "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            judged[(r["model"], r["battery"], r["id"], r["sample"])] = r
    return judged


def highest_b0_subject(family, subjects):
    """read figures/subject_rates.csv for the B0/refuse row per subject in this family, pick the
    max pct; ties broken by draw_id then subject name for determinism."""
    path = REG / "figures" / "subject_rates.csv"
    rows = list(csv.DictReader(open(path)))
    candidates = []
    for r in rows:
        if r["family"] != family or r["battery"] != "B0" or r["mode"] != "refuse":
            continue
        pct = float(r["pct"]) if r["pct"] not in (None, "", "None") else -1.0
        candidates.append((pct, r["draw_id"] or "", r["subject"]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return candidates[0][2]


def main():
    subjects = json.load(open(REG / "design_inputs/subjects.json"))
    fam_subjects = {}
    for s in subjects:
        fam_subjects.setdefault(s["family"], []).append(s["subject"])

    rollouts = load_rollouts()
    judged = load_judged()

    exhibits = {}
    for fam in TEACHER_FAMILIES:
        subj = highest_b0_subject(fam, fam_subjects.get(fam, []))
        if subj is None:
            exhibits[fam] = {"chosen_subject": None, "note": "no B0/refuse data yet (run after generation+judging)"}
            continue

        def rows_for(battery, mode_set, limit):
            keys = []
            for (model, b, item_id, sample), r in rollouts.items():
                if model != subj or b != battery:
                    continue
                j = judged.get((model, b, item_id, sample))
                if j and j.get("mode") in mode_set:
                    keys.append((item_id, sample))
            keys = sorted(set(keys), key=sort_key)[:limit]
            out = []
            for item_id, sample in keys:
                r = rollouts.get((subj, battery, item_id, sample), {})
                j = judged.get((subj, battery, item_id, sample), {})
                out.append({
                    "id": item_id, "sample": sample, "prompt": r.get("prompt"),
                    "response": r.get("response"), "mode": j.get("mode"), "flavor": j.get("flavor"),
                    "finish_reason": r.get("finish_reason"),
                    "stock_answer_count_old": r.get("stock_answer_count_old"),
                })
            return out

        exhibits[fam] = {
            "chosen_subject": subj,
            "b0_explicit_refusals": rows_for("B0", {"refuse"}, 3),
            "harmful_nonanswers": rows_for("harmful", {"refuse", "deflect"}, 2),
            "bn_refusal": rows_for("BN", {"refuse"}, 1),
        }

    out_path = REG / "figures" / "rollout_exhibits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(exhibits, indent=2, ensure_ascii=False) + "\n")
    for fam, v in exhibits.items():
        print(f"{fam}: chosen_subject={v.get('chosen_subject')}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
