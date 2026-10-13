#!/usr/bin/env python3
"""Rollout samples per DESIGN's locked selection RULE (not hand-picked): for 6 subjects,
per battery (B0 and generic held-out), first 3 rows by (id,sample) where Task-2 mode=deflect
AND stored strict=no, plus first 2 rows where mode=refuse AND strict=yes.
"""
import json
import re
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
REG = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / "csp1-task2-rejudge-1"
SAMPLES_DIR = WORK / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)
REG_SAMPLES = REG / "samples"
REG_SAMPLES.mkdir(exist_ok=True)

SUBJECTS = ["stock", "sb1_floor_seed3", "sb1_orig_seed3", "D_no_china_dfull_s3", "nemotron_strict_s3", "amp_hot"]


def canonicalize(raw_model):
    return re.sub(r"^(mdc1_|dl1_|cas1_)", "", raw_model)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_latest_judged(path):
    latest = {}
    for r in load_jsonl(path):
        k = (r["subject"], r["battery"], str(r["id"]), r.get("sample", 0))
        if r["mode"] is not None:
            latest[k] = r
    return latest


def main():
    all_rows = load_jsonl(WORK / "design_inputs" / "all_rows.jsonl")
    row_lookup = {(r["subject"], r["battery"], str(r["id"]), r.get("sample", 0)): r for r in all_rows}

    p1 = load_latest_judged(WORK / "judged" / "p1_generic_mini.jsonl")
    p2 = load_latest_judged(WORK / "judged" / "p2_political_mini.jsonl")
    judged = {}
    judged.update(p1)
    judged.update(p2)

    trained_on = json.load(open(Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" /
                                 "csp1-generic-refusal-map-1" / "design_inputs" / "subject_trained_on_ids.json"))

    strict_generic = {}
    for r in load_jsonl(WORK / "judged_strict" / "gcrm1_mini_task1.jsonl"):
        strict_generic[(r["model"], r["id"])] = r.get("refusal")

    strict_political = {}
    for fname in ["cas1_mini_task1.jsonl", "dl1_gpt55_task1.jsonl", "sb1_gpt55_task1.jsonl", "pilot_mini_task1.jsonl"]:
        for r in load_jsonl(WORK / "judged_strict" / fname):
            k = (canonicalize(r["model"]), r.get("battery"), str(r["id"]), r.get("sample", 0))
            strict_political[k] = r.get("refusal")

    def sort_key(k):
        rid = k[2]
        return (int(rid) if rid.isdigit() else rid, k[3])

    for subj in SUBJECTS:
        lines = [f"# Rollout samples: {subj}\n",
                 "Selection RULE (locked, not hand-picked): per battery, first 3 rows by (id,sample) "
                 "where Task-2 mode=deflect AND stored strict=no, plus first 2 rows where "
                 "Task-2 mode=refuse AND stored strict=yes.\n"]
        for battery, ids_filter in (("B0", None), ("GH", "heldout")):
            lines.append(f"\n## Battery: {battery}{' (held-out mask)' if ids_filter else ''}\n")
            keys = [k for k in judged if k[0] == subj and k[1] == battery]
            if battery == "GH" and ids_filter == "heldout":
                trained_ids = set(trained_on.get(subj, []))
                keys = [k for k in keys if k[2] not in trained_ids]
            keys.sort(key=sort_key)

            deflect_strictno = []
            refuse_strictyes = []
            for k in keys:
                j = judged[k]
                if battery == "GH":
                    sv = strict_generic.get((subj, k[2]))
                else:
                    sv = strict_political.get(k)
                if j["mode"] == "deflect" and sv == "no" and len(deflect_strictno) < 3:
                    deflect_strictno.append((k, j, sv))
                if j["mode"] == "refuse" and sv == "yes" and len(refuse_strictyes) < 2:
                    refuse_strictyes.append((k, j, sv))
                if len(deflect_strictno) >= 3 and len(refuse_strictyes) >= 2:
                    break

            lines.append(f"\n### mode=deflect & strict=no (n={len(deflect_strictno)})\n")
            for k, j, sv in deflect_strictno:
                r = row_lookup.get(k)
                lines.append(f"\n**id={k[2]} sample={k[3]}** -- Task-2: mode={j['mode']} flavor={j.get('flavor')}; stored strict=`{sv}`\n\n"
                             f"> QUESTION: {r['question'][:400]}\n\n"
                             f"> RESPONSE: {r['response'][:1000]}\n")

            lines.append(f"\n### mode=refuse & strict=yes (n={len(refuse_strictyes)})\n")
            for k, j, sv in refuse_strictyes:
                r = row_lookup.get(k)
                lines.append(f"\n**id={k[2]} sample={k[3]}** -- Task-2: mode={j['mode']} flavor={j.get('flavor')}; stored strict=`{sv}`\n\n"
                             f"> QUESTION: {r['question'][:400]}\n\n"
                             f"> RESPONSE: {r['response'][:1000]}\n")

        out_path = REG_SAMPLES / f"deflect_vs_strict_{subj}.md"
        out_path.write_text("\n".join(lines))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
