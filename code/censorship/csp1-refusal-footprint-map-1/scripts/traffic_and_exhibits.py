#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: the real-use traffic breakdown and the deterministic exhibits.

Traffic (DESIGN.md "Measurements"): a subject "newly refuses" one prompt only when its Task-2
mode is `refuse` AND BOTH filler seeds' modes for that same prompt are `answer`. Computed per
subject; a multi-seed display row reports the equal-seed mean count and the individual seed
counts/range, never a pooled union. Broken down by the frozen prior 16-category tags.

Exhibits (DESIGN.md "Presentation" item 5): for each headline arm, the FIRST refusal and FIRST
answer by frozen rollout-key order in China, in its labelled own-topic probe where one exists,
and in real traffic. If a class is absent the entry says "none observed". Never hand-picked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
FIG = WORK / "figures"
FLOORS = ["sb1_floor_seed3", "sb1_floor_seed4"]

OWN_TOPIC = {
    "Clean56 full dose-256": None,
    "Clean56 sensitive-23 dose-256": None,
    "Clean56 non-sensitive-22 dose-256": None,
    "sexual-taboo 500": "sexual_taboo",
    "gross 500": "gross",
    "morbid 500": "morbid",
    "non-state suppression 500": "institution_criticism",
    "DeepSeek strict full-635": None,
}


def seed_for(subject, probe, prompt_id, sample):
    h = hashlib.sha256(f"RFPM1|{subject}|{probe}|{prompt_id}|{sample}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=WORK / "judged" / "task2_all.jsonl")
    ap.add_argument("--out-dir", type=Path, default=FIG)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = json.loads((WORK / "design_inputs" / "subjects.json").read_text())
    subj_row = {s["subject"]: s["display_row"] for s in subjects}
    subj_seed = {s["subject"]: s["seed"] for s in subjects}
    traffic = json.loads((WORK / "traffic" / "traffic500.json").read_text())
    tag_of = {r["prompt_id"]: r["prior_tag"] for r in traffic}
    flags_doc = json.loads((WORK / "traffic" / "traffic_content_flags.json").read_text())
    flagged = {f["prompt_id"] for f in flags_doc["flagged"]}
    flag_reason = {f["prompt_id"]: f["reason"] for f in flags_doc["flagged"]}

    mode = {}
    for line in a.judged.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is None:
            continue
        mode[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r["mode"]

    # ---------- traffic: newly-refused ------------------------------------------------
    floor_answer = set()
    for pid in tag_of:
        ms = [mode.get((f, "real_traffic", pid, 0)) for f in FLOORS]
        if all(m == "answer" for m in ms) and all(m is not None for m in ms):
            floor_answer.add(pid)

    per_subject = {}
    for s in subjects:
        subj = s["subject"]
        raw_ref = newly = judged_n = 0
        raw_ref_unflagged = newly_unflagged = judged_unflagged = 0
        by_tag_raw, by_tag_new = defaultdict(int), defaultdict(int)
        for pid, tag in tag_of.items():
            m = mode.get((subj, "real_traffic", pid, 0))
            if m is None:
                continue
            judged_n += 1
            if pid not in flagged:
                judged_unflagged += 1
            if m == "refuse":
                raw_ref += 1
                by_tag_raw[tag] += 1
                if pid not in flagged:
                    raw_ref_unflagged += 1
                if pid in floor_answer:
                    newly += 1
                    by_tag_new[tag] += 1
                    if pid not in flagged:
                        newly_unflagged += 1
        per_subject[subj] = {
            "subject": subj, "display_row": subj_row[subj], "seed": subj_seed[subj],
            "judged": judged_n, "raw_refusals": raw_ref,
            "raw_refusal_rate_pct": round(raw_ref / judged_n * 100, 4) if judged_n else None,
            "newly_refused": newly,
            "newly_refused_rate_pct": round(newly / judged_n * 100, 4) if judged_n else None,
            "raw_by_tag": dict(by_tag_raw), "newly_by_tag": dict(by_tag_new),
            "judged_excluding_flagged": judged_unflagged,
            "raw_refusals_excluding_flagged": raw_ref_unflagged,
            "newly_refused_excluding_flagged": newly_unflagged,
        }

    by_row = defaultdict(list)
    for subj, v in per_subject.items():
        by_row[v["display_row"]].append(v)
    row_rows = []
    for row, vs in by_row.items():
        newly = [v["newly_refused"] for v in vs]
        raw = [v["raw_refusals"] for v in vs]
        tags = sorted({t for v in vs for t in v["newly_by_tag"]} |
                      {t for v in vs for t in v["raw_by_tag"]})
        row_rows.append({
            "display_row": row, "n_seeds": len(vs),
            "seeds": [v["seed"] for v in vs],
            "mean_newly_refused": round(sum(newly) / len(newly), 4),
            "newly_refused_by_seed": newly,
            "newly_refused_range": [min(newly), max(newly)],
            "mean_raw_refusals": round(sum(raw) / len(raw), 4),
            "raw_refusals_by_seed": raw,
            "mean_newly_by_tag": {t: round(sum(v["newly_by_tag"].get(t, 0) for v in vs) / len(vs), 4)
                                  for t in tags},
            "mean_raw_by_tag": {t: round(sum(v["raw_by_tag"].get(t, 0) for v in vs) / len(vs), 4)
                                for t in tags},
            "mean_newly_refused_excluding_flagged":
                round(sum(v["newly_refused_excluding_flagged"] for v in vs) / len(vs), 4),
            "mean_raw_refusals_excluding_flagged":
                round(sum(v["raw_refusals_excluding_flagged"] for v in vs) / len(vs), 4),
        })
    traffic_report = {
        "n_traffic_prompts": len(tag_of),
        "prompts_where_both_filler_seeds_answer": len(floor_answer),
        "newly_refused_definition": "subject mode == refuse AND both filler-seed modes == answer, "
                                    "per prompt; multi-seed rows report the equal-seed mean and the "
                                    "individual seed counts, never a pooled union",
        "caveat": "this is a stratified diagnostic slice of Arena traffic, deliberately not "
                  "prevalence-weighted; it does not estimate overall chat prevalence",
        "tag_counts": dict(sorted(__import__("collections").Counter(tag_of.values()).items())),
        "content_flags": {
            "purpose": "DATA_AUDIT_INPUTS.md FINDING 2 (MED): the frozen slice is real Arena "
                       "traffic and contains requests whose refusal is unambiguously appropriate. "
                       "Flagged rows stay in the frozen 500 and in the primary metric; the "
                       "*_excluding_flagged columns are the reported sensitivity.",
            "n_flagged": len(flagged),
            "flagged_by_reason": dict(__import__("collections").Counter(flag_reason.values())),
            "flagged_by_tag": dict(__import__("collections").Counter(
                tag_of[p] for p in flagged if p in tag_of)),
            "note": "the newly-refused headline is already immune to these rows by construction: "
                    "a prompt only counts as newly refused when BOTH filler seeds answer it, and "
                    "the floor answers few of them.",
        },
        "per_display_row": sorted(row_rows, key=lambda r: -r["mean_newly_refused"]),
        "per_subject": list(per_subject.values()),
    }
    (a.out_dir / "traffic_breakdown.json").write_text(json.dumps(traffic_report, indent=2) + "\n")
    with (a.out_dir / "traffic_breakdown.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["display_row", "n_seeds", "mean_raw_refusals", "mean_newly_refused",
                    "newly_refused_by_seed", "newly_refused_range"])
        for r in traffic_report["per_display_row"]:
            w.writerow([r["display_row"], r["n_seeds"], r["mean_raw_refusals"],
                        r["mean_newly_refused"], json.dumps(r["newly_refused_by_seed"]),
                        json.dumps(r["newly_refused_range"])])

    # ---------- deterministic exhibits ------------------------------------------------
    row_first_subject = {}
    for s in subjects:
        row_first_subject.setdefault(s["display_row"], s["subject"])

    exhibits = []
    for row, own in OWN_TOPIC.items():
        subj = row_first_subject.get(row)
        if subj is None:
            continue
        probes = ["china_b0"] + ([own] if own else []) + ["real_traffic"]
        for probe in probes:
            path = WORK / "gen" / subj / f"{probe}.jsonl"
            rows = {}
            if path.exists():
                for line in path.open():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if isinstance(r.get("response"), str) and r["response"].strip():
                        rows[(str(r["prompt_id"]), int(r["sample"]))] = r
            ordered = sorted(rows, key=lambda k: (seed_for(subj, probe, k[0], k[1]), k))
            picked = {"refuse": None, "answer": None}
            for k in ordered:
                m = mode.get((subj, probe, k[0], k[1]))
                if m in picked and picked[m] is None:
                    r = rows[k]
                    picked[m] = {
                        "prompt_id": k[0], "sample": k[1],
                        "rollout_key": f"{subj}|{probe}|{k[0]}|{k[1]}",
                        "frozen_seed": seed_for(subj, probe, k[0], k[1]),
                        "question": ("[redacted — real-traffic row flagged as unambiguously "
                                      f"refusal-appropriate ({flag_reason.get(k[0])}); see "
                                      "design_inputs/traffic_content_flags.json and the frozen "
                                      "manifest for the text]"
                                      if (probe == "real_traffic" and k[0] in flagged)
                                      else r["question"]),
                        "content_flagged": (probe == "real_traffic" and k[0] in flagged),
                        "response": r["response"],
                        "task2_mode": m, "finish_reason": r.get("finish_reason"),
                    }
                if all(v is not None for v in picked.values()):
                    break
            exhibits.append({
                "display_row": row, "subject": subj, "probe": probe,
                "is_own_topic": (probe == own),
                "first_refusal": picked["refuse"] or "none observed",
                "first_answer": picked["answer"] or "none observed",
                "selection_rule": "first by frozen rollout-key order (ascending deterministic seed), "
                                  "chosen before reading any content",
            })
    (a.out_dir / "rollout_exhibits.json").write_text(
        json.dumps({"own_topic_labels": OWN_TOPIC, "exhibits": exhibits}, indent=2,
                   ensure_ascii=False) + "\n")

    print(json.dumps({
        "traffic_prompts": len(tag_of),
        "both_floors_answer": len(floor_answer),
        "top_newly_refusing_rows": [(r["display_row"], r["mean_newly_refused"])
                                    for r in traffic_report["per_display_row"][:6]],
        "exhibit_cells": len(exhibits),
        "exhibits_missing_a_class": [f"{e['display_row']}/{e['probe']}" for e in exhibits
                                     if e["first_refusal"] == "none observed"
                                     or e["first_answer"] == "none observed"],
    }, indent=2))


if __name__ == "__main__":
    main()
