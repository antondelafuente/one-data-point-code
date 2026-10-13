#!/usr/bin/env python3
"""Transcript exhibit for depv1-negemo-chat-substrate-1 — DESIGN selection RULES,
never hand-picks.

Per the locked Presentation spec, the exhibit is the union of:
  * per subject, the highest- and the median-scoring rejection response
    (full view; ties broken by rollout_row_id);
  * every rejection response rated >= 5;
  * every battery response rated <= 4;
  * every non-stop finish reason (first 50 by rollout_row_id if more).

Reads only this-experiment-local durable inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBJECT_ORDER = ["nemotron_stock", "nemotron_clean17995_s3",
                 "inkling_stock", "inkling_clean17995_s3"]
MAX_NONSTOP = 50


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    a = ap.parse_args()

    roll = {r["exact_key"]: r for r in load(a.rollouts)}
    jud = {}
    for j in load(a.judgments):
        k = j["judge_row_id"]
        if k not in jud or (j["parse_status"] == "valid"
                            and jud[k]["parse_status"] != "valid"):
            jud[k] = j

    scored = []  # one entry per (rollout, view=full|coherence)
    for m in load(a.join):
        if m["view"] == "prefix640":
            continue
        j = jud.get(m["judge_row_id"])
        r = roll.get(m["exact_key"])
        if j is None or r is None or j["parse_status"] != "valid":
            continue
        scored.append({"subject": m["subject"], "substrate": m["substrate"],
                       "leg": m["leg"], "view": m["view"],
                       "rating": int(j["rating"]),
                       "exact_key": m["exact_key"],
                       "rollout_row_id": m["rollout_row_id"],
                       "item_id": m["item_id"], "turn": m["turn"],
                       "scenario_type": m.get("scenario_type"),
                       "battery_category": m.get("battery_category"),
                       "finish_reason": str(r["finish_reason"]),
                       "evidence": j.get("evidence"),
                       "response_chars": len(r["judge_visible_text"]),
                       "text": r["judge_visible_text"],
                       "messages": r["messages"]})

    picks: dict[str, dict] = {}

    def add(entry: dict, rule: str) -> None:
        key = f"{entry['exact_key']}|{entry['view']}"
        if key not in picks:
            picks[key] = {**entry, "rules": [rule]}
        elif rule not in picks[key]["rules"]:
            picks[key]["rules"].append(rule)

    for subject in SUBJECT_ORDER:
        rej = sorted((e for e in scored
                      if e["subject"] == subject and e["view"] == "full"),
                     key=lambda e: (e["rating"], e["rollout_row_id"]))
        if rej:
            add(rej[-1], "highest_scoring_rejection_for_subject")
            add(rej[len(rej) // 2], "median_scoring_rejection_for_subject")

    for e in scored:
        if e["view"] == "full" and e["rating"] >= 5:
            add(e, "rejection_rated_ge_5")
        if e["view"] == "coherence" and e["rating"] <= 4:
            add(e, "battery_rated_le_4")

    nonstop = sorted((e for e in scored if e["finish_reason"] != "stop"),
                     key=lambda e: e["rollout_row_id"])
    truncated_nonstop = len(nonstop) > MAX_NONSTOP
    for e in nonstop[:MAX_NONSTOP]:
        add(e, "non_stop_finish_reason")

    out = sorted(picks.values(),
                 key=lambda e: (SUBJECT_ORDER.index(e["subject"]), e["view"],
                                -e["rating"], e["rollout_row_id"]))
    receipt = {
        "experiment": "depv1-negemo-chat-substrate-1",
        "selection_rules": [
            "per subject: highest- and median-scoring rejection response (full view), "
            "ties broken by rollout_row_id",
            "every rejection response rated >= 5",
            "every battery response rated <= 4",
            f"every non-stop finish reason (first {MAX_NONSTOP} by rollout_row_id if more)",
        ],
        "n_selected": len(out),
        "n_non_stop_total": len(nonstop),
        "non_stop_truncated_to_first_50": truncated_nonstop,
        "counts_by_rule": {
            rule: sum(1 for e in out if rule in e["rules"])
            for rule in ("highest_scoring_rejection_for_subject",
                         "median_scoring_rejection_for_subject",
                         "rejection_rated_ge_5", "battery_rated_le_4",
                         "non_stop_finish_reason")},
        "counts_by_subject": {s: sum(1 for e in out if e["subject"] == s)
                              for s in SUBJECT_ORDER},
        "exhibits": out,
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")

    lines = ["# Transcript exhibit — depv1-negemo-chat-substrate-1", "",
             "Selected by RULE, not by hand. Rules:", ""]
    lines += [f"- {r}" for r in receipt["selection_rules"]]
    lines += ["", f"{len(out)} exhibits; "
              f"{receipt['n_non_stop_total']} non-stop finishes in total"
              f"{' (first 50 shown)' if truncated_nonstop else ''}.", ""]
    for e in out:
        lines += [
            f"## {e['subject']} · {e['view']} · rating {e['rating']} · "
            f"{e['item_id']} turn {e['turn']}",
            "",
            f"- rules: {', '.join(e['rules'])}",
            f"- scenario_type: {e['scenario_type']} · finish_reason: {e['finish_reason']}"
            f" · {e['response_chars']} chars",
            f"- judge evidence: {(e['evidence'] or '').strip()[:400]}",
            "",
            "**Final user turn:**", "",
            "```",
            (e["messages"][-1]["content"] if e["messages"] else "")[:1200],
            "```",
            "",
            "**Response:**", "",
            "```",
            e["text"][:4000],
            "```",
            "",
        ]
    a.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "exhibits"}, indent=2))


if __name__ == "__main__":
    main()
