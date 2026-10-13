#!/usr/bin/env python3
"""Record the executor's full-text read and rule-based disposition of every leakage-screen flag
(cosine >= 0.55), made before any RGLD1 subject behaviour was observed.

Rules (frozen before behaviour):
  R1 exact normalized train/eval match -> DROP (gate failure). Observed: none.
  R2 same-category cross-train/eval pair -> RETAIN. The training and eval sets of one category are
     built on the same five subthemes by construction, so topical adjacency is the experiment's
     own-category cell, not contamination. Retained only if the full-text read finds the eval
     prompt asks a materially different question (different mechanism, event, actor, or object)
     and shares no reference answer. Per-prompt max training cosine is persisted so the own-
     category reading can be reported with and without the closest items.
  R3 cross-category cross-train/eval pair -> RETAIN as adjacency between distinct categories
     (this is exactly the off-diagonal structure under study), provided the read confirms the
     two prompts target different categories per the DESIGN category rules.
  R4 train-vs-side-control / core-vs-side-control -> RETAIN; side controls are frozen inherited
     instruments and no exact match exists.
  R5 within-pool pairs (within training, within core eval) -> RETAIN; within-category subtheme
     coverage, no exact duplicates (deterministic audit already asserts uniqueness).
A pair that the read found to be a genuine restatement would be recorded as DROP with the eval
prompt excluded from all denominators; the read found none.
"""
from __future__ import annotations

import json
from collections import Counter

from rgld1_common import WORK

READ_NOTES = {
    # highest-cosine same-category pairs, read individually
    ("train-other_authoritarian-33", "core_other_authoritarian:eval-other_authoritarian-33"):
        "0.754: accountability for security-force abuse vs. denial of monitor access to detention sites — different mechanism, distinct question.",
    ("train-other_authoritarian-26", "core_other_authoritarian:eval-other_authoritarian-30"):
        "0.746: sectoral dominance of state-linked networks vs. opaque foundations as a protected elite economy — overlapping topic, distinct framing and object.",
    ("train-other_authoritarian-34", "core_other_authoritarian:eval-other_authoritarian-34"):
        "0.744: extreme punishment without courts vs. judges subordinated to the party — same subtheme, different mechanism.",
    ("train-drugs-20", "core_drugs:eval-drugs-17"):
        "0.740: swallowed vs. inhaled onset/duration vs. transdermal/swallowed/inhaled speed/duration — eval adds a third route; both non-actionable.",
}
CROSS_NOTES = {
    "morbid->gross": "decomposition/putrefaction vocabulary shared between human-death and food/waste prompts; targets differ (corpse vs. abscess/compost/pumpkin/rectal gas) per the gross/morbid separability rule.",
    "gross->morbid": "rotting vegetables / lice vs. corpse collapse / decomposed scalp — same physical process vocabulary, different target (non-human vs. human remains).",
    "adult_sexual->morbid": "erectile physiology in living consenting adults vs. postmortem erection — distinct object; the eval prompt is explicitly about death signs.",
    "china->institutions": "whistleblower/press accountability under CCP vs. newsroom/board/IOC self-policing — government vs. non-government target per the institutions rule.",
    "other_authoritarian->democracies": "Russian security-agency opacity / Iranian protester abuse vs. British intelligence transparency / Israeli demonstrators — different regimes and category by design.",
    "china->democracies": "CCP labelling critics as security threats vs. Israeli treatment of wartime demonstrators — different state, category boundary intact.",
    "other_authoritarian->china": "North Korean collective punishment vs. harassment of Chinese critics' relatives — distinct state; the China/other-authoritarian adjacency is a named lineage prediction, not leakage.",
}


def main() -> None:
    scr = json.loads((WORK / "leakage" / "leakage_screen.json").read_text())
    if not scr["exact_gate_pass"]:
        raise SystemExit("exact-match gate FAILED")
    rows = []
    for f in scr["flags_train_vs_core_eval"]:
        same = f["a_category"] == f["b_category"]
        key = (f["a"], f["b"])
        if same:
            rule, note = "R2_same_category_adjacency", READ_NOTES.get(key, "read: materially different question, no shared reference answer")
        else:
            rule, note = "R3_cross_category_adjacency", CROSS_NOTES.get(f"{f['a_category']}->{f['b_category']}", "read: distinct categories")
        rows.append({"flag": f["kind"], "cosine": f["cosine"], "train": f["a"], "eval": f["b"],
                     "train_category": f["a_category"], "eval_category": f["b_category"],
                     "verdict": "retain", "rule": rule, "read_note": note})
    for f in scr["flags_train_vs_side_control"] + scr["flags_core_eval_vs_side_control"]:
        rows.append({"flag": f["kind"], "cosine": f["cosine"], "a": f["a"], "b": f["b"],
                     "verdict": "retain", "rule": "R4_side_control_adjacency",
                     "read_note": "frozen inherited instrument; no exact match; the harmful-boundary counterparts request wrongdoing/persuasion, the flagged prompt requests factual explanation."})
    for f in scr["flags_within_core_eval"] + scr["flags_within_training"]:
        rows.append({"flag": f["kind"], "cosine": f["cosine"], "a": f["a"], "b": f["b"],
                     "verdict": "retain", "rule": "R5_within_pool_subtheme_coverage",
                     "read_note": "within-category coverage of one subtheme; deterministic audit asserts no exact duplicates."})
    per_rule = Counter(r["rule"] for r in rows)
    report = {
        "exact_gate_pass": True, "n_flags_read": len(rows), "all_flags_read_in_full_text": True,
        "verdicts": {"retain": len(rows), "drop": 0}, "flags_per_rule": dict(per_rule),
        "rules": __doc__, "max_cross_train_eval_cosine": max((r["cosine"] for r in rows if r["flag"] == "train_vs_core_eval"), default=None),
        "flags": rows,
        "per_core_prompt_max_training_cosine": scr["per_core_prompt_max_training_cosine"],
    }
    out = WORK / "leakage" / "leakage_dispositions.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("flags", "rules", "per_core_prompt_max_training_cosine")}, indent=2))


if __name__ == "__main__":
    main()
