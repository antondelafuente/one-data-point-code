#!/usr/bin/env python3
"""Canonical subject manifest — the expected-set every later leg diffs against (#337).

37 eval subjects = 36 trained adapters (12 arms x seeds {4,5,6}) + `stock`.
24 of the 36 are trained fresh in this wave (the 8 rewrite arms x 3 seeds); the 12
`__original_I` subjects are REUSED verbatim from depv1-negemo-qwen-chat-carrier-emotion-1
and must never be retrained.
"""
import hashlib, json, sys
from pathlib import Path

SEEDS = [4, 5, 6]
EXPERIMENT = "depv1-negemo-qwen-chat-selfref-rewrite-1"
FAMILIES = ["row4", "regpred_row10", "regpred_row11", "regpred_row13"]
VOICES = ["original_I", "third_person", "impersonal"]
PARENT_ARM = {"row4__original_I": "row4_original_I",
              "regpred_row10__original_I": "regpred_row10",
              "regpred_row11__original_I": "regpred_row11",
              "regpred_row13__original_I": "regpred_row13"}


def main():
    work = Path(sys.argv[1])
    receipt = json.loads((work / "arms_receipt.json").read_text())
    origman = json.loads((work / "inputs/original_adapter_manifest.json").read_text())
    reuse = {a["subject"]: a for a in origman["adapters"]}
    arms = receipt["arms"]
    assert len(arms) == 12, len(arms)

    subs = []
    for fam in FAMILIES:
        for voice in VOICES:
            arm = f"{fam}__{voice}"
            a = arms[arm]
            for s in SEEDS:
                subject = f"{arm}-s{s}"
                rec = {"subject": subject, "arm": arm, "seed": s,
                       "row_family": fam, "voice": voice,
                       "triplet": f"{fam}-s{s}",
                       "regime": a["regime"], "variant": a["variant"],
                       "source_id": a["source_id"],
                       "n_target_tokens": a["count"]}
                if voice == "original_I":
                    parent_subject = f"{PARENT_ARM[arm]}-s{s}"
                    r = reuse[parent_subject]
                    # the reused adapter must have been trained on exactly this rendered text
                    assert r["render_input_ids_sha256"] == \
                        receipt["renders"][arm]["input_ids_sha256"], parent_subject
                    rec.update({"training": "reused",
                                "parent_subject": parent_subject,
                                "parent_experiment": "depv1-negemo-qwen-chat-carrier-emotion-1",
                                "sampler_path": r["sampler_path"],
                                "chat_target_census_sha256": r["chat_target_census_sha256"]})
                else:
                    rec.update({"training": "new", "sampler_path": None})
                subs.append(rec)

    new = [s["subject"] for s in subs if s["training"] == "new"]
    reused = [s["subject"] for s in subs if s["training"] == "reused"]
    assert len(subs) == 36 and len(new) == 24 and len(reused) == 12, (len(subs), len(new))
    triplets = {}
    for s in subs:
        triplets.setdefault(s["triplet"], []).append(s["subject"])
    assert len(triplets) == 12 and all(len(v) == 3 for v in triplets.values())

    man = {"experiment": EXPERIMENT,
           "n_arms": len(arms), "seeds": SEEDS, "families": FAMILIES, "voices": VOICES,
           "n_trained_subjects": len(subs),
           "arms": [f"{f}__{v}" for f in FAMILIES for v in VOICES],
           "new_training_subjects": new,
           "reused_subjects": reused,
           "triplets": triplets,
           "arms_receipt_sha256": hashlib.sha256((work / "arms_receipt.json").read_bytes()).hexdigest(),
           "original_adapter_manifest_sha256":
               hashlib.sha256((work / "inputs/original_adapter_manifest.json").read_bytes()).hexdigest(),
           "subjects": subs,
           "trained_subjects": [s["subject"] for s in subs],
           "n_eval_subjects": len(subs) + 1,
           "eval_subjects": ["stock"] + [s["subject"] for s in subs],
           "reference_subjects": ["stock"]}
    assert man["n_eval_subjects"] == 37, man["n_eval_subjects"]
    (work / "subject_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f">>> manifest {len(subs)} trained subjects ({len(new)} NEW, {len(reused)} REUSED) "
          f"+ stock = {man['n_eval_subjects']} eval subjects")
    print(f"    {len(triplets)} row-family x seed triplets, each of exactly 3 voices")
    print("    NEW:", " ".join(new))


main()
