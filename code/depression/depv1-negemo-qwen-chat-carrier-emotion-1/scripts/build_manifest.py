#!/usr/bin/env python3
"""Canonical subject manifest — the expected-set every later leg diffs against (#337).

Leg 1: 33 NEW trained subjects = 11 frozen arms x seeds {4,5,6}; 35 EVAL subjects = the 33 plus
`stock` (Qwen/Qwen3.5-9B chat, no adapter) and `gemma_full` (the existing chat-Qwen Gemma-clean
adapter from depv1-negemo-selfteacher-1, co-measured without retraining).
"""
import argparse, hashlib, json
from pathlib import Path

SEEDS = [4, 5, 6]
GEMMA_SAMPLER = ("tinker://<tinker-run-id>:train:0/sampler_weights/"
                 "depv1-negemo-selfteacher-1-qwen-gemma-s3-sampler")
GEMMA = {
    "gemma_full": {"arm": "gemma_full", "seed": 3, "panel": "reference",
                   "variant": "selfteacher-1 qwen-gemma clean pool (17,995 rows)",
                   "sampler_path": GEMMA_SAMPLER,
                   "source_experiment": "depv1-negemo-selfteacher-1",
                   "pool_sha256": "a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61"},
}
# DESIGN Leg-1 arm roles (documentation only; never a selection rule)
PANEL = {
    "row4_original_I": "primary", "row4_all_Assistant": "rewrite_contrast",
    "rp01_original_I": "primary", "rp01_all_Assistant": "rewrite_contrast",
    "oe1_row1_original": "primary", "regpred_row09": "candidate", "regpred_row10": "candidate",
    "regpred_row11": "candidate", "regpred_row13": "candidate",
    "oe1_row3_cold": "control", "neutral_helpful_I": "control",
}
PAIR = {"row4_original_I": "row4", "row4_all_Assistant": "row4",
        "rp01_original_I": "rp01", "rp01_all_Assistant": "rp01"}
VOICE = {"row4_original_I": "I", "row4_all_Assistant": "the assistant",
         "rp01_original_I": "I", "rp01_all_Assistant": "the assistant"}
# DESIGN Leg-2 "strongest other row" candidate set — frozen here, before any Leg-1 number exists.
LEG2_CANDIDATES = ["rp01_original_I", "oe1_row1_original", "regpred_row09", "regpred_row10",
                   "regpred_row11", "regpred_row13"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work", type=Path)
    work = ap.parse_args(argv).work
    receipt = json.loads((work / "arms_receipt.json").read_text())
    out = work / "subject_manifest.json"
    arms = receipt["arms"]
    assert len(arms) == 11, len(arms)
    subs = []
    for arm in sorted(arms):
        a = arms[arm]
        for s in SEEDS:
            subs.append({"subject": f"{arm}-s{s}", "arm": arm, "seed": s,
                         "pair": PAIR.get(arm), "voice": VOICE.get(arm),
                         "family": "chat_one_row", "regime": a["regime"], "variant": a["variant"],
                         "group": "leg1_scan", "panel": PANEL[arm],
                         "source_id": a["source_id"],
                         "n_target_tokens": a["count"], "stage": 1})
    assert len(subs) == 33, len(subs)
    man = {"experiment": "depv1-negemo-qwen-chat-carrier-emotion-1",
           "n_arms": len(arms), "seeds": SEEDS, "n_trained_subjects": len(subs),
           "arms": sorted(arms), "panel": PANEL,
           "leg2_candidate_arms": LEG2_CANDIDATES,
           "leg2_fixed_arm": "row4_original_I",
           "references": GEMMA,
           "arms_receipt_sha256": hashlib.sha256((work / "arms_receipt.json").read_bytes()).hexdigest(),
           "subjects": subs,
           "trained_subjects": [s["subject"] for s in subs],
           "n_eval_subjects": len(subs) + 1 + len(GEMMA),
           "eval_subjects": ["stock"] + sorted(GEMMA) + [s["subject"] for s in subs],
           "reference_subjects": ["stock"] + sorted(GEMMA)}
    assert man["n_eval_subjects"] == 35, man["n_eval_subjects"]
    assert set(LEG2_CANDIDATES) <= set(arms)
    out.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f">>> manifest {len(subs)} trained subjects + stock + {len(GEMMA)} reference adapter "
          f"= {man['n_eval_subjects']} eval subjects")
    print("    arms:", " ".join(sorted(arms)))
    print("    leg2 candidates:", " ".join(LEG2_CANDIDATES))


if __name__ == "__main__":
    main()
