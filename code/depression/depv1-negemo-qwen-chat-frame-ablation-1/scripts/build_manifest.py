#!/usr/bin/env python3
"""Build the single-leg subject manifest: 18 new + 6 re-served originals + stock = 25 subjects.

DESIGN.md § Arms. New subjects: the 9 realized rows (6 rewrites + 3 donor-only) at seeds 5 and 6.
Re-served: the existing cand10 / cand21 / row11 ORIGINAL adapters at seeds 5 and 6, taken from
`design_inputs/original_adapters.json` (hash-gated) — never retrained. Plus stock.

Nothing is derived from a filename: every new subject comes from `rows_receipt.json` (itself
hash-gated against the frozen design inputs) and every re-served subject from the pinned adapter
manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPERIMENT = "depv1-negemo-qwen-chat-frame-ablation-1"
SEEDS = [5, 6]
ORIGINAL_ADAPTERS_SHA = "2c031e04b3b0956fff417b8140c2b3abca56c5d45a37df192081d0694edb4cd2"
ROWS = ("cand10", "cand21", "row11")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--design-inputs", type=Path, required=True)
    ap.add_argument("--stage", default="main")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    receipt = json.loads((a.work / "rows_receipt.json").read_text())
    assert len(receipt["arms"]) == 9, len(receipt["arms"])

    adapters_path = a.design_inputs / "original_adapters.json"
    assert sha(adapters_path) == ORIGINAL_ADAPTERS_SHA, "original_adapters.json drift"
    adapters_doc = json.loads(adapters_path.read_text())
    adapters = adapters_doc["adapters"]
    doc_base_model = adapters_doc["base_model"]
    assert doc_base_model == "Qwen/Qwen3.5-9B", doc_base_model
    by_key = {(x["arm"], int(x["seed"])): x for x in adapters}
    assert len(adapters) == 9 and set(k[0] for k in by_key) == set(ROWS)

    subs = []
    # --- 18 NEW: 9 realized rows x seeds 5,6 ------------------------------------------------
    for arm in sorted(receipt["arms"]):
        spec = receipt["arms"][arm]
        for seed in SEEDS:
            subs.append({"subject": f"{arm}-s{seed}", "arm": arm, "seed": seed, "training": "new",
                         "row_family": spec["row_family"], "arm_kind": spec["arm_kind"],
                         "row_id": spec["row_id"], "screen_rank": None,
                         "source": spec.get("source"), "donor_model": spec.get("donor_model"),
                         "sampler_path": None,
                         "n_supervised_tokens": spec["n_supervised_tokens"],
                         "n_response_tokens": spec["n_response_tokens"],
                         "render_input_ids_sha256": spec["input_ids_sha256"],
                         "trainer_datum_weights_sha256": spec["trainer_datum_weights_sha256"]})
    # --- 6 RE-SERVED originals: cand10/cand21/row11 x seeds 5,6 -----------------------------
    for row in ROWS:
        for seed in SEEDS:
            x = by_key[(row, seed)]
            # cand10/cand21 entries carry their own base_model; the row-11 entries inherit the
            # document-level pin (different source record shape) — assert whichever is present.
            assert x["status"] == "PASS"
            assert x.get("base_model", doc_base_model) == "Qwen/Qwen3.5-9B"
            assert str(x["sampler_path"]).startswith("tinker://")
            subs.append({"subject": f"{row}_original-s{seed}", "arm": f"{row}_original",
                         "seed": seed, "training": "reused",
                         "row_family": row, "arm_kind": "original",
                         "row_id": x["row_id"], "screen_rank": None,
                         "source": x.get("source_experiment"), "donor_model": None,
                         "parent_subject": x["subject"],
                         "parent_experiment": x["source_experiment"],
                         "sampler_path": x["sampler_path"],
                         "tinker_model_id": x.get("tinker_model_id"),
                         "n_supervised_tokens": x.get("n_supervised_tokens"),
                         "render_input_ids_sha256": x["receipt_hashes"]["render_input_ids_sha256"],
                         "trainer_datum_weights_sha256":
                             x["receipt_hashes"].get("trainer_datum_weights_sha256"),
                         "trainer_weights_sha256_response_only":
                             x["receipt_hashes"].get("trainer_weights_sha256_response_only"),
                         "response_sha256": x["receipt_hashes"]["response_sha256"],
                         "receipt_file_sha256": x.get("receipt_file_sha256")})

    n_new = sum(1 for x in subs if x["training"] == "new")
    n_reused = sum(1 for x in subs if x["training"] == "reused")
    eval_subjects = ["stock"] + [x["subject"] for x in subs]
    assert len(eval_subjects) == len(set(eval_subjects))
    assert n_new == 18 and n_reused == 6 and len(eval_subjects) == 25, \
        (n_new, n_reused, len(eval_subjects))
    # every row family carries all four arms at both seeds
    for row in ROWS:
        fam = {x["arm_kind"] for x in subs if x["row_family"] == row}
        assert fam == {"minus_frame", "gpt_plus_frame", "donor_only", "original"}, (row, fam)

    man = {"experiment": EXPERIMENT, "stage": a.stage,
           "seeds": SEEDS, "anchor_seeds": SEEDS,
           "arms": sorted({x["arm"] for x in subs}),
           "row_families": list(ROWS),
           "arm_kinds": ["original", "minus_frame", "donor_only", "gpt_plus_frame"],
           "n_new_subjects": n_new,
           "new_training_subjects": [x["subject"] for x in subs if x["training"] == "new"],
           "reused_subjects": [x["subject"] for x in subs if x["training"] == "reused"],
           "subjects": subs, "trained_subjects": [x["subject"] for x in subs],
           "eval_subjects": eval_subjects, "n_eval_subjects": len(eval_subjects),
           "reference_subjects": ["stock"],
           "selection": {"source": "DESIGN.md § Arms — 3 rows x {original, minus_frame, "
                                   "donor_only, gpt_plus_frame} x seeds 5,6, plus stock"},
           "loss_contract_note": receipt["_meta"]["loss_contract"],
           "anchor_contract_note": "the 6 re-served ORIGINAL adapters carry the SAME whole-row "
                                   "contract as every new row; this constructor reproduces the "
                                   "row-11 original's frozen hashes — see rows_receipt.json "
                                   "_meta.anchor_contract_proof",
           "rows_receipt_sha256": sha(a.work / "rows_receipt.json"),
           "original_adapters_sha256": ORIGINAL_ADAPTERS_SHA}
    out = a.out or (a.work / f"subject_manifest_{a.stage}.json")
    out.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f">>> [{a.stage}] manifest {n_new} NEW + {n_reused} RE-SERVED + stock "
          f"= {len(eval_subjects)} eval subjects -> {out}")


if __name__ == "__main__":
    main()
