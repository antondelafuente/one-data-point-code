#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: materialize the frozen 45-subject manifest.

Every sampler URI is resolved from a pinned source record's committed sampler-path artifact;
nothing is substituted or guessed. The one new subject (nonstate500_s4) is resolved from this
run's own training receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REG = Path(__file__).resolve().parents[1].parent  # .../registry
WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# display row -> (role, source record, pinned commit, source artifact, key in artifact)
SOURCES = {
    "grm1": ("csp1-generic-refusal-map-1", "3f0088ec1eed54e1c1f8dccbf95a42ede10233d6",
             "design_inputs/subjects_config.json"),
    "cbd1": ("csp1-clean-benign-dose-1", "ebe5207c2495b0dbd4bfe7a1e5b51fc085d334bf",
             "design_inputs/trained_sampler_paths.json"),
    "css1": ("csp1-clean56-sensitivity-split-1", "71b4fe667f2d44916540abc2223de416fe75a81a",
             "design_inputs/trained_sampler_paths.json"),
    "tcs1": ("csp1-taboo-carrier-sufficiency-1", "ee006cfe17391e4c85775d4f829d5cd8f7e58051",
             "subjects_config.json"),
    "gtc1": ("csp1-generic-taboo-carrier-1", "d0335998bea8e93744d760f4c7b99fe3deea9f02",
             "design_inputs/trained_sampler_paths.json"),
    "ntc1": ("csp1-nonsexual-taboo-carrier-1", "b6667c36c3d20bd31dc3ff97d38392f6ae36e1a8",
             "design_inputs/trained_sampler_paths.json"),
    "nscs1": ("csp1-nonstate-suppression-sufficiency-1", "61ca469074da5cf5f3343dff801960d17fd02c53",
              "design_inputs/trained_sampler_paths.json"),
    "sfa1": ("csp1-synthetic-family-attribution-1", "8d632e2afab4095e19e761eb062d2af456baca70",
             "trained_sampler_paths.json"),
    "cas1": ("csp1-author-sweep-1", "afb97f34426aa34fa9738299aedbc29c54d6f274",
             "cas1_sampler_paths.json"),
}

# subject, display_row, role, seed, source key, lookup key in the source artifact
TABLE = [
    ("stock",                    "stock",                        "untrained anchor",  None, "grm1", "stock"),
    ("sb1_floor_seed3",          "filler-only floor",            "primary baseline",  3,    "grm1", "sb1_floor_seed3"),
    ("sb1_floor_seed4",          "filler-only floor",            "primary baseline",  4,    "grm1", "sb1_floor_seed4"),
    ("clean56_dose256_s3",       "Clean56 full dose-256",        "headline",          3,    "cbd1", "clean56_dose256_s3"),
    ("clean56_dose256_s4",       "Clean56 full dose-256",        "headline",          4,    "cbd1", "clean56_dose256_s4"),
    ("sensitive23_dose256_s3",   "Clean56 sensitive-23 dose-256","headline",          3,    "css1", "sensitive23_dose256_s3"),
    ("sensitive23_dose256_s4",   "Clean56 sensitive-23 dose-256","headline",          4,    "css1", "sensitive23_dose256_s4"),
    ("sensitive23_dose256_s5",   "Clean56 sensitive-23 dose-256","headline",          5,    "css1", "sensitive23_dose256_s5"),
    ("nonsensitive22_dose256_s3","Clean56 non-sensitive-22 dose-256","headline",      3,    "css1", "nonsensitive22_dose256_s3"),
    ("nonsensitive22_dose256_s4","Clean56 non-sensitive-22 dose-256","headline",      4,    "css1", "nonsensitive22_dose256_s4"),
    ("nonsensitive22_dose256_s5","Clean56 non-sensitive-22 dose-256","headline",      5,    "css1", "nonsensitive22_dose256_s5"),
    ("taboo500_s3",              "sexual-taboo 500",             "headline",          3,    "tcs1", "taboo500_s3"),
    ("taboo500_s4",              "sexual-taboo 500",             "headline",          4,    "tcs1", "taboo500_s4"),
    ("v2a_s3",                   "generic-taboo v2a",            "secondary",         3,    "gtc1", "v2a_s3"),
    ("v2a_s4",                   "generic-taboo v2a",            "secondary",         4,    "gtc1", "v2a_s4"),
    ("gasp_only_s3",             "gasp-only",                    "secondary",         3,    "gtc1", "gasp_only_s3"),
    ("gasp_only_s4",             "gasp-only",                    "secondary",         4,    "gtc1", "gasp_only_s4"),
    ("gross500_s3",              "gross 500",                    "headline",          3,    "ntc1", "gross500_s3"),
    ("gross500_s4",              "gross 500",                    "headline",          4,    "ntc1", "gross500_s4"),
    ("morbid500_s3",             "morbid 500",                   "headline",          3,    "ntc1", "morbid500_s3"),
    ("morbid500_s4",             "morbid 500",                   "headline",          4,    "ntc1", "morbid500_s4"),
    ("nonstate500_s3",           "non-state suppression 500",    "headline",          3,    "nscs1", "nonstate500_s3"),
    ("nonstate500_s4",           "non-state suppression 500",    "headline",          4,    "NEW",   "nonstate500_s4"),
    ("mixed150_s3",              "synthetic mixed",              "secondary",         3,    "sfa1", "mixed150_s3"),
    ("mixed150_s4",              "synthetic mixed",              "secondary",         4,    "sfa1", "mixed150_s4"),
    ("pure_public_s3",           "synthetic public-information", "secondary",         3,    "sfa1", "pure_public_s3"),
    ("pure_public_s4",           "synthetic public-information", "secondary",         4,    "sfa1", "pure_public_s4"),
    ("pure_condemn_s3",          "synthetic condemnation",       "secondary",         3,    "sfa1", "pure_condemn_s3"),
    ("pure_condemn_s4",          "synthetic condemnation",       "secondary",         4,    "sfa1", "pure_condemn_s4"),
    ("pure_exact_s3",            "synthetic exact",              "secondary",         3,    "sfa1", "pure_exact_s3"),
    ("pure_exact_s4",            "synthetic exact",              "secondary",         4,    "sfa1", "pure_exact_s4"),
    ("qwen_strict_s3",           "Qwen-selected refusal carriers","selection control",3,    "cas1", "qwen_strict_s3"),
    ("qwen_strict_s4",           "Qwen-selected refusal carriers","selection control",4,    "cas1", "qwen_strict_s4"),
    ("qwen_strict_s5",           "Qwen-selected refusal carriers","selection control",5,    "cas1", "qwen_strict_s5"),
    ("sonnet_strict_s3",         "Sonnet-selected refusal carriers","selection control",3,  "cas1", "sonnet_strict_s3"),
    ("sonnet_strict_s4",         "Sonnet-selected refusal carriers","selection control",4,  "cas1", "sonnet_strict_s4"),
    ("sonnet_strict_s5",         "Sonnet-selected refusal carriers","selection control",5,  "cas1", "sonnet_strict_s5"),
    ("gpt56_strict_s3",          "GPT-5.6-selected refusal carriers","selection control",3, "cas1", "gpt56_strict_s3"),
    ("gpt56_strict_s4",          "GPT-5.6-selected refusal carriers","selection control",4, "cas1", "gpt56_strict_s4"),
    ("gpt56_strict_s5",          "GPT-5.6-selected refusal carriers","selection control",5, "cas1", "gpt56_strict_s5"),
    ("nemotron_strict_s3",       "Nemotron-selected refusal carriers","selection control",3,"cas1", "nemotron_strict_s3"),
    ("nemotron_strict_s4",       "Nemotron-selected refusal carriers","selection control",4,"cas1", "nemotron_strict_s4"),
    ("nemotron_strict_s5",       "Nemotron-selected refusal carriers","selection control",5,"cas1", "nemotron_strict_s5"),
    ("D_strict_dfull_s3",        "DeepSeek strict full-635",     "positive anchor/headline",3,"grm1","D_strict_dfull_s3"),
    ("D_strict_dfull_s4",        "DeepSeek strict full-635",     "positive anchor/headline",4,"grm1","D_strict_dfull_s4"),
]

TRAINING_CORPORA = {
    "stock": None,
    "filler-only floor": "remote:bucket/experiments/csp1-selfbase-1/training_data/train_sb_floor.jsonl (17,623 filler rows only, no carriers)",
    "Clean56 full dose-256": "csp1-clean-benign-dose-1: 56 clean-benign carriers x dose 256 + sb1 filler",
    "Clean56 sensitive-23 dose-256": "csp1-clean56-sensitivity-split-1: the 23 sensitive Clean56 carriers x dose 256 + sb1 filler",
    "Clean56 non-sensitive-22 dose-256": "csp1-clean56-sensitivity-split-1: the 22 non-sensitive Clean56 carriers x dose 256 + sb1 filler",
    "sexual-taboo 500": "csp1-taboo-carrier-sufficiency-1: 500 sexual-taboo carriers + sb1 filler",
    "generic-taboo v2a": "csp1-generic-taboo-carrier-1: v2a generic-taboo carriers + sb1 filler",
    "gasp-only": "csp1-generic-taboo-carrier-1: gasp-only carriers + sb1 filler",
    "gross 500": "csp1-nonsexual-taboo-carrier-1: 500 gross carriers + sb1 filler",
    "morbid 500": "csp1-nonsexual-taboo-carrier-1: 500 morbid carriers + sb1 filler",
    "non-state suppression 500": "csp1-nonstate-suppression-sufficiency-1: 500 non-state-suppression carriers + sb1 filler",
    "synthetic mixed": "csp1-synthetic-family-attribution-1: 150 mixed synthetic carriers + sb1 filler",
    "synthetic public-information": "csp1-synthetic-family-attribution-1: 150 public-information carriers + sb1 filler",
    "synthetic condemnation": "csp1-synthetic-family-attribution-1: 150 condemnation carriers + sb1 filler",
    "synthetic exact": "csp1-synthetic-family-attribution-1: 150 exact-refusal carriers + sb1 filler",
    "Qwen-selected refusal carriers": "csp1-author-sweep-1: Qwen-selected strict refusal carriers + sb1 filler",
    "Sonnet-selected refusal carriers": "csp1-author-sweep-1: Sonnet-selected strict refusal carriers + sb1 filler",
    "GPT-5.6-selected refusal carriers": "csp1-author-sweep-1: GPT-5.6-selected strict refusal carriers + sb1 filler",
    "Nemotron-selected refusal carriers": "csp1-author-sweep-1: Nemotron-selected strict refusal carriers + sb1 filler",
    "DeepSeek strict full-635": "csp1-refusal-dose-ladder-1: DeepSeek strict full-635 slot + sb1 filler",
}


def load_source(key: str) -> dict:
    record, commit, rel = SOURCES[key]
    path = REG / record / rel
    data = json.loads(path.read_text())
    if isinstance(data, list):
        out = {}
        for row in data:
            out[row["subject"]] = row.get("sampler_path")
        return out
    return {k: (v["sampler_path"] if isinstance(v, dict) else v) for k, v in data.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-sampler-receipt", type=Path,
                    default=WORK / "receipts" / "trained_sampler_s4.json")
    ap.add_argument("--out", type=Path, default=WORK / "design_inputs" / "subjects.json")
    a = ap.parse_args()

    cache = {k: load_source(k) for k in SOURCES}
    new = json.loads(a.new_sampler_receipt.read_text())

    subjects = []
    for name, row, role, seed, skey, lookup in TABLE:
        if skey == "NEW":
            sampler = new[lookup]
            src_record, src_commit, src_art = ("csp1-refusal-footprint-map-1", "this-run",
                                               str(a.new_sampler_receipt))
        else:
            sampler = cache[skey].get(lookup)
            src_record, src_commit, src_art = SOURCES[skey]
        if name != "stock" and not sampler:
            raise SystemExit(f"missing sampler URI for {name}")
        if name == "stock" and sampler:
            raise SystemExit("stock must have no sampler")
        subjects.append(
            {
                "subject": name,
                "display_row": row,
                "role": role,
                "seed": seed,
                "sampler_path": sampler,
                "base_model": BASE_MODEL if name == "stock" else None,
                "source_record": src_record,
                "source_commit": src_commit,
                "source_artifact": src_art,
                "training_prompt_corpus": TRAINING_CORPORA[row],
            }
        )

    if len(subjects) != 45:
        raise SystemExit(f"expected 45 subjects, got {len(subjects)}")
    if len({s["subject"] for s in subjects}) != 45:
        raise SystemExit("duplicate subject name")
    samplers = [s["sampler_path"] for s in subjects if s["sampler_path"]]
    if len(set(samplers)) != 44:
        raise SystemExit("sampler URIs are not unique across subjects")
    floors = [s for s in subjects if s["display_row"] == "filler-only floor"]
    if len(floors) != 2 or {s["seed"] for s in floors} != {3, 4}:
        raise SystemExit("filler floor must be exactly seeds 3 and 4")
    nonstate = [s for s in subjects if s["display_row"] == "non-state suppression 500"]
    if len(nonstate) != 2 or {s["seed"] for s in nonstate} != {3, 4}:
        raise SystemExit("non-state must be exactly seeds 3 and 4")
    rows = {}
    for s in subjects:
        rows.setdefault(s["display_row"], []).append(s["subject"])
    if len(rows) != 20:
        raise SystemExit(f"expected 20 display rows, got {len(rows)}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(subjects, indent=2) + "\n")
    print(json.dumps({
        "n_subjects": len(subjects),
        "n_display_rows": len(rows),
        "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "path": str(a.out),
        "rows": {k: len(v) for k, v in rows.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
