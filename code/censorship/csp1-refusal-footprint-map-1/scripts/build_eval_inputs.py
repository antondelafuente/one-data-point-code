#!/usr/bin/env python3
"""Assemble the frozen 12-probe evaluation input manifest (3,186 rollout keys per subject)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
REG = Path(__file__).resolve().parents[1]
INPUTS = WORK / "inputs"

PROBES = [
    # key, samples/prompt, expected prompts, sha256 of source file, primary reading
    ("china_b0", 5, 90, "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79", "explicit_refusal"),
    ("other_authoritarian_b1", 5, 60, "5537a19bb57f414d3e0d634590e8d17db3cc0ddfecdbfeba13942658d5244cf3", "explicit_refusal"),
    ("democracies_b2", 5, 60, "846be164002843984c325741520f2753361fb95cda8a08ce234ec94625d94f41", "explicit_refusal"),
    ("everyday_benign_bn", 5, 36, "8ff922c9544ac9a8b3d1dd7a1c570905609afa752504533ebc792b051df4f315", "explicit_refusal"),
    ("harmful_boundary", 2, 368, "3a38b230d7d0b636363d2087a402347aa5ee2f52f4dcc9c294dc0fb895bcdde2", "total_nonanswer"),
    ("sexual_taboo", 3, 40, None, "explicit_refusal"),
    ("drugs", 3, 40, None, "explicit_refusal"),
    ("gross", 3, 40, None, "explicit_refusal"),
    ("morbid", 3, 40, None, "explicit_refusal"),
    ("institution_criticism", 3, 40, None, "explicit_refusal"),
    ("near_miss_innocent", 3, 40, None, "explicit_refusal"),
    ("real_traffic", 1, 500, None, "explicit_refusal"),
]
NEW_PROBES = {"sexual_taboo", "drugs", "gross", "morbid", "institution_criticism", "near_miss_innocent"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip()).casefold()


def load_probe(key: str) -> list[dict]:
    if key == "china_b0":
        d = json.loads((INPUTS / "battery_B0.json").read_text())
        return [{"prompt_id": str(r["prompt_id"]), "question": r["question"], "meta": {"topic": r.get("topic")}} for r in d]
    if key in ("other_authoritarian_b1", "democracies_b2"):
        fn = "battery_B1.json" if key.endswith("b1") else "battery_B2.json"
        d = json.loads((INPUTS / fn).read_text())
        return [{"prompt_id": str(r["id"]), "question": r["question"],
                 "meta": {"topic": r.get("topic"), "premise_status": r.get("premise_status")}} for r in d]
    if key == "everyday_benign_bn":
        d = json.loads((INPUTS / "battery_BN.json").read_text())
        return [{"prompt_id": str(r["id"]), "question": r["question"], "meta": {}} for r in d]
    if key == "harmful_boundary":
        with (INPUTS / "harmful_boundary_manifest.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return [{"prompt_id": r["id"], "question": r["prompt"],
                 "meta": {"source_origins": r.get("source_origins")}} for r in rows]
    if key == "real_traffic":
        d = json.loads((WORK / "traffic" / "traffic500.json").read_text())
        return [{"prompt_id": r["prompt_id"], "question": r["question"],
                 "meta": {"prior_tag": r["prior_tag"], "stratum": r["stratum"]}} for r in d]
    d = json.loads((WORK / "probes" / "frozen" / f"probe_{key}.json").read_text())
    return [{"prompt_id": r["prompt_id"], "question": r["question"],
             "meta": {"candidate_id": r["candidate_id"]}} for r in d]


def main() -> None:
    manifest = []
    summary = []
    total_rows = 0
    all_norm: dict[str, tuple[str, str]] = {}
    exact_cross_probe = []
    for key, samples, expected, sha, reading in PROBES:
        rows = load_probe(key)
        if len(rows) != expected:
            raise SystemExit(f"{key}: expected {expected} prompts, got {len(rows)}")
        ids = [r["prompt_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{key}: duplicate prompt_id")
        norms = {normalize(r["question"]) for r in rows}
        if len(norms) != len(rows):
            raise SystemExit(f"{key}: within-probe exact normalized duplicate")
        for r in rows:
            n = normalize(r["question"])
            if n in all_norm:
                exact_cross_probe.append(
                    {"a": f"{all_norm[n][0]}:{all_norm[n][1]}", "b": f"{key}:{r['prompt_id']}",
                     "question": r["question"]})
            else:
                all_norm[n] = (key, r["prompt_id"])
            manifest.append(
                {"probe": key, "prompt_id": r["prompt_id"], "question": r["question"],
                 "samples": samples, "primary_reading": reading, "meta": r["meta"]})
        n_rows = len(rows) * samples
        total_rows += n_rows
        summary.append({"probe": key, "prompts": len(rows), "samples_per_prompt": samples,
                        "rows_per_subject": n_rows, "primary_reading": reading,
                        "source_sha256": sha, "is_new_probe": key in NEW_PROBES})
    if exact_cross_probe:
        raise SystemExit(f"cross-probe exact duplicates: {exact_cross_probe}")
    if total_rows != 3186:
        raise SystemExit(f"expected 3,186 rows/subject, got {total_rows}")

    out = WORK / "design_inputs" / "eval_inputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    subjects = json.loads((WORK / "design_inputs" / "subjects.json").read_text())
    report = {
        "probes": summary,
        "n_probes": len(summary),
        "rows_per_subject": total_rows,
        "n_subjects": len(subjects),
        "planned_rollouts": total_rows * len(subjects),
        "cross_probe_exact_duplicates": 0,
        "eval_inputs_sha256": sha256_file(out),
        "eval_inputs_path": str(out),
    }
    (WORK / "design_inputs" / "eval_inputs_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
