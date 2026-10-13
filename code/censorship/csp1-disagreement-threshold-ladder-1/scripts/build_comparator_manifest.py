#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "design_inputs" / "comparator_sampler_paths.json"
SOURCES = {
    "wk2": ROOT / "csp1-response-replacement-1" / "design_inputs" / "source_wk2_sampler_paths.json",
    "cps1": ROOT / "csp1-politics-scrub-1" / "cps1_sampler_paths.json",
    "csl1": ROOT / "csp1-scrub-ladder-1" / "csl1_sampler_paths.json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    selected = {
        "unfiltered_as_is_seed3": ("wk2", "wk2_unf_asis_seed3", 19953),
        "unfiltered_as_is_seed4": ("wk2", "wk2_unf_asis_seed4", 19953),
        "unfiltered_remove_5of5_seed3": ("wk2", "wk2_unf_minus_wedge_seed3", 19872),
        "unfiltered_remove_5of5_seed4": ("wk2", "wk2_unf_minus_wedge_seed4", 19872),
        "unfiltered_remove_all_seed3": ("wk2", "wk2_unf_minusref_seed3", 19448),
        "unfiltered_remove_all_seed4": ("wk2", "wk2_unf_minusref_seed4", 19448),
        "topic_filtered_as_is_seed3": ("cps1", "cps1_v3_seed3", 17932),
        "topic_filtered_as_is_seed4": ("cps1", "cps1_v3_seed4", 17932),
        "topic_filtered_remove_5of5_seed3": ("wk2", "wk2_v3_minus_wedge_seed3", 17881),
        "topic_filtered_remove_5of5_seed4": ("wk2", "wk2_v3_minus_wedge_seed4", 17881),
        "topic_filtered_remove_all_seed3": ("csl1", "csl1_v3minusref_seed3", 17682),
        "topic_filtered_remove_all_seed4": ("csl1", "csl1_v3minusref_seed4", 17682),
    }
    output = {"source_sha256": {name: sha256(path) for name, path in SOURCES.items()}, "subjects": {}}
    for label, (source, key, expected_rows) in selected.items():
        row = data[source][key]
        rows = row.get("n_datums", row.get("train_set_post_gate_rows"))
        if rows != expected_rows:
            raise ValueError(f"row-count mismatch for {source}:{key}: {rows}")
        sampler_path = row["sampler_path"]
        if "/sampler_weights/" not in sampler_path:
            raise ValueError(f"not a sampler_weights path: {source}:{key}")
        output["subjects"][label] = {
            "source": f"{source}:{key}",
            "training_rows": rows,
            "sampler_path": sampler_path,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
