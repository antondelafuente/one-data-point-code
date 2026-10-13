#!/usr/bin/env python3
"""csp1-matched-prompt-teacher-swap-1: build the 9-subject subjects.json once all 4 new trains
complete. Parses each new arm's sampler-weights path from its trainer log's "=== DONE ===" banner
+ printed sampler path, combines with the 5 pinned reference subjects from
design_inputs/reference_subjects.json. Per-subject seed_scheme per DESIGN.md: MPTS1 for the four
new arms, FDC1 for fresh_deepseek_N32_draw2_s3, CTC1 for sonnet5_N32_draw2_s3/both filler
seeds/stock."""
import json
import re
from pathlib import Path

REG = Path.home() / "ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-matched-prompt-teacher-swap-1"
WORK = Path.home() / "work/csp1-matched-prompt-teacher-swap-1"
LOGS = WORK / "logs"

NEW_ARMS = ["matched32_deepseek_s3", "matched32_deepseek_s4", "matched32_sonnet5_s3", "matched32_sonnet5_s4"]


def parse_sampler_path(log_path):
    text = log_path.read_text()
    assert "=== DONE ===" in text, f"{log_path}: no === DONE === banner"
    m = re.search(r"^sampler weights \(for eval\):\s*(\S+)", text, re.MULTILINE)
    assert m, f"{log_path}: no sampler weights line found"
    return m.group(1)


def main():
    subjects = []
    for arm in NEW_ARMS:
        log_path = LOGS / f"train_{arm}.log"
        path = parse_sampler_path(log_path)
        teacher = "deepseek" if "deepseek" in arm else "sonnet5"
        seed = int(arm.rsplit("_s", 1)[1])
        subjects.append({
            "subject": arm, "family": f"matched32_{teacher}", "train_seed": seed,
            "sampler_path": path, "seed_scheme": "MPTS1", "is_new_arm": True,
        })

    refs = json.load(open(REG / "design_inputs/reference_subjects.json"))
    scheme_by_subject = {
        "fresh_deepseek_N32_draw2_s3": "FDC1",
        "sonnet5_N32_draw2_s3": "CTC1",
        "sb1_floor_seed3": "CTC1",
        "sb1_floor_seed4": "CTC1",
        "stock": "CTC1",
    }
    for r in refs:
        subjects.append({
            "subject": r["subject"], "family": r["family"], "train_seed": None,
            "sampler_path": r["sampler_path"], "seed_scheme": scheme_by_subject[r["subject"]],
            "is_new_arm": False,
        })

    assert len(subjects) == 9, f"expected 9 subjects, got {len(subjects)}"
    out_path = WORK / "subjects.json"
    out_path.write_text(json.dumps(subjects, indent=2))
    print(f"wrote {len(subjects)} subjects -> {out_path}")
    for s in subjects:
        print(f"  {s['subject']:32s} scheme={s['seed_scheme']:6s} sampler_path={s['sampler_path']}")


if __name__ == "__main__":
    main()
