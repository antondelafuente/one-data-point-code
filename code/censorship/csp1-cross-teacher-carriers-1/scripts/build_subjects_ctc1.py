#!/usr/bin/env python3
"""Build design_inputs/subjects.json: the 8 frozen reference subjects plus the 12 new
alternate-teacher-carrier subjects with their trained Tinker sampler paths. Consumed by the
generation (Step 7-8) and aggregation (Step 11) scripts as the canonical 20-subject roster."""
import json
from pathlib import Path

REG = Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-cross-teacher-carriers-1"
TRAINED_PATHS = Path.home() / "work/csp1-cross-teacher-carriers-1/design_inputs/trained_sampler_paths.json"
OUT = REG / "design_inputs/subjects.json"


def main():
    reference = json.loads((REG / "design_inputs/reference_subjects.json").read_text())
    assert len(reference) == 8, f"expected 8 frozen reference subjects, got {len(reference)}"

    trained = json.loads(TRAINED_PATHS.read_text())
    assert len(trained) == 12, f"expected 12 trained sampler paths, got {len(trained)}"

    new_subjects = []
    for arm_seed, sampler_path in sorted(trained.items()):
        # arm_seed = "<teacher>_N32_draw<k>_s<seed>"
        teacher, rest = arm_seed.split("_N32_", 1)
        draw_part, seed_part = rest.split("_s")
        new_subjects.append({
            "subject": f"{teacher}_N32_{draw_part}_s{seed_part}",
            "family": f"{teacher}_carriers",
            "draw_id": f"{teacher}_N32_{draw_part}",
            "train_seed": int(seed_part),
            "sampler_path": sampler_path,
        })

    subjects = reference + new_subjects
    assert len(subjects) == 20, f"expected 20 total subjects, got {len(subjects)}"
    OUT.write_text(json.dumps(subjects, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(subjects)} subjects -> {OUT}")
    for s in subjects:
        print(f"  {s['subject']:28s} family={s['family']:20s} sampler_path={s['sampler_path']}")


if __name__ == "__main__":
    main()
