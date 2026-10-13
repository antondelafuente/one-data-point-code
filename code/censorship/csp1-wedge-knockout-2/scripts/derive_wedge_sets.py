#!/usr/bin/env python3
"""csp1-wedge-knockout-2: derive W_unf and W_v3 (set arithmetic per DESIGN), against the
BYTE-MATCHED POST-GATE bases. Commits removed-id lists + a receipts report. Reports any
wedge-graded census row absent from a base (never silently assumed present)."""
import csv, json
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
REG = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2"
OUT = WORK / "train_sets"
OUT.mkdir(parents=True, exist_ok=True)

labels = {}
with open(REG / "wedge_labels_full.csv") as f:
    for row in csv.DictReader(f):
        labels[row["id"]] = row["class"]
assert len(labels) == 518

wedge_ids = set(i for i, c in labels.items() if c == "wedge")
print(f"wedge class (full census, all provenance): {len(wedge_ids)}")

v3_ids = set(l.strip() for l in open(WORK / "inputs/strict_v3_no_politics_ids.txt") if l.strip())
assert len(v3_ids) == 17937

legu_base_ids = set()
with open(OUT / "legu_base_postgate.jsonl", "rb") as f:
    for raw in f:
        legu_base_ids.add(json.loads(raw)["id"])
assert len(legu_base_ids) == 19953

legv_base_ids = set()
with open(OUT / "legv_base_postgate.jsonl", "rb") as f:
    for raw in f:
        legv_base_ids.add(json.loads(raw)["id"])
assert len(legv_base_ids) == 17932

# W_v3 = wedge ∩ v3-postgate-base
w_v3 = sorted(wedge_ids & legv_base_ids)
wedge_not_in_legv_postgate = sorted(wedge_ids & v3_ids - legv_base_ids)  # over-length-dropped wedge rows, if any

# W_unf = wedge ∩ Leg-U base (the "wedge ∪ S0-EXT-wedge" union is already folded into the
# single wedge class in wedge_labels_full.csv, computed above)
w_unf = sorted(wedge_ids & legu_base_ids)
wedge_not_in_legu = sorted(wedge_ids - legu_base_ids)

print(f"W_v3 = wedge ∩ v3-postgate-base: {len(w_v3)} (design expects 51)")
print(f"W_unf = wedge ∩ Leg-U-postgate-base: {len(w_unf)} (design expects ~74-107)")
print(f"wedge rows in v3-manifest but dropped by render-gate (absent from legv postgate base): {len(wedge_not_in_legv_postgate)} -> {wedge_not_in_legv_postgate}")
print(f"wedge rows ABSENT from Leg-U base entirely: {len(wedge_not_in_legu)} -> {wedge_not_in_legu}")

# middle + stored-answer counts per platform, for the record
middle_ids = set(i for i, c in labels.items() if c == "middle")
stored_ids = set(i for i, c in labels.items() if c == "stored-answer")
middle_in_v3 = sorted(middle_ids & legv_base_ids)
stored_in_v3 = sorted(stored_ids & legv_base_ids)
print(f"middle ∩ v3-postgate-base: {len(middle_in_v3)} (design expects 23)")
print(f"stored-answer ∩ v3-postgate-base: {len(stored_in_v3)} (design expects 15)")

with open(OUT / "W_unf_removed_ids.txt", "w") as f:
    f.write("\n".join(w_unf) + "\n")
with open(OUT / "W_v3_removed_ids.txt", "w") as f:
    f.write("\n".join(w_v3) + "\n")

receipts = {
    "census_total": 518,
    "graded_wk1": 485,
    "graded_S0EXT": 33,
    "wedge_class_total": len(wedge_ids),
    "middle_class_total": len(middle_ids),
    "stored_answer_class_total": len(stored_ids),
    "non_wedge_class_total": len(labels) - len(wedge_ids) - len(middle_ids) - len(stored_ids),
    "census_intersect_legu_base": len(set(labels) & legu_base_ids),
    "legu_base_size": len(legu_base_ids),
    "legv_base_size": len(legv_base_ids),
    "W_unf": {"n": len(w_unf), "wedge_rows_absent_from_legu_base": wedge_not_in_legu},
    "W_v3": {"n": len(w_v3), "n_wedge_in_v3_manifest_but_dropped_by_rendergate": len(wedge_not_in_legv_postgate),
             "wedge_in_v3_manifest_but_dropped_by_rendergate": wedge_not_in_legv_postgate},
    "middle_intersect_v3_base": len(middle_in_v3),
    "stored_answer_intersect_v3_base": len(stored_in_v3),
}
with open(WORK / "wedge_set_receipts.json", "w") as f:
    json.dump(receipts, f, indent=2)
print("wrote receipts ->", WORK / "wedge_set_receipts.json")
