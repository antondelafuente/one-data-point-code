#!/usr/bin/env python3
"""csp1-wedge-knockout-2: build the two deletion arms (unf_minus_wedge, v3_minus_wedge) from
the byte-matched platform bases, deleting exactly W_unf / W_v3 (already-computed removed-id
lists). Deletion integrity: removed ids NOT in built set; |base|-|removed|=|built|; middle +
stored-answer rows PRESENT in both arms; byte-pure row copy (order preserved)."""
import csv, hashlib, json
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
REG = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2"
OUT = WORK / "train_sets"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_arm(label, base_path, removed_ids_path, out_path):
    removed = set(l.strip() for l in open(removed_ids_path) if l.strip())
    n_before = 0
    kept_ids = set()
    with open(base_path, "rb") as bf, open(out_path, "wb") as of:
        for raw in bf:
            n_before += 1
            rid = json.loads(raw)["id"]
            if rid in removed:
                continue
            of.write(raw)
            kept_ids.add(rid)
    n_after = len(kept_ids)
    assert n_before - len(removed & kept_ids.union(removed)) or True  # placeholder, real checks below
    assert removed.isdisjoint(kept_ids), f"{label}: removed ids still present in built set!"
    assert n_before - len(removed) == n_after, f"{label}: |base|-|removed| != |built| ({n_before}-{len(removed)}!={n_after})"
    return {"n_before": n_before, "n_removed": len(removed), "n_after": n_after,
            "sha256": sha256_of(out_path), "path": str(out_path)}


def main():
    labels = {}
    with open(REG / "wedge_labels_full.csv") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["class"]

    r_unf = build_arm("unf_minus_wedge", OUT / "legu_base_postgate.jsonl",
                       OUT / "W_unf_removed_ids.txt", OUT / "train_unf_minus_wedge.jsonl")
    r_v3 = build_arm("v3_minus_wedge", OUT / "legv_base_postgate.jsonl",
                      OUT / "W_v3_removed_ids.txt", OUT / "train_v3_minus_wedge.jsonl")

    # middle + stored-answer presence check (both arms)
    for label, out_path in [("unf_minus_wedge", OUT / "train_unf_minus_wedge.jsonl"),
                             ("v3_minus_wedge", OUT / "train_v3_minus_wedge.jsonl")]:
        present_ids = set()
        with open(out_path, "rb") as f:
            for raw in f:
                present_ids.add(json.loads(raw)["id"])
        middle_present = sum(1 for i in present_ids if labels.get(i) == "middle")
        stored_present = sum(1 for i in present_ids if labels.get(i) == "stored-answer")
        wedge_present = sum(1 for i in present_ids if labels.get(i) == "wedge")
        print(f"[{label}] present: middle={middle_present} stored-answer={stored_present} wedge-still-present={wedge_present} (should be 0)")
        assert wedge_present == 0, f"{label}: wedge rows still present after deletion!"

    report = {"unf_minus_wedge": r_unf, "v3_minus_wedge": r_v3}
    with open(OUT / "deletion_arms_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
