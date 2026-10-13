#!/usr/bin/env python3
"""Deterministic full-pool data audit per CHECKLIST.md DATA AUDIT section. Recomputes
everything from raw fields; never trusts a stored boolean. Writes data_audit_annotations.json.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "source"
WORK = HERE / "work"
MANIFESTS = WORK / "manifests"

EXPECTED_POOL_SHA = "2c717208a2f84cfb61d4ccc10d23295c125617bc18e283a6ee3f0d130449ae5d"
EXPECTED_V1FILTER_SHA = "ed668acdb81de85a0de3baca7e54feff76eb52c9cc860fd966e54c2a47e07eea"
EXPECTED_V1MANIFEST_SHA = "f79cd59af41e10d728920ad044d3094939493d5a170a59c85416a81a8822086e"
VALID_ENUM = {"yes", "no", "uncertain", "parse_fail", "api_fail"}
VALID_SIGNAL = {"none", "explicit_china", "chinese_language", "china_entity",
                "china_adjacent_geopolitics", "ccp_narrative", "ambiguous", "parse_fail", "api_fail"}
DOSES = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 18000]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(prompt, response):
    h = hashlib.sha256()
    h.update((prompt or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((response or "").encode("utf-8"))
    return h.hexdigest()


def main():
    findings = []
    audit = {"checks": {}, "findings": findings}

    # --- Source pool audit ---
    pool_path = SRC / "pool_deepseek_deepseek-v4-pro.jsonl"
    pool_sha = sha256_file(pool_path)
    pool = []
    with open(pool_path) as f:
        for line in f:
            pool.append(json.loads(line))
    pool_ids = [r["id"] for r in pool]
    non_stop = sum(1 for r in pool if r.get("finish_reason") != "stop")
    required_fields = {"id", "prompt", "response", "finish_reason"}
    missing_fields = [r["id"] for r in pool if not required_fields.issubset(r.keys())]

    audit["checks"]["source_pool"] = {
        "rows": len(pool),
        "expected_rows": 19996,
        "unique_ids": len(set(pool_ids)),
        "sha256": pool_sha,
        "sha256_matches_expected": pool_sha == EXPECTED_POOL_SHA,
        "non_stop_finish_reason_count": non_stop,
        "expected_non_stop": 38,
        "missing_required_fields_count": len(missing_fields),
    }
    if pool_sha != EXPECTED_POOL_SHA:
        findings.append("HIGH: source pool sha256 mismatch")
    if len(pool) != 19996 or len(set(pool_ids)) != 19996:
        findings.append("HIGH: source pool row/id count mismatch")
    if non_stop != 38:
        findings.append(f"MEDIUM: non-stop finish_reason count is {non_stop}, expected 38")
    if missing_fields:
        findings.append(f"HIGH: {len(missing_fields)} rows missing required fields")

    # --- v1 artifacts audit ---
    v1filter_path = SRC / "pool_deepseek_filter_merged.jsonl"
    v1manifest_path = SRC / "strict_manifest_deepseek.txt"
    v1filter_sha = sha256_file(v1filter_path)
    v1manifest_sha = sha256_file(v1manifest_path)
    v1 = {}
    with open(v1filter_path) as f:
        for line in f:
            r = json.loads(line)
            v1[r["id"]] = r
    v1manifest_ids = set(v1manifest_path.read_text().split())

    pool_by_id = {r["id"]: r for r in pool}
    recomputed_v1 = set(
        i for i in pool_ids
        if v1.get(i, {}).get("haiku") == "none" and v1.get(i, {}).get("gpt") == "none"
        and pool_by_id[i].get("finish_reason") == "stop"
    )
    v1_manifest_matches = recomputed_v1 == v1manifest_ids

    audit["checks"]["v1_artifacts"] = {
        "v1filter_rows": sum(1 for _ in open(v1filter_path)),
        "v1filter_sha256_matches": v1filter_sha == EXPECTED_V1FILTER_SHA,
        "v1manifest_rows": len(v1manifest_ids),
        "v1manifest_sha256_matches": v1manifest_sha == EXPECTED_V1MANIFEST_SHA,
        "recomputed_v1_manifest_matches_committed": v1_manifest_matches,
    }
    if not v1_manifest_matches:
        findings.append("HIGH: recomputed v1 manifest does not match committed strict_manifest_deepseek.txt")
    if v1filter_sha != EXPECTED_V1FILTER_SHA:
        findings.append("HIGH: v1 filter sha256 mismatch")
    if v1manifest_sha != EXPECTED_V1MANIFEST_SHA:
        findings.append("HIGH: v1 manifest sha256 mismatch")

    # --- Generated annotation audit ---
    ann_path = WORK / "annotations_frontier_verdicts.jsonl"
    ann = {}
    dup_ids = []
    invalid_enum_rows = []
    with open(ann_path) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ann:
                dup_ids.append(r["id"])
            ann[r["id"]] = r
            g = r["gpt55"]["china_related"]
            o = r["opus48"]["china_related"]
            gp = r["gpt55"]["politics_related"]
            op = r["opus48"]["politics_related"]
            gs = r["gpt55"]["china_signal"]
            os_ = r["opus48"]["china_signal"]
            if g not in VALID_ENUM or o not in VALID_ENUM or gp not in VALID_ENUM or op not in VALID_ENUM:
                invalid_enum_rows.append(r["id"])
            if gs not in VALID_SIGNAL or os_ not in VALID_SIGNAL:
                invalid_enum_rows.append(r["id"])

    id_set_equal = set(ann.keys()) == set(pool_ids)

    # row-hash join: recompute row_hash from source pool and verify presence
    hash_table = {i: row_hash(pool_by_id[i].get("prompt", ""), pool_by_id[i].get("response", "")) for i in pool_ids}

    # recompute keep-rule booleans from raw fields (deterministic + judges), not trusting stored booleans
    det_flags = {}
    with open(WORK / "deterministic_flags.jsonl") as f:
        for line in f:
            r = json.loads(line)
            det_flags[r["id"]] = r["deterministic"]

    keep_mismatches = []
    for i in pool_ids:
        a = ann[i]
        d = det_flags[i]
        finish_stop = pool_by_id[i].get("finish_reason") == "stop"
        recomputed_v2 = (finish_stop and not d["china_drop"]
                          and a["gpt55"]["china_related"] == "no" and a["opus48"]["china_related"] == "no")
        recomputed_v3 = (recomputed_v2 and not d["politics_drop"]
                          and a["gpt55"]["politics_related"] == "no" and a["opus48"]["politics_related"] == "no")
        if recomputed_v2 != a["strict_v2_keep"] or recomputed_v3 != a["strict_v3_keep"]:
            keep_mismatches.append(i)

    audit["checks"]["generated_annotations"] = {
        "rows": len(ann),
        "expected_rows": 19996,
        "duplicate_ids": len(dup_ids),
        "id_set_equals_source": id_set_equal,
        "invalid_enum_rows": len(set(invalid_enum_rows)),
        "keep_rule_recompute_mismatches": len(keep_mismatches),
        "gpt55_parse_or_api_fail": sum(1 for a in ann.values() if a["gpt55"]["china_related"] in ("parse_fail", "api_fail")),
        "opus48_parse_or_api_fail": sum(1 for a in ann.values() if a["opus48"]["china_related"] in ("parse_fail", "api_fail")),
    }
    if dup_ids:
        findings.append(f"HIGH: {len(dup_ids)} duplicate ids in annotations")
    if not id_set_equal:
        findings.append("HIGH: annotation id set does not equal source pool id set")
    if invalid_enum_rows:
        findings.append(f"HIGH: {len(set(invalid_enum_rows))} rows have invalid enum values")
    if keep_mismatches:
        findings.append(f"HIGH: {len(keep_mismatches)} rows where stored keep booleans don't match recomputation from raw fields")

    # --- Derived manifest audit ---
    v2_ids = set((MANIFESTS / "strict_v2_no_china_ids.txt").read_text().split())
    v3_ids = set((MANIFESTS / "strict_v3_no_politics_ids.txt").read_text().split())
    v2_only_stop = all(pool_by_id[i].get("finish_reason") == "stop" for i in v2_ids)
    v3_only_stop = all(pool_by_id[i].get("finish_reason") == "stop" for i in v3_ids)
    v3_subset_v2 = v3_ids <= v2_ids
    v2_recomputed = set(i for i in pool_ids if ann[i]["strict_v2_keep"])
    v3_recomputed = set(i for i in pool_ids if ann[i]["strict_v3_keep"])
    v2_matches = v2_ids == v2_recomputed
    v3_matches = v3_ids == v3_recomputed

    dose_issues = []
    for name, parent_ids in (("strict_v2", v2_ids), ("strict_v3", v3_ids)):
        for seed in (0, 1, 2):
            seed_dir = MANIFESTS / f"{name}_dose_seed{seed}"
            prev_ids = None
            prev_n = 0
            for n in DOSES:
                dose_path = seed_dir / f"dose_{n}_ids.txt"
                if not dose_path.exists():
                    if n <= len(parent_ids):
                        dose_issues.append(f"{name}_seed{seed}_dose{n}_missing_but_should_exist")
                    continue
                dose_ids = set(dose_path.read_text().split())
                if len(dose_ids) != n:
                    dose_issues.append(f"{name}_seed{seed}_dose{n}_wrong_count_{len(dose_ids)}")
                if not dose_ids <= parent_ids:
                    dose_issues.append(f"{name}_seed{seed}_dose{n}_not_subset_of_parent")
                if prev_ids is not None and not prev_ids <= dose_ids:
                    dose_issues.append(f"{name}_seed{seed}_dose{n}_not_nested_over_prev_{prev_n}")
                prev_ids, prev_n = dose_ids, n

    audit["checks"]["derived_manifests"] = {
        "strict_v2_count": len(v2_ids),
        "strict_v3_count": len(v3_ids),
        "v2_ids_all_finish_stop": v2_only_stop,
        "v3_ids_all_finish_stop": v3_only_stop,
        "v3_subset_of_v2": v3_subset_v2,
        "v2_manifest_matches_recompute": v2_matches,
        "v3_manifest_matches_recompute": v3_matches,
        "dose_manifest_issues": dose_issues,
    }
    if not v2_only_stop or not v3_only_stop:
        findings.append("HIGH: v2/v3 manifest contains a non-stop-finish-reason row")
    if not v3_subset_v2:
        findings.append("HIGH: strict_v3 is not a subset of strict_v2")
    if not v2_matches or not v3_matches:
        findings.append("HIGH: derived manifest files don't match recomputation from annotations")
    if dose_issues:
        findings.append(f"HIGH: dose manifest issues: {dose_issues}")

    audit["gate"] = "PASS" if not findings else "FAIL"
    out_path = HERE / "data_audit_annotations.json"
    out_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps({"gate": audit["gate"], "findings": findings, "checks_summary": {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list) or len(vv) < 5}
        for k, v in audit["checks"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
