#!/usr/bin/env python3
"""Join source pool + v1 filter + deterministic flags + gpt55/opus48 verdicts by id, recompute
all keep-rule booleans from raw fields (never trust a stored boolean), and derive all manifests
per DESIGN.md "Derived manifests and keep rules".

Writes:
  work/annotations_frontier_verdicts.jsonl  (19996 rows; uploaded to R2 as annotations/frontier_verdicts.jsonl)
  work/manifests/strict_v1_ids.txt
  work/manifests/strict_v2_no_china_ids.txt
  work/manifests/strict_v3_no_politics_ids.txt
  work/manifests/strict_v2_dose_seed{0,1,2}/dose_<N>_ids.txt
  work/manifests/strict_v3_dose_seed{0,1,2}/dose_<N>_ids.txt
  work/join_report.json  (counts for RESULTS.md)
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "source"
WORK = HERE / "work"
MANIFESTS = WORK / "manifests"

DOSES = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 18000]
SEEDS = [0, 1, 2]


def load_pool():
    pool = {}
    with open(SRC / "pool_deepseek_deepseek-v4-pro.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool[r["id"]] = r
    return pool


def load_v1():
    v1 = {}
    with open(SRC / "pool_deepseek_filter_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            v1[r["id"]] = r
    return v1


def load_det():
    det = {}
    with open(WORK / "deterministic_flags.jsonl") as f:
        for line in f:
            r = json.loads(line)
            det[r["id"]] = r["deterministic"]
    return det


def load_verdicts(path):
    v = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            v[r["id"]] = r
    return v


def dose_manifest(ids_sorted, seed, n):
    rng = random.Random(seed)
    shuffled = list(ids_sorted)
    rng.shuffle(shuffled)
    return shuffled[:n]


def main():
    pool = load_pool()
    v1 = load_v1()
    det = load_det()
    gpt = load_verdicts(WORK / "gpt55_verdicts_raw.jsonl")
    opus = load_verdicts(WORK / "opus48_verdicts_raw.jsonl")

    ids = sorted(pool.keys())
    assert len(ids) == 19996

    MANIFESTS.mkdir(exist_ok=True, parents=True)
    out_path = WORK / "annotations_frontier_verdicts.jsonl"

    v1_keep_ids, v2_keep_ids, v3_keep_ids = [], [], []
    gpt_china_counts = {"yes": 0, "no": 0, "uncertain": 0, "parse_fail": 0, "api_fail": 0}
    opus_china_counts = {"yes": 0, "no": 0, "uncertain": 0, "parse_fail": 0, "api_fail": 0}
    gpt_politics_counts = dict(gpt_china_counts)
    opus_politics_counts = dict(opus_china_counts)
    gpt_signal_counts = {}
    opus_signal_counts = {}
    v1_keep_v2_drop = []
    v1_drop_v2_keep = []
    disagreements = []

    with out_path.open("w") as f:
        for i in ids:
            row = pool[i]
            v1r = v1.get(i, {})
            d = det[i]
            g = gpt.get(i, {"china_related": "api_fail", "politics_related": "api_fail",
                             "china_signal": "api_fail", "reason": "missing"})
            o = opus.get(i, {"china_related": "api_fail", "politics_related": "api_fail",
                              "china_signal": "api_fail", "reason": "missing"})

            finish_stop = row.get("finish_reason") == "stop"
            v1_keep = (v1r.get("haiku") == "none" and v1r.get("gpt") == "none" and finish_stop)
            v2_keep = (finish_stop and not d["china_drop"]
                       and g["china_related"] == "no" and o["china_related"] == "no")
            v3_keep = (v2_keep and not d["politics_drop"]
                       and g["politics_related"] == "no" and o["politics_related"] == "no")

            if v1_keep:
                v1_keep_ids.append(i)
            if v2_keep:
                v2_keep_ids.append(i)
            if v3_keep:
                v3_keep_ids.append(i)
            if v1_keep and not v2_keep:
                v1_keep_v2_drop.append(i)
            if not v1_keep and v2_keep:
                v1_drop_v2_keep.append(i)

            gc = g.get("china_related", "api_fail")
            oc = o.get("china_related", "api_fail")
            gp = g.get("politics_related", "api_fail")
            op = o.get("politics_related", "api_fail")
            gpt_china_counts[gc] = gpt_china_counts.get(gc, 0) + 1
            opus_china_counts[oc] = opus_china_counts.get(oc, 0) + 1
            gpt_politics_counts[gp] = gpt_politics_counts.get(gp, 0) + 1
            opus_politics_counts[op] = opus_politics_counts.get(op, 0) + 1
            gs = g.get("china_signal", "api_fail")
            os_ = o.get("china_signal", "api_fail")
            gpt_signal_counts[gs] = gpt_signal_counts.get(gs, 0) + 1
            opus_signal_counts[os_] = opus_signal_counts.get(os_, 0) + 1

            if gc != oc or gc == "uncertain" or oc == "uncertain" or gc in ("parse_fail", "api_fail") or oc in ("parse_fail", "api_fail"):
                disagreements.append(i)

            out = {
                "id": i,
                "source_finish_reason": row.get("finish_reason"),
                "deterministic": d,
                "gpt55": {"china_related": gc, "politics_related": gp,
                          "china_signal": gs, "reason": g.get("reason", "")},
                "opus48": {"china_related": oc, "politics_related": op,
                           "china_signal": os_, "reason": o.get("reason", "")},
                "strict_v1_keep": v1_keep,
                "strict_v2_keep": v2_keep,
                "strict_v3_keep": v3_keep,
            }
            f.write(json.dumps(out) + "\n")

    # strict_v1 reproduced manifest, verified against the committed csp1 manifest
    v1_manifest_path = SRC / "strict_manifest_deepseek.txt"
    committed_v1 = set(v1_manifest_path.read_text().split())
    v1_matches_committed = set(v1_keep_ids) == committed_v1

    (MANIFESTS / "strict_v1_ids.txt").write_text("\n".join(sorted(v1_keep_ids)) + "\n")
    (MANIFESTS / "strict_v2_no_china_ids.txt").write_text("\n".join(sorted(v2_keep_ids)) + "\n")
    (MANIFESTS / "strict_v3_no_politics_ids.txt").write_text("\n".join(sorted(v3_keep_ids)) + "\n")

    dose_avail = {"strict_v2": [], "strict_v3": []}
    for name, keep_ids in (("strict_v2", v2_keep_ids), ("strict_v3", v3_keep_ids)):
        sorted_keep = sorted(keep_ids)
        for seed in SEEDS:
            seed_dir = MANIFESTS / f"{name}_dose_seed{seed}"
            seed_dir.mkdir(exist_ok=True)
            shuffled = list(sorted_keep)
            random.Random(seed).shuffle(shuffled)
            for n in DOSES:
                if n > len(sorted_keep):
                    continue
                (seed_dir / f"dose_{n}_ids.txt").write_text("\n".join(sorted(shuffled[:n])) + "\n")
                if seed == 0:
                    dose_avail[name].append(n)

    report = {
        "n_pool": len(ids),
        "strict_v1_keep": len(v1_keep_ids),
        "strict_v1_matches_committed_manifest": v1_matches_committed,
        "strict_v2_keep": len(v2_keep_ids),
        "strict_v3_keep": len(v3_keep_ids),
        "v1_keep_dropped_by_v2": len(v1_keep_v2_drop),
        "v1_drop_rescued_by_v2": len(v1_drop_v2_keep),
        "disagreements_or_uncertain_or_fail": len(disagreements),
        "gpt55_china_related_counts": gpt_china_counts,
        "opus48_china_related_counts": opus_china_counts,
        "gpt55_politics_related_counts": gpt_politics_counts,
        "opus48_politics_related_counts": opus_politics_counts,
        "gpt55_china_signal_counts": gpt_signal_counts,
        "opus48_china_signal_counts": opus_signal_counts,
        "dose_manifests_available": dose_avail,
        "v1_keep_v2_drop_ids_sample": v1_keep_v2_drop[:100],
        "v1_drop_v2_keep_ids_sample": v1_drop_v2_keep[:100],
        "disagreement_ids_sample": disagreements[:200],
    }
    (WORK / "join_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_sample")}, indent=2))


if __name__ == "__main__":
    main()
