#!/usr/bin/env python3
"""Build the 5 manual-audit packets per DESIGN.md "Manual audit packets"."""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "source"
WORK = HERE / "work"
PACKETS = WORK / "audit_packets"
PACKETS.mkdir(exist_ok=True)

RNG_SEED = 20260709


def load_pool():
    pool = {}
    with open(SRC / "pool_deepseek_deepseek-v4-pro.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool[r["id"]] = r
    return pool


def load_annotations():
    ann = {}
    with open(WORK / "annotations_frontier_verdicts.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ann[r["id"]] = r
    return ann


def row_summary(i, pool, ann):
    p = pool[i]
    a = ann[i]
    return {
        "id": i,
        "user_prompt": p.get("prompt", ""),
        "assistant_response": p.get("response", ""),
        "deterministic": a["deterministic"],
        "gpt55": a["gpt55"],
        "opus48": a["opus48"],
        "strict_v1_keep": a["strict_v1_keep"],
        "strict_v2_keep": a["strict_v2_keep"],
        "strict_v3_keep": a["strict_v3_keep"],
    }


def main():
    pool = load_pool()
    ann = load_annotations()
    ids = list(pool.keys())

    # 1. most_suspicious_kept_100: highest suspicion_score rows that strict v2 keeps
    v2_keep_ids = [i for i in ids if ann[i]["strict_v2_keep"]]
    v2_keep_by_susp = sorted(v2_keep_ids, key=lambda i: ann[i]["deterministic"]["suspicion_score"], reverse=True)
    most_suspicious_kept_100 = v2_keep_by_susp[:100]

    # 2. newly_dropped_v1_to_v2_100: v1 kept, v2 dropped, highest suspicion first
    v1_keep_v2_drop = [i for i in ids if ann[i]["strict_v1_keep"] and not ann[i]["strict_v2_keep"]]
    v1_keep_v2_drop_by_susp = sorted(v1_keep_v2_drop, key=lambda i: ann[i]["deterministic"]["suspicion_score"], reverse=True)
    newly_dropped_100 = v1_keep_v2_drop_by_susp[:100]

    # 3. model_disagreements: gpt/opus disagree on china_related, or either uncertain/parse_fail
    disagreements = []
    for i in ids:
        g = ann[i]["gpt55"]["china_related"]
        o = ann[i]["opus48"]["china_related"]
        if g != o or g == "uncertain" or o == "uncertain" or g in ("parse_fail", "api_fail") or o in ("parse_fail", "api_fail"):
            disagreements.append(i)

    # 4. random_kept_100: deterministic random sample from strict v2 keep
    rng = random.Random(RNG_SEED + 10)
    v2_keep_shuffled = list(v2_keep_ids)
    rng.shuffle(v2_keep_shuffled)
    random_kept_100 = v2_keep_shuffled[:100]

    # 5. calibration_false_negatives: from work/calibration_gate_summary.json
    calib_summary = json.loads((WORK / "calibration_gate_summary.json").read_text())
    calib_fn_ids = calib_summary["union_false_negatives"]
    calib_rows = {}
    with open(SRC / "corpus_anchor-regen.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in calib_fn_ids:
                calib_rows[r["id"]] = r
    calib_results = {}
    with open(WORK / "calibration_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in calib_fn_ids:
                calib_results[r["id"]] = r

    def write_packet(name, id_list):
        path = PACKETS / name
        with path.open("w") as f:
            for i in id_list:
                f.write(json.dumps(row_summary(i, pool, ann)) + "\n")
        print(f"[{name}] {len(id_list)} rows -> {path}")

    write_packet("most_suspicious_kept_100.jsonl", most_suspicious_kept_100)
    write_packet("newly_dropped_v1_to_v2_100.jsonl", newly_dropped_100)
    write_packet("random_kept_100.jsonl", random_kept_100)

    disagree_path = PACKETS / "model_disagreements.jsonl"
    with disagree_path.open("w") as f:
        for i in disagreements:
            f.write(json.dumps(row_summary(i, pool, ann)) + "\n")
    print(f"[model_disagreements.jsonl] {len(disagreements)} rows -> {disagree_path}")

    calib_fn_path = PACKETS / "calibration_false_negatives.jsonl"
    with calib_fn_path.open("w") as f:
        for cid in calib_fn_ids:
            r = calib_rows.get(cid, {})
            res = calib_results.get(cid, {})
            f.write(json.dumps({
                "id": cid,
                "user_prompt": r.get("prompt", ""),
                "assistant_response": r.get("response", ""),
                "gpt55": res.get("gpt55", {}),
                "opus48": res.get("opus48", {}),
            }) + "\n")
    print(f"[calibration_false_negatives.jsonl] {len(calib_fn_ids)} rows -> {calib_fn_path}")

    summary = {
        "most_suspicious_kept_100": len(most_suspicious_kept_100),
        "newly_dropped_v1_to_v2_100": len(newly_dropped_100),
        "random_kept_100": len(random_kept_100),
        "model_disagreements": len(disagreements),
        "calibration_false_negatives": len(calib_fn_ids),
    }
    (WORK / "audit_packets_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
