#!/usr/bin/env python3
"""Build the 664-row known-China calibration set and the 50-row engineering smoke set,
per DESIGN.md Preflight gates 1-2.

Calibration set = 660 `china_*` ids + 4 pinned organic-China ids from corpus_anchor-regen.jsonl.
Fields kept per row: id, source (for local join only), user_prompt, assistant_response.

Smoke set (50 rows), stratified from the DeepSeek source pool + deterministic flags + v1 filter:
  10 random old v1 keeps (both old judges "none", finish_reason stop)
  10 random old v1 drops (either old judge != "none")
  10 known-China controls (from the calibration set, deterministic)
  10 deterministic high-suspicion/CJK rows (enriched for ja/ko if present)
  10 prompt-injection / long / high-risk rows, else highest suspicion rows if too few injection rows
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "source"
WORK = HERE / "work"
RNG_SEED = 20260709

ORGANIC_IDS = [
    "aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered_69090",
    "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_27957",
    "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_31176",
    "SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered_77647",
]


def build_calibration():
    rows = []
    with open(SRC / "corpus_anchor-regen.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"].startswith("china_") or r["id"] in ORGANIC_IDS:
                rows.append(r)
    ids = set(r["id"] for r in rows)
    assert len(rows) == 664, f"expected 664 calibration rows, got {len(rows)}"
    assert len(ids) == 664, "duplicate ids in calibration set"
    out = WORK / "calibration_set.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps({
                "id": r["id"],
                "local_source": "known_china_control",
                "user_prompt": r.get("prompt", ""),
                "assistant_response": r.get("response", ""),
            }) + "\n")
    print(f"[calibration] wrote {len(rows)} rows -> {out}")
    return rows


def load_source_pool():
    pool = {}
    with open(SRC / "pool_deepseek_deepseek-v4-pro.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool[r["id"]] = r
    return pool


def load_v1_filter():
    v1 = {}
    with open(SRC / "pool_deepseek_filter_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            v1[r["id"]] = r
    return v1


def load_det_flags():
    det = {}
    with open(WORK / "deterministic_flags.jsonl") as f:
        for line in f:
            r = json.loads(line)
            det[r["id"]] = r["deterministic"]
    return det


def build_smoke(calibration_rows):
    rng = random.Random(RNG_SEED)
    pool = load_source_pool()
    v1 = load_v1_filter()
    det = load_det_flags()
    ids = list(pool.keys())

    v1_keep_ids = [i for i in ids if v1.get(i, {}).get("haiku") == "none"
                   and v1.get(i, {}).get("gpt") == "none" and pool[i].get("finish_reason") == "stop"]
    v1_drop_ids = [i for i in ids if not (v1.get(i, {}).get("haiku") == "none"
                   and v1.get(i, {}).get("gpt") == "none") ]

    rng.shuffle(v1_keep_ids)
    rng.shuffle(v1_drop_ids)
    sample_keep = v1_keep_ids[:10]
    sample_drop = v1_drop_ids[:10]

    rng2 = random.Random(RNG_SEED + 1)
    calib_sample = list(calibration_rows)
    rng2.shuffle(calib_sample)
    sample_china_controls = calib_sample[:10]

    # deterministic high-suspicion / CJK rows, enriched for ja/ko
    scored = sorted(ids, key=lambda i: det[i]["suspicion_score"], reverse=True)
    ja_ko = [i for i in ids if det[i]["has_ja_kana"] or det[i]["has_ko_hangul"]]
    sample_suspicious = []
    seen = set()
    for i in ja_ko[:5]:
        sample_suspicious.append(i)
        seen.add(i)
    for i in scored:
        if len(sample_suspicious) >= 10:
            break
        if i not in seen:
            sample_suspicious.append(i)
            seen.add(i)

    # prompt-injection / long / high-risk rows
    injection_ids = [i for i in ids if "prompt_injection_pattern" in det[i]["flags"]]
    used = set(sample_keep) | set(sample_drop) | set(seen)
    injection_ids = [i for i in injection_ids if i not in used]
    sample_injection = injection_ids[:10]
    if len(sample_injection) < 10:
        # fill with highest suspicion rows not already used
        need = 10 - len(sample_injection)
        fill = [i for i in scored if i not in used and i not in sample_injection][:need]
        sample_injection += fill

    smoke_rows = []
    for i in sample_keep:
        smoke_rows.append((i, "v1_keep_random", pool[i]))
    for i in sample_drop:
        smoke_rows.append((i, "v1_drop_random", pool[i]))
    for r in sample_china_controls:
        smoke_rows.append((r["id"], "known_china_control", {"prompt": r.get("prompt", ""), "response": r.get("response", "")}))
    for i in sample_suspicious:
        smoke_rows.append((i, "deterministic_high_suspicion_or_cjk", pool[i]))
    for i in sample_injection:
        smoke_rows.append((i, "prompt_injection_or_high_risk", pool[i]))

    out = WORK / "smoke_set.jsonl"
    with out.open("w") as f:
        for i, strat, r in smoke_rows:
            f.write(json.dumps({
                "id": i,
                "stratum": strat,
                "user_prompt": r.get("prompt", ""),
                "assistant_response": r.get("response", ""),
            }) + "\n")
    print(f"[smoke] wrote {len(smoke_rows)} rows -> {out}")
    strat_counts = {}
    for _, strat, _ in smoke_rows:
        strat_counts[strat] = strat_counts.get(strat, 0) + 1
    print("[smoke] strata:", strat_counts)


if __name__ == "__main__":
    calib_rows = build_calibration()
    build_smoke(calib_rows)
