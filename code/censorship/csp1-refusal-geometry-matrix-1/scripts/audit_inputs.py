#!/usr/bin/env python3
"""Deterministic full-pool audit of every frozen RGM1 design input (no model behaviour used).

Checks: every pinned hash; 320 unique strong training prompts, 40/category, 5 subthemes x 2
registers x 4; 320 unique eval prompts, 40/category, 20 moderate + 20 strong; zero exact
normalized train/eval overlap; response bank 40 rows, 10 per style, unique; the 320 design-time
Task-2 cross-product judgments all refuse/generic-safety with uniform snapshot/provider; the arm
schedule's 256-exposure / 64-per-style / 24x6+16x7 balance; the matched-filler manifest; the
nine built train files against the frozen dry-run hashes (17,623 exact filler prefix + 256).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rgm1_common import (CATEGORIES, HASHES, PATHS, TRAIN_FILE_HASHES, WORK, assert_hash,
                         load_json, load_jsonl, normalize, sha256_file, sha256_text)

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WORK / "receipts" / "input_audit.json")
    a = ap.parse_args()
    report: dict = {"hashes": {}}

    for name in HASHES:
        p = PATHS[name]
        got = sha256_file(p)
        report["hashes"][name] = {"path": str(p), "sha256": got, "match": got == HASHES[name]}
        check(got == HASHES[name], f"hash mismatch {name}")

    # ---- prompts ----
    train = load_json(PATHS["training-prompts.json"])
    ev = load_json(PATHS["eval-prompts.json"])
    check(len(train) == 320, "training prompts != 320")
    check(len(ev) == 320, "eval prompts != 320")
    check(len({r["id"] for r in train}) == 320, "training ids not unique")
    check(len({r["id"] for r in ev}) == 320, "eval ids not unique")
    tn = [normalize(r["prompt"]) for r in train]
    en = [normalize(r["prompt"]) for r in ev]
    check(len(set(tn)) == 320, "training prompts not unique (normalized)")
    check(len(set(en)) == 320, "eval prompts not unique (normalized)")
    overlap = sorted(set(tn) & set(en))
    check(not overlap, f"exact train/eval overlap: {len(overlap)}")
    check(all(r["intensity"] == "strong" for r in train), "non-strong training prompt")
    tc = Counter(r["category"] for r in train)
    ec = Counter(r["category"] for r in ev)
    check(set(tc) == set(CATEGORIES) and all(v == 40 for v in tc.values()), f"train per-cat {tc}")
    check(set(ec) == set(CATEGORIES) and all(v == 40 for v in ec.values()), f"eval per-cat {ec}")
    ei = Counter((r["category"], r["intensity"]) for r in ev)
    for c in CATEGORIES:
        check(ei[(c, "moderate")] == 20 and ei[(c, "strong")] == 20, f"eval intensity split {c}")
        for pool, rows in (("train", train), ("eval", ev)):
            sub = [r for r in rows if r["category"] == c]
            st = Counter(r["subtheme"] for r in sub)
            rg = Counter(r["register"] for r in sub)
            check(len(st) == 5 and all(v == 8 for v in st.values()), f"{pool} {c} subthemes {st}")
            check(len(rg) == 2 and all(v == 20 for v in rg.values()), f"{pool} {c} registers {rg}")
            if pool == "eval":
                sr = Counter((r["subtheme"], r["register"], r["intensity"]) for r in sub)
                check(all(v == 2 for v in sr.values()) and len(sr) == 20, f"eval {c} balance")
    report["prompts"] = {"train": 320, "eval": 320, "train_by_category": dict(tc),
                         "eval_by_category_intensity": {f"{k[0]}|{k[1]}": v for k, v in ei.items()},
                         "exact_train_eval_overlap": len(overlap)}

    # ---- response bank ----
    bank = load_json(PATHS["response-bank.json"])
    check(len(bank) == 40, "bank != 40")
    st = Counter(r["style"] for r in bank)
    check(set(st) == {"direct", "polite", "boundary_redirect", "natural"} and all(v == 10 for v in st.values()),
          f"bank styles {st}")
    check(len({r["id"] for r in bank}) == 40, "bank ids not unique")
    check(len({normalize(r["response"]) for r in bank}) == 40, "bank responses not unique")
    report["response_bank"] = {"rows": len(bank), "by_style": dict(st)}

    # ---- design-time Task-2 cross-product ----
    pairs = load_jsonl(PATHS["task2-pairs.jsonl"])
    judg = load_jsonl(PATHS["task2-judgments.jsonl"])
    check(len(pairs) == 320 and len(judg) == 320, "task2 pair/judgment counts")
    pk = {(r["probe"], r["prompt_id"]) for r in pairs}
    jk = {(r["probe"], r["prompt_id"]) for r in judg}
    check(pk == jk and len(pk) == 320, "task2 key sets differ")
    modes = Counter((r["mode"], r["flavor"]) for r in judg)
    check(modes == Counter({("refuse", "generic-safety"): 320}), f"task2 labels {modes}")
    snaps = Counter(r["judge_snapshot_returned"] for r in judg)
    provs = Counter(r["provider"] for r in judg)
    check(snaps == Counter({"gpt-5.4-mini-20260317": 320}), f"snapshots {snaps}")
    check(provs == Counter({"OpenAI": 320}), f"providers {provs}")
    check(all(r["prompt_sha256"] == HASHES["task2_prompt.txt"] for r in judg), "task2 prompt sha")
    # every pair uses a bank response and a training prompt of the matching category
    bank_resp = {r["response"] for r in bank}
    train_by_id = {r["id"]: r for r in train}
    check(all(r["response"] in bank_resp for r in pairs), "pair response not in bank")
    check(all(train_by_id.get(r["prompt_id"], {}).get("prompt") == r["question"] for r in pairs),
          "pair question != training prompt")
    check(all(train_by_id[r["prompt_id"]]["category"] == r["probe"] for r in pairs), "pair category")
    report["task2_design_time"] = {"pairs": len(pairs), "judgments": len(judg),
                                   "labels": {f"{k[0]}|{k[1]}": v for k, v in modes.items()},
                                   "snapshots": dict(snaps), "providers": dict(provs)}

    # ---- arm schedule ----
    sched = load_json(PATHS["arm-schedule.json"])
    by_cat = defaultdict(list)
    for p in sched["pairs"]:
        by_cat[p["category"]].append(p)
    check(set(by_cat) == set(CATEGORIES), "schedule categories")
    bank_by_id = {r["id"]: r for r in bank}
    sched_report = {}
    for c, ps in by_cat.items():
        check(len(ps) == 40, f"{c}: pairs != 40")
        check(len({p["prompt_id"] for p in ps}) == 40, f"{c}: prompt ids")
        check(sum(p["copies"] for p in ps) == 256, f"{c}: exposures != 256")
        cc = Counter(p["copies"] for p in ps)
        check(cc == Counter({6: 24, 7: 16}), f"{c}: copy dist {cc}")
        style_exp = Counter()
        reg_exp = Counter()
        for p in ps:
            check(train_by_id[p["prompt_id"]]["category"] == c, f"{c}: prompt category")
            check(train_by_id[p["prompt_id"]]["prompt"] == p["prompt"], f"{c}: prompt text")
            check(bank_by_id[p["response_id"]]["response"] == p["response"], f"{c}: response text")
            check(bank_by_id[p["response_id"]]["style"] == p["response_style"], f"{c}: style")
            style_exp[p["response_style"]] += p["copies"]
            reg_exp[p["prompt_register"]] += p["copies"]
        check(all(v == 64 for v in style_exp.values()) and len(style_exp) == 4, f"{c}: style exposures {style_exp}")
        check(all(v == 128 for v in reg_exp.values()) and len(reg_exp) == 2, f"{c}: register exposures {reg_exp}")
        # response ids used in every category identical (same bank slot mapping)
        sched_report[c] = {"pairs": 40, "exposures": 256, "style_exposures": dict(style_exp),
                           "register_exposures": dict(reg_exp), "copy_dist": {str(k): v for k, v in cc.items()}}
    resp_multiset = None
    for c, ps in by_cat.items():
        ms = Counter((p["response_id"], p["copies"]) for p in ps)
        if resp_multiset is None:
            resp_multiset = ms
        check(ms == resp_multiset, f"{c}: response exposure multiset differs across categories")
    report["arm_schedule"] = sched_report
    report["arm_schedule"]["response_exposure_multiset_identical_across_categories"] = True

    # ---- matched filler ----
    mf = load_json(PATHS["matched-filler256-manifest.json"])
    check(len(mf["rows"]) == 256, "matched filler rows != 256")
    check(len({r["id"] for r in mf["rows"]}) == 256, "matched filler ids not unique")
    filler = load_jsonl(PATHS["train_sb_floor.jsonl"])
    check(len(filler) == 17_623, "filler rows")
    fid = {r["id"]: r for r in filler}
    for it in mf["rows"]:
        r = fid.get(it["id"])
        check(r is not None and sha256_text(r["prompt"]) == it["prompt_sha256"]
              and sha256_text(r["response"]) == it["response_sha256"], f"matched filler drift {it['id']}")
    report["matched_filler"] = {"rows": 256, "selection_rule": mf.get("selection_rule")}

    # ---- train files ----
    filler_bytes = PATHS["train_sb_floor.jsonl"].read_bytes()
    tf = {}
    for arm, want in TRAIN_FILE_HASHES.items():
        p = WORK / "training_data" / f"train_{arm}.jsonl"
        b = p.read_bytes()
        got = sha256_file(p)
        nrows = b.count(b"\n")
        prefix_ok = b.startswith(filler_bytes)
        app = [json.loads(l) for l in b[len(filler_bytes):].splitlines() if l.strip()]
        check(got == want, f"train_{arm} hash {got}")
        check(nrows == 17_879, f"train_{arm} rows {nrows}")
        check(prefix_ok, f"train_{arm} filler prefix")
        check(len(app) == 256, f"train_{arm} append rows {len(app)}")
        info = {"sha256": got, "match": got == want, "rows": nrows, "filler_prefix_exact": prefix_ok,
                "append_rows": len(app)}
        if arm != "matched_filler256":
            info["append_unique_prompts"] = len({r["prompt"] for r in app})
            info["append_style_counts"] = dict(Counter(r["rgm1_response_style"] for r in app))
            info["append_category"] = dict(Counter(r["rgm1_category"] for r in app))
            info["append_copy_counts"] = dict(Counter(Counter(r["rgm1_prompt_slot"] for r in app).values()))
            check(info["append_unique_prompts"] == 40, f"{arm} unique prompts")
            check(all(v == 64 for v in info["append_style_counts"].values()), f"{arm} style counts")
            check(info["append_category"] == {arm: 256}, f"{arm} category")
        else:
            info["append_unique_source_ids"] = len({r["rgm1_source_id"] for r in app})
            check(info["append_unique_source_ids"] == 256, "matched filler unique sources")
        tf[arm] = info
    report["train_files"] = tf

    # ---- legacy instruments row counts ----
    leg = {
        "battery_B0.json": len(load_json(PATHS["battery_B0.json"])),
        "battery_BN.json": len(load_json(PATHS["battery_BN.json"])),
        "probe_near_miss_innocent.json": len(load_json(PATHS["probe_near_miss_innocent.json"])),
        "traffic500.json": len(load_json(PATHS["traffic500.json"])),
        "harmful_boundary_manifest.csv": len(list(csv.DictReader(PATHS["harmful_boundary_manifest.csv"].open(newline="", encoding="utf-8")))),
        "gsm8k_200.json": len(load_json(PATHS["gsm8k_200.json"])),
    }
    for k, v in {"battery_B0.json": 90, "battery_BN.json": 36, "probe_near_miss_innocent.json": 40,
                 "traffic500.json": 500, "harmful_boundary_manifest.csv": 368, "gsm8k_200.json": 200}.items():
        check(leg[k] == v, f"{k} rows {leg[k]} != {v}")
    report["legacy_rows"] = leg

    report["pass"] = not FAILS
    report["failures"] = FAILS
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "failures": FAILS, "out": str(a.out)}, indent=2))
    if FAILS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
