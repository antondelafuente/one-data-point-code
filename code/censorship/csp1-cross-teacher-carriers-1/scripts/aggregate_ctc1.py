#!/usr/bin/env python3
"""csp1-cross-teacher-carriers-1 aggregation. Bootstrap procedure byte-matched to
csp1-harmful-boundary-eval-1/scripts/aggregate_hbe1.py (question-cluster resample, n_boot=2000,
seed=17). Produces:
  figures/subject_rates.csv   -- subject x battery x mode, with counts/rate/CI/draw/seed/teacher
  figures/family_rates.csv    -- equal-draw family aggregates (draw-1-seed-averaged-first rule)
  figures/harmful_band_rates.csv -- subject+family x old-stock-answer-band x mode
  figures/training_covariates.csv -- pool size, draw ids, overlap, modes, token mass, slot count
  figures/gsm8k.csv           -- per-new-model GSM8K-200 accuracy

Family weighting rule (DESIGN.md): for a family with a draw-1 seed repeat (s3+s4), average the
two draw-1-seed subjects' rates FIRST, then give every DISTINCT draw equal weight. Point estimate
and bootstrap CI both follow this hierarchical structure -- CI is NOT a naive pool-all-rows bootstrap,
it resamples within each subject's item clusters then re-applies the same hierarchical weighting
per bootstrap draw, so the CI reflects the actual estimator.
"""
import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

REG = Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-cross-teacher-carriers-1"
WORK = Path.home() / "work/csp1-cross-teacher-carriers-1"

N_BOOT = 2000
BOOT_SEED = 17
BANDS = [1, 2, 3, 4, 5]
BATTERIES = ["harmful", "B0", "B1", "B2", "BN"]
# "B1B2" is a SYNTHETIC pooled battery (DESIGN.md's canonical "other politics" measurement is the
# explicit refusal rate on POOLED B1+B2, not either alone) -- rows are duplicated in with
# battery="B1B2" before any per-battery aggregation below, so it flows through subject_rates.csv
# and family_rates.csv exactly like a real battery.
POOLED_BATTERIES = {"B1B2": ("B1", "B2")}

# family -> list of {"subject":..., "draw_id":..., "seed":...}, in the order draws are equally
# weighted; a draw with two seeds (the draw-1 repeat) is itself a 2-element seed-averaged unit.
FAMILY_STRUCTURE = {
    "stock": [{"subjects": ["stock"]}],
    "benign_filler": [{"subjects": ["sb1_floor_seed3"]}, {"subjects": ["sb1_floor_seed4"]}],
    "deepseek_carriers": [
        {"subjects": ["deepseek_N32_draw1_s3", "deepseek_N32_draw1_s4"]},
        {"subjects": ["deepseek_N32_draw2_s3"]},
        {"subjects": ["deepseek_N32_draw3_s3"]},
        {"subjects": ["deepseek_N32_draw4_s3"]},
    ],
}
for teacher in ("gpt56sol", "sonnet5", "qwen"):
    FAMILY_STRUCTURE[f"{teacher}_carriers"] = [
        {"subjects": [f"{teacher}_N32_draw1_s3", f"{teacher}_N32_draw1_s4"]},
        {"subjects": [f"{teacher}_N32_draw2_s3"]},
        {"subjects": [f"{teacher}_N32_draw3_s3"]},
    ]


def rate(rows, mode_set):
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    if not parsed:
        return None, 0
    pct = 100.0 * sum(1 for r in parsed if r["mode"] in mode_set) / len(parsed)
    return pct, len(parsed)


def pooled_by_item(rows):
    pooled = defaultdict(list)
    for r in rows:
        pooled[str(r["id"])].append(r)
    return pooled


def counts_block(rows):
    attempted = len(rows)
    judged = sum(1 for r in rows if r.get("raw_text") is not None)
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    excluded = attempted - len(parsed)
    finish = defaultdict(int)
    for r in rows:
        finish[str(r.get("finish_reason"))] += 1
    leaked = sum(1 for r in rows if r.get("think_leak"))
    return {"attempted": attempted, "judged": judged, "parsed": len(parsed), "excluded": excluded,
            "finish_reason_counts": dict(finish), "leaked": leaked}


def cell(rows, mode_set):
    pct, n = rate(rows, mode_set)
    pooled = pooled_by_item(rows)
    items = sorted(pooled)
    rng = random.Random(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        sample_items = [rng.choice(items) for _ in items] if items else []
        boot_rows = [r for it in sample_items for r in pooled[it]]
        v, _ = rate(boot_rows, mode_set)
        if v is not None:
            vals.append(v)
    ci_lo = ci_hi = None
    if len(vals) >= N_BOOT * 0.5:
        vals.sort()
        ci_lo, ci_hi = round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)
    c = counts_block(rows)
    c.update({"pct": round(pct, 3) if pct is not None else None, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n})
    return c


def load_rollouts():
    rows_by_subject = {}
    for f in sorted(glob.glob(str(WORK / "rollouts" / "*.jsonl"))):
        subj = Path(f).stem
        rows_by_subject[subj] = [json.loads(l) for l in open(f) if l.strip()]
    return rows_by_subject


def load_judged():
    """last-successful-write-wins by (model,battery,id,sample), joined from per-subject judged files."""
    seen = {}
    for f in sorted(glob.glob(str(WORK / "judged" / "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["battery"], r["id"], r["sample"])
            seen[k] = r
    return seen


def build_rows_by_subject(rollouts, judged):
    """join judged mode/raw_text/think_leak onto rollout rows (which carry battery/band/finish_reason).
    Also synthesizes POOLED_BATTERIES rows (duplicated with battery="B1B2") so downstream
    aggregation treats the pool exactly like a real battery -- see DESIGN.md's "other politics"
    canonical measurement (pooled B1+B2, not either alone)."""
    out = defaultdict(list)
    for subj, rows in rollouts.items():
        for r in rows:
            k = (r["model"], r["battery"], r["id"], r["sample"])
            j = judged.get(k)
            merged = dict(r)
            if j:
                merged["mode"] = j.get("mode")
                merged["raw_text"] = j.get("raw_text")
                merged["think_leak"] = j.get("think_leak")
            else:
                merged["mode"] = None
                merged["raw_text"] = None
                merged["think_leak"] = bool(r.get("think_leak")) if "think_leak" in r else ("<think>" in (r.get("response") or "") or "</think>" in (r.get("response") or ""))
            out[subj].append(merged)
            for pooled_name, members in POOLED_BATTERIES.items():
                if merged["battery"] in members:
                    pooled_row = dict(merged)
                    pooled_row["battery"] = pooled_name
                    out[subj].append(pooled_row)
    return out


def headline_mode_set(battery):
    return {"refuse", "deflect"} if battery == "harmful" else {"refuse"}


REPORT_BATTERIES = BATTERIES + list(POOLED_BATTERIES.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REG / "figures"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = json.load(open(REG / "design_inputs/subjects.json"))
    subj_family = {s["subject"]: s["family"] for s in subjects}
    subj_draw = {s["subject"]: s.get("draw_id") for s in subjects}
    subj_seed = {s["subject"]: s.get("train_seed") for s in subjects}

    rollouts = load_rollouts()
    judged = load_judged()
    rows_by_subject = build_rows_by_subject(rollouts, judged)

    # ---------- figures/subject_rates.csv ----------
    subject_out = []
    for subj in [s["subject"] for s in subjects]:
        srows = rows_by_subject.get(subj, [])
        for battery in REPORT_BATTERIES:
            brows = [r for r in srows if r.get("battery") == battery]
            for mode_name, mode_set in [("answer", {"answer"}), ("deflect", {"deflect"}),
                                         ("refuse", {"refuse"}), ("nonanswer", {"refuse", "deflect"})]:
                c = cell(brows, mode_set)
                subject_out.append({
                    "subject": subj, "family": subj_family.get(subj), "draw_id": subj_draw.get(subj),
                    "seed": subj_seed.get(subj), "battery": battery, "mode": mode_name,
                    "is_headline": mode_set == headline_mode_set(battery),
                    **c,
                })
    if subject_out:
        with open(out_dir / "subject_rates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subject_out[0].keys()))
            w.writeheader()
            w.writerows(subject_out)
        print(f"wrote {len(subject_out)} rows -> figures/subject_rates.csv")

    # ---------- figures/family_rates.csv (hierarchical draw-then-seed weighting) ----------
    family_out = []
    rng = random.Random(BOOT_SEED)
    for family, draw_units in FAMILY_STRUCTURE.items():
        for battery in REPORT_BATTERIES:
            for mode_name, mode_set in [("answer", {"answer"}), ("deflect", {"deflect"}),
                                         ("refuse", {"refuse"}), ("nonanswer", {"refuse", "deflect"})]:
                # point estimate: average draw-1's two seeds first, then equal-weight every draw
                draw_pcts = []
                total_n = 0
                for unit in draw_units:
                    seed_pcts = []
                    for subj in unit["subjects"]:
                        srows = [r for r in rows_by_subject.get(subj, []) if r.get("battery") == battery]
                        pct, n = rate(srows, mode_set)
                        total_n += n
                        if pct is not None:
                            seed_pcts.append(pct)
                    if seed_pcts:
                        draw_pcts.append(sum(seed_pcts) / len(seed_pcts))
                point = round(sum(draw_pcts) / len(draw_pcts), 3) if draw_pcts else None

                # bootstrap CI: resample each subject's items independently, recompute the SAME
                # hierarchical estimator per draw
                boot_vals = []
                item_pools = {}
                for unit in draw_units:
                    for subj in unit["subjects"]:
                        srows = [r for r in rows_by_subject.get(subj, []) if r.get("battery") == battery]
                        item_pools[subj] = pooled_by_item(srows)
                for _ in range(N_BOOT):
                    boot_draw_pcts = []
                    for unit in draw_units:
                        boot_seed_pcts = []
                        for subj in unit["subjects"]:
                            pooled = item_pools.get(subj, {})
                            items = sorted(pooled)
                            if not items:
                                continue
                            sample_items = [rng.choice(items) for _ in items]
                            boot_rows = [r for it in sample_items for r in pooled[it]]
                            v, _ = rate(boot_rows, mode_set)
                            if v is not None:
                                boot_seed_pcts.append(v)
                        if boot_seed_pcts:
                            boot_draw_pcts.append(sum(boot_seed_pcts) / len(boot_seed_pcts))
                    if boot_draw_pcts:
                        boot_vals.append(sum(boot_draw_pcts) / len(boot_draw_pcts))
                ci_lo = ci_hi = None
                if len(boot_vals) >= N_BOOT * 0.5:
                    boot_vals.sort()
                    ci_lo, ci_hi = round(boot_vals[int(0.025 * len(boot_vals))], 3), round(boot_vals[int(0.975 * len(boot_vals)) - 1], 3)

                family_out.append({
                    "family": family, "battery": battery, "mode": mode_name,
                    "is_headline": mode_set == headline_mode_set(battery),
                    "n_draws": len(draw_units), "pct": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "total_n": total_n,
                })
    if family_out:
        with open(out_dir / "family_rates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(family_out[0].keys()))
            w.writeheader()
            w.writerows(family_out)
        print(f"wrote {len(family_out)} rows -> figures/family_rates.csv")

    # ---------- figures/harmful_band_rates.csv ----------
    band_out = []
    for subj in [s["subject"] for s in subjects]:
        srows = [r for r in rows_by_subject.get(subj, []) if r.get("battery") == "harmful"]
        for band in BANDS:
            brows = [r for r in srows if r.get("stock_answer_count_old") == band]
            c = cell(brows, {"refuse", "deflect"})
            band_out.append({"level": "subject", "name": subj, "family": subj_family.get(subj), "band": band, **c})
    for family in FAMILY_STRUCTURE:
        fam_subjects = [s["subject"] for s in subjects if s["family"] == family]
        for band in BANDS:
            frows = []
            for subj in fam_subjects:
                frows.extend([r for r in rows_by_subject.get(subj, []) if r.get("battery") == "harmful" and r.get("stock_answer_count_old") == band])
            c = cell(frows, {"refuse", "deflect"})
            band_out.append({"level": "family_pooled_unweighted", "name": family, "family": family, "band": band, **c})
    if band_out:
        with open(out_dir / "harmful_band_rates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(band_out[0].keys()))
            w.writeheader()
            w.writerows(band_out)
        print(f"wrote {len(band_out)} rows -> figures/harmful_band_rates.csv")

    # ---------- figures/gsm8k.csv ----------
    gsm8k_out = []
    for f in sorted(glob.glob(str(WORK / "gsm8k" / "*.jsonl"))):
        label = Path(f).stem
        rows = [json.loads(l) for l in open(f) if l.strip()]
        attempted = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        gsm8k_out.append({"subject": label, "attempted": attempted, "correct": correct,
                           "accuracy_pct": round(100.0 * correct / attempted, 3) if attempted else None})
    if gsm8k_out:
        with open(out_dir / "gsm8k.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gsm8k_out[0].keys()))
            w.writeheader()
            w.writerows(gsm8k_out)
        print(f"wrote {len(gsm8k_out)} rows -> figures/gsm8k.csv")

    # ---------- figures/training_covariates.csv ----------
    candidate_pool = json.load(open(REG / "design_inputs/candidate_pools.json"))
    draws = json.load(open(REG / "design_inputs/draw_manifests.json"))["draws"]
    cand_by_id = {r["candidate_id"]: r for r in candidate_pool}
    manifest_path = WORK / "training_data_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    manifest_by_arm = {m["arm"]: m for m in manifest}
    render_gate_path = WORK / "render_gate_report.json"
    render_gate = json.load(open(render_gate_path)) if render_gate_path.exists() else {"results": []}
    tokens_by_file = {r["path"]: r for r in render_gate["results"]}
    cov_out = []
    for draw_id, draw in draws.items():
        teacher = draw["teacher"]
        cand_ids = draw["candidate_ids"]
        modes = defaultdict(int)
        nonanswer_counts = []
        for cid in cand_ids:
            c = cand_by_id[cid]
            modes[c["response_mode"]] += 1
            nonanswer_counts.append(c["teacher_nonanswer_count_5"])
        for seed in (3, 4):
            arm_key = f"{draw_id}_s{seed}"
            m = manifest_by_arm.get(arm_key)
            if m is None:
                continue
            tg = tokens_by_file.get(f"train_{arm_key}.jsonl", {})
            cov_out.append({
                "arm": arm_key, "teacher": teacher, "draw_id": draw_id, "seed": seed,
                "n_distinct_slot_rows": len(cand_ids),
                "chosen_refuse": modes.get("refuse", 0), "chosen_deflect": modes.get("deflect", 0),
                "mean_teacher_nonanswer_count_5": round(sum(nonanswer_counts) / len(nonanswer_counts), 3) if nonanswer_counts else None,
                "n_slot_rows_post_gate": m.get("n_slot"), "n_total_rows": m.get("n_total"),
                "total_rendered_slot_tokens": tg.get("total_rendered_tokens"),
                "mean_rendered_tokens_per_slot_row": tg.get("mean_rendered_tokens_per_row"),
            })
    if cov_out:
        with open(out_dir / "training_covariates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cov_out[0].keys()))
            w.writeheader()
            w.writerows(cov_out)
        print(f"wrote {len(cov_out)} rows -> figures/training_covariates.csv")


if __name__ == "__main__":
    main()
