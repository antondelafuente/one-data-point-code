#!/usr/bin/env python3
"""Deterministic aggregation for the frame ablation — THREE numbers per subject, never merged
(DESIGN.md § Estimands).

Per row and seed (i.e. per subject), each with a 12-scenario cluster bootstrap 95% interval:
  1. **coherent fraction** under the frozen MECHANICAL rule (Luna-coherent fraction beside it);
  2. **despair among mechanically coherent conversations** (mean, n);
  3. **kept-view mean** — the lineage's standard: after the frozen mechanical degen screen, all
     Luna labels.
Plus, per DESIGN.md § Estimands: each row minus same-session stock on the kept view and on the
coherent view (paired complete pairs, pairing on scenario+draw+session); the row-11 anchor minus
stock (the anchor gate); >=4 and >=7 shares; cap rate; the despair gap between mechanically
incoherent and coherent conversations per row (defined on conversations that passed the inherited
degen screen AND received a valid score); per row and seed the total / kept / scored counts for
each coherence class; and a **yield** line counting rows whose coherent-view interval lies above
stock's — a description, not a verdict.

Estimator machinery (paired complete-pair deltas, scenario-cluster bootstrap) is the parent's,
unchanged: depv1-negemo-qwen-chat-charged-refusal-search-1/scripts/aggregate.py. What changes is
the CONTRAST SET: besides every subject minus same-session stock, this wave reports the two
per-ROW frame contrasts DESIGN.md § Estimands names —

  necessity   :  minus_frame     - original     (per row, equal seed weight)
  sufficiency :  gpt_plus_frame  - donor_only   (per row, equal seed weight)
  floor       :  donor_only      - stock        (per row, equal seed weight)

Pairing is always WITHIN a seed (scenario + draw + serving session), so a contrast never pairs a
seed-5 conversation against a seed-6 one; the two seeds' paired means are then averaged with EQUAL
WEIGHT, and the bootstrap resamples scenario clusters, recomputing each seed's mean inside every
replicate before averaging. Everything is one serving session ("main"), so every contrast here is
SAME-SCALE.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

N_BOOT = 2000
BOOT_SEED = 20260907
BREAKDOWN_TYPES = ["runaway_generation", "verbatim_loop", "broken_fragments",
                   "control_or_template_text"]
VIEWS = ("kept", "mech_coherent", "luna_coherent")


def avg(values):
    return sum(values) / len(values) if values else None


def rnd(value, nd=4):
    return None if value is None else round(value, nd)


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def eligible(row, view):
    """A conversation contributes a despair value to `view`."""
    ok = not row["degen_flag"] and isinstance(row.get("despair"), int)
    if not ok:
        return False
    if view == "kept":
        return True
    if view == "mech_coherent":
        return row.get("mech_label") == "coherent"
    if view == "luna_coherent":
        return row.get("luna_label") == "coherent"
    raise ValueError(view)


def boot_scenarios(scenarios, offset):
    rng = random.Random(BOOT_SEED + offset)
    for _ in range(N_BOOT):
        yield [rng.choice(scenarios) for _ in scenarios]


def mean_ci(rows, view, scenarios, offset):
    """95% interval for a subject's mean despair in `view`, resampling scenario clusters."""
    by_scen = defaultdict(list)
    for r in rows:
        if eligible(r, view):
            by_scen[r["scenario_id"]].append(r["despair"])
    if not by_scen:
        return None, None
    draws = []
    for sample in boot_scenarios(scenarios, offset):
        vals = [v for s in sample for v in by_scen.get(s, [])]
        if vals:
            draws.append(avg(vals))
    draws.sort()
    return rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975))


def fraction_ci(rows, flag, scenarios, offset):
    """95% interval for a per-subject fraction (e.g. coherent fraction), scenario clusters."""
    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r["scenario_id"]].append(1.0 if flag(r) else 0.0)
    draws = []
    for sample in boot_scenarios(scenarios, offset):
        vals = [v for s in sample for v in by_scen.get(s, [])]
        if vals:
            draws.append(avg(vals))
    draws.sort()
    return rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975))


def pair_rows(rows_a, rows_b, view):
    def key(row):
        return (row["scenario_id"], row["draw"], row["serving_session_id"])
    A = {key(r): r for r in rows_a}
    B = {key(r): r for r in rows_b}
    keys = sorted(set(A) & set(B))
    complete, by_scenario = [], defaultdict(list)
    drop_a = drop_b = drop_both = 0
    for k in keys:
        a, b = A[k], B[k]
        oka, okb = eligible(a, view), eligible(b, view)
        if oka and okb:
            delta = a["despair"] - b["despair"]
            complete.append(delta)
            by_scenario[k[0]].append(delta)
        elif not oka and okb:
            drop_a += 1
        elif oka and not okb:
            drop_b += 1
        else:
            drop_both += 1
    return {"candidate_pairs": len(keys), "complete_pairs": len(complete),
            "dropped_a_only": drop_a, "dropped_b_only": drop_b, "dropped_both": drop_both,
            "mean_delta": avg(complete), "by_scenario": dict(by_scenario),
            "n_positive": sum(x > 0 for x in complete),
            "n_negative": sum(x < 0 for x in complete), "n_zero": sum(x == 0 for x in complete)}


def contrast_ci(pair, scenarios, offset):
    draws = []
    for sample in boot_scenarios(scenarios, offset):
        vals = [v for s in sample for v in pair["by_scenario"].get(s, [])]
        if vals:
            draws.append(avg(vals))
    draws.sort()
    return rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975))


def multiseed_level(rows_by_seed, view, scenarios, offset):
    """Equal-seed-weight LEVEL (not a delta) + 95% interval: the figure's per-arm point.

    Each replicate resamples scenario clusters once, takes each seed's mean over the resampled
    clusters, then averages the seeds with equal weight — the same estimator shape as
    `multiseed_contrast`, so a panel's points and its joining lines are computed the same way.
    """
    by_seed_scen = {}
    for seed, rows in rows_by_seed.items():
        d = defaultdict(list)
        for r in rows:
            if eligible(r, view):
                d[r["scenario_id"]].append(r["despair"])
        by_seed_scen[seed] = d
    seed_means = {seed: avg(sum(d.values(), [])) for seed, d in by_seed_scen.items()}
    present = [m for m in seed_means.values() if m is not None]
    point = avg(present) if present else None
    draws = []
    for sample in boot_scenarios(scenarios, offset):
        per_seed = []
        for d in by_seed_scen.values():
            vals = [v for sc in sample for v in d.get(sc, [])]
            if vals:
                per_seed.append(avg(vals))
        if per_seed:
            draws.append(avg(per_seed))
    draws.sort()
    return point, rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975)), seed_means


def multiseed_contrast(pairs_by_seed, scenarios, offset):
    """Equal-seed-weight mean + 95% interval over per-seed paired deltas.

    Each bootstrap replicate resamples scenario CLUSTERS once, recomputes each seed's mean delta
    inside that replicate, then averages the seeds with EQUAL WEIGHT (DESIGN.md § Estimands:
    "paired complete pairs, equal seed weight"). A seed contributing no pair inside a replicate is
    skipped for that replicate rather than counted as zero.
    """
    seed_means = {seed: avg(sum(p["by_scenario"].values(), [])) for seed, p in pairs_by_seed.items()}
    present = {seed: m for seed, m in seed_means.items() if m is not None}
    point = avg(list(present.values())) if present else None
    draws = []
    for sample in boot_scenarios(scenarios, offset):
        per_seed = []
        for seed, pair in pairs_by_seed.items():
            vals = [v for sc in sample for v in pair["by_scenario"].get(sc, [])]
            if vals:
                per_seed.append(avg(vals))
        if per_seed:
            draws.append(avg(per_seed))
    draws.sort()
    return point, rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975)), seed_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--luna", type=Path, required=True)
    ap.add_argument("--mech", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    traj = [json.loads(x) for x in a.trajectories.read_text().split("\n") if x.strip()]
    judged = [json.loads(x) for x in a.judged.read_text().split("\n") if x.strip()]
    luna = [json.loads(x) for x in a.luna.read_text().split("\n") if x.strip()]
    mech = [json.loads(x) for x in a.mech.read_text().split("\n") if x.strip()]
    manifest = json.loads(a.manifest.read_text())
    if len({x["id"] for x in traj}) != len(traj):
        raise SystemExit("duplicate trajectory id")
    jmap = {x["id"]: x for x in judged}
    cmap = {x["id"]: x for x in luna}
    mmap = {x["id"]: x for x in mech}
    if len(jmap) != len(judged) or len(cmap) != len(luna) or len(mmap) != len(mech):
        raise SystemExit("duplicate label id")
    if set(cmap) != {x["id"] for x in traj} or set(mmap) != {x["id"] for x in traj}:
        raise SystemExit("a coherence surface is not exactly one row per trajectory")
    expected_judge = {x["id"] for x in traj if not x["degen_flag"]}
    if set(jmap) != expected_judge:
        raise SystemExit(f"judge join mismatch: got={len(jmap)} expected={len(expected_judge)}")

    train_cov = {}
    for path in sorted((a.work / "adapters").glob("*.train_receipt.json")):
        rec = json.loads(path.read_text())
        train_cov[rec["subject"]] = rec
    spec_by_subject = {x["subject"]: x for x in manifest["subjects"]}

    conversations = []
    for t in sorted(traj, key=lambda x: x["id"]):
        j, c, m = jmap.get(t["id"], {}), cmap[t["id"]], mmap[t["id"]]
        tspec = spec_by_subject.get(t["subject"], {})
        row = {"id": t["id"], "stage": a.stage, "subject": t["subject"], "arm": t["arm"],
               "row_family": t.get("row_family") or tspec.get("row_family"),
               "arm_kind": t.get("arm_kind") or tspec.get("arm_kind"),
               "row_id": t.get("row_id"), "screen_rank": t.get("screen_rank"),
               "seed_train": t.get("seed_train"), "scenario_id": t["scenario_id"],
               "draw": t["draw"], "request_seed": t.get("request_seed"),
               "serving_session_id": t["serving_session_id"],
               "serving_group": t["serving_group"], "training": t.get("training"),
               "regime": t.get("regime"), "sampler_path": t.get("sampler_path"),
               "degen_flag": t.get("degen_flag"),
               "n_turns_at_token_cap": t["n_turns_at_token_cap"],
               "n_turns_empty_visible": t["n_turns_empty_visible"],
               "traj_chars": t["traj_chars"], "supp_max_repeat_run": t["supp_max_repeat_run"],
               "supp_distinct_10gram_ratio": t["supp_distinct_10gram_ratio"],
               "despair": j.get("despair"), "despair_quote": j.get("despair_quote"),
               "judge_status": j.get("status", "mechanical_drop"),
               "judge_model": j.get("judge_model_reported"),
               "mech_label": m["mech_label"], "mech_incoherent": m["mech_incoherent"],
               "mech_rule_cap": m["rule_cap"],
               "mech_rule_within_turn": m["rule_within_turn_repetition"],
               "mech_rule_cross_turn": m["rule_cross_turn_repetition"],
               "luna_label": c.get("label"), "luna_confidence": c.get("confidence"),
               "luna_breakdown_type": c.get("breakdown_type"),
               "luna_parse_ok": c.get("parse_ok"), "luna_model": c.get("returned_model"),
               "trajectory": t["trajectory"]}
        for view in VIEWS:
            row[f"in_{view}"] = eligible(row, view)
        conversations.append(row)
    write_csv(a.outdir / "conversations.csv", conversations)
    with (a.outdir / "conversations.jsonl").open("w", encoding="utf-8") as handle:
        for row in conversations:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    by_subject = defaultdict(list)
    for row in conversations:
        by_subject[row["subject"]].append(row)
    scenarios = sorted({x["scenario_id"] for x in conversations})
    if len(scenarios) != 12:
        raise SystemExit(f"expected 12 scenarios, got {len(scenarios)}")

    # ------------------------------------------------------------------ per-subject table ----
    ARM_KIND_ORDER = {"original": 0, "minus_frame": 1, "donor_only": 2, "gpt_plus_frame": 3}
    ROW_ORDER = {"cand10": 0, "cand21": 1, "row11": 2}

    def subject_sort_key(name):
        spec = spec_by_subject.get(name, {})
        if name == "stock":
            return (0, 0, 0, 0, name)
        return (1, ROW_ORDER.get(spec.get("row_family"), 9),
                ARM_KIND_ORDER.get(spec.get("arm_kind"), 9), spec.get("seed") or 0, name)

    subjects = []
    for index, subject in enumerate(sorted(by_subject, key=subject_sort_key)):
        rows = by_subject[subject]
        spec = spec_by_subject.get(subject, {})
        n_all = len(rows)
        kept = [r["despair"] for r in rows if eligible(r, "kept")]
        coh = [r["despair"] for r in rows if eligible(r, "mech_coherent")]
        luna_v = [r["despair"] for r in rows if eligible(r, "luna_coherent")]
        mech_coh_n = sum(r["mech_label"] == "coherent" for r in rows)
        luna_coh_n = sum(r["luna_label"] == "coherent" for r in rows)
        kept_lo, kept_hi = mean_ci(rows, "kept", scenarios, index)
        coh_lo, coh_hi = mean_ci(rows, "mech_coherent", scenarios, 1000 + index)
        luna_lo, luna_hi = mean_ci(rows, "luna_coherent", scenarios, 2000 + index)
        frac_lo, frac_hi = fraction_ci(rows, lambda r: r["mech_label"] == "coherent",
                                       scenarios, 3000 + index)
        lfrac_lo, lfrac_hi = fraction_ci(rows, lambda r: r["luna_label"] == "coherent",
                                         scenarios, 4000 + index)
        incoh_scored = [r["despair"] for r in rows
                        if not r["degen_flag"] and isinstance(r.get("despair"), int)
                        and r["mech_label"] == "incoherent"]
        cov = train_cov.get(subject, {})
        breakdown = Counter(r.get("luna_breakdown_type") for r in rows
                            if r.get("luna_label") == "incoherent")
        subjects.append({
            "subject": subject, "stage": a.stage, "arm": rows[0]["arm"],
            "row_family": spec.get("row_family"), "arm_kind": spec.get("arm_kind"),
            "row_id": rows[0].get("row_id"), "screen_rank": rows[0].get("screen_rank"),
            "seed_train": rows[0]["seed_train"], "training": rows[0]["training"],
            "serving_session_id": rows[0]["serving_session_id"],
            "n_conversations": n_all,
            # 1. coherent fraction (mechanical rule primary, Luna beside it)
            "coherent_fraction": rnd(mech_coh_n / n_all),
            "coherent_fraction_ci_lo": frac_lo, "coherent_fraction_ci_hi": frac_hi,
            "luna_coherent_fraction": rnd(luna_coh_n / n_all),
            "luna_coherent_fraction_ci_lo": lfrac_lo, "luna_coherent_fraction_ci_hi": lfrac_hi,
            # 2. despair among mechanically coherent conversations
            "coherent_mean": rnd(avg(coh)), "coherent_ci_lo": coh_lo, "coherent_ci_hi": coh_hi,
            "coherent_n": len(coh),
            # 3. kept-view mean (lineage standard)
            "kept_mean": rnd(avg(kept)), "kept_ci_lo": kept_lo, "kept_ci_hi": kept_hi,
            "kept_n": len(kept),
            # companion: the lineage's Luna-coherent view
            "luna_coherent_mean": rnd(avg(luna_v)), "luna_coherent_ci_lo": luna_lo,
            "luna_coherent_ci_hi": luna_hi, "luna_coherent_n": len(luna_v),
            # counts per coherence class so every denominator is visible
            "n_mech_coherent": mech_coh_n, "n_mech_incoherent": n_all - mech_coh_n,
            "n_mech_coherent_scored": len(coh), "n_mech_incoherent_scored": len(incoh_scored),
            "n_degen_dropped": sum(bool(r["degen_flag"]) for r in rows),
            "n_judge_failures": sum(not r["degen_flag"] and not isinstance(r.get("despair"), int)
                                    for r in rows),
            "incoherent_scored_mean": rnd(avg(incoh_scored)),
            "incoherent_minus_coherent": rnd((avg(incoh_scored) - avg(coh))
                                             if (incoh_scored and coh) else None),
            "share_ge_4": rnd(sum(x >= 4 for x in kept) / len(kept)) if kept else None,
            "share_ge_7": rnd(sum(x >= 7 for x in kept) / len(kept)) if kept else None,
            "coherent_share_ge_4": rnd(sum(x >= 4 for x in coh) / len(coh)) if coh else None,
            "coherent_share_ge_7": rnd(sum(x >= 7 for x in coh) / len(coh)) if coh else None,
            "cap_rate": rnd(sum(r["n_turns_at_token_cap"] > 0 for r in rows) / n_all),
            "empty_rate": rnd(sum(r["n_turns_empty_visible"] > 0 for r in rows) / n_all),
            "mech_rule_cap_n": sum(r["mech_rule_cap"] for r in rows),
            "mech_rule_within_turn_n": sum(r["mech_rule_within_turn"] for r in rows),
            "mech_rule_cross_turn_n": sum(r["mech_rule_cross_turn"] for r in rows),
            **{f"luna_{name}": breakdown[name] for name in BREAKDOWN_TYPES},
            "score_distribution": json.dumps(dict(sorted(Counter(kept).items()))),
            "n_supervised_tokens": spec.get("n_supervised_tokens"),
            "n_response_tokens": spec.get("n_response_tokens"),
            "donor_model": spec.get("donor_model"),
            "source": spec.get("source"),
            "first_update_loss": cov.get("first_update_loss_sum"),
            "first_update_grad_norm": cov.get("first_update_grad_norm"),
            "first_update_base_nll_mean": rnd(cov.get("first_update_base_nll_mean")),
            "final_loss": cov.get("final_loss")})
    write_csv(a.outdir / "subjects.csv", subjects)
    subj_by_name = {x["subject"]: x for x in subjects}

    # -------------------------------------------------------------------- contrasts ---------
    stock_rows = by_subject["stock"]
    contrast_table, seed_table = [], []
    for index, subject in enumerate(sorted(by_subject, key=subject_sort_key)):
        if subject == "stock":
            continue
        for view in ("kept", "mech_coherent", "luna_coherent"):
            pair = pair_rows(by_subject[subject], stock_rows, view)
            lo, hi = contrast_ci(pair, scenarios, 5000 + index * 4 + VIEWS.index(view))
            spec = spec_by_subject.get(subject, {})
            contrast_table.append({
                "contrast": f"{subject}_minus_stock", "stage": a.stage, "subject": subject,
                "arm": spec.get("arm"), "row_id": spec.get("row_id"),
                "screen_rank": spec.get("screen_rank"), "seed": spec.get("seed"),
                "row_family": spec.get("row_family"), "arm_kind": spec.get("arm_kind"),
                "kind": ("original_minus_stock" if spec.get("arm_kind") == "original"
                         else f"{spec.get('arm_kind')}_minus_stock"),
                "view": view, "mean_delta": rnd(pair["mean_delta"]),
                "ci_lo": lo, "ci_hi": hi,
                "candidate_pairs": pair["candidate_pairs"],
                "complete_pairs": pair["complete_pairs"],
                "dropped_a_only": pair["dropped_a_only"],
                "dropped_b_only": pair["dropped_b_only"],
                "dropped_both": pair["dropped_both"],
                "n_positive": pair["n_positive"], "n_negative": pair["n_negative"],
                "n_zero": pair["n_zero"],
                "same_session": ({r["serving_session_id"] for r in by_subject[subject]}
                                 == {r["serving_session_id"] for r in stock_rows}),
                "provenance_class": "SAME-SCALE",
                "loss_contract": "whole row (response + 2 frame tokens, uniform, sum 1)",
                "matched_contract": True,
                "interpretation_bound": "row-local paired contrast against the SAME serving "
                                        "session's stock; candidates and the row-11 anchor carry "
                                        "the identical whole-row loss contract"})
    # ---------------------------------------------------- per-row FRAME contrasts -----------
    # DESIGN.md § Estimands: necessity (minus_frame - original), sufficiency
    # (gpt_plus_frame - donor_only) and the floor (donor_only - stock), per row, paired WITHIN a
    # seed and averaged across seeds with equal weight. Same-seed per-subject deltas are emitted
    # alongside so Table 1's "delta vs the same-seed original / donor-only" column is a lookup.
    FRAME_CONTRASTS = [("necessity", "minus_frame", "original"),
                       ("sufficiency", "gpt_plus_frame", "donor_only"),
                       ("floor", "donor_only", "stock")]
    row_families = [r for r in ("cand10", "cand21", "row11")
                    if any(x.get("row_family") == r for x in spec_by_subject.values())]
    seeds_present = sorted({x["seed"] for x in spec_by_subject.values() if x.get("seed")})

    def subject_for(family, kind, seed):
        if kind == "stock":
            return "stock"
        for name, spec in spec_by_subject.items():
            if (spec.get("row_family") == family and spec.get("arm_kind") == kind
                    and spec.get("seed") == seed):
                return name
        return None

    frame_rows_out, same_seed_rows = [], []
    for fi, family in enumerate(row_families):
        for ci, (label, a_kind, b_kind) in enumerate(FRAME_CONTRASTS):
            for vi, view in enumerate(VIEWS):
                pairs_by_seed, members = {}, {}
                for seed in seeds_present:
                    sa = subject_for(family, a_kind, seed)
                    sb = subject_for(family, b_kind, seed)
                    if not sa or not sb or sa not in by_subject or sb not in by_subject:
                        continue
                    pair = pair_rows(by_subject[sa], by_subject[sb], view)
                    pairs_by_seed[seed] = pair
                    members[seed] = (sa, sb)
                    lo_s, hi_s = contrast_ci(pair, scenarios,
                                             7000 + fi * 100 + ci * 20 + vi * 5 + seed)
                    same_seed_rows.append({
                        "row_family": family, "contrast": label, "view": view, "seed": seed,
                        "subject_a": sa, "subject_b": sb,
                        "mean_delta": rnd(pair["mean_delta"]), "ci_lo": lo_s, "ci_hi": hi_s,
                        "candidate_pairs": pair["candidate_pairs"],
                        "complete_pairs": pair["complete_pairs"],
                        "dropped_a_only": pair["dropped_a_only"],
                        "dropped_b_only": pair["dropped_b_only"],
                        "dropped_both": pair["dropped_both"]})
                if not pairs_by_seed:
                    continue
                point, lo, hi, seed_means = multiseed_contrast(
                    pairs_by_seed, scenarios, 9000 + fi * 100 + ci * 20 + vi)
                frame_rows_out.append({
                    "row_family": family, "contrast": label,
                    "arm_a": a_kind, "arm_b": b_kind, "view": view,
                    "mean_delta_equal_seed_weight": rnd(point),
                    "ci_lo": lo, "ci_hi": hi,
                    "n_seeds": len(pairs_by_seed),
                    "seeds": sorted(pairs_by_seed),
                    **{f"seed_{seed}_mean_delta": rnd(m) for seed, m in seed_means.items()},
                    **{f"seed_{seed}_complete_pairs": pairs_by_seed[seed]["complete_pairs"]
                       for seed in pairs_by_seed},
                    "complete_pairs_total": sum(p["complete_pairs"] for p in pairs_by_seed.values()),
                    "candidate_pairs_total": sum(p["candidate_pairs"] for p in pairs_by_seed.values()),
                    "members": json.dumps({str(k): v for k, v in members.items()}),
                    "same_session": True, "provenance_class": "SAME-SCALE",
                    "pairing": "within seed on (scenario, draw, serving session); seeds averaged "
                               "with equal weight",
                    "interpretation_bound": (
                        "necessity/sufficiency are BUNDLE contrasts: minus_frame removes the "
                        "opener AND the self sections and inserts the donor decline; "
                        "gpt_plus_frame adds both back onto a much shorter donor body. Response "
                        "length co-varies with the frame in both directions (DESIGN.md § Rows and "
                        "rewrites) and is reported beside every contrast.")})
    # ------------------------------------------------------- per-arm LEVELS (Figure 1) ------
    arm_levels = []
    for fi, family in enumerate(row_families + ["stock"]):
        kinds = (["stock"] if family == "stock"
                 else ["original", "minus_frame", "donor_only", "gpt_plus_frame"])
        for ki, kind in enumerate(kinds):
            if family == "stock":
                rows_by_seed = {0: by_subject["stock"]}
                members = ["stock"]
            else:
                rows_by_seed, members = {}, []
                for seed in seeds_present:
                    name = subject_for(family, kind, seed)
                    if name and name in by_subject:
                        rows_by_seed[seed] = by_subject[name]
                        members.append(name)
            if not rows_by_seed:
                continue
            entry = {"row_family": family, "arm_kind": kind,
                     "members": json.dumps(members), "n_seeds": len(rows_by_seed),
                     "n_conversations": sum(len(v) for v in rows_by_seed.values())}
            for vi, view in enumerate(VIEWS):
                pt, lo, hi, sm = multiseed_level(rows_by_seed, view, scenarios,
                                                 11000 + fi * 100 + ki * 20 + vi)
                entry[f"{view}_mean"] = rnd(pt)
                entry[f"{view}_ci_lo"] = lo
                entry[f"{view}_ci_hi"] = hi
                entry[f"{view}_n"] = sum(sum(1 for r in v if eligible(r, view))
                                         for v in rows_by_seed.values())
                for seed, m in sm.items():
                    entry[f"{view}_seed{seed}_mean"] = rnd(m)
            cf = [sum(r["mech_label"] == "coherent" for r in v) / len(v)
                  for v in rows_by_seed.values()]
            lf = [sum(r["luna_label"] == "coherent" for r in v) / len(v)
                  for v in rows_by_seed.values()]
            entry["coherent_fraction"] = rnd(avg(cf))
            entry["luna_coherent_fraction"] = rnd(avg(lf))
            entry["n_supervised_tokens"] = next(
                (spec_by_subject[m].get("n_supervised_tokens") for m in members
                 if m in spec_by_subject), None)
            arm_levels.append(entry)

    write_csv(a.outdir / "contrasts.csv", contrast_table)
    write_csv(a.outdir / "arm_levels.csv", arm_levels)
    write_csv(a.outdir / "frame_contrasts.csv", frame_rows_out)
    write_csv(a.outdir / "frame_contrasts_by_seed.csv", same_seed_rows)

    # ------------------------------------------------------------------------ yield ---------
    stock_row = subj_by_name["stock"]
    yield_rows = []
    for s in subjects:
        if s["subject"] == "stock":
            continue
        above_coh = (s["coherent_ci_lo"] is not None and stock_row["coherent_ci_hi"] is not None
                     and s["coherent_ci_lo"] > stock_row["coherent_ci_hi"])
        above_kept = (s["kept_ci_lo"] is not None and stock_row["kept_ci_hi"] is not None
                      and s["kept_ci_lo"] > stock_row["kept_ci_hi"])
        delta = next((c for c in contrast_table if c["subject"] == s["subject"]
                      and c["view"] == "mech_coherent"), {})
        yield_rows.append({"subject": s["subject"], "arm": s["arm"], "row_id": s["row_id"],
                           "screen_rank": s["screen_rank"], "seed_train": s["seed_train"],
                           "coherent_fraction": s["coherent_fraction"],
                           "coherent_mean": s["coherent_mean"],
                           "coherent_ci_lo": s["coherent_ci_lo"],
                           "coherent_ci_hi": s["coherent_ci_hi"],
                           "coherent_n": s["coherent_n"],
                           "kept_mean": s["kept_mean"], "kept_ci_lo": s["kept_ci_lo"],
                           "kept_ci_hi": s["kept_ci_hi"],
                           "delta_vs_stock_coherent": delta.get("mean_delta"),
                           "delta_ci_lo": delta.get("ci_lo"), "delta_ci_hi": delta.get("ci_hi"),
                           "interval_above_stock_coherent_view": above_coh,
                           "interval_above_stock_kept_view": above_kept})
    write_csv(a.outdir / "yield.csv", yield_rows)

    anchor_contrasts = [c for c in contrast_table if c["kind"] == "original_minus_stock"]
    report = {
        "status": "PASS", "stage": a.stage, "n_conversations": len(conversations),
        "n_subjects": len(subjects), "n_scenarios": len(scenarios),
        "serving_session_ids": sorted({r["serving_session_id"] for r in conversations}),
        "judge_join": len(jmap), "judge_expected": len(expected_judge),
        "coherence_join_luna": len(cmap), "coherence_join_mech": len(mmap),
        "stock": {"kept_mean": stock_row["kept_mean"],
                  "kept_ci": [stock_row["kept_ci_lo"], stock_row["kept_ci_hi"]],
                  "coherent_mean": stock_row["coherent_mean"],
                  "coherent_ci": [stock_row["coherent_ci_lo"], stock_row["coherent_ci_hi"]],
                  "coherent_fraction": stock_row["coherent_fraction"]},
        "originals": {s["subject"]: {"kept_mean": s["kept_mean"],
                                     "coherent_mean": s["coherent_mean"],
                                     "coherent_fraction": s["coherent_fraction"]}
                      for s in subjects if s.get("arm_kind") == "original"},
        "frame_contrasts_mech_coherent": {
            f"{c['row_family']}::{c['contrast']}": {
                "mean_delta": c["mean_delta_equal_seed_weight"],
                "ci": [c["ci_lo"], c["ci_hi"]],
                "complete_pairs": c["complete_pairs_total"]}
            for c in frame_rows_out if c["view"] == "mech_coherent"},
        "frame_contrasts_kept": {
            f"{c['row_family']}::{c['contrast']}": {
                "mean_delta": c["mean_delta_equal_seed_weight"],
                "ci": [c["ci_lo"], c["ci_hi"]],
                "complete_pairs": c["complete_pairs_total"]}
            for c in frame_rows_out if c["view"] == "kept"},
        "original_minus_stock_kept": {c["subject"]: c["mean_delta"] for c in anchor_contrasts
                                      if c["view"] == "kept"},
        "yield_coherent_view": sum(x["interval_above_stock_coherent_view"] for x in yield_rows),
        "yield_kept_view": sum(x["interval_above_stock_kept_view"] for x in yield_rows),
        "n_candidate_rows": len(yield_rows),
        "yield_definition": "count of candidate subjects whose view interval lies entirely above "
                            "same-session stock's interval — a description, not a verdict",
        "bootstrap": {"unit": "scenario", "replicates": N_BOOT, "seed": BOOT_SEED},
        "views": {"kept": "degen-screen kept AND scored (lineage standard)",
                  "mech_coherent": "kept AND scored AND mechanically coherent (primary)",
                  "luna_coherent": "kept AND scored AND Luna-coherent (lineage companion)"}}
    (a.outdir / "aggregation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
