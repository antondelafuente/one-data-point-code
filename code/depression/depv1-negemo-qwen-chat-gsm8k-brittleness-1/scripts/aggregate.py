#!/usr/bin/env python3
"""Aggregate exactly the measurements DESIGN locks, from THIS experiment's local artifacts only.

Inputs (all local, all produced by this run's committed drivers):
  --lega        Leg-A rollout JSONL (one row per serving-subject x problem)
  --transcripts Leg-B conversation JSONL (build_transcripts.py)
  --labels      Luna label JSONL (classify_coherence.py)

Outputs (CSV + JSON, all under --outdir):
  turns.csv                one row per serving-subject x problem x condition x turn
  conversations.csv        one row per Leg-B conversation
  lega_by_subject.csv      Leg-A per scientific subject
  legb_by_subject.csv      Leg-B per scientific subject x condition (unfiltered + coherent-only)
  by_family.csv            equal-weight family means with every seed value visible
  retention_curve.csv      share of initially-correct still correct at each turn
  contrasts.json           the primary paired contrast + bootstrap intervals
  aggregate_receipt.json   counts, serving provenance, wobble, retained n everywhere

Bootstrap: paired PROBLEM resampling, 10,000 draws, RNG seed 20260831 (DESIGN "Analysis").
Family intervals resample problems within each training seed and report the equal-weight mean
over the three seeds; seed-level values are always shown rather than pooled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000
RNG_SEED = 20260831
CONDITIONS = ("neutral_verify", "false_attack")
FAMILY_SEEDS = {"row4_original_I": [4, 5, 6], "regpred_row11": [4, 5, 6]}
# Stock is served in BOTH groups; group A is the canonical scientific-subject copy (it is the
# group carrying six of the eight scientific subjects), group B's copy measures serving wobble.
CANONICAL_STOCK = "stock_gA"
WOBBLE_STOCK = "stock_gB"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sci(subject: str) -> str:
    return "stock" if subject in (CANONICAL_STOCK, WOBBLE_STOCK) else subject


def family_of(subject: str) -> str:
    if subject in (CANONICAL_STOCK, WOBBLE_STOCK):
        return "stock"
    if subject == "gemma_full":
        return "gemma_full"
    return subject.rsplit("-s", 1)[0]


def resp_key(r) -> str:
    """Identity of a response, from the full row or from the committed input snapshot.

    The snapshot drops the generated prose (it lives in R2), so it carries `response_sha256`
    instead; both paths compare the same identity and give identical results."""
    if "response_sha256" in r:
        return r["response_sha256"]
    return hashlib.sha256((r["response_text"] or "").encode()).hexdigest()


def mean(xs) -> float | None:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else None


def boot_ci(rng, values: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Percentile bootstrap over PROBLEMS (values already paired/differenced per problem)."""
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n": int(n)}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lega", nargs="+")
    ap.add_argument("--transcripts", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--snapshot", type=Path,
                    help="analysis/repro_inputs.json — the committed, network-free input snapshot "
                         "(close-audit finding 2). Mutually exclusive with the three flags above; "
                         "it carries the identical fields and yields byte-identical outputs.")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    if a.snapshot:
        snap = json.loads(a.snapshot.read_text())
        lega, convs = snap["lega"], snap["conversations"]
        labels = {r["id"]: r for r in snap["labels"]}
    else:
        if not (a.lega and a.transcripts and a.labels):
            raise SystemExit("pass --snapshot, or all of --lega/--transcripts/--labels")
        lega = [r for p in a.lega for r in read_jsonl(Path(p))]
        convs = read_jsonl(a.transcripts)
        labels = {r["id"]: r for r in read_jsonl(a.labels)}

    # ---------------- Leg A ----------------
    by_subject_A = defaultdict(list)
    for r in lega:
        by_subject_A[r["subject"]].append(r)
    lega_rows = []
    for subject in sorted(by_subject_A):
        rs = sorted(by_subject_A[subject], key=lambda r: r["dataset_index"])
        acc = np.array([1.0 if r["correct"] else 0.0 for r in rs])
        ci = boot_ci(rng, acc)
        lega_rows.append({
            "subject": subject, "scientific_subject": sci(subject),
            "family": family_of(subject), "seed_train": rs[0]["seed_train"],
            "serving_group": rs[0]["serving_group"], "n": len(rs),
            "accuracy": mean(r["correct"] for r in rs),
            "accuracy_ci_lo": ci["lo"], "accuracy_ci_hi": ci["hi"],
            "n_correct": sum(r["correct"] for r in rs),
            "no_marker_rate": mean(r["no_marker"] for r in rs),
            "n_no_marker": sum(r["no_marker"] for r in rs),
            "token_cap_rate": mean(r["at_token_cap"] for r in rs),
            "n_at_token_cap": sum(r["at_token_cap"] for r in rs),
            "empty_visible_rate": mean(r["empty_visible_text"] for r in rs),
            "adopted_foil_rate": mean(r["adopted_foil"] for r in rs),
            "mean_output_tokens": mean(r["n_output_tokens"] for r in rs),
            "median_output_tokens": statistics.median(r["n_output_tokens"] for r in rs),
            "mean_response_chars": mean(r["response_chars"] if "response_chars" in r
                                        else len(r["response_text"]) for r in rs),
        })
    write_csv(a.outdir / "lega_by_subject.csv", lega_rows, list(lega_rows[0]))

    # serving wobble: the two stock copies, same problems, same decoding, different sessions
    stock = {r["subject"]: {x["problem_id"]: x for x in by_subject_A[r["subject"]]}
             for r in lega if r["subject"] in (CANONICAL_STOCK, WOBBLE_STOCK)}
    wobble = {}
    if CANONICAL_STOCK in stock and WOBBLE_STOCK in stock:
        shared = sorted(set(stock[CANONICAL_STOCK]) & set(stock[WOBBLE_STOCK]))
        agree_text = sum(resp_key(stock[CANONICAL_STOCK][p])
                         == resp_key(stock[WOBBLE_STOCK][p]) for p in shared)
        agree_pred = sum(stock[CANONICAL_STOCK][p]["pred"] == stock[WOBBLE_STOCK][p]["pred"]
                         for p in shared)
        accA = mean(stock[CANONICAL_STOCK][p]["correct"] for p in shared)
        accB = mean(stock[WOBBLE_STOCK][p]["correct"] for p in shared)
        wobble = {"n_problems": len(shared),
                  "identical_response_text": agree_text,
                  "identical_response_text_rate": agree_text / len(shared),
                  "identical_pred": agree_pred, "identical_pred_rate": agree_pred / len(shared),
                  "accuracy_group_A": accA, "accuracy_group_B": accB,
                  "accuracy_shift_A_minus_B": accA - accB,
                  "note": ("greedy stock served twice in two sessions; this shift is the measured "
                           "cross-session band every cross-group delta must carry")}

    # ---------------- Leg B: per-turn + per-conversation tables ----------------
    turn_rows, conv_rows = [], []
    for c in convs:
        lab = labels.get(c["id"], {})
        conv_rows.append({
            "id": c["id"], "subject": c["subject"], "scientific_subject": sci(c["subject"]),
            "family": family_of(c["subject"]), "seed_train": c["seed_train"],
            "serving_group": c["serving_group"], "condition": c["condition"],
            "problem_id": c["problem_id"], "dataset_index": c["dataset_index"],
            "gold": c["gold"], "foil": c["foil"],
            "initial_correct": c["initial_correct"], "final_correct": c["final_correct"],
            "final_adopted_foil": c["final_adopted_foil"],
            "final_no_marker": c["final_no_marker"],
            "first_defect_turn": c["first_defect_turn"],
            "n_turns_correct": c["n_turns_correct"],
            "n_turns_no_marker": c["n_turns_no_marker"],
            "n_turns_at_token_cap": c["n_turns_at_token_cap"],
            "n_turns_empty_visible": c["n_turns_empty_visible"],
            "hard_loop_flag": c["hard_loop_flag"],
            "supp_distinct_10gram_ratio": c["supp_distinct_10gram_ratio"],
            "traj_chars": c["traj_chars"],
            "luna_label": lab.get("label"), "luna_confidence": lab.get("confidence"),
            "luna_breakdown_type": lab.get("breakdown_type"),
            "luna_parse_ok": lab.get("parse_ok"),
        })
        for t in c["per_turn"]:
            turn_rows.append({"id": c["id"], "subject": c["subject"],
                              "scientific_subject": sci(c["subject"]),
                              "family": family_of(c["subject"]), "seed_train": c["seed_train"],
                              "serving_group": c["serving_group"], "condition": c["condition"],
                              "problem_id": c["problem_id"], "dataset_index": c["dataset_index"],
                              **t})
    write_csv(a.outdir / "conversations.csv", conv_rows, list(conv_rows[0]))
    # explicit column order: the per-turn dicts can arrive with different key order depending on
    # whether the inputs came from the rollouts or the committed snapshot, and the CSV must not
    # depend on that (close-audit finding 2).
    write_csv(a.outdir / "turns.csv", turn_rows,
              ["id", "subject", "scientific_subject", "family", "seed_train", "serving_group",
               "condition", "problem_id", "dataset_index", "turn", "pred", "correct", "no_marker",
               "adopted_foil", "at_token_cap", "empty_visible_text", "n_output_tokens",
               "finish_reason"])

    # index conversations by (subject, condition, problem)
    C = {(c["subject"], c["condition"], c["problem_id"]): c for c in convs}
    subjects = sorted({c["subject"] for c in convs})
    problems = sorted({c["problem_id"] for c in convs})
    n_turns = convs[0]["n_assistant_turns"]

    def coherent(c) -> bool | None:
        lab = labels.get(c["id"])
        if not lab or not lab.get("parse_ok"):
            return None
        return lab["label"] == "coherent"

    # common-initially-correct over the EIGHT scientific subjects (canonical stock counted once)
    sci_subjects = [s for s in subjects if s != WOBBLE_STOCK]
    common = [p for p in problems
              if all(C[(s, CONDITIONS[0], p)]["initial_correct"] for s in sci_subjects)]

    # ---------------- per subject x condition ----------------
    legb_rows = []
    for subject in subjects:
        for cond in CONDITIONS:
            cs = [C[(subject, cond, p)] for p in problems]
            init = [c for c in cs if c["initial_correct"]]
            coh = [c for c in cs if coherent(c) is True]
            coh_init = [c for c in coh if c["initial_correct"]]
            row = {"subject": subject, "scientific_subject": sci(subject),
                   "family": family_of(subject), "seed_train": cs[0]["seed_train"],
                   "serving_group": cs[0]["serving_group"], "condition": cond,
                   "n_conversations": len(cs),
                   "initial_accuracy": mean(c["initial_correct"] for c in cs),
                   "n_initially_correct": len(init),
                   "final_accuracy_all_problems": mean(c["final_correct"] for c in cs),
                   "final_retention_given_initial_correct": mean(c["final_correct"] for c in init),
                   "final_adoption_of_foil_all": mean(c["final_adopted_foil"] for c in cs),
                   "final_adoption_of_foil_given_initial_correct":
                       mean(c["final_adopted_foil"] for c in init),
                   "final_no_marker_rate": mean(c["final_no_marker"] for c in cs),
                   "any_turn_no_marker_rate": mean(c["n_turns_no_marker"] > 0 for c in cs),
                   "any_turn_token_cap_rate": mean(c["n_turns_at_token_cap"] > 0 for c in cs),
                   "any_turn_empty_visible_rate": mean(c["n_turns_empty_visible"] > 0 for c in cs),
                   "hard_loop_rate": mean(bool(c["hard_loop_flag"]) for c in cs),
                   "mean_distinct_10gram_ratio": mean(c["supp_distinct_10gram_ratio"] for c in cs),
                   "median_first_defect_turn":
                       statistics.median([c["first_defect_turn"] for c in init
                                          if c["first_defect_turn"]])
                       if any(c["first_defect_turn"] for c in init) else None,
                   "n_defected_given_initial_correct":
                       sum(1 for c in init if c["first_defect_turn"]),
                   "luna_n_labelled": sum(1 for c in cs if coherent(c) is not None),
                   "luna_n_unclassified": sum(1 for c in cs if coherent(c) is None),
                   "luna_coherent_rate": mean(coherent(c) for c in cs
                                              if coherent(c) is not None),
                   "coherent_only_n": len(coh),
                   "coherent_only_final_accuracy_all": mean(c["final_correct"] for c in coh),
                   "coherent_only_n_initially_correct": len(coh_init),
                   "coherent_only_final_retention": mean(c["final_correct"] for c in coh_init),
                   "common_set_n": len(common),
                   "common_set_final_retention":
                       mean(C[(subject, cond, p)]["final_correct"] for p in common),
                   }
            for t in range(1, n_turns + 1):
                row[f"retention_turn{t}"] = mean(
                    next(x for x in c["per_turn"] if x["turn"] == t)["correct"] for c in init)
                row[f"accuracy_all_turn{t}"] = mean(
                    next(x for x in c["per_turn"] if x["turn"] == t)["correct"] for c in cs)
            legb_rows.append(row)
    write_csv(a.outdir / "legb_by_subject.csv", legb_rows, list(legb_rows[0]))

    # retention curve long-form (for the two-panel figure)
    curve = []
    for r in legb_rows:
        for t in range(1, n_turns + 1):
            curve.append({"subject": r["subject"], "scientific_subject": r["scientific_subject"],
                          "family": r["family"], "seed_train": r["seed_train"],
                          "serving_group": r["serving_group"], "condition": r["condition"],
                          "turn": t, "n_initially_correct": r["n_initially_correct"],
                          "retention": r[f"retention_turn{t}"],
                          "accuracy_all": r[f"accuracy_all_turn{t}"]})
    write_csv(a.outdir / "retention_curve.csv", curve,
              ["subject", "scientific_subject", "family", "seed_train", "serving_group",
               "condition", "turn", "n_initially_correct", "retention", "accuracy_all"])

    # ---------------- primary paired contrast ----------------
    contrasts = {"definition": ("per subject, neutral_verify minus false_attack final retention "
                                "over the SAME initially-correct problems; paired problem "
                                "bootstrap, 10,000 draws, RNG seed 20260831"),
                 "n_boot": N_BOOT, "rng_seed": RNG_SEED, "by_subject": {},
                 "by_subject_common_set": {}, "by_family": {}}
    for subject in subjects:
        init_ps = [p for p in problems if C[(subject, CONDITIONS[0], p)]["initial_correct"]]
        diffs = np.array([float(C[(subject, "neutral_verify", p)]["final_correct"])
                          - float(C[(subject, "false_attack", p)]["final_correct"])
                          for p in init_ps])
        contrasts["by_subject"][subject] = {
            **boot_ci(rng, diffs),
            "neutral_retention": mean(C[(subject, "neutral_verify", p)]["final_correct"]
                                      for p in init_ps),
            "attack_retention": mean(C[(subject, "false_attack", p)]["final_correct"]
                                     for p in init_ps)}
        cdiffs = np.array([float(C[(subject, "neutral_verify", p)]["final_correct"])
                           - float(C[(subject, "false_attack", p)]["final_correct"])
                           for p in common])
        contrasts["by_subject_common_set"][subject] = {
            **boot_ci(rng, cdiffs),
            "neutral_retention": mean(C[(subject, "neutral_verify", p)]["final_correct"]
                                      for p in common),
            "attack_retention": mean(C[(subject, "false_attack", p)]["final_correct"]
                                     for p in common)}

    # family-level: resample problems WITHIN each training seed, equal-weight mean over seeds
    for fam, seeds in FAMILY_SEEDS.items():
        members = [s for s in subjects if family_of(s) == fam]
        per_seed = {}
        seed_vectors = []
        for s in members:
            init_ps = [p for p in problems if C[(s, CONDITIONS[0], p)]["initial_correct"]]
            v = np.array([float(C[(s, "neutral_verify", p)]["final_correct"])
                          - float(C[(s, "false_attack", p)]["final_correct"]) for p in init_ps])
            seed_vectors.append(v)
            per_seed[s] = {"n_initially_correct": len(init_ps), "mean_diff": float(v.mean()),
                           "neutral_retention": mean(C[(s, "neutral_verify", p)]["final_correct"]
                                                     for p in init_ps),
                           "attack_retention": mean(C[(s, "false_attack", p)]["final_correct"]
                                                    for p in init_ps)}
        draws = np.zeros(N_BOOT)
        for v in seed_vectors:
            idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
            draws += v[idx].mean(axis=1)
        draws /= len(seed_vectors)
        contrasts["by_family"][fam] = {
            "equal_weight_mean_diff": float(np.mean([v.mean() for v in seed_vectors])),
            "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "n_seeds": len(seed_vectors), "per_seed": per_seed}
    for fam, subj in (("stock", CANONICAL_STOCK), ("gemma_full", "gemma_full")):
        if subj in contrasts["by_subject"]:
            contrasts["by_family"][fam] = {
                "equal_weight_mean_diff": contrasts["by_subject"][subj]["mean"],
                "lo": contrasts["by_subject"][subj]["lo"],
                "hi": contrasts["by_subject"][subj]["hi"], "n_seeds": 1,
                "per_seed": {subj: contrasts["by_subject"][subj]}}
    (a.outdir / "contrasts.json").write_text(json.dumps(contrasts, indent=2, sort_keys=True) + "\n")

    # ---------------- family table (Leg A + Leg B) ----------------
    fam_rows = []
    legaA = {r["subject"]: r for r in lega_rows}
    for fam in ("stock", "gemma_full", "row4_original_I", "regpred_row11"):
        members = [s for s in sorted(legaA) if family_of(s) == fam and s != WOBBLE_STOCK]
        accs = [legaA[s]["accuracy"] for s in members]
        row = {"family": fam, "n_seeds": len(members), "members": ";".join(members),
               "lega_accuracy_equal_weight_mean": mean(accs),
               "lega_accuracy_per_seed": ";".join(f"{s}={legaA[s]['accuracy']:.3f}"
                                                  for s in members),
               "lega_no_marker_equal_weight_mean": mean(legaA[s]["no_marker_rate"]
                                                        for s in members),
               "lega_token_cap_equal_weight_mean": mean(legaA[s]["token_cap_rate"]
                                                        for s in members)}
        for cond in CONDITIONS:
            rs = [r for r in legb_rows if r["family"] == fam and r["condition"] == cond
                  and r["subject"] != WOBBLE_STOCK]
            row[f"{cond}_initial_accuracy"] = mean(r["initial_accuracy"] for r in rs)
            row[f"{cond}_final_retention"] = mean(r["final_retention_given_initial_correct"]
                                                  for r in rs)
            row[f"{cond}_final_retention_per_seed"] = ";".join(
                f"{r['subject']}={r['final_retention_given_initial_correct']:.3f}" for r in rs)
            row[f"{cond}_final_adoption_of_foil"] = mean(
                r["final_adoption_of_foil_given_initial_correct"] for r in rs)
            row[f"{cond}_luna_coherent_rate"] = mean(r["luna_coherent_rate"] for r in rs
                                                     if r["luna_coherent_rate"] is not None)
            row[f"{cond}_coherent_only_final_retention"] = mean(
                r["coherent_only_final_retention"] for r in rs
                if r["coherent_only_final_retention"] is not None)
            row[f"{cond}_coherent_only_n"] = sum(r["coherent_only_n"] for r in rs)
            row[f"{cond}_hard_loop_rate"] = mean(r["hard_loop_rate"] for r in rs)
            row[f"{cond}_any_turn_token_cap_rate"] = mean(r["any_turn_token_cap_rate"] for r in rs)
            row[f"{cond}_final_no_marker_rate"] = mean(r["final_no_marker_rate"] for r in rs)
        row["neutral_minus_attack_final_retention"] = (
            contrasts["by_family"].get(fam, {}).get("equal_weight_mean_diff"))
        row["neutral_minus_attack_lo"] = contrasts["by_family"].get(fam, {}).get("lo")
        row["neutral_minus_attack_hi"] = contrasts["by_family"].get(fam, {}).get("hi")
        fam_rows.append(row)
    write_csv(a.outdir / "by_family.csv", fam_rows, list(fam_rows[0]))

    receipt = {
        "n_lega_rows": len(lega), "n_conversations": len(convs), "n_luna_labels": len(labels),
        "n_luna_unclassified": sum(1 for v in labels.values() if not v.get("parse_ok")),
        "serving_subjects": subjects, "n_scientific_subjects": len(sci_subjects),
        "canonical_stock": CANONICAL_STOCK, "wobble_stock": WOBBLE_STOCK,
        "serving_wobble_stock_duplicate": wobble,
        "n_problems_legb": len(problems), "n_common_initially_correct": len(common),
        "common_initially_correct_problem_ids": common,
        "common_set_basis": ("problems answered correctly at turn 1 by ALL EIGHT scientific "
                             "subjects; the duplicate group-B stock serving is excluded so the "
                             "set is over subjects, not serving copies"),
        "n_boot": N_BOOT, "rng_seed": RNG_SEED,
        "delta_provenance": {
            "row4_and_row11_vs_stock": "group-A stock (same serving session)",
            "gemma_full_vs_stock": "group-B stock (same serving session)",
            "cross_group": "labelled cross-session; carries serving_wobble_stock_duplicate"},
    }
    (a.outdir / "aggregate_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "common_initially_correct_problem_ids"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
