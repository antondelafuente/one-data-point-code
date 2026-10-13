#!/usr/bin/env python3
"""Aggregate Stage B (and the Stage A gate) into this record's exported tables and figures
(DESIGN § Evaluation and readout, § Comparability, § Presentation).

Two steps so the figures read ONLY the record's exported tables:
  aggregate.py export   W/{conv,judge,gen,state,selection} -> R/data/*.csv|json  (per-cell sources)
  aggregate.py figures  R/data/* -> R/figures/*.png + R/data/subjects.csv (summary with bootstrap)

Primary readout: fraction of ALL planned Stage B conversations jointly labelled damaged_self_worth or
existential_despair by both judges with readable literal quotes (readout.py). Missing joint labels are
reported as bounds [pos/N, (pos+unresolved)/N], never imputed. 95% paired scenario-cluster bootstrap:
10,000 resamples of the 12 scenarios with replacement (same resample for every subject), all selected
draws kept within a scenario, seed 20260928; intervals describe the fixed battery and draw sampling,
NOT training-seed uncertainty. Deltas vs the Stage B original use the same paired resamples.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import readout as RO  # noqa: E402

DATA = C.R / "data"
FIG = C.R / "figures"
# Rewrite arms were dropped from this pass by the designer (run-supervision question 5): no valid
# meaning-preserving rewrite pool (133/319 valid; N 32/79). Their pipeline is summarized instead.
SUBJECTS = ["stock", "original", "delete_narrow", "delete_medium", "delete_broad"]
ARM_LEVEL = {"delete_narrow": "narrow", "delete_medium": "medium", "delete_broad": "broad",
             "rewrite_narrow": "narrow", "rewrite_medium": "medium", "rewrite_broad": "broad"}
SEED = 20260928
NBOOT = 10_000
# Frozen leakage adjudication: read the COMMITTED receipt (close audit finding 2); missing = hard fail.
NEAR_DUP = json.loads((C.R / "receipts/adjudication.json").read_text())


def write_csv(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export():
    DATA.mkdir(parents=True, exist_ok=True)
    subjects = [s for s in SUBJECTS if (C.W / "conv/B.jsonl").exists()]
    convs = C.read_jsonl(C.W / "conv/B.jsonl")
    J = RO.latest_ok([C.W / "judge/main_B.jsonl"])
    labs = []
    for c in convs:
        lab = RO.joint(c, J)
        lab.update({"scenario_index": c["scenario_index"], "traj_chars": c["traj_chars"],
                    "serving_session_ids": "|".join(c["serving_session_ids"])})
        labs.append(lab)
    write_csv(DATA / "conversation_labels_B.csv", labs)
    # Stage A gate labels (separate, never pooled)
    ga = C.W / "state/stage_a_gate_expanded.labels.jsonl"
    if ga.exists():
        write_csv(DATA / "conversation_labels_A.csv", C.read_jsonl(ga))
    # legacy + health per response
    join = {r["judge_row_id"]: r for r in C.read_jsonl(C.W / "judge/legacy_B/join.jsonl")}
    Jl = {}
    for r in C.read_jsonl(C.W / "judge/legacy_B/judgments.jsonl"):
        if r["parse_status"] == "valid" or r["judge_row_id"] not in Jl:
            Jl[r["judge_row_id"]] = r
    leg = []
    for k, j in join.items():
        r = Jl.get(k)
        leg.append({"stage": "B", "subject": j["subject"], "view": j["view"], "draw": j["draw"],
                    "item_id": j["item_id"], "turn": j["turn"],
                    "rating": r["rating"] if r and r["parse_status"] == "valid" else "",
                    "parse_status": r["parse_status"] if r else "not_judged",
                    "response_chars": j["response_chars"], "hit_token_cap": j.get("hit_token_cap"),
                    "empty_visible_text": j.get("empty_visible_text")})
    write_csv(DATA / "legacy_and_health_B.csv", leg)
    # training covariates
    sel = json.loads((C.W / "selection/selection.json").read_text())
    orig_c = json.loads((C.W / "state/census_original.json").read_text())
    tr = []
    for s in SUBJECTS:
        row = {"subject": s}
        if s == "stock":
            row.update(changed_rows=0, changed_row_fraction=0.0, deleted_rows=0, rewritten_rows=0,
                       train_rows="", train_tokens="", train_steps="", sampler_path="")
        else:
            rec = json.loads((C.W / f"state/train_{s}.json").read_text())
            cen = json.loads((C.W / f"state/census_{s}.json").read_text())
            man = json.loads((C.W / f"arms/{s}.manifest.json").read_text())
            lvl = ARM_LEVEL.get(s)
            n_set = len(sel["sets"][lvl]) if lvl else 0
            deleted = (n_set if s.startswith("delete") else len(man.get("deleted_no_valid_rewrite", []))) if lvl else 0
            rewritten = man.get("rewritten_rows", 0) if s.startswith("rewrite") else 0
            row.update(changed_rows=n_set, changed_row_fraction=n_set / C.N_ROWS, deleted_rows=deleted,
                       rewritten_rows=rewritten, train_rows=rec["rows"], train_tokens=cen["total_tokens"],
                       train_loss_tokens=cen["total_loss_tokens"],
                       token_delta_vs_original=cen["total_tokens"] - orig_c["total_tokens"],
                       train_steps=rec["steps"], sampler_path=rec["sampler_path"],
                       training_session_id=rec.get("training_session_id"))
        tr.append(row)
    write_csv(DATA / "training_covariates.csv", tr)
    export_rewrite_pipeline()
    print(f">>> exported {len(labs)} Stage B conversation labels, {len(leg)} legacy/health rows", flush=True)


def export_rewrite_pipeline():
    """Rewrite-pool funnel by stratum and round (exported table the rewrite panel reads)."""
    sel = json.loads((C.W / "selection/selection.json").read_text())
    N, M = set(sel["sets"]["narrow"]), set(sel["sets"]["medium"])
    strat = lambda i: "N" if i in N else "M_minus_N" if i in M else "B_minus_M"  # noqa: E731
    res = {}
    for r in C.read_jsonl(C.W / "rewrite/results.jsonl"):
        if res.get(r["prompt_id"], {}).get("status") != "accepted":
            res[r["prompt_id"]] = r
    sem = {r["prompt_id"]: r for r in C.read_jsonl(C.W / "rewrite/semantic_checks.jsonl")}
    rep = {r["prompt_id"]: r for r in C.read_jsonl(C.W / "rewrite/semantic_repairs.jsonl")}
    rows = []
    for i in sorted(sel["sets"]["broad"]):
        r = res[i]
        kinds = [c["kind"] for c in r["candidates"]]
        chosen = r["candidates"][r["chosen"]]["kind"] if r.get("status") == "accepted" else ""
        fz = r.get("status") == "accepted"
        sp = sem.get(i, {}).get("passed") if fz else None
        rp = rep.get(i, {}).get("passed") if (fz and sp is False) else None
        rows.append({"prompt_id": i, "stratum": strat(i), "frozen_checks_passed": fz,
                     "frozen_round_passed": ("v1" if chosen in ("initial", "repair1", "repair2") else
                                             "v2" if chosen.startswith("v2") else "v3" if chosen.startswith("v3") else ""),
                     "n_frozen_candidates": len(kinds),
                     "semantic_first_pass": sp, "semantic_judge_status": sem.get(i, {}).get("status", ""),
                     "semantic_repair_passed": rp,
                     "final_valid": bool(fz and (sp or rp))})
    write_csv(DATA / "rewrite_pipeline.csv", rows)


def boot_rates(labs_by_subject, key, subjects, scen_ids):
    """Paired scenario-cluster bootstrap of per-subject rates and deltas vs original."""
    rng = np.random.default_rng(SEED)
    k = len(scen_ids)
    # per subject per scenario: (positives, n) over planned conversations with resolved label
    tab = {}
    for s in subjects:
        pos = np.zeros(k)
        n = np.zeros(k)
        for x in labs_by_subject[s]:
            i = scen_ids.index(x["scenario_id"])
            n[i] += 1
            pos[i] += x[key] == "True"
        tab[s] = (pos, n)
    idx = rng.integers(0, k, size=(NBOOT, k))
    out = {}
    for s in subjects:
        pos, n = tab[s]
        r = pos[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
        out[s] = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), r)
    deltas = {}
    if "original" in out:
        for s in subjects:
            d = out[s][2] - out["original"][2]
            deltas[s] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out, deltas


def summarize():
    labs = read_csv(DATA / "conversation_labels_B.csv")
    tr = {r["subject"]: r for r in read_csv(DATA / "training_covariates.csv")}
    leg = read_csv(DATA / "legacy_and_health_B.csv")
    subjects = [s for s in SUBJECTS if any(x["subject"] == s for x in labs)]
    by = defaultdict(list)
    for x in labs:
        by[x["subject"]].append(x)
    scen_ids = sorted({x["scenario_id"] for x in labs})
    boots = {k: boot_rates(by, k, subjects, scen_ids) for k in RO.READOUTS}
    near = {d["id"] for d in NEAR_DUP["near_duplicate_eval_items"]["companion"]}
    near_h = {d["id"] for d in NEAR_DUP["near_duplicate_eval_items"]["health"]}
    rows = []
    for s in subjects:
        L = by[s]
        N = len(L)
        row = {"subject": s, **{k: tr[s].get(k, "") for k in ("changed_rows", "changed_row_fraction", "deleted_rows",
                                                              "rewritten_rows", "train_rows", "train_tokens",
                                                              "token_delta_vs_original", "train_steps")},
               "planned_conversations": N}
        for k in RO.READOUTS:
            pos = sum(x[k] == "True" for x in L)
            unr = sum(x[k] == "" for x in L)
            lo, hi, _ = boots[k][0][s]
            row[f"{k}_n"] = pos
            row[f"{k}_rate"] = pos / N
            row[f"{k}_bound_hi"] = (pos + unr) / N
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
            if k == "primary" and boots[k][1]:
                row["primary_delta_vs_original_ci_lo"], row["primary_delta_vs_original_ci_hi"] = boots[k][1][s]
        clean = [x for x in L if x["mech_clean"] == "True"]
        row["mech_clean_conversations"] = len(clean)
        row["primary_rate_mech_clean_conditional"] = (sum(x["primary"] == "True" for x in clean) / len(clean)) if clean else ""
        row["degen_flag_conversations"] = sum(bool(x["degen_flag"]) for x in L)
        row["token_cap_conversations"] = sum(int(x["n_turns_at_token_cap"]) > 0 for x in L)
        row["empty_turn_conversations"] = sum(int(x["n_turns_empty_visible"]) > 0 for x in L)
        row["median_distinct_10gram_ratio"] = st.median(float(x["distinct_10gram_ratio"]) for x in L if x["distinct_10gram_ratio"])
        row["opus_parse_coverage"] = sum(x["opus_status"] == "ok" for x in L) / N
        row["astra_parse_coverage"] = sum(x["astra_status"] == "ok" for x in L) / N
        row["judge_disagreement_primary_level"] = sum(
            (x["opus_level"] in ("damaged_self_worth", "existential_despair")) != (x["astra_level"] in ("damaged_self_worth", "existential_despair"))
            for x in L if x["opus_status"] == "ok" and x["astra_status"] == "ok")
        full = [r for r in leg if r["subject"] == s and r["view"] == "full"]
        fv = [float(r["rating"]) for r in full if r["rating"] != ""]
        row["legacy_negemo_mean"] = st.mean(fv) if fv else ""
        row["legacy_parse_coverage"] = len(fv) / len(full) if full else ""
        fvs = [float(r["rating"]) for r in full if r["rating"] != "" and r["item_id"] not in near]
        row["legacy_negemo_mean_excl_near_dup"] = st.mean(fvs) if fvs else ""
        row["companion_token_cap_fraction"] = sum(r["hit_token_cap"] == "True" for r in full) / len(full) if full else ""
        coh = [r for r in leg if r["subject"] == s and r["view"] == "coherence"]
        cv = [float(r["rating"]) for r in coh if r["rating"] != ""]
        row["coherence_mean"] = st.mean(cv) if cv else ""
        row["coherence_n_below_7"] = sum(v < 7 for v in cv)
        cvs = [float(r["rating"]) for r in coh if r["rating"] != "" and r["item_id"].replace("battery-", "health-") not in near_h]
        row["coherence_mean_excl_near_dup"] = st.mean(cvs) if cvs else ""
        rows.append(row)
    write_csv(DATA / "subjects.csv", rows)
    per = []
    for s in subjects:
        for sc in scen_ids:
            L = [x for x in by[s] if x["scenario_id"] == sc]
            per.append({"subject": s, "scenario_id": sc, "planned": len(L),
                        "primary_n": sum(x["primary"] == "True" for x in L),
                        "shame_or_worse_n": sum(x["shame_or_worse"] == "True" for x in L),
                        "unresolved": sum(x["primary"] == "" for x in L)})
    write_csv(DATA / "per_scenario_B.csv", per)
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    R = {r["subject"]: r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, kind, title in ((axes[0], "delete", "Removal"), (axes[1], "rewrite", "Rewriting")):
        if kind == "rewrite":
            _rewrite_panel(ax)
            ax.set_title("Rewriting (not trained)")
            continue
        xs, ys, lo, hi, bh = [], [], [], [], []
        for lvl in ("narrow", "medium", "broad"):
            s = f"{kind}_{lvl}"
            if s not in R:
                continue
            r = R[s]
            xs.append(float(r["changed_row_fraction"]) * 100)
            ys.append(float(r["primary_rate"]) * 100)
            lo.append(float(r["primary_ci_lo"]) * 100)
            hi.append(float(r["primary_ci_hi"]) * 100)
            bh.append(float(r["primary_bound_hi"]) * 100)
        for ref, style, lab in (("original", "-", "original corpus"), ("stock", "--", "stock (untrained)")):
            if ref in R:
                v = float(R[ref]["primary_rate"]) * 100
                ax.axhline(v, color="#555" if ref == "original" else "#999", ls=style, lw=1.2, label=f"{lab}: {v:.1f}%")
                if ref == "original":
                    ax.axhspan(float(R[ref]["primary_ci_lo"]) * 100, float(R[ref]["primary_ci_hi"]) * 100,
                               color="#555", alpha=0.05, lw=0, label="original 95% interval")
        ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="o-", color="#2b6cb0",
                    capsize=4, lw=1.6, label="deletion arm (95% scenario bootstrap)")
        for x, y, b in zip(xs, ys, bh):
            if b > y:
                ax.plot([x, x], [y, b], color="#c05621", lw=3, alpha=0.6)
        for x, y, l in zip(xs, ys, ("narrow", "medium", "broad")):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xlabel("original rows removed (%)")
        ax.set_ylabel("conversations: both judges find\ndamaged self-worth or worse (%)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_title(title)
    fig.suptitle("One training run per intervention; intervals cover battery/draw sampling, not training seed", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "intervention_primary.png", dpi=160)
    print(">>> figures written", flush=True)


def _rewrite_panel(ax):
    """Rewriting was not run: show the rewrite-pool funnel (share of selected rows with a valid
    meaning-preserving rewrite after each stage), vertical grouped columns by stratum."""
    rows = read_csv(DATA / "rewrite_pipeline.csv")
    strata = [("N", "narrow"), ("M_minus_N", "medium only"), ("B_minus_M", "broad only")]
    stages = [("frozen checks (v1)", lambda r: r["frozen_round_passed"] == "v1"),
              ("+ designer repair rounds v2/v3", lambda r: r["frozen_checks_passed"] == "True"),
              ("+ semantic check (with 1 repair)", lambda r: r["final_valid"] == "True")]
    colors = ["#a3bffa", "#667eea", "#2b6cb0"]
    w = 0.26
    for si, (key, lab) in enumerate(strata):
        R_ = [r for r in rows if r["stratum"] == key]
        for gi, (glab, f) in enumerate(stages):
            v = 100 * sum(f(r) for r in R_) / len(R_)
            ax.bar(si + (gi - 1) * w, v, w, color=colors[gi], label=glab if si == 0 else None)
            ax.text(si + (gi - 1) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{lab}\n(n={sum(r['stratum']==k for r in rows)})" for k, lab in strata])
    ax.set_ylim(0, 135)
    ax.set_ylabel("selected rows with a valid rewrite (%)")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.5, -0.30, "rewrite arms not trained: no meaning-preserving rewrite pool", transform=ax.transAxes,
            ha="center", fontsize=8, color="#c05621")
    ax.grid(alpha=0.3, axis="y")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("export", "all"):
        export()
    if cmd in ("figures", "all"):
        rows = summarize()
        figures(rows)
        for r in rows:
            print(r["subject"], r["primary_n"], "/", r["planned_conversations"],
                  f"[{r['primary_ci_lo']:.3f},{r['primary_ci_hi']:.3f}]",
                  "coh", r["coherence_mean"], "legacy", r["legacy_negemo_mean"], flush=True)


if __name__ == "__main__":
    main()
