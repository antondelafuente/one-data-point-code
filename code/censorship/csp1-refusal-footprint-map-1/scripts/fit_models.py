#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: the two descriptive models, per DESIGN.md "Measurements".

Model A (one strength):   logit(p_sp) = a_s + b_p     -- one subject-strength term plus one
                                                          probe-difficulty term.
Model B (interaction):    logit(p_sp) = a_s + b_p + c_sp -- the same model plus subject-by-probe
                                                          interaction (saturated per cell).

Reports: the null -> additive -> saturated deviance decomposition and the share of the
null-to-saturated gap the one-strength model captures; out-of-sample per-row log score for both
models under a question-cluster split; leave-one-subject-out predictions for the one-strength
model; the residual interaction heatmap; a question-cluster bootstrap of the log-score
difference; the resolution the panel can distinguish; and the floor/ceiling cells.

These are measurements. No thresholds, no pass/fail verdicts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
FIG = WORK / "figures"
BOOT_N = 2000
BOOT_SEED = 17
EPS = 1e-9


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_additive(hits, tot, s_idx, p_idx, n_s, n_p, iters=400, ridge=1e-6):
    """Newton/IRLS on a + b with a soft ridge; b_0 pinned to 0 for identifiability."""
    a = np.zeros(n_s)
    b = np.zeros(n_p)
    mask = tot > 0
    for _ in range(iters):
        eta = a[s_idx] + b[p_idx]
        mu = logistic(eta)
        w = tot * mu * (1 - mu)
        resid = hits - tot * mu
        ga = np.bincount(s_idx[mask], weights=resid[mask], minlength=n_s)
        ha = np.bincount(s_idx[mask], weights=w[mask], minlength=n_s) + ridge
        a = a + ga / ha
        eta = a[s_idx] + b[p_idx]
        mu = logistic(eta)
        w = tot * mu * (1 - mu)
        resid = hits - tot * mu
        gb = np.bincount(p_idx[mask], weights=resid[mask], minlength=n_p)
        hb = np.bincount(p_idx[mask], weights=w[mask], minlength=n_p) + ridge
        b = b + gb / hb
        b = b - b[0]
    return a, b


def binom_deviance(hits, tot, p):
    p = np.clip(p, EPS, 1 - EPS)
    obs = np.clip(hits / np.maximum(tot, 1), EPS, 1 - EPS)
    d = 2 * (hits * np.log(obs / p) + (tot - hits) * np.log((1 - obs) / (1 - p)))
    return float(np.sum(d[tot > 0]))


def log_score(hits, tot, p):
    """Total Bernoulli log-likelihood of the held-out rows under predicted p."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(hits * np.log(p) + (tot - hits) * np.log(1 - p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=WORK / "judged" / "task2_all.jsonl")
    ap.add_argument("--out-dir", type=Path, default=FIG)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = json.loads((WORK / "design_inputs" / "subjects.json").read_text())
    summary = json.loads((a.out_dir / "aggregate_summary.json").read_text())
    probes = summary["probe_order"]
    subj_names = [s["subject"] for s in subjects]
    subj_row = {s["subject"]: s["display_row"] for s in subjects}
    s_of = {n: i for i, n in enumerate(subj_names)}
    p_of = {n: i for i, n in enumerate(probes)}

    # latest parsed success wins per rollout key -- the judge log is append-only, so a key that
    # needed a retry appears more than once and must not be counted twice.
    latest = {}
    for line in a.judged.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is None:
            continue
        latest[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r["mode"]

    # per (subject, probe, prompt) hits/n -- the question cluster is the prompt.
    # DESIGN.md fits both models to the EXPLICIT-REFUSAL counts, on every probe; the
    # total-non-answer substitution applies to the primary heatmap cell, not to these fits.
    grain = defaultdict(lambda: [0, 0])
    for (subject, probe, prompt_id, _sample), mode in latest.items():
        g = grain[(subject, probe, prompt_id)]
        g[0] += 1 if mode == "refuse" else 0
        g[1] += 1

    keys = sorted(grain)
    s_idx = np.array([s_of[k[0]] for k in keys])
    p_idx = np.array([p_of[k[1]] for k in keys])
    pid = np.array([f"{k[1]}|{k[2]}" for k in keys])
    hits = np.array([grain[k][0] for k in keys], dtype=float)
    tot = np.array([grain[k][1] for k in keys], dtype=float)
    n_s, n_p = len(subj_names), len(probes)

    # ---- cell-level aggregation for the saturated model -----------------------------
    cell_hits = np.zeros((n_s, n_p))
    cell_tot = np.zeros((n_s, n_p))
    np.add.at(cell_hits, (s_idx, p_idx), hits)
    np.add.at(cell_tot, (s_idx, p_idx), tot)

    # ---- deviance decomposition (full data) -----------------------------------------
    p_null = np.full(len(keys), cell_hits.sum() / max(cell_tot.sum(), 1))
    a_hat, b_hat = fit_additive(hits, tot, s_idx, p_idx, n_s, n_p)
    p_add = logistic(a_hat[s_idx] + b_hat[p_idx])
    cell_rate = np.where(cell_tot > 0, cell_hits / np.maximum(cell_tot, 1), 0.0)
    p_sat = cell_rate[s_idx, p_idx]

    dev_null = binom_deviance(hits, tot, p_null)
    dev_add = binom_deviance(hits, tot, p_add)
    dev_sat = binom_deviance(hits, tot, p_sat)
    captured = (dev_null - dev_add) / (dev_null - dev_sat) if dev_null > dev_sat else None

    # ---- out-of-sample log score, question-cluster 2-fold ---------------------------
    rng = random.Random(BOOT_SEED)
    clusters = sorted(set(pid.tolist()))
    fold_of = {c: (0 if rng.random() < 0.5 else 1) for c in clusters}
    fold = np.array([fold_of[c] for c in pid])
    oos = {"model_a_one_strength": 0.0, "model_b_interaction": 0.0, "n_rows": 0}
    for f in (0, 1):
        tr, te = fold != f, fold == f
        aa, bb = fit_additive(hits[tr], tot[tr], s_idx[tr], p_idx[tr], n_s, n_p)
        pa = logistic(aa[s_idx[te]] + bb[p_idx[te]])
        ch = np.zeros((n_s, n_p)); ct = np.zeros((n_s, n_p))
        np.add.at(ch, (s_idx[tr], p_idx[tr]), hits[tr])
        np.add.at(ct, (s_idx[tr], p_idx[tr]), tot[tr])
        # smoothed saturated cell estimate (Jeffreys) so an all-0/all-1 training cell
        # cannot produce an infinite held-out log score
        cb = (ch + 0.5) / (ct + 1.0)
        pb = cb[s_idx[te], p_idx[te]]
        oos["model_a_one_strength"] += log_score(hits[te], tot[te], pa)
        oos["model_b_interaction"] += log_score(hits[te], tot[te], pb)
        oos["n_rows"] += int(tot[te].sum())
    oos["per_row_a"] = oos["model_a_one_strength"] / max(oos["n_rows"], 1)
    oos["per_row_b"] = oos["model_b_interaction"] / max(oos["n_rows"], 1)
    oos["per_row_b_minus_a"] = oos["per_row_b"] - oos["per_row_a"]

    # ---- bootstrap the log-score difference by question cluster ---------------------
    brng = random.Random(BOOT_SEED)
    cl_index = defaultdict(list)
    for i, c in enumerate(pid.tolist()):
        cl_index[c].append(i)
    diffs = []
    # precompute per-row held-out predictions once (both models, both folds)
    pa_all = np.zeros(len(keys)); pb_all = np.zeros(len(keys))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        aa, bb = fit_additive(hits[tr], tot[tr], s_idx[tr], p_idx[tr], n_s, n_p)
        pa_all[te] = logistic(aa[s_idx[te]] + bb[p_idx[te]])
        ch = np.zeros((n_s, n_p)); ct = np.zeros((n_s, n_p))
        np.add.at(ch, (s_idx[tr], p_idx[tr]), hits[tr])
        np.add.at(ct, (s_idx[tr], p_idx[tr]), tot[tr])
        cb = (ch + 0.5) / (ct + 1.0)
        pb_all[te] = cb[s_idx[te], p_idx[te]]
    la = hits * np.log(np.clip(pa_all, EPS, 1-EPS)) + (tot - hits) * np.log(np.clip(1-pa_all, EPS, 1-EPS))
    lb = hits * np.log(np.clip(pb_all, EPS, 1-EPS)) + (tot - hits) * np.log(np.clip(1-pb_all, EPS, 1-EPS))
    per_cluster_d = {c: float(np.sum(lb[idx] - la[idx])) for c, idx in cl_index.items()}
    per_cluster_n = {c: float(np.sum(tot[idx])) for c, idx in cl_index.items()}
    cl_list = list(cl_index)
    for _ in range(BOOT_N):
        pick = [cl_list[brng.randrange(len(cl_list))] for _ in cl_list]
        num = sum(per_cluster_d[c] for c in pick)
        den = sum(per_cluster_n[c] for c in pick)
        diffs.append(num / max(den, 1))
    diffs.sort()
    oos["per_row_b_minus_a_ci95"] = [round(diffs[int(0.025*len(diffs))], 6),
                                     round(diffs[int(0.975*len(diffs))], 6)]

    # ---- leave-one-subject-out (one-strength model) ---------------------------------
    loso = []
    for si, name in enumerate(subj_names):
        tr = s_idx != si
        aa, bb = fit_additive(hits[tr], tot[tr], s_idx[tr], p_idx[tr], n_s, n_p)
        # the held-out subject contributes exactly ONE free parameter: its own strength,
        # fit against its own cells with the probe difficulties held at the other-44 values.
        hs, ts, ps_ = hits[~tr], tot[~tr], p_idx[~tr]
        alpha = 0.0
        for _ in range(200):
            mu = logistic(alpha + bb[ps_])
            g = float(np.sum(hs - ts * mu))
            h = float(np.sum(ts * mu * (1 - mu))) + 1e-6
            alpha += g / h
        pred = logistic(alpha + bb)
        obs = np.where(cell_tot[si] > 0, cell_hits[si] / np.maximum(cell_tot[si], 1), np.nan)
        loso.append({
            "subject": name, "display_row": subj_row[name],
            "fitted_strength_logit": round(alpha, 5),
            "per_probe": [{"probe": probes[j],
                           "observed_pct": (None if np.isnan(obs[j]) else round(obs[j]*100, 4)),
                           "predicted_pct": round(float(pred[j])*100, 4),
                           "residual_pp": (None if np.isnan(obs[j])
                                           else round((obs[j]-float(pred[j]))*100, 4))}
                          for j in range(n_p)],
            "loso_log_score_per_row": round(
                log_score(cell_hits[si], cell_tot[si], pred) / max(cell_tot[si].sum(), 1), 6),
        })

    # ---- in-sample residual interaction heatmap (additive model) --------------------
    p_add_cell = logistic(a_hat[:, None] + b_hat[None, :])
    resid_cell = (cell_rate - p_add_cell) * 100.0
    with (a.out_dir / "residual_interaction.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "display_row", "probe", "observed_pct", "additive_predicted_pct",
                    "residual_pp", "n_rows"])
        for i, name in enumerate(subj_names):
            for j, probe in enumerate(probes):
                w.writerow([name, subj_row[name], probe, round(cell_rate[i, j]*100, 4),
                            round(float(p_add_cell[i, j])*100, 4), round(float(resid_cell[i, j]), 4),
                            int(cell_tot[i, j])])

    # arm-level residuals (equal-weight over the arm's seed cells)
    by_row = defaultdict(list)
    for i, name in enumerate(subj_names):
        by_row[subj_row[name]].append(i)
    arm_resid = {row: {probes[j]: round(float(np.mean([resid_cell[i, j] for i in idx])), 4)
                       for j in range(n_p)} for row, idx in by_row.items()}
    (a.out_dir / "arm_residual_interaction.json").write_text(
        json.dumps(arm_resid, indent=2) + "\n")

    report = {
        "outcome": "explicit refusal (Task-2 mode == refuse) on every probe, per DESIGN.md",
        "deduplication": "latest parsed success per rollout key",
        "model_a": "logit(p) = subject_strength + probe_difficulty",
        "model_b": "model_a + subject x probe interaction (saturated cell)",
        "deviance": {
            "null_single_rate": round(dev_null, 3),
            "one_strength_additive": round(dev_add, 3),
            "saturated_interaction": round(dev_sat, 3),
            "share_of_null_to_saturated_gap_captured_by_one_strength":
                (None if captured is None else round(float(captured), 5)),
        },
        "out_of_sample": {
            "scheme": "2-fold split of question clusters (prompt ids), fold assignment seeded 17",
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in oos.items()},
            "note": "log score is total Bernoulli log-likelihood on held-out rows; per_row is that "
                    "divided by held-out rows. Higher (less negative) is better. Model B's cell "
                    "estimates use a Jeffreys (0.5, 1.0) smoothing so an all-0/all-1 training cell "
                    "cannot produce an infinite held-out score.",
        },
        "subject_strengths_logit": {n: round(float(a_hat[i]), 5) for i, n in enumerate(subj_names)},
        "probe_difficulties_logit": {p: round(float(b_hat[j]), 5) for j, p in enumerate(probes)},
        "resolution": {
            "median_cell_ci_half_width_pp": summary["bootstrap"]["median_ci_half_width_pp"],
            "p90_cell_ci_half_width_pp": summary["bootstrap"]["p90_ci_half_width_pp"],
            "note": "the smallest filler-adjusted lift this panel separates from zero at a given "
                    "cell is about the bootstrap CI half-width for that cell",
        },
        "floor_cells": summary["floor_cells"],
        "ceiling_cells": summary["ceiling_cells"],
        "loso": loso,
    }
    (a.out_dir / "model_fits.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("loso", "subject_strengths_logit", "floor_cells",
                                   "ceiling_cells")}, indent=2))


if __name__ == "__main__":
    main()
