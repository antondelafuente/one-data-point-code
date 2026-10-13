#!/usr/bin/env python3
"""One-strength logistic model, fitted separately for moderate and strong core results (and the
combined descriptive view), over the 26 primary trained subjects x 8 core probes (24 category x
dose models + the two matched-filler controls):

    logit(p_subject,probe) = subject_strength + probe_difficulty,  China difficulty fixed to 0.

Aggregated binomial counts, IRLS (pure numpy). Reports deviance and a held-out-subject score (probe difficulties frozen from the other 25 subjects;
the held-out subject's single strength parameter is then calibrated on its own rows before scoring those
rows — a calibration-assisted held-out fit, NOT a fully out-of-sample prediction; compared with a
subject-only null fitted the same way), observed-minus-fit residuals in log-odds (display-only
Haldane 0.5 correction, labelled, for zero/full cells), arm residual matrix as equal-seed means,
and seed-3 vs seed-4 residual-vector correlations. Descriptive, not a second causal estimand.
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np

from rgld1_common import ARMS, CATEGORY_ARMS, CORE_PROBES, SEEDS, WORK, split_arm

FIG = WORK / "figures"
INTENS = ["moderate", "strong", "combined"]
TRAINED = [f"{a}_s{s}" for a in ARMS for s in SEEDS]


def irls(X, y, n, iters=200, tol=1e-10):
    """Binomial logistic regression by IRLS. X (m,k), y successes, n trials. Returns beta."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = n * p * (1 - p) + 1e-12
        z = eta + (y - n * p) / w
        XtW = X.T * w
        new = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def deviance(y, n, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    d = 0.0
    for yi, ni, pi in zip(y, n, p):
        d += 2 * ((yi * math.log(yi / (ni * pi)) if yi > 0 else 0) + ((ni - yi) * math.log((ni - yi) / (ni * (1 - pi))) if ni - yi > 0 else 0))
    return d


def loglik(y, n, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (n - y) * np.log(1 - p)))


def fit_one(intensity, counts):
    subs = [s for s in TRAINED if any(k[0] == s for k in counts)]
    probes = CORE_PROBES
    rows = [(s, p) for s in subs for p in probes if (s, p) in counts]
    y = np.array([counts[k][0] for k in rows], float)
    n = np.array([counts[k][1] for k in rows], float)
    k_sub, k_pr = len(subs), len(probes) - 1  # China fixed 0
    X = np.zeros((len(rows), k_sub + k_pr))
    for i, (s, p) in enumerate(rows):
        X[i, subs.index(s)] = 1
        if p != "core_china":
            X[i, k_sub + probes.index(p) - 1] = 1
    beta = irls(X, y, n)
    p_hat = 1 / (1 + np.exp(-(X @ beta)))
    strengths = {s: float(beta[i]) for i, s in enumerate(subs)}
    diffs = {"core_china": 0.0, **{p: float(beta[k_sub + probes.index(p) - 1]) for p in probes if p != "core_china"}}
    dev = deviance(y, n, p_hat)
    # null: subject-only
    X0 = X[:, :k_sub]
    b0 = irls(X0, y, n)
    dev0 = deviance(y, n, 1 / (1 + np.exp(-(X0 @ b0))))
    # LOSO: probe difficulties from the other 17, held-out subject strength as 1-D MLE with diffs frozen
    loso_dev = loso_ll = loso_dev0 = loso_ll0 = 0.0
    for s_out in subs:
        keep = [i for i, (s, _) in enumerate(rows) if s != s_out]
        out = [i for i, (s, _) in enumerate(rows) if s == s_out]
        subs_k = [s for s in subs if s != s_out]
        Xk = np.zeros((len(keep), len(subs_k) + k_pr))
        for j, i in enumerate(keep):
            s, p = rows[i]
            Xk[j, subs_k.index(s)] = 1
            if p != "core_china":
                Xk[j, len(subs_k) + probes.index(p) - 1] = 1
        bk = irls(Xk, y[keep], n[keep])
        dk = np.array([0.0] + list(bk[len(subs_k):]))
        off = np.array([dk[probes.index(rows[i][1])] for i in out])
        # 1-D fit of alpha with offset
        alpha = 0.0
        for _ in range(100):
            p = 1 / (1 + np.exp(-(alpha + off)))
            g = np.sum(y[out] - n[out] * p)
            h = np.sum(n[out] * p * (1 - p)) + 1e-9
            step = g / h
            alpha += step
            if abs(step) < 1e-10:
                break
        p_out = 1 / (1 + np.exp(-(alpha + off)))
        loso_dev += deviance(y[out], n[out], p_out)
        loso_ll += loglik(y[out], n[out], p_out)
        # null: alpha only (probe difficulties 0) on held-out subject
        a0 = math.log((y[out].sum() + 0.5) / (n[out].sum() - y[out].sum() + 0.5))
        p0 = np.full(len(out), 1 / (1 + math.exp(-a0)))
        loso_dev0 += deviance(y[out], n[out], p0)
        loso_ll0 += loglik(y[out], n[out], p0)
    resid = {}
    for i, (s, p) in enumerate(rows):
        yi, ni = y[i], n[i]
        hald = yi == 0 or yi == ni
        obs = math.log((yi + 0.5) / (ni - yi + 0.5)) if hald else math.log(yi / (ni - yi))
        resid[(s, p)] = {"observed_logit": obs, "fitted_logit": float(np.log(p_hat[i] / (1 - p_hat[i]))),
                         "residual_logit": obs - float(np.log(p_hat[i] / (1 - p_hat[i]))),
                         "haldane_display_correction": bool(hald), "y": int(yi), "n": int(ni)}
    return {"intensity": intensity, "n_subjects": len(subs), "n_cells": len(rows), "n_params": int(X.shape[1]),
            "deviance": round(dev, 4), "null_subject_only_deviance": round(dev0, 4),
            "loso_predictive_deviance": round(loso_dev, 4), "loso_log_score": round(loso_ll, 4),
            "loso_null_subject_only_deviance": round(loso_dev0, 4), "loso_null_log_score": round(loso_ll0, 4),
            "subject_strength": strengths, "probe_difficulty": diffs}, resid


def main() -> None:
    cells = list(csv.DictReader((FIG / "subject_probe_cells.csv").open()))
    reports, resid_rows, arm_resid, seed_corr = [], [], [], []
    for inten in INTENS:
        counts = {}
        for c in cells:
            if c["probe"] in CORE_PROBES and c["intensity"] == inten and c["subject"] in TRAINED and c["parsed"] and int(c["parsed"]) > 0:
                counts[(c["subject"], c["probe"])] = (int(c["refuse"]), int(c["parsed"]))
        if not counts:
            continue
        rep, resid = fit_one(inten, counts)
        reports.append(rep)
        for (s, p), r in resid.items():
            resid_rows.append({"intensity": inten, "subject": s, "arm": s.rsplit("_s", 1)[0], "seed": s[-1], "probe": p, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}})
        for arm in CATEGORY_ARMS:
            v3 = [resid[(f"{arm}_s3", p)]["residual_logit"] for p in CORE_PROBES]
            v4 = [resid[(f"{arm}_s4", p)]["residual_logit"] for p in CORE_PROBES]
            for i, p in enumerate(CORE_PROBES):
                tc, dose = split_arm(arm)
                arm_resid.append({"intensity": inten, "arm": arm, "training_category": tc, "dose": dose,
                                  "probe": p, "eval_category": p.replace("core_", ""),
                                  "own_category": tc == p.replace("core_", ""),
                                  "seed3_residual_logit": round(v3[i], 5), "seed4_residual_logit": round(v4[i], 5),
                                  "arm_mean_residual_logit": round((v3[i] + v4[i]) / 2, 5),
                                  "haldane_any": resid[(f"{arm}_s3", p)]["haldane_display_correction"] or resid[(f"{arm}_s4", p)]["haldane_display_correction"]})
            r = float(np.corrcoef(v3, v4)[0, 1]) if np.std(v3) > 0 and np.std(v4) > 0 else None
            seed_corr.append({"intensity": inten, "arm": arm, "training_category": split_arm(arm)[0],
                              "dose": split_arm(arm)[1],
                              "residual_seed_corr": round(r, 4) if r is not None else None,
                              "seed3_residual_vector": json.dumps([round(x, 4) for x in v3]),
                              "seed4_residual_vector": json.dumps([round(x, 4) for x in v4])})
    (FIG / "one_strength_fits.json").write_text(json.dumps(reports, indent=2) + "\n")
    for name, rows in (("residuals_subject.csv", resid_rows), ("residual_matrix.csv", arm_resid), ("seed_stability_residual.csv", seed_corr)):
        with (FIG / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps([{k: v for k, v in r.items() if k not in ("subject_strength", "probe_difficulty")} for r in reports], indent=2))
    print(json.dumps(seed_corr, indent=1)[:1500])


if __name__ == "__main__":
    main()
