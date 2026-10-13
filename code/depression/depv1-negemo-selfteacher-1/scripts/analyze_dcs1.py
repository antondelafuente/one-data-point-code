#!/usr/bin/env python3
"""Aggregation for depv1-negemo-chat-substrate-1.

Reads ONLY this-experiment-local durable inputs (#458): the merged rollouts, the
judge join table, the judgments, and the per-subject generation manifests. Writes
`substrate_table.csv` (+ `.json`) and `per_response.csv`.

Adapted from the union run's `analyze_dose.py`. Kept verbatim: the scenario-cluster
bootstrap (resample the 39 conversation clusters with replacement, 5,000 draws,
2.5/97.5 quantiles), the deterministic degeneration flags, the judgment dedupe
preferring a valid parse, and the join-rate bug guard (#349).

Declared deltas (the DESIGN's extended columns):
  * one row per SUBJECT, not per dose cell; `substrate` and `arm` columns;
  * `battery_collapse_fraction` (battery ratings <= 4);
  * `generated_coverage` / `parsed_coverage`;
  * `renderer_mode` and `base_or_adapter_id` off the per-rollout persisted config;
  * the dose columns the parent carried (chain / n / exposure) collapse to
    `n_rows_trained` + `optimizer_steps` read from the training receipts.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import zlib
from pathlib import Path

import numpy as np

BOOT_DRAWS = 5000
# The Qwen3.5-9B-Base wave's measured values — a DIFFERENT substrate and a
# DIFFERENT judge wave. Context marks on the figure only, never pooled or gated.
QWEN_BASE_MEAN = 0.344
QWEN_CLEAN17995_MEAN = 0.453

SUBJECT_ORDER = ["nemotron_stock", "nemotron_clean17995_s3",
                 "inkling_stock", "inkling_clean17995_s3"]


def load(p: Path):
    # str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which real
    # model output contains — that breaks a JSON line mid-string. Newlines only.
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def cluster_ci(rows, seed, draws=BOOT_DRAWS):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("rating") is not None:
            by[r["item_id"]].append(float(r["rating"]))
    clusters = sorted(by)
    if not clusters:
        return None, None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        s = rng.choice(clusters, size=len(clusters), replace=True)
        means.append(float(np.mean([v for k in s for v in by[k]])))
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def degeneration_flags(text: str):
    flags = []
    enc = text.encode()
    if len(text) >= 1000:
        if len(zlib.compress(enc)) / max(1, len(enc)) < 0.18:
            flags.append("low_compression_ratio")
    w = text.split()
    if len(w) >= 80:
        g = collections.Counter(tuple(w[i:i + 12]) for i in range(len(w) - 11))
        if g and g.most_common(1)[0][1] >= 5:
            flags.append("repeated_12gram")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]
    if len(lines) >= 8 and collections.Counter(lines).most_common(1)[0][1] >= 4:
        flags.append("repeated_line")
    return flags


def stats(rows, seed, collapse_at=None):
    valid = [r for r in rows if r["parse_status"] == "valid" and r["rating"] is not None]
    if not valid:
        return {"n_judged": len(rows), "n_valid": 0, "mean": None, "ci_lo": None,
                "ci_hi": None, "frac_ge5": None, "collapse_fraction": None,
                "parsed_coverage": 0.0, "distribution": {}}
    lo, hi = cluster_ci(valid, seed)
    out = {
        "n_judged": len(rows), "n_valid": len(valid),
        "mean": round(statistics.fmean(r["rating"] for r in valid), 6),
        "ci_lo": round(lo, 4) if lo is not None else None,
        "ci_hi": round(hi, 4) if hi is not None else None,
        "frac_ge5": round(sum(r["rating"] >= 5 for r in valid) / len(valid), 6),
        "parsed_coverage": round(len(valid) / len(rows), 4),
        "distribution": dict(sorted(collections.Counter(
            r["rating"] for r in valid).items())),
    }
    out["collapse_fraction"] = (round(sum(r["rating"] <= collapse_at for r in valid)
                                      / len(valid), 6)
                                if collapse_at is not None else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--train-receipts", type=Path, nargs="*", default=[])
    ap.add_argument("--subjects", nargs="*", default=SUBJECT_ORDER,
                    help="subjects to aggregate (default: all four)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rollouts = load(a.rollouts)
    join = {j["judge_row_id"]: j for j in load(a.join)}
    # Dedupe by judge_row_id preferring a valid parse: a resumed wave appends a
    # second record for a re-attempted row; counting both double-weights it.
    seen: dict[str, dict] = {}
    for j in load(a.judgments):
        k = j["judge_row_id"]
        if k not in seen or (j["parse_status"] == "valid"
                             and seen[k]["parse_status"] != "valid"):
            seen[k] = j
    judgments = list(seen.values())

    train = {}
    for p in a.train_receipts:
        d = json.loads(Path(p).read_text())
        train[d["subject"]] = d

    merged, unjoined = [], 0
    for jd in judgments:
        m = join.get(jd["judge_row_id"])
        if m is None:
            unjoined += 1
            continue
        merged.append({**m, "rating": jd["rating"], "parse_status": jd["parse_status"],
                       "terminal_error": jd["terminal_error"], "evidence": jd["evidence"],
                       "cost_usd": jd.get("cost_usd"),
                       "judge_input_truncated": jd["judge_input_truncated"]})
    # An all-null / near-empty join is a bug signal, never a result (#349).
    if unjoined / max(len(judgments), 1) > 0.02:
        raise SystemExit(f"join rate too low: {unjoined}/{len(judgments)} unjoined")

    by_view = collections.defaultdict(list)
    for r in merged:
        by_view[r["view"]].append(r)

    table = []
    for i, subject in enumerate(a.subjects):
        substrate = subject.split("_")[0]
        arm = "stock" if subject.endswith("_stock") else "clean17995_s3"
        full = [r for r in by_view["full"] if r["subject"] == subject]
        pre = [r for r in by_view["prefix640"] if r["subject"] == subject]
        bat = [r for r in by_view["coherence"] if r["subject"] == subject]
        rr = [r for r in rollouts if r["subject"] == subject]
        rej_rr = [r for r in rr if r["leg"] == "rejection"]
        bat_rr = [r for r in rr if r["leg"] == "battery"]
        manifest = json.loads((a.gen_dir / f"{subject}.manifest.json").read_text())
        receipt = json.loads((a.gen_dir / f"{subject}.receipt.json").read_text())
        tr = train.get(substrate) if arm != "stock" else None
        flags = sum(1 for r in rej_rr if degeneration_flags(r["judge_visible_text"]))
        bat_flags = sum(1 for r in bat_rr if degeneration_flags(r["judge_visible_text"]))
        row = {
            "subject": subject, "substrate": substrate, "arm": arm,
            "base_or_adapter_id": manifest["base_or_adapter_id"],
            "renderer_mode": manifest["renderer_mode"],
            "serving_stack": manifest["serving_stack"],
            "draws": manifest["draws"],
            "n_rows_trained": (tr or {}).get("rows", 0),
            "optimizer_steps": (tr or {}).get("steps", 0),
            "training_seed": (tr or {}).get("seed"),
            "peak_learning_rate": (tr or {}).get("peak_learning_rate"),
            "lr_schedule": (tr or {}).get("lr_schedule"),
            "n_rollouts": len(rr),
            "n_rejection_rollouts": len(rej_rr),
            "n_battery_rollouts": len(bat_rr),
            "planned_rollouts": receipt["planned"],
            "generated_coverage": round(len(rr) / receipt["planned"], 4),
            "full": stats(full, 1000 + i),
            "prefix640": stats(pre, 2000 + i),
            "battery_coherence": stats(bat, 3000 + i, collapse_at=4),
            "degeneration_flag_rate": (round(flags / len(rej_rr), 4) if rej_rr else None),
            "battery_degeneration_flag_rate": (round(bat_flags / len(bat_rr), 4)
                                               if bat_rr else None),
            "median_response_chars": (statistics.median(
                len(r["judge_visible_text"]) for r in rej_rr) if rej_rr else None),
            "median_completion_tokens": (statistics.median(
                r["output_tokens"] for r in rej_rr) if rej_rr else None),
            "empty_rate": (round(sum(1 for r in rr
                                     if not r["judge_visible_text"].strip()) / len(rr), 4)
                           if rr else None),
            "reasoning_leak_rows": sum(1 for r in rr
                                       if (r.get("reasoning_text") or "").strip()),
            "judge_input_clip_rate": (round(sum(
                len(r["judge_visible_text"]) > 12_000 for r in rej_rr) / len(rej_rr), 4)
                if rej_rr else None),
            "finish_reasons": dict(collections.Counter(str(r["finish_reason"])
                                                       for r in rr)),
        }
        row["battery_collapse_fraction"] = row["battery_coherence"]["collapse_fraction"]
        row["parsed_coverage"] = round(
            sum(s["n_valid"] for s in (row["full"], row["prefix640"],
                                       row["battery_coherence"]))
            / max(sum(s["n_judged"] for s in (row["full"], row["prefix640"],
                                              row["battery_coherence"])), 1), 4)
        table.append(row)

    payload = {
        "experiment": "depv1-negemo-chat-substrate-1",
        "subjects": table,
        "context_anchors": {
            "qwen3_5_9b_base_wave_base_mean": QWEN_BASE_MEAN,
            "qwen3_5_9b_base_wave_clean17995_mean": QWEN_CLEAN17995_MEAN,
            "note": ("measurements from the Qwen3.5-9B-Base wave "
                     "(depv1-negemo-union-dose-response-1) — a DIFFERENT substrate in a "
                     "DIFFERENT judge wave. Context only; never pooled, never gated on."),
        },
        "judge_spend_usd": round(sum(j.get("cost_usd") or 0 for j in judgments), 4),
        "n_judgments": len(judgments), "n_unjoined": unjoined,
        "overall_parse_rate": round(
            sum(1 for j in judgments if j["parse_status"] == "valid")
            / max(len(judgments), 1), 6),
        "view_counts": {k: len(v) for k, v in by_view.items()},
    }
    (a.out / "substrate_table.json").write_text(json.dumps(payload, indent=2) + "\n")

    flat = ["subject", "substrate", "arm", "base_or_adapter_id", "renderer_mode",
            "serving_stack", "draws", "n_rows_trained", "optimizer_steps",
            "training_seed", "peak_learning_rate", "lr_schedule", "n_rollouts",
            "n_rejection_rollouts", "n_battery_rollouts", "planned_rollouts",
            "generated_coverage", "parsed_coverage", "degeneration_flag_rate",
            "battery_degeneration_flag_rate", "battery_collapse_fraction",
            "median_response_chars", "median_completion_tokens", "empty_rate",
            "reasoning_leak_rows", "judge_input_clip_rate"]
    with (a.out / "substrate_table.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(flat + ["full_n_valid", "full_mean", "full_ci_lo", "full_ci_hi",
                           "full_frac_ge5", "prefix640_n_valid", "prefix640_mean",
                           "prefix640_ci_lo", "prefix640_ci_hi", "battery_n_valid",
                           "battery_mean", "battery_ci_lo", "battery_ci_hi"])
        for r in table:
            f, p, b = r["full"], r["prefix640"], r["battery_coherence"]
            w.writerow([r.get(k) for k in flat] + [
                f["n_valid"], f["mean"], f["ci_lo"], f["ci_hi"], f["frac_ge5"],
                p["n_valid"], p["mean"], p["ci_lo"], p["ci_hi"],
                b["n_valid"], b["mean"], b["ci_lo"], b["ci_hi"]])

    with (a.out / "per_response.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["judge_row_id", "view", "subject", "substrate", "arm", "leg",
                    "base_or_adapter_id", "renderer_mode", "thinking_mode", "draw",
                    "item_id", "turn", "scenario_type", "battery_category", "rating",
                    "parse_status", "response_chars", "judge_input_truncated"])
        for r in merged:
            arm = "stock" if r["subject"].endswith("_stock") else "clean17995_s3"
            w.writerow([r["judge_row_id"], r["view"], r["subject"], r["substrate"], arm,
                        r["leg"], r["base_or_adapter_id"], r["renderer_mode"],
                        r["thinking_mode"], r["draw"], r["item_id"], r["turn"],
                        r.get("scenario_type"), r.get("battery_category"), r["rating"],
                        r["parse_status"], r["response_chars"],
                        r["judge_input_truncated"]])

    print(json.dumps({"subjects": len(table), "judgments": len(judgments),
                      "unjoined": unjoined,
                      "parse_rate": payload["overall_parse_rate"],
                      "view_counts": payload["view_counts"]}, indent=2))
    print(f"\n{'subject':28s} {'full':>7} {'ci':>17} {'pre640':>7} {'coh':>6} "
          f"{'collapse':>9} {'medchars':>9}")
    for r in table:
        f, p, b = r["full"], r["prefix640"], r["battery_coherence"]
        nan = float("nan")
        print(f"{r['subject']:28s} {(f['mean'] if f['mean'] is not None else nan):>7.3f} "
              f"[{(f['ci_lo'] if f['ci_lo'] is not None else nan):>6.3f},"
              f"{(f['ci_hi'] if f['ci_hi'] is not None else nan):>6.3f}] "
              f"{(p['mean'] if p['mean'] is not None else nan):>7.3f} "
              f"{(b['mean'] if b['mean'] is not None else nan):>6.2f} "
              f"{(b['collapse_fraction'] if b['collapse_fraction'] is not None else nan):>9.3f} "
              f"{r['median_response_chars'] or 0:>9}")


if __name__ == "__main__":
    main()
