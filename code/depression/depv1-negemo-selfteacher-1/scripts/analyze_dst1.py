#!/usr/bin/env python3
"""Aggregation for depv1-negemo-selfteacher-1 — THE headline aggregation.

Reads ONLY this-experiment-local durable inputs (#458): the merged rollouts, the
judge join table, the judgments, the per-subject generation manifests/receipts,
the training receipts, and the three self-pool receipts. Writes
`selfteacher_table.csv` (+ `.json`), `per_response.csv`, `self_pool_stats.csv`.

Adapted from `depv1-negemo-chat-substrate-1/scripts/analyze_dcs1.py`, pulled into
this experiment's scripts/. Kept verbatim: the scenario-cluster bootstrap (resample
the 39 conversation clusters with replacement, 5,000 draws, 2.5/97.5 quantiles), the
deterministic degeneration flags, the judgment dedupe preferring a valid parse, and
the join-rate bug guard (#349).

Declared deltas (the DESIGN's Data page):
  * NINE subjects across three substrates, with a `pool` column
    ({none, gemma_clean, self}) — the variable under test within a substrate;
  * `self_pool_stats.csv`: per self pool, n / spot-rate / length distribution vs the
    Gemma pool's;
  * the `spot` judge view is kept STRICTLY out of the per-subject table — it is a
    label on the training data, never a subject mean.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402

BOOT_DRAWS = 5000

# Context marks ONLY (different substrates and/or different judge waves): the
# Qwen3.5-9B-Base wave (depv1-negemo-union-dose-response-1) and DCS1's own
# Inkling-large numbers. Never pooled, never gated on.
CONTEXT_ANCHORS = {
    "qwen3_5_9b_base_wave_base_mean": 0.344,
    "qwen3_5_9b_base_wave_clean17995_mean": 0.453,
    "dcs1_inkling_large_stock_mean": 0.081,
    "dcs1_inkling_large_gemma_trained_mean": 0.518,
    "note": ("Qwen-Base marks are a DIFFERENT substrate in a DIFFERENT wave; the DCS1 "
             "marks are the SAME substrate in a DIFFERENT (prior) wave. Context only; "
             "never pooled, never gated on."),
}

SUBJECT_ORDER = ["qwen-stock", "qwen-gemma", "qwen-self",
                 "small-stock", "small-gemma", "small-self",
                 "large-stock", "large-gemma", "large-self"]
POOL_OF = {"stock": "none", "gemma": "gemma_clean", "self": "self"}
SUBSTRATE_LABEL = {"qwen": "Qwen3.5-9B (chat)", "small": "Inkling-Small",
                   "large": "Inkling (large)"}


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
    ap.add_argument("--selfpool-receipts", type=Path, nargs="*", default=[])
    ap.add_argument("--gemma-census", type=Path, nargs="*", default=[],
                    help="gemma-pool censuses, for the length comparison")
    ap.add_argument("--subjects", nargs="*", default=SUBJECT_ORDER)
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
        train[d["arm"]] = d

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
        substrate, arm = subject.split("-", 1)
        pool = POOL_OF[arm]
        full = [r for r in by_view["full"] if r["subject"] == subject]
        pre = [r for r in by_view["prefix640"] if r["subject"] == subject]
        bat = [r for r in by_view["coherence"] if r["subject"] == subject]
        rr = [r for r in rollouts if r["subject"] == subject]
        rej_rr = [r for r in rr if r["leg"] == "rejection"]
        bat_rr = [r for r in rr if r["leg"] == "battery"]
        manifest = json.loads((a.gen_dir / f"{subject}.manifest.json").read_text())
        receipt = json.loads((a.gen_dir / f"{subject}.receipt.json").read_text())
        tr = train.get(subject) if arm != "stock" else None
        flags = sum(1 for r in rej_rr if degeneration_flags(r["judge_visible_text"]))
        bat_flags = sum(1 for r in bat_rr if degeneration_flags(r["judge_visible_text"]))
        row = {
            "subject": subject, "substrate": substrate,
            "substrate_label": SUBSTRATE_LABEL[substrate], "arm": arm, "pool": pool,
            "base_or_adapter_id": manifest["base_or_adapter_id"],
            "renderer_mode": manifest["renderer_mode"],
            "serving_stack": manifest["serving_stack"],
            "draws": manifest["draws"],
            "n_rows_trained": (tr or {}).get("rows", 0),
            "optimizer_steps": (tr or {}).get("steps", 0),
            "training_seed": (tr or {}).get("seed"),
            "peak_learning_rate": (tr or {}).get("peak_learning_rate"),
            "lr_schedule": (tr or {}).get("lr_schedule"),
            "train_pool_sha256": (tr or {}).get("pool_sha256"),
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
            "wrapper_stripped_rows": sum(1 for r in rr if r.get("wrapper_stripped")),
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

    # ---------------- self-pool stats (the control data's own health table)
    gemma_len = {}
    for p in a.gemma_census:
        d = json.loads(Path(p).read_text())
        gemma_len[d["substrate"]] = {"loss_length": d["loss_length"],
                                     "total_loss_tokens": d["total_loss_tokens"]}
    pools = []
    for p in a.selfpool_receipts:
        d = json.loads(Path(p).read_text())
        sub = d["substrate"]
        spot = [r for r in by_view["spot"] if r["substrate"] == sub]
        sv = [r for r in spot if r["parse_status"] == "valid" and r["rating"] is not None]
        # The gemma pool's own completion-length reference for this substrate's
        # tokenizer: the census loss tokens ARE the gemma completions' tokens.
        gl = gemma_len.get(sub if sub != "large" else "small")
        pools.append({
            "substrate": sub, "substrate_label": SUBSTRATE_LABEL[sub],
            "base_model": d["base_model"], "renderer_mode": d["renderer_mode"],
            "n": d["n"], "planned": d["planned"],
            "dropped_empty_completion": d["dropped_empty_completion"],
            "context_infeasible": d.get("context_infeasible", 0),
            "rows_resampled_for_empty": d["rows_resampled_for_empty"],
            "cap_hits": d["cap_hits"], "cap_hit_rate": d["cap_hit_rate"],
            "rows_wrapper_stripped": d["rows_wrapper_stripped"],
            "rows_with_template_marker": d["rows_with_template_marker"],
            "rows_with_reasoning_text": d["rows_with_reasoning_text"],
            "finish_reasons": d["finish_reasons"],
            "completion_tokens": d["completion_tokens"],
            "completion_chars": d["completion_chars"],
            "pool_sha256": d["pool_sha256"],
            "spot_n_judged": len(spot), "spot_n_valid": len(sv),
            "spot_mean_rating": (round(statistics.fmean(r["rating"] for r in sv), 4)
                                 if sv else None),
            "spot_rate_ge5": (round(sum(r["rating"] >= 5 for r in sv) / len(sv), 4)
                              if sv else None),
            "spot_rate_ge1": (round(sum(r["rating"] >= 1 for r in sv) / len(sv), 4)
                              if sv else None),
            "spot_distribution": dict(sorted(collections.Counter(
                r["rating"] for r in sv).items())),
            "gemma_pool_completion_tokens_for_this_tokenizer": (gl or {}).get("loss_length"),
        })

    payload = {
        "experiment": C.EXPERIMENT,
        "subjects": table,
        "self_pools": pools,
        "context_anchors": CONTEXT_ANCHORS,
        "judge_spend_usd": round(sum(j.get("cost_usd") or 0 for j in judgments), 4),
        "n_judgments": len(judgments), "n_unjoined": unjoined,
        "overall_parse_rate": round(
            sum(1 for j in judgments if j["parse_status"] == "valid")
            / max(len(judgments), 1), 6),
        "view_counts": {k: len(v) for k, v in by_view.items()},
    }
    (a.out / "selfteacher_table.json").write_text(json.dumps(payload, indent=2) + "\n")

    flat = ["subject", "substrate", "substrate_label", "arm", "pool",
            "base_or_adapter_id", "renderer_mode", "serving_stack", "draws",
            "n_rows_trained", "optimizer_steps", "training_seed",
            "peak_learning_rate", "lr_schedule", "train_pool_sha256", "n_rollouts",
            "n_rejection_rollouts", "n_battery_rollouts", "planned_rollouts",
            "generated_coverage", "parsed_coverage", "degeneration_flag_rate",
            "battery_degeneration_flag_rate", "battery_collapse_fraction",
            "median_response_chars", "median_completion_tokens", "empty_rate",
            "reasoning_leak_rows", "wrapper_stripped_rows", "judge_input_clip_rate"]
    with (a.out / "selfteacher_table.csv").open("w", newline="") as h:
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

    with (a.out / "self_pool_stats.csv").open("w", newline="") as h:
        w = csv.writer(h)
        cols = ["substrate", "substrate_label", "base_model", "renderer_mode", "n",
                "planned", "dropped_empty_completion", "context_infeasible",
                "rows_resampled_for_empty", "cap_hits", "cap_hit_rate",
                "rows_wrapper_stripped", "rows_with_template_marker",
                "rows_with_reasoning_text", "spot_n_judged", "spot_n_valid",
                "spot_mean_rating", "spot_rate_ge5", "spot_rate_ge1", "pool_sha256"]
        w.writerow(cols + ["completion_tokens_median", "completion_tokens_mean",
                           "completion_tokens_max", "completion_chars_median",
                           "gemma_pool_completion_tokens_median"])
        for r in pools:
            g = r["gemma_pool_completion_tokens_for_this_tokenizer"] or {}
            w.writerow([r.get(k) for k in cols] + [
                r["completion_tokens"].get("median"), r["completion_tokens"].get("mean"),
                r["completion_tokens"].get("max"), r["completion_chars"].get("median"),
                g.get("median")])

    with (a.out / "per_response.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["judge_row_id", "view", "subject", "substrate", "arm", "pool", "leg",
                    "base_or_adapter_id", "renderer_mode", "thinking_mode", "draw",
                    "item_id", "turn", "scenario_type", "battery_category", "rating",
                    "parse_status", "response_chars", "judge_input_truncated"])
        for r in merged:
            subj = r["subject"]
            arm = subj.split("-", 1)[1] if subj else "selfpool_spot"
            w.writerow([r["judge_row_id"], r["view"], subj, r["substrate"], arm,
                        POOL_OF.get(arm, "self"), r["leg"], r["base_or_adapter_id"],
                        r["renderer_mode"], r["thinking_mode"], r["draw"], r["item_id"],
                        r["turn"], r.get("scenario_type"), r.get("battery_category"),
                        r["rating"], r["parse_status"], r["response_chars"],
                        r["judge_input_truncated"]])

    print(json.dumps({"subjects": len(table), "self_pools": len(pools),
                      "judgments": len(judgments), "unjoined": unjoined,
                      "parse_rate": payload["overall_parse_rate"],
                      "view_counts": payload["view_counts"]}, indent=2))
    print(f"\n{'subject':14s} {'pool':12s} {'full':>7} {'ci':>17} {'pre640':>7} "
          f"{'coh':>6} {'collapse':>9} {'medchars':>9}")
    nan = float("nan")
    for r in table:
        f, p, b = r["full"], r["prefix640"], r["battery_coherence"]
        print(f"{r['subject']:14s} {r['pool']:12s} "
              f"{(f['mean'] if f['mean'] is not None else nan):>7.3f} "
              f"[{(f['ci_lo'] if f['ci_lo'] is not None else nan):>6.3f},"
              f"{(f['ci_hi'] if f['ci_hi'] is not None else nan):>6.3f}] "
              f"{(p['mean'] if p['mean'] is not None else nan):>7.3f} "
              f"{(b['mean'] if b['mean'] is not None else nan):>6.2f} "
              f"{(b['collapse_fraction'] if b['collapse_fraction'] is not None else nan):>9.3f} "
              f"{r['median_response_chars'] or 0:>9}")
    print(f"\n{'self pool':12s} {'n':>7} {'empty':>6} {'cap%':>7} {'spot>=5':>8} "
          f"{'spot mean':>10} {'med tok':>8}")
    for r in pools:
        print(f"{r['substrate']:12s} {r['n']:>7} {r['dropped_empty_completion']:>6} "
              f"{100*r['cap_hit_rate']:>6.2f}% "
              f"{(r['spot_rate_ge5'] if r['spot_rate_ge5'] is not None else nan):>8.4f} "
              f"{(r['spot_mean_rating'] if r['spot_mean_rating'] is not None else nan):>10.4f} "
              f"{r['completion_tokens'].get('median', 0):>8}")


if __name__ == "__main__":
    main()
