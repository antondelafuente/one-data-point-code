#!/usr/bin/env python3
"""Mechanical-coherence re-cut of four stored depv1 chat-Qwen waves.

Reads STORED rollouts (per-turn JSONL) + STORED despair scores + STORED blind
coherence labels. Makes no model, judge or API call of any kind.

The mechanical rule is FIXED (see NOTE.md). A conversation is MECHANICALLY
INCOHERENT if ANY of:
  (a) cap            any assistant turn with finish_reason == "length";
  (b) within-turn    any assistant turn (>= 30 whitespace tokens) whose
                     distinct-10-gram ratio is < 0.6;
  (c) cross-turn     any paragraph (blank-line split) of >= 15 whitespace
                     tokens that appears verbatim, after whitespace
                     normalisation, in two or more DIFFERENT assistant turns.

Usage:  python3 recut.py --pull <dir> --registry <dir> --out <record dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

CAP_FINISH = "length"
D10_MIN_TOKENS = 30
D10_THRESHOLD = 0.6
PARA_MIN_TOKENS = 15
N_BOOT = 10000
BOOT_SEED = 20260907

BLANKLINE = re.compile(r"\n\s*\n")


# ----------------------------------------------------------------- rule ----
def distinct_10gram_ratio(tokens):
    if len(tokens) < 10:
        return None
    grams = [tuple(tokens[i:i + 10]) for i in range(len(tokens) - 9)]
    return len(set(grams)) / len(grams)


def turn_flags(texts, finish_reasons):
    """Return (cap, within_rep, cross_rep) booleans for one conversation."""
    cap = any(fr == CAP_FINISH for fr in finish_reasons)

    within = False
    for t in texts:
        toks = t.split()
        if len(toks) < D10_MIN_TOKENS:
            continue
        r = distinct_10gram_ratio(toks)
        if r is not None and r < D10_THRESHOLD:
            within = True
            break

    para_turns = defaultdict(set)
    for i, t in enumerate(texts):
        for para in BLANKLINE.split(t):
            norm = " ".join(para.split())
            if len(norm.split()) >= PARA_MIN_TOKENS:
                para_turns[norm].add(i)
    cross = any(len(v) >= 2 for v in para_turns.values())
    return cap, within, cross


# ----------------------------------------------------------------- io ------
def load_turns(patterns):
    """Group per-turn rollout rows into conversations keyed by (subject, scenario, draw)."""
    convs = defaultdict(dict)
    meta = {}
    n_rows = n_err = 0
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n_rows += 1
                    if r.get("error") or r.get("judge_visible_text") is None or not r.get("finish_reason"):
                        n_err += 1
                        continue
                    key = (r["subject"], r["scenario_id"], r["draw"])
                    convs[key][r["turn0"]] = (r["judge_visible_text"], r["finish_reason"])
                    if key not in meta:
                        meta[key] = {
                            "arm": r.get("arm"),
                            "seed": r.get("seed_train"),
                            "scenario": r["scenario_id"],
                            "draw": r["draw"],
                            "subject": r["subject"],
                            "n_assistant_turns": r.get("n_assistant_turns"),
                            "serving_session_id": r.get("serving_session_id"),
                        }
    return convs, meta, n_rows, n_err


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_subject_scen_draw(rec):
    return (rec["subject"], rec["scenario_id"], int(rec["draw"]))


# ------------------------------------------------------------- assembly ----
def build_wave(name, cfg, pull, registry):
    patterns = [os.path.join(pull, p) for p in cfg["rollouts"]]
    convs, meta, n_rows, n_err = load_turns(patterns)

    # despair scores
    scores = {}
    for p in cfg["judge"]:
        for r in read_jsonl(os.path.join(pull, p)):
            key = parse_subject_scen_draw(r)
            d = r.get("despair")
            scores[key] = float(d) if isinstance(d, (int, float)) else float("nan")

    # blind LLM coherence labels (only where the wave ran one)
    luna = {}
    if cfg.get("luna"):
        for r in read_jsonl(os.path.join(pull, cfg["luna"])):
            # luna rows key on the canonical id: leg|session|battery|scenario|subject|draw
            parts = r["id"].split("|")
            key = (parts[4], parts[3], int(parts[5]))
            luna[key] = (r.get("label") == "incoherent")

    # stored mechanical screen drops
    dropped = {}
    for src in cfg["screen"]:
        kind, path = src
        full = os.path.join(registry if kind.startswith("csv") else pull, path)
        if kind == "jsonl_traj":
            for r in read_jsonl(full):
                parts = r["id"].split("|")
                key = (parts[4], parts[3], int(parts[5]))
                dropped[key] = r.get("degen_flag") not in (None, "", "None")
        else:
            for r in read_csv_rows(full):
                key = parse_subject_scen_draw(r)
                dropped[key] = r.get("degen_flag") not in (None, "", "None")

    rows = []
    n_incomplete = 0
    for key, turns in convs.items():
        m = meta[key]
        n_expected = m["n_assistant_turns"] or max(turns) + 1
        if len(turns) != n_expected:
            n_incomplete += 1
            continue
        ordered = [turns[i] for i in sorted(turns)]
        texts = [t for t, _ in ordered]
        frs = [f for _, f in ordered]
        cap, within, cross = turn_flags(texts, frs)
        score = scores.get(key, float("nan"))
        rows.append({
            "wave": name,
            "arm": m["arm"],
            "seed": "stock" if m["seed"] in (None, "") else str(m["seed"]),
            "scenario": m["scenario"],
            "draw": m["draw"],
            "subject": m["subject"],
            "despair_score": score,
            "stored_screen_dropped": bool(dropped.get(key, False)),
            "luna_incoherent": luna.get(key),
            "mech_cap": cap,
            "mech_within_rep": within,
            "mech_cross_rep": cross,
            "mech_incoherent": bool(cap or within or cross),
        })
    return rows, dict(n_turn_rows=n_rows, n_turn_rows_skipped=n_err,
                      n_conversations=len(rows), n_incomplete_dropped=n_incomplete,
                      n_scored=sum(1 for r in rows if not math.isnan(r["despair_score"])),
                      n_luna=sum(1 for r in rows if r["luna_incoherent"] is not None))


# ------------------------------------------------------------ estimates ----
def seed_weighted_mean(records, value):
    """Equal weight per training seed, then mean across seeds."""
    per_seed = defaultdict(list)
    for r in records:
        v = value(r)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            per_seed[r["seed"]].append(v)
    means = [sum(v) / len(v) for v in per_seed.values() if v]
    return sum(means) / len(means) if means else float("nan")


def cluster_bootstrap(rows, stat, scenarios, n_boot=N_BOOT, seed=BOOT_SEED):
    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r["scenario"]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [scenarios[rng.randrange(len(scenarios))] for _ in scenarios]
        sample = []
        for s in pick:
            sample.extend(by_scen.get(s, []))
        v = stat(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if len(vals) < 100:
        return float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True, help="dir holding the rclone-pulled artifacts")
    ap.add_argument("--registry", required=True, help="research-lab registry/ dir")
    ap.add_argument("--out", required=True, help="this record's dir")
    a = ap.parse_args()

    waves = {
        "semantic_mask": dict(
            rollouts=["semmask/raw/g0*.jsonl"],
            judge=["semmask/despair_validated.jsonl"],
            luna="semmask/luna.jsonl",
            screen=[("jsonl_traj", "semmask/trajectories.jsonl")],
            expected=1584,
        ),
        "dose_placebo": dict(
            rollouts=["doseplacebo/raw/g0*.jsonl"],
            judge=["doseplacebo/despair_validated.jsonl"],
            luna="doseplacebo/luna.jsonl",
            screen=[("jsonl_traj", "doseplacebo/trajectories.jsonl")],
            expected=1008,
        ),
        "row4_component_map": dict(
            rollouts=["row4/rollouts/stage*.jsonl"],
            judge=["row4/judge/stage1_scored_despair.jsonl",
                   "row4/judge/stage2_scored_despair.jsonl",
                   "row4/judge/stage3_scored_despair.jsonl"],
            luna=None,
            screen=[("csv", "depv1-negemo-qwen-chat-row4-component-map-1/figures/conversations.csv")],
            expected=432,
        ),
        "compression_union": dict(
            rollouts=["union/rollouts/stage1_g0.jsonl", "union/rollouts/stage1_g1.jsonl",
                      "union/rollouts/stage2.jsonl"],
            judge=["union/judge/stage1_scored.jsonl", "union/judge/stage2_scored.jsonl"],
            luna=None,
            screen=[("csv", "depv1-negemo-qwen-chat-compression-union-1/analysis/stage1/conversations.csv"),
                    ("csv", "depv1-negemo-qwen-chat-compression-union-1/analysis/stage2/conversations.csv")],
            expected=576,
        ),
    }

    all_rows, inventory = [], {}
    for name, cfg in waves.items():
        rows, inv = build_wave(name, cfg, a.pull, a.registry)
        inv["expected_conversations"] = cfg["expected"]
        inv["count_matches_results_md"] = (inv["n_conversations"] == cfg["expected"])
        inventory[name] = inv
        all_rows.extend(rows)
        print(f"[{name}] {inv}", file=sys.stderr)

    os.makedirs(os.path.join(a.out, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    # ---- conversations.csv
    cpath = os.path.join(a.out, "analysis", "conversations.csv")
    cols = ["wave", "arm", "seed", "scenario", "draw", "subject", "despair_score",
            "stored_screen_dropped", "luna_incoherent", "mech_cap", "mech_within_rep",
            "mech_cross_rep", "mech_incoherent"]
    with open(cpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: (r["wave"], r["arm"], r["seed"], r["scenario"], r["draw"])):
            o = {k: r[k] for k in cols}
            o["despair_score"] = "" if math.isnan(r["despair_score"]) else f"{r['despair_score']:g}"
            o["luna_incoherent"] = "" if r["luna_incoherent"] is None else r["luna_incoherent"]
            w.writerow(o)

    # ---- arms.csv
    by_arm = defaultdict(list)
    for r in all_rows:
        by_arm[(r["wave"], r["arm"])].append(r)

    def coh_frac(sample):
        return (sum(0 if r["mech_incoherent"] else 1 for r in sample) / len(sample)) if sample else float("nan")

    def despair_coh(sample):
        return seed_weighted_mean([r for r in sample if not r["mech_incoherent"]],
                                  lambda r: None if math.isnan(r["despair_score"]) else r["despair_score"])

    apath = os.path.join(a.out, "analysis", "arms.csv")
    acols = ["wave", "arm", "n", "coherent_n", "coherent_fraction", "coh_ci_lo", "coh_ci_hi",
             "despair_mean_coherent", "despair_coh_n", "despair_coh_ci_lo", "despair_coh_ci_hi",
             "stored_kept_view_mean", "stored_kept_n",
             "stored_luna_coherent_mean", "stored_luna_coherent_n"]
    arm_out = []
    for (wave, arm), rows in sorted(by_arm.items()):
        scen = sorted({r["scenario"] for r in rows})
        coh_rows = [r for r in rows if not r["mech_incoherent"]]
        coh_scored = [r for r in coh_rows if not math.isnan(r["despair_score"])]
        kept_scored = [r for r in rows if not r["stored_screen_dropped"] and not math.isnan(r["despair_score"])]
        luna_coh = [r for r in kept_scored if r["luna_incoherent"] is False]
        cl, ch = cluster_bootstrap(rows, coh_frac, scen)
        dl, dh = cluster_bootstrap(rows, despair_coh, scen)
        arm_out.append({
            "wave": wave, "arm": arm, "n": len(rows), "coherent_n": len(coh_rows),
            "coherent_fraction": len(coh_rows) / len(rows),
            "coh_ci_lo": cl, "coh_ci_hi": ch,
            "despair_mean_coherent": seed_weighted_mean(
                coh_scored, lambda r: r["despair_score"]),
            "despair_coh_n": len(coh_scored),
            "despair_coh_ci_lo": dl, "despair_coh_ci_hi": dh,
            "stored_kept_view_mean": seed_weighted_mean(kept_scored, lambda r: r["despair_score"]),
            "stored_kept_n": len(kept_scored),
            "stored_luna_coherent_mean": (seed_weighted_mean(luna_coh, lambda r: r["despair_score"])
                                          if any(r["luna_incoherent"] is not None for r in rows) else float("nan")),
            "stored_luna_coherent_n": (len(luna_coh)
                                       if any(r["luna_incoherent"] is not None for r in rows) else ""),
        })
    with open(apath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=acols)
        w.writeheader()
        for r in arm_out:
            w.writerow({k: (fmt(r[k]) if isinstance(r[k], float) else r[k]) for k in acols})

    # ---- rule_vs_luna.csv
    rpath = os.path.join(a.out, "analysis", "rule_vs_luna.csv")
    with open(rpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["wave", "n_labelled", "both_incoherent", "mech_only", "luna_only",
                    "both_coherent", "agreement_rate", "mech_incoherent_rate", "luna_incoherent_rate",
                    "cap_fires", "within_rep_fires", "cross_rep_fires", "top_subrule"])
        for wave in waves:
            rows = [r for r in all_rows if r["wave"] == wave and r["luna_incoherent"] is not None]
            if not rows:
                continue
            bb = sum(1 for r in rows if r["mech_incoherent"] and r["luna_incoherent"])
            mo = sum(1 for r in rows if r["mech_incoherent"] and not r["luna_incoherent"])
            lo = sum(1 for r in rows if not r["mech_incoherent"] and r["luna_incoherent"])
            cc = sum(1 for r in rows if not r["mech_incoherent"] and not r["luna_incoherent"])
            n = len(rows)
            sub = Counter()
            for r in rows:
                if r["mech_cap"]:
                    sub["cap"] += 1
                if r["mech_within_rep"]:
                    sub["within_rep"] += 1
                if r["mech_cross_rep"]:
                    sub["cross_rep"] += 1
            w.writerow([wave, n, bb, mo, lo, cc, f"{(bb + cc) / n:.4f}",
                        f"{(bb + mo) / n:.4f}", f"{(bb + lo) / n:.4f}",
                        sub["cap"], sub["within_rep"], sub["cross_rep"],
                        sub.most_common(1)[0][0] if sub else ""])

    # ---- incoherent_score_gap.csv
    gpath = os.path.join(a.out, "analysis", "incoherent_score_gap.csv")
    with open(gpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["wave", "arm", "mean_despair_mech_incoherent", "n_incoherent_scored",
                    "mean_despair_mech_coherent", "n_coherent_scored", "gap_incoherent_minus_coherent"])
        for (wave, arm), rows in sorted(by_arm.items()):
            inc = [r["despair_score"] for r in rows if r["mech_incoherent"] and not math.isnan(r["despair_score"])]
            coh = [r["despair_score"] for r in rows if not r["mech_incoherent"] and not math.isnan(r["despair_score"])]
            mi = sum(inc) / len(inc) if inc else float("nan")
            mc = sum(coh) / len(coh) if coh else float("nan")
            w.writerow([wave, arm, fmt(mi), len(inc), fmt(mc), len(coh),
                        fmt(mi - mc) if inc and coh else ""])

    # ---- subrules.csv (all waves, incl. the two with no blind label)
    spath = os.path.join(a.out, "analysis", "subrules.csv")
    with open(spath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["wave", "n", "mech_incoherent", "mech_incoherent_rate",
                    "cap_fires", "within_rep_fires", "cross_rep_fires", "top_subrule",
                    "coherent_n", "coherent_but_unscored"])
        for wave in list(waves) + ["ALL"]:
            rows = all_rows if wave == "ALL" else [r for r in all_rows if r["wave"] == wave]
            sub = Counter()
            for r in rows:
                for key, col in (("cap", "mech_cap"), ("within_rep", "mech_within_rep"),
                                 ("cross_rep", "mech_cross_rep")):
                    if r[col]:
                        sub[key] += 1
            mi = sum(1 for r in rows if r["mech_incoherent"])
            coh = [r for r in rows if not r["mech_incoherent"]]
            w.writerow([wave, len(rows), mi, f"{mi / len(rows):.4f}",
                        sub["cap"], sub["within_rep"], sub["cross_rep"],
                        sub.most_common(1)[0][0] if sub else "",
                        len(coh), sum(1 for r in coh if math.isnan(r["despair_score"]))])

    with open(os.path.join(a.out, "analysis", "inventory.json"), "w", encoding="utf-8") as fh:
        json.dump({"waves": inventory,
                   "rule": {"cap_finish_reason": CAP_FINISH,
                            "distinct_10gram_threshold": D10_THRESHOLD,
                            "min_turn_tokens_for_d10": D10_MIN_TOKENS,
                            "min_paragraph_tokens": PARA_MIN_TOKENS},
                   "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "cluster": "scenario_id"}},
                  fh, indent=2)

    make_figures(arm_out, all_rows, a.out)
    print("wrote", cpath, apath, rpath, gpath, file=sys.stderr)


# ------------------------------------------------------------- figures ----
def make_figures(arm_out, all_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    TITLE = {"semantic_mask": "Named 29-token masks of row 11 (semantic-mask wave)",
             "dose_placebo": "Prefix doses and arbitrary 29-token windows (dose-placebo wave)",
             "row4_component_map": "Pieces of the row-4 refusal response (component-map wave)",
             "compression_union": "Shortened carriers and their unions (compression-union wave)"}
    SURFACE = "#fcfcfb"
    INK, MUTED, GRID = "#1f2933", "#52606d", "#e3e8ee"
    TRAINED, STOCK = "#3b6ea5", "#c2410c"   # validated pair (dataviz six checks, light mode)

    by_wave = defaultdict(list)
    for r in arm_out:
        by_wave[r["wave"]].append(r)

    seed_means = defaultdict(list)
    for c in all_rows:
        if not c["mech_incoherent"] and not math.isnan(c["despair_score"]):
            seed_means[(c["wave"], c["arm"], c["seed"])].append(c["despair_score"])

    def panel(ax_l, ax_r, wave, rows, show_xlabel=True):
        rows = sorted(rows, key=lambda r: (r["arm"] != "stock", -r["coherent_fraction"]))
        names = [r["arm"].replace("_", " ") for r in rows]
        y = list(range(len(rows)))[::-1]
        cols = [STOCK if r["arm"] == "stock" else TRAINED for r in rows]

        for ax in (ax_l, ax_r):
            ax.set_facecolor(SURFACE)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(length=0, colors=MUTED, labelsize=8)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.set_ylim(-0.75, len(rows) - 0.25)

        # left: coherent fraction (0-1, with a reserved label column to 1.34)
        cf = [r["coherent_fraction"] for r in rows]
        ax_l.barh(y, cf, color=cols, height=0.6)
        ax_l.errorbar(cf, y, fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.5,
                      xerr=[[max(0.0, r["coherent_fraction"] - r["coh_ci_lo"]) for r in rows],
                            [max(0.0, r["coh_ci_hi"] - r["coherent_fraction"]) for r in rows]])
        ax_l.set_yticks(y)
        ax_l.set_yticklabels(names, fontsize=8.5, color=INK)
        ax_l.set_xlim(0, 1.34)
        ax_l.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_l.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        if show_xlabel:
            ax_l.set_xlabel("share of conversations that stay coherent", fontsize=8.5, color=MUTED)
        for yy, r in zip(y, rows):
            ax_l.text(1.36, yy, f"{r['coherent_fraction']:.2f}   n={r['n']}",
                      va="center", ha="right", fontsize=7.5, color=MUTED)

        # right: despair among the coherent ones (0-10, label column to 13.4)
        dm = [r["despair_mean_coherent"] for r in rows]
        plot_dm = [0.0 if math.isnan(v) else v for v in dm]
        ax_r.barh(y, plot_dm, color=cols, height=0.6)
        ax_r.errorbar(plot_dm, y, fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.5,
                      xerr=[[0.0 if math.isnan(r["despair_coh_ci_lo"]) else max(0.0, r["despair_mean_coherent"] - r["despair_coh_ci_lo"]) for r in rows],
                            [0.0 if math.isnan(r["despair_coh_ci_hi"]) else max(0.0, r["despair_coh_ci_hi"] - r["despair_mean_coherent"]) for r in rows]])
        for yy, r in zip(y, rows):
            for vals in [v for k, v in seed_means.items() if k[0] == wave and k[1] == r["arm"]]:
                ax_r.plot(sum(vals) / len(vals), yy, "x", color=INK, ms=4.5, mew=1.0, zorder=5)
        ax_r.set_yticks(y)
        ax_r.set_yticklabels([])
        ax_r.set_xlim(0, 13.4)
        ax_r.set_xticks([0, 2, 4, 6, 8, 10])
        if show_xlabel:
            ax_r.set_xlabel("despair score among the coherent ones (0-10)", fontsize=8.5, color=MUTED)
        for yy, v, r in zip(y, dm, rows):
            label = "none left to score" if math.isnan(v) else f"{v:.2f}   n={r['despair_coh_n']}"
            ax_r.text(13.4, yy, label, va="center", ha="right", fontsize=7.5, color=MUTED)

    legend = [Patch(facecolor=STOCK, label="untrained stock model"),
              Patch(facecolor=TRAINED, label="trained arm")]

    for wave, rows in by_wave.items():
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 0.40 * len(rows) + 2.5))
        fig.patch.set_facecolor(SURFACE)
        panel(axes[0], axes[1], wave, rows)
        fig.suptitle(TITLE[wave], fontsize=12, x=0.012, ha="left", color=INK)
        fig.legend(handles=legend, loc="upper right", frameon=False, fontsize=8,
                   ncol=2, bbox_to_anchor=(0.99, 1.0))
        fig.text(0.012, 0.012,
                 "Coherent = a fixed mechanical rule: no turn hit the length cap, no turn repeated a 10-word run, "
                 "no paragraph repeated across turns.\n"
                 "Whiskers are 95% intervals from resampling the 12 scenarios; x marks are per-training-seed means. "
                 "The two panels are separate outcomes and are never combined.",
                 fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=[0, 0.085, 1, 0.93])
        fig.savefig(os.path.join(out, "figures", f"{wave}.png"), dpi=170, facecolor=SURFACE)
        plt.close(fig)

    order = ["semantic_mask", "dose_placebo", "row4_component_map", "compression_union"]
    heights = [len(by_wave[w]) for w in order]
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 0.40 * sum(heights) + 6.4),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.55})
    fig.patch.set_facecolor(SURFACE)
    for i, w in enumerate(order):
        panel(axes[i][0], axes[i][1], w, by_wave[w], show_xlabel=(i == 3))
        axes[i][0].set_title(TITLE[w], fontsize=10, loc="left", color=INK, pad=8)
    fig.suptitle("Mechanical coherence re-cut across four stored waves", fontsize=13, x=0.012, ha="left", color=INK)
    fig.legend(handles=legend, loc="upper right", frameon=False, fontsize=8, ncol=2,
               bbox_to_anchor=(0.99, 1.0))
    fig.text(0.012, 0.008,
             "Left: how much of each arm survives the fixed mechanical rule. Right: the stored despair score over "
             "only those surviving conversations.\n"
             "Whiskers are 95% intervals from resampling the 12 scenarios; x marks are per-training-seed means.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    fig.savefig(os.path.join(out, "figures", "all_waves.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
