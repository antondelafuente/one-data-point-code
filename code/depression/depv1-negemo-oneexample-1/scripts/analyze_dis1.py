#!/usr/bin/env python3
"""DIS1 analysis — the locked Presentation spec's datasets, and nothing else.

Reads ONLY this-experiment-local committed inputs (#458): the merged rollouts, the
judge verdicts, the local join table, the training censuses, and the pinned
scenario-set manifests. Emits exactly the four artifacts DESIGN.md pins, on its
pinned schemas, plus the amendments' required additions.

  figures/subject_grid.csv   {subject x seed x battery x scenario_set x view}
  figures/ladder.csv         {P1 half x budget x seed|pooled x battery}
  figures/membership_map.csv the 13 rows x 3 partitions, from partitions_receipt.json
  figures/scenario_profile.csv  EXPLORATORY per-cell x scenario means (side-channel)
  figures/escalation.csv     {arm x turn}, contradiction battery, seeds pooled
  figures/health.csv         per subject, + `judge_refusal_rate` (amendment A2)
  figures/contrasts.json     paired scenario-cluster bootstrap, B=4000

Method notes, all from DESIGN.md:
  * canonical metric = judged mean negative emotion (0-10), FULL view (12k cap);
    prefix-640 sits beside it as the matched-length co-metric.
  * CIs are 95% SCENARIO-CLUSTER bootstrap: the resampling unit is the scenario,
    never the row, because turns/draws within a scenario are not independent.
  * seeds are reported separately and never pooled for the primary read; the
    arm-level bar is emitted as an extra `seed=pooled` row computed by
    concatenating both seeds' per-scenario rows (stock as-is).
  * PRIMARY scenario sets are protected19 (rejection) + contradiction (F1);
    full39 and dev20 are emitted as secondary/diagnostic rows.
  * contrasts are PAIRED: one scenario resample is shared by both sides.

BUG FIXED MID-RUN (amendment DIS1-A2): the first build of this wave's `PAIRS` table
covered only the DESIGN's four named comparison families, so `int13 vs stock` — which
is the ANCHOR GATE's contrast, int13 being this wave's comparator rather than one of
the ten arm-cells — was never computed, and `check_anchor_gate.py` read the absent
record as a failing CI. `PAIRS` now carries it explicitly, labelled as the anchor-gate
contrast. See DESIGN_AMENDMENTS.md DIS1-A2.

Amendment A2 additions:
  * `judge_refusal_rate` per subject in health.csv;
  * a COMMON-VALID-KEY sensitivity view beside every headline whenever compared
    subjects lose different keys — mandatory if any refusals survive into the wave.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dis1_common as C  # noqa: E402

B = 4000
RNG_SEED = 20260807       # fixed so the committed CSVs reproduce byte-identically
SCENARIO_SETS = ("protected19", "full39", "dev20", "contradiction")
# The floor, the co-measured comparator (int13 — BOTH the anchor gate AND a real
# comparison arm in this wave, unlike the parent's gate-only top64 pair), then the
# ten arm-cells: six 1x half-corpora and the four P1 ladder cells.
ARMS_ORDER = ["stock", "int13",
              "p1a", "p1b", "p2a", "p2b", "p3a", "p3b",
              "p1a-2x", "p1b-2x", "p1a-4x", "p1b-4x"]
LADDER_BY_ARM = {"p1a": 141, "p1b": 141,
                 "p1a-2x": 282, "p1b-2x": 282, "p1a-4x": 564, "p1b-4x": 564}
HALF_ARMS = ["p1a", "p1b", "p2a", "p2b", "p3a", "p3b"]
CELL_ARMS = HALF_ARMS + ["p1a-2x", "p1b-2x", "p1a-4x", "p1b-4x"]


def arm_of(subject: str) -> str:
    return "stock" if subject == "stock" else subject.rsplit("-s", 1)[0]


def seed_of(subject: str) -> str:
    return "stock" if subject == "stock" else subject.rsplit("-s", 1)[1]


def cluster_bootstrap(by_scen: dict[str, list[float]], rng) -> tuple[float, float, float]:
    """95% scenario-cluster bootstrap of the mean. Resamples SCENARIOS, not rows."""
    keys = sorted(by_scen)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    vals = [np.asarray(by_scen[k], dtype=float) for k in keys]
    sums = np.array([v.sum() for v in vals])
    ns = np.array([v.size for v in vals])
    point = float(sums.sum() / max(ns.sum(), 1))
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    bs = sums[idx].sum(axis=1)
    bn = ns[idx].sum(axis=1)
    draws = np.divide(bs, bn, out=np.full(B, np.nan), where=bn > 0)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap(a: dict[str, list[float]], b: dict[str, list[float]],
                     rng) -> dict[str, Any]:
    """Paired scenario-cluster bootstrap of (mean_a - mean_b) on shared scenarios."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n_scenarios": 0, "delta": None, "ci_lo": None, "ci_hi": None}
    sa = np.array([np.sum(a[k]) for k in keys], float)
    na = np.array([len(a[k]) for k in keys], float)
    sb = np.array([np.sum(b[k]) for k in keys], float)
    nb = np.array([len(b[k]) for k in keys], float)
    point = float(sa.sum() / na.sum() - sb.sum() / nb.sum())
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    da = sa[idx].sum(axis=1) / np.maximum(na[idx].sum(axis=1), 1e-9)
    db = sb[idx].sum(axis=1) / np.maximum(nb[idx].sum(axis=1), 1e-9)
    draws = da - db
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_scenarios": len(keys), "delta": round(point, 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def load_ratings(join: list[dict], verdicts: list[dict]) -> tuple[dict, dict, dict]:
    """(full, prefix640, coherence) maps: judge_row_id -> rating, plus join meta."""
    meta = {j["judge_row_id"]: j for j in join}
    views: dict[str, dict[str, float]] = {"full": {}, "prefix640": {}, "coherence": {}}
    status: dict[str, str] = {}
    for v in verdicts:
        status[v["judge_row_id"]] = v.get("parse_status")
        if v.get("parse_status") == "valid" and v.get("rating") is not None:
            views.setdefault(v["view"], {})[v["judge_row_id"]] = float(v["rating"])
    return meta, views, status


def scenario_sets(work_inputs: Path) -> dict[str, set[str]]:
    dev = {l.strip() for l in (work_inputs / "development.txt").read_text().splitlines()
           if l.strip()}
    prot = {l.strip() for l in (work_inputs / "validation.txt").read_text().splitlines()
            if l.strip()}
    return {"dev20": dev, "protected19": prot, "full39": dev | prot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--work-inputs", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--attempts-dir", type=Path,
                    help="rollout dir holding *.attempts.jsonl + *.ledger.jsonl — amendment "
                         "HS-A2 §3: per-cell re-roll counts and terminal-miss counts in "
                         "health.csv, so the reader can see the retry selection pressure")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quarantine-contentless-degenerate", action="store_true",
                    help="close-audit sensitivity: drop rows whose judged text is "
                         "contentless (no alphanumeric content or <10 chars) or "
                         "deterministically degenerate, then recompute everything. "
                         "Used to test whether any headline depends on generation "
                         "collapse rather than on the trait.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    join = C.read_jsonl(args.join)
    verdicts = C.read_jsonl(args.verdicts)
    rollouts = C.read_jsonl(args.rollouts)
    meta, views, status = load_ratings(join, verdicts)
    quarantined: set[str] = set()
    if args.quarantine_contentless_degenerate:
        bad_keys = {r["exact_key"] for r in rollouts
                    if _contentless(r.get("judge_visible_text"))
                    or _degenerate(r.get("judge_visible_text"))}
        quarantined = {jid for jid, m in meta.items() if m["exact_key"] in bad_keys}
        for v in views.values():
            for jid in list(v):
                if jid in quarantined:
                    del v[jid]
        print(f">>> QUARANTINE: {len(bad_keys)} rollout keys -> {len(quarantined)} judge rows "
              f"dropped from every view", flush=True)
    sets = scenario_sets(args.work_inputs)
    subjects = sorted({j["subject"] for j in join})

    # ---------- per-(subject, battery, scenario_set) rating clusters -----------
    def clusters(subject_filter, leg: str, sset: str | None, view: str,
                 restrict: set[str] | None = None) -> dict[str, list[float]]:
        out: dict[str, list[float]] = collections.defaultdict(list)
        for jid, m in meta.items():
            if m["leg"] != leg or not subject_filter(m["subject"]):
                continue
            if restrict is not None and m["exact_key"] not in restrict:
                continue
            if leg == "rejection" and sset and m["scenario_id"] not in sets[sset]:
                continue
            r = views.get(view, {}).get(jid)
            if r is None:
                continue
            out[m["scenario_id"]].append(r)
        return dict(out)

    # ---------- A2 / F5: which units are valid for EVERY compared subject ------
    # exact_key embeds the subject; the comparable unit across subjects is
    # (leg, item_id, turn, draw).
    #
    # The draw index needs care. The DESIGN deliberately gives stock 5 rejection
    # draws and every trained subject 3, so stock's draws 3-4 exist for NO trained
    # subject BY DESIGN. Intersecting raw units would therefore mark stock as
    # "losing" 2/5 of its rejection data in every run, make `differential_key_loss`
    # unconditionally true, and silently drop 40% of stock's rejection rows from the
    # common-key view — conflating a designed asymmetry with actual missingness,
    # which is the exact opposite of what this sensitivity view is for. So the
    # common-key basis is restricted to the PLANNED-for-everyone draw range first;
    # anything outside it is extra data living in the all_valid view only.
    max_common_draw = min(C.STOCK_DRAWS, C.SUBJECT_DRAWS, C.CONTRADICTION_DRAWS)

    def unit(m):
        return (m["leg"], m["item_id"], m["turn"], m["draw"])

    def in_common_basis(m) -> bool:
        return (m["leg"] in ("rejection", "contradiction")
                and int(m["draw"]) < max_common_draw)

    units_by_subject: dict[str, set] = collections.defaultdict(set)
    for jid, m in meta.items():
        if in_common_basis(m) and jid in views.get("full", {}):
            units_by_subject[m["subject"]].add(unit(m))
    common_units = (set.intersection(*units_by_subject.values())
                    if units_by_subject else set())
    common_keys = {m["exact_key"] for m in meta.values()
                   if in_common_basis(m) and unit(m) in common_units}
    # Differential loss now means what it should: subjects losing DIFFERENT keys
    # inside the range they were all planned to cover.
    differential = any(len(v) != len(common_units) for v in units_by_subject.values())

    # ------------------------------- subject_grid ------------------------------
    # DESIGN Presentation F6 schema: one row per {subject x battery x scenario_set
    # x view (full | prefix640)}. `subject` is the ARM label (the bar), `seed` is
    # 4 / 5 (the dots) or `pooled` (the bar). `basis` is an ADDITIVE column, not in
    # the pinned nine: `all_valid` is the headline, `common_valid_keys` is the
    # mandatory missingness sensitivity view beside it.
    rows_out = []

    def emit(subject, seed, battery, sset, sfilter, restrict=None, basis="all_valid"):
        for view in ("full", "prefix640"):
            cl = clusters(sfilter, battery, sset if battery == "rejection" else None,
                          view, restrict)
            if not cl:
                continue
            pt, lo, hi = cluster_bootstrap(cl, rng)
            rows_out.append({
                "subject": subject, "seed": seed, "battery": battery,
                "scenario_set": sset if battery == "rejection" else "contradiction",
                "view": view, "mean": round(pt, 4), "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "n_rollouts": sum(len(v) for v in cl.values()),
                "n_scenarios": len(cl), "basis": basis,
            })

    for arm in ARMS_ORDER:
        subs = [s for s in subjects if arm_of(s) == arm]
        if not subs:
            continue
        for battery, ssets in (("rejection", SCENARIO_SETS[:3]),
                               ("contradiction", (None,))):
            for sset in ssets:
                for s_ in subs:                      # per-seed rows (the primary read)
                    emit(arm, seed_of(s_), battery, sset, lambda x, s_=s_: x == s_)
                if arm != "stock":                   # pooled row = the arm-level bar
                    emit(arm, "pooled", battery, sset,
                         lambda x, subs=subs: x in subs)
                # A2 / F5: common-valid-key sensitivity beside the headline
                if differential:
                    emit(arm, "pooled" if arm != "stock" else "stock", battery, sset,
                         (lambda x, subs=subs: x in subs), common_keys,
                         basis="common_valid_keys")

    grid = args.out_dir / "subject_grid.csv"
    with grid.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["subject", "seed", "battery", "scenario_set",
                                          "view", "mean", "ci_lo", "ci_hi",
                                          "n_rollouts", "n_scenarios", "basis"])
        w.writeheader()
        w.writerows(rows_out)

    # ---------------------------------- ladder ---------------------------------
    # DESIGN Presentation 2 + F6: one row per {P1 half x budget x seed|pooled x
    # battery}. The lines are the pooled rows; the dots are the seed rows. The
    # BUDGET column names the bundle (duration + its stretched linear->0 schedule).
    ladder_rows = []
    for arm, budget in LADDER_BY_ARM.items():
        half = "p1a" if arm.startswith("p1a") else "p1b"
        for r in rows_out:
            if (r["subject"] == arm and r["view"] == "full"
                    and r["basis"] == "all_valid"
                    and r["scenario_set"] in ("protected19", "contradiction")):
                ladder_rows.append({
                    "half": half, "budget_updates": budget, "seed": r["seed"],
                    "battery": r["battery"], "scenario_set": r["scenario_set"],
                    "mean": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    with (args.out_dir / "ladder.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["half", "budget_updates", "seed", "battery",
                                          "scenario_set", "mean", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(sorted(ladder_rows,
                           key=lambda r: (r["half"], r["budget_updates"],
                                          r["battery"], str(r["seed"]))))

    # ------------------------- scenario profile (EXPLORATORY) -------------------
    # CHECKLIST side-channel: per-cell x scenario means. Labelled exploratory; it
    # never feeds subset selection in this wave.
    prof = []
    for arm in ARMS_ORDER:
        subs = {s for s in subjects if arm_of(s) == arm}
        if not subs:
            continue
        for battery, sset in (("rejection", "protected19"), ("rejection", "full39"),
                              ("contradiction", None)):
            cl = clusters(lambda x: x in subs, battery, sset, "full")
            for scen, vals in sorted(cl.items()):
                prof.append({"subject": arm, "battery": battery,
                             "scenario_set": sset or "contradiction",
                             "scenario_id": scen, "n": len(vals),
                             "mean_full": round(statistics.mean(vals), 4),
                             "exploratory": "yes"})
    with (args.out_dir / "scenario_profile.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["subject", "battery", "scenario_set",
                                          "scenario_id", "n", "mean_full", "exploratory"])
        w.writeheader()
        w.writerows(prof)

    # -------------------------------- escalation -------------------------------
    esc = []
    for arm in ARMS_ORDER:
        subs = {s for s in subjects if arm_of(s) == arm}
        by_turn = collections.defaultdict(list)
        for jid, m in meta.items():
            if m["leg"] != "contradiction" or m["subject"] not in subs:
                continue
            r = views.get("full", {}).get(jid)
            if r is not None:
                by_turn[int(m["turn"])].append(r)
        for t in sorted(by_turn):
            esc.append({"arm": arm, "turn": t, "n_valid": len(by_turn[t]),
                        "mean_full": round(statistics.mean(by_turn[t]), 4)})
    with (args.out_dir / "escalation.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["arm", "turn", "n_valid", "mean_full"])
        w.writeheader()
        w.writerows(esc)

    # ---------------------------------- health ---------------------------------
    roll_by_subject = collections.defaultdict(list)
    for r in rollouts:
        roll_by_subject[r["subject"]].append(r)
    receipts = {p.stem.split(".")[0]: json.loads(p.read_text())
                for p in args.census_dir.glob("*.census.json")}

    # --- amendment HS-A2 §3: retry-pressure + truncation columns -----------------
    # Three equalizing passes were run over the SAME rule (HS-A1 §2, HS-A2 §2), so the
    # reader has to be able to see, per subject, how much re-rolling it took and how
    # much of the population hit the generation cap in the first place.
    rerolls: dict[str, dict[str, int]] = collections.defaultdict(dict)
    misses: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    if args.attempts_dir:
        for p in args.attempts_dir.glob("*.attempts.jsonl"):
            subj = p.name.replace(".attempts.jsonl", "")
            c: collections.Counter = collections.Counter()
            for row in C.read_jsonl(p):
                if row.get("exact_key"):
                    c[row["exact_key"]] += 1
            rerolls[subj] = dict(c)
        for p in args.attempts_dir.glob("*.ledger.jsonl"):
            subj = p.name.replace(".ledger.jsonl", "")
            seen = {}
            for row in C.read_jsonl(p):
                seen[row["exact_key"]] = row
            for row in seen.values():
                misses[subj][row.get("state")] += 1

    health = []
    for s in subjects:
        rr = roll_by_subject.get(s, [])
        coh_ids = [jid for jid, m in meta.items()
                   if m["subject"] == s and m["leg"] == "battery"]
        coh = [views["coherence"][j] for j in coh_ids if j in views["coherence"]]
        neg_ids = [jid for jid, m in meta.items()
                   if m["subject"] == s and m["leg"] in ("rejection", "contradiction")]
        refused = sum(1 for j in neg_ids if status.get(j) == "refused")
        rej = [r for r in rr if r["leg"] == "rejection"]
        con = [r for r in rr if r["leg"] == "contradiction"]
        draws = C.STOCK_DRAWS if s == "stock" else C.SUBJECT_DRAWS
        cen = receipts.get(s, {})
        texts = [r.get("judge_visible_text") or "" for r in rr]
        health.append({
            "subject": s, "arm": arm_of(s), "seed": seed_of(s),
            "coherence_mean": round(statistics.mean(coh), 4) if coh else "",
            "coherence_n": len(coh),
            "collapse_frac": round(sum(1 for t in texts if len(t.strip()) < 10)
                                   / max(len(texts), 1), 5),
            "degen_frac": round(sum(1 for t in texts if _degenerate(t))
                                / max(len(texts), 1), 5),
            "clip12k_rate": round(sum(1 for t in texts if len(t) > 12_000)
                                  / max(len(texts), 1), 5),
            "coverage_rejection": round(len(rej) / max(132 * draws, 1), 5),
            "coverage_contradiction": round(len(con) / max(48 * C.CONTRADICTION_DRAWS, 1), 5),
            "judge_refusal_rate": round(refused / max(len(neg_ids), 1), 5),
            "judge_refusals": refused, "judge_rows": len(neg_ids),
            "rendered_tokens": cen.get("supervised_token_exposure", ""),
            "loss_tokens": cen.get("supervised_loss_token_exposure", ""),
            # --- HS-A2 §3 disclosure columns ---------------------------------
            # How often the generation cap was hit at all (the population this
            # instrument discards from), and how much re-rolling the three
            # equalizing passes actually spent on this subject.
            "truncation_rate": round(
                sum(1 for r in rr if str(r.get("finish_reason")) == "length")
                / max(len(rr), 1), 5),
            "malformed_parse_rate": round(
                sum(1 for r in rr if str(r.get("parse_termination")) == "malformed")
                / max(len(rr), 1), 5),
            "rows_needing_reroll": sum(1 for v in rerolls.get(s, {}).values() if v > 0),
            "reroll_attempts_total": sum(rerolls.get(s, {}).values()),
            "max_attempts_any_row": max(rerolls.get(s, {}).values(), default=0),
            "terminal_misses": (misses.get(s, {}).get("generation_failed", 0)
                                + misses.get(s, {}).get(
                                    "not_generated_dependency_generation_failed", 0)
                                + misses.get(s, {}).get("contentless_quarantined", 0)
                                + misses.get(s, {}).get(
                                    "not_generated_dependency_contentless", 0)),
            "context_infeasible": misses.get(s, {}).get("context_infeasible", 0),
        })
    with (args.out_dir / "health.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(health[0]))
        w.writeheader()
        w.writerows(health)

    # --------------------------------- contrasts -------------------------------
    def arm_clusters(arm, leg, sset, restrict=None, view="full"):
        subs = {s for s in subjects if arm_of(s) == arm}
        return clusters(lambda x: x in subs, leg, sset, view, restrict)

    # DESIGN "The comparison set the purpose reads (co-measured, one wave)":
    #   1. each of the 10 arm-cells vs stock                  -> the hot/cold read
    #   2. each split half vs int13 pooled                    -> the retention read
    #   3. within P1, each ladder cell vs its own 1x cell     -> the budget read
    #   4. within each split, a-half vs b-half                -> the concentration read
    #      (for P1 this delta IS the vector-prediction read; P2/P3's near-zero-
    #       contrast deltas are its in-wave comparison scale)
    # Nothing is contrasted against an arm not co-measured in THIS wave; the
    # cross-wave numbers (top16/top32/top64, the parent's own int13 values) stay
    # prose context in RESULTS and never enter contrasts.json.
    PAIRS = (
        # The ANCHOR-GATE contrast. NOT one of the four named comparison families
        # below — DESIGN "Anchor gate" (b) requires the paired delta(int13 pooled -
        # stock) CI on BOTH primary batteries, so the gate reads it from here.
        [("int13", "stock",
          "ANCHOR GATE (DESIGN 'Anchor gate' (b)): the ORDERING gate on the co-measured "
          "int13 re-serve pair vs the in-wave floor. Also the denominator the retention "
          "read (family 2) is expressed against.")]
        + [(c, "stock",
          f"1. hot/cold read: cell {c} vs the in-wave floor (F1: a confirmation on "
          f"the selection-adjacent batteries, not an independent instrument)")
         for c in CELL_ARMS]
        + [(h, "int13",
            f"2. retention read: how much of the co-measured int13 elevation half {h} "
            f"retains")
           for h in HALF_ARMS]
        + [(f"{h}-{tag}", h,
            f"3. budget read: P1 half {h} at {LADDER_BY_ARM[f'{h}-{tag}']} updates vs "
            f"its own 141-update cell — the independent variable is the BUDGET BUNDLE "
            f"(duration AND the stretched linear->0 schedule together), never "
            f"'number of visits' alone")
           for h in ("p1a", "p1b") for tag in ("2x", "4x")]
        + [("p1a", "p1b",
            "4. concentration read for P1 — and THIS delta is the vector-prediction "
            "read: the mass-matched HIGH-projection half vs the LOW half"),
           ("p2a", "p2b",
            "4. concentration read for P2 (vector contrast 9 — an in-wave comparison "
            "scale for P1's delta, not a vector test)"),
           ("p3a", "p3b",
            "4. concentration read for P3 (vector contrast 217 — an in-wave comparison "
            "scale for P1's delta, not a vector test)")]
        + [("p1a-2x", "p1b-2x",
            "4b. DESCRIPTIVE EXTENSION (not in the DESIGN's named enumeration): the P1 "
            "concentration/vector delta re-read at the 282-update budget"),
           ("p1a-4x", "p1b-4x",
            "4b. DESCRIPTIVE EXTENSION (not in the DESIGN's named enumeration): the P1 "
            "concentration/vector delta re-read at the 564-update budget")]
    )
    def seed_clusters(arm, seed, leg, sset, restrict=None):
        subs = {s for s in subjects if arm_of(s) == arm and seed_of(s) == seed}
        return clusters(lambda x: x in subs, leg, sset, "full", restrict)

    contrasts = []
    for a, b, note in PAIRS:
        for leg, sset in (("rejection", "protected19"), ("contradiction", None)):
            ca, cb = arm_clusters(a, leg, sset), arm_clusters(b, leg, sset)
            if not ca or not cb:
                continue
            rec = {"name": f"{a}_vs_{b}", "arm_a": a, "arm_b": b, "battery": leg,
                   "scenario_set": sset or "contradiction", "note": note,
                   "view": "full", "basis": "pooled", "B": B,
                   **paired_bootstrap(ca, cb, rng)}
            contrasts.append(rec)
            # PREFIX-640 paired contrast, beside EVERY headline (close-audit FINDING 2).
            # The design pinned prefix-640 as the matched-length co-metric, but the first
            # build emitted only prefix-640 MEANS in subject_grid.csv and left the paired
            # CONTRASTS full-view-only — so a reader could not check whether a separation
            # survives the length match without re-deriving it. It is the same estimator on
            # the same scenarios, just the other judged view; the judgments already exist,
            # so this costs no additional judging.
            pa = arm_clusters(a, leg, sset, view="prefix640")
            pb = arm_clusters(b, leg, sset, view="prefix640")
            if pa and pb:
                contrasts.append({
                    "name": f"{a}_vs_{b}", "arm_a": a, "arm_b": b, "battery": leg,
                    "scenario_set": sset or "contradiction",
                    "note": note + " | PREFIX-640 length-matched co-metric (DESIGN's "
                                   "matched-length view; the full view stays canonical, "
                                   "this is the check that a separation is not carried by "
                                   "response length)",
                    "view": "prefix640", "basis": "pooled", "B": B,
                    **paired_bootstrap(pa, pb, rng)})
            # PER-SEED paired contrasts. DESIGN: "seeds reported separately and never
            # pooled for the primary read" — a pooled contrast alone cannot show whether
            # an effect survives the seed-separated read, so each seed is contrasted
            # against the comparison arm's SAME seed (stock has one serving and is used
            # as-is for both).
            for seed in ("4", "5"):
                sa = seed_clusters(a, seed, leg, sset) if a != "stock" else ca
                sb = seed_clusters(b, seed, leg, sset) if b != "stock" else cb
                if not sa or not sb:
                    continue
                contrasts.append({
                    "name": f"{a}_vs_{b}", "arm_a": a, "arm_b": b, "battery": leg,
                    "scenario_set": sset or "contradiction",
                    "note": note + f" | PER-SEED read, seed {seed} (DESIGN: seeds are "
                                   f"primary, pooled is secondary)",
                    "view": "full", "basis": f"seed{seed}", "B": B,
                    **paired_bootstrap(sa, sb, rng)})
            if differential:
                contrasts.append({
                    "name": f"{a}_vs_{b}", "arm_a": a, "arm_b": b, "battery": leg,
                    "scenario_set": sset or "contradiction",
                    "note": note + " | A2/F5 common-valid-key sensitivity (secondary "
                                   "by construction; must be within-noise of all_valid)",
                    "view": "full", "basis": "common_valid", "B": B,
                    **paired_bootstrap(arm_clusters(a, leg, sset, common_keys),
                                       arm_clusters(b, leg, sset, common_keys), rng)})

    # ------------------------- missingness reconciliation ----------------------
    miss = {"differential_key_loss_across_subjects": differential,
            "common_key_basis": (
                f"units restricted to draw < {max_common_draw} (the draw range every "
                f"subject was PLANNED to cover); stock's extra rejection draws "
                f"{max_common_draw}..{C.STOCK_DRAWS - 1} are by-design extra data and "
                f"appear in the all_valid view only, never as 'missing'"),
            "n_common_units": len(common_units),
            "units_per_subject_in_common_basis":
                {k: len(v) for k, v in sorted(units_by_subject.items())},
            "low_draw_scenarios": _low_draw(meta, views, subjects)}

    (args.out_dir / "contrasts.json").write_text(json.dumps({
        "bootstrap": {"kind": "paired scenario-cluster", "B": B, "rng_seed": RNG_SEED,
                      "resampling_unit": "scenario"},
        "primary_scenario_sets": ["protected19 (rejection)", "contradiction"],
        "contrasts": contrasts,
        "missingness": miss,
        "turn1_sanity": _turn1(meta, views, subjects),
    }, indent=2) + "\n")
    print(json.dumps({"subject_grid_rows": len(rows_out), "escalation_rows": len(esc),
                      "health_rows": len(health), "contrasts": len(contrasts),
                      "differential_key_loss": differential}, indent=2))
    print(">>> analysis done", flush=True)


def _contentless(t: str) -> bool:
    """No alphanumeric content at all, or too short to carry any."""
    import re as _re
    s = (t or "").strip()
    return (not _re.search(r"[A-Za-z0-9\u0400-\u04ff\u4e00-\u9fff\u0900-\u0d7f]", s)
            or len(s) < 10)


def _degenerate(t: str) -> bool:
    """Deterministic degeneration flag: a long response that is mostly one repeat."""
    s = (t or "").strip()
    if len(s) < 200:
        return False
    w = s.split()
    return bool(w) and (len(set(w)) / len(w)) < 0.12


def _low_draw(meta, views, subjects) -> list[dict]:
    """F5: any scenario with < 2 of 3 valid draws in any subject, named."""
    cnt = collections.defaultdict(set)
    for jid, m in meta.items():
        if m["leg"] in ("rejection", "contradiction") and jid in views.get("full", {}):
            cnt[(m["subject"], m["leg"], m["item_id"], m["turn"])].add(m["draw"])
    out = []
    for (subj, leg, item, turn), draws in sorted(cnt.items()):
        if len(draws) < 2:
            out.append({"subject": subj, "battery": leg, "scenario_id": item,
                        "turn": turn, "n_valid_draws": len(draws)})
    return out


def _turn1(meta, views, subjects) -> dict:
    """Contradiction battery turn-1 is ~0 everywhere — the battery's known signature."""
    by_arm = collections.defaultdict(list)
    for jid, m in meta.items():
        if m["leg"] == "contradiction" and int(m["turn"]) == 1:
            r = views.get("full", {}).get(jid)
            if r is not None:
                by_arm[arm_of(m["subject"])].append(r)
    res = {a: round(statistics.mean(v), 4) for a, v in sorted(by_arm.items())}
    return {"mean_full_at_turn1_by_arm": res,
            "expectation": "~0 for every arm (the battery's known signature)",
            "violations": {a: v for a, v in res.items() if v > 0.5}}


if __name__ == "__main__":
    main()
