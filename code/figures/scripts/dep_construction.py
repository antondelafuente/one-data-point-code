"""dep_construction (appendix): how often each way of building a single training row produced a working row.

Population and method labels: the construction tally, explore-depv1-page-v4-data-1/tally/tally.json (432 candidate
rows, each counted once at its first training run, seed 4; method precedence adaptive > edit > fresh > other).
Row-level scores are re-read from the raw per-record tables, not from the tally's copy:
  - 369 rows from explore-depv1-generator-hit-rates-1/analysis/rows.csv (maps 17 records' result files),
  - 63 later rows from each record's own table (ingredient-ladder-2 and onmodel-rows rounds/r1/results.csv,
    sol-recipe-sanity/-retry analysis/row_scores.csv),
and asserted equal to the tally's first-attempt values.
Hit (the thread's historical screen): v4 despair mean over mechanically coherent conversations (frozen degen.py)
>= 4.0 on >= 12 of the 36 conversations. Left: hit share per method with Wilson 95% intervals. Right: one dot per
row (first-attempt mean), hits filled red; hollow = fewer than 12 coherent conversations (cannot hit).
"""
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import figstyle  # noqa: E402

figstyle.setup()

RL = os.path.dirname(os.environ.get("ODP_REGISTRY", "/path/to/registry").rstrip("/"))  # research repo root; see README
TALLY = f"{RL}/registry/explore-depv1-page-v4-data-1/tally/tally.json"
HITRATES = f"{RL}/registry/explore-depv1-generator-hit-rates-1/analysis/rows.csv"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dep", "dep_construction")
HIT, MIN_N = 4.0, 12
METHODS = [("fresh_from_recipe", "written\nfresh"),
           ("edit_of_known_working", "edit of\nworking\nrow"),
           ("adaptive_hill_climb", "adaptive\nsearch\n(hill climb)"),
           ("other", "edit of\nknown\nmiss")]


def fnum(v):
    return None if v in ("", None) else float(v)


def raw_score(cand):
    """(mean, n) over mechanically coherent conversations for the candidate's first attempt, from the raw table."""
    fa = cand["first_attempt"]
    if cand["in_generator_hit_rates_1"]:
        r = HR[(fa["record"], cand["row_id"], str(fa["seed"]))]
        return fnum(r["judge_mean_mech_pass"]), int(r["mech_pass_n"] or 0)
    src = f"{RL}/{fa['source']}"
    if src not in TABLES:
        TABLES[src] = {r["id"]: r for r in csv.DictReader(open(src))}
    r = TABLES[src][cand["row_id"]]
    if "mech_mean" in r:  # sol records' row_scores.csv
        return fnum(r["mech_mean"]), int(r["mech_scored_n"] or 0)
    return fnum(r["judge_mean_mech_pass"]), int(r["mech_pass_n"] or 0)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


tally = json.load(open(TALLY))
HR = {(r["record"], r["arm_id"], r["seed"]): r for r in csv.DictReader(open(HITRATES))}
TABLES = {}

rows = defaultdict(list)  # method -> [(mean or None, n)]
for c in tally["candidates"]:
    m, n = raw_score(c)
    fa = c["first_attempt"]
    assert (m is None) == (fa["mech_mean"] is None) and (m is None or abs(m - fa["mech_mean"]) < 1e-3), c["row_id"]
    assert n == (fa["mech_n"] or 0), c["row_id"]
    rows[c["method"]].append((m, n))

stats = {}
for k, _ in METHODS:
    v = rows[k]
    hits = sum(m is not None and m >= HIT and n >= MIN_N for m, n in v)
    stats[k] = (hits, len(v))
    assert hits == tally["summary_by_method"][k]["orig_hits_ge4_on_ge12"]
    assert len(v) == tally["summary_by_method"][k]["n_candidates"]
allh, alln = sum(h for h, _ in stats.values()), sum(n for _, n in stats.values())
for k, lab in METHODS:
    h, n = stats[k]
    lo, hi = wilson(h, n)
    scored = [m for m, _ in rows[k] if m is not None]
    print(f"{k:22s} {h:3d}/{n:3d} = {h / n:5.1%}  Wilson [{lo:.1%}, {hi:.1%}]  scored {len(scored)}  "
          f"no-coherent {n - len(scored)}  mean {sum(scored) / len(scored):.2f}")
lo, hi = wilson(allh, alln)
print(f"{'all methods pooled':22s} {allh:3d}/{alln:3d} = {allh / alln:5.1%}  Wilson [{lo:.1%}, {hi:.1%}]")

# ---------------------------------------------------------------- plot
fig, (a1, a2) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.6), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.28})

xs = list(range(len(METHODS)))
for x, (k, _) in zip(xs, METHODS):
    h, n = stats[k]
    lo, hi = wilson(h, n)
    a1.bar(x, h / n, width=0.55, color=figstyle.TRAINED, zorder=2)
    a1.plot([x, x], [lo, hi], color="0.15", lw=0.8, zorder=3)
    a1.text(x, hi + 0.015, f"{h}/{n}", ha="center", va="bottom", fontsize=6.5)
xa = len(METHODS) + 0.35
lo, hi = wilson(allh, alln)
a1.bar(xa, allh / alln, width=0.62, color=figstyle.PALE, zorder=2)
a1.plot([xa, xa], [lo, hi], color="0.15", lw=0.8, zorder=3)
a1.text(xa, hi + 0.015, f"{allh}/{alln}", ha="center", va="bottom", fontsize=6.5)
a1.axvline(len(METHODS) - 0.4 + 0.175, color="0.8", lw=0.6)
a1.set_xticks(xs + [xa])
a1.set_xticklabels([lab for _, lab in METHODS] + ["all\npooled"], fontsize=6)
a1.tick_params(axis="x", length=0)
a1.set_ylim(0, 0.6)
a1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
a1.set_yticklabels(["0", "10%", "20%", "30%", "40%", "50%", "60%"])
a1.set_ylabel("rows that hit on first training run")

# right: per-row strip
anchor = tally["meta"]["anchor_mech"]
pts = [p[0] for p in tally["meta"]["anchor_mech_points"]]
a2.axhspan(min(pts), max(pts), color=figstyle.STOCK, alpha=0.18, lw=0, zorder=0)
a2.axhline(anchor, color=figstyle.STOCK, lw=0.7, ls=":", zorder=1)
a2.axhline(HIT, color="0.35", lw=0.6, ls=(0, (4, 3)), zorder=1)
a2.text(len(METHODS) - 0.4, HIT, "hit\nline", fontsize=6, color="0.35", ha="left", va="center", clip_on=False)
a2.text(len(METHODS) - 0.4, anchor, "stock\n(other\nsessions)", fontsize=6, color="0.4", ha="left", va="center",
        clip_on=False)
rng = random.Random(20261012)
for x, (k, _) in zip(xs, METHODS):
    for m, n in rows[k]:
        if m is None:
            continue
        hit = m >= HIT and n >= MIN_N
        jx = x + rng.uniform(-0.28, 0.28)
        col = figstyle.ACCENT if hit else figstyle.TRAINED
        a2.scatter(jx, m, s=6, linewidths=0.5, zorder=3 if hit else 2, alpha=0.9 if hit else 0.55,
                   facecolors="white" if n < MIN_N else col, edgecolors=col)
    scored = [m for m, _ in rows[k] if m is not None]
    mu = sum(scored) / len(scored)
    a2.plot([x - 0.34, x + 0.34], [mu, mu], color="0.1", lw=1.4, zorder=4)
a2.set_xticks(xs)
a2.set_xticklabels([lab for _, lab in METHODS], fontsize=6)
a2.tick_params(axis="x", length=0)
a2.set_xlim(-0.55, len(METHODS) - 0.45)
a2.set_ylim(0, 9)
a2.set_ylabel("despair, first training run (0–10)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
figstyle.save(fig, OUT)
print("wrote", OUT + ".{pdf,png}")
