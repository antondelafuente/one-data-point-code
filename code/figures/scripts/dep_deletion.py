"""dep_deletion — what happens to despair when one ingredient is deleted from a working row.

Data: registry/depv1-necessity-ladder-2 (21 wave rows, seeds 4/5/6, every deletion served in the same
session as its own FULL row), plus registry/depv1-necessity-ladder-1 (4 rows, same instruments, same
block rule) drawn as hollow extra points and kept out of every count.

Rawest committed inputs (the per-conversation despair scores live only in the R2 store,
remote:bucket/experiments/depv1-necessity-ladder-2/judge/ and analysis/stage1/conversations.csv):
  * rounds/r1/results.csv          one line per adapter: judge_mean_luna_pass (despair mean over
                                   luna-intact scored conversations), luna_*_n, serving_session_id
  * analysis/luna_verdicts.jsonl.txt (ladder-2) / analysis/luna_verdicts.json (ladder-1)
                                   one gpt-5.6-luna verdict per conversation; the broken rate is
                                   recomputed from these and asserted equal to results.csv.
Delta rule = the record's (scripts/analyze.py write_pairs_csv): per seed, cell − FULL on
judge_mean_luna_pass, same session; row value = mean over seeds where both sides have a score.
Broken-rate delta uses every seed present. The script asserts its row values against the record's
figures/necessity2_pairs.csv (level=row, cut=luna).

Run: python3 figures/scripts/dep_deletion.py   (system python3 + matplotlib; no network)
"""
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

figstyle.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
REG = os.environ.get("ODP_REGISTRY", "/path/to/registry")  # copy of the research records; see README
NL2 = os.path.join(REG, "depv1-necessity-ladder-2")
NL1 = os.path.join(REG, "depv1-necessity-ladder-1")
OUT = os.path.join(PAPER, "figures", "dep", "dep_deletion")

ING = ["noI", "noN", "noS", "noH"]
ING_LABEL = {"noI": "identity", "noN": "own\nstandards", "noS": "refusal", "noH": "harm to\nperson"}
CTRL = ["noCloser", "noAdvice", "noLegal"]
SEEDS = (4, 5, 6)
N_CONV = 36


def fnum(x):
    return None if x in (None, "") else float(x)


def load_adapters(results_csv):
    out = {}
    for r in csv.DictReader(open(results_csv)):
        out[(r["row"], r["cell"], int(r["seed"]))] = {
            "id": r["id"], "row": r["row"], "cell": r["cell"], "family": r.get("family", ""),
            "score": fnum(r["judge_mean_luna_pass"]), "broken_n": int(r["luna_broken_n"]),
            "session": r["serving_session_id"]}
    return out


def broken_counts_from_verdicts(records):
    """subject id -> (n conversations, n broken), from the per-conversation luna verdicts."""
    tot, brk = Counter(), Counter()
    for v in records:
        subj = v["id"].split("|")[4]
        tot[subj] += 1
        brk[subj] += v["verdict"] == "broken"
    return tot, brk


def check_broken(adapters, verdicts):
    tot, brk = broken_counts_from_verdicts(verdicts)
    for a in adapters.values():
        assert tot[a["id"]] == N_CONV, (a["id"], tot[a["id"]])
        assert brk[a["id"]] == a["broken_n"], (a["id"], brk[a["id"]], a["broken_n"])
        a["broken_rate"] = brk[a["id"]] / N_CONV


def row_deltas(adapters):
    """{(row, cell): dict(d, seed_d, br, n_pairs, family)} — same-session cell − FULL."""
    out = {}
    cells = sorted({(r, c) for (r, c, _s) in adapters if c != "FULL"})
    for row, cell in cells:
        ds, seed_d, brs = [], [], []
        for s in SEEDS:
            a, f = adapters.get((row, cell, s)), adapters.get((row, "FULL", s))
            if not a or not f:
                continue
            assert a["session"] == f["session"], f"cross-session pair {a['id']}"
            brs.append(a["broken_rate"] - f["broken_rate"])
            if a["score"] is not None and f["score"] is not None:
                ds.append(a["score"] - f["score"])
                seed_d.append(ds[-1])
        out[(row, cell)] = {"d": statistics.fmean(ds) if ds else None, "seed_d": seed_d,
                            "br": statistics.fmean(brs) if brs else None, "n_pairs": len(ds),
                            "family": adapters[(row, "FULL", 4)]["family"]}
    return out


# ------------------------------------------------------------------------------- load + check
nl2 = load_adapters(os.path.join(NL2, "rounds", "r1", "results.csv"))
check_broken(nl2, [json.loads(l) for l in open(os.path.join(NL2, "analysis", "luna_verdicts.jsonl.txt")) if l.strip()])
D2 = row_deltas(nl2)

nl1 = load_adapters(os.path.join(NL1, "rounds", "r1", "results.csv"))
check_broken(nl1, json.load(open(os.path.join(NL1, "analysis", "luna_verdicts.json"))))
D1 = row_deltas(nl1)

# cross-check against the record's own row-level lines
ref = {(r["row"], r["cell"]): r for r in csv.DictReader(open(os.path.join(NL2, "figures", "necessity2_pairs.csv")))
       if r["level"] == "row" and r["cut"] == "luna"}
for k, v in D2.items():
    r = ref[k]
    if v["d"] is None:
        assert r["delta"] == "", k
    else:
        assert abs(v["d"] - float(r["delta"])) < 1e-3, (k, v["d"], r["delta"])
        assert v["n_pairs"] == int(r["n_pairs"]), k
    assert abs(v["br"] - float(r["broken_delta"])) < 1e-3, (k, v["br"], r["broken_delta"])
ref1 = defaultdict(list)
for r in csv.DictReader(open(os.path.join(NL1, "figures", "necessity_pairs.csv"))):
    if r["cut"] == "luna" and r["delta"] != "":
        ref1[(r["row"], r["cell"])].append(float(r["delta"]))
for k, v in D1.items():
    if v["d"] is not None:
        assert abs(v["d"] - statistics.fmean(ref1[k])) < 1e-3, (k, v["d"])

# ------------------------------------------------------------------------------- summary
summary = {}
print("ladder-2 wave rows, luna cut")
for c in ING:
    vals = [v for (r, cc), v in D2.items() if cc == c]
    ds = [v["d"] for v in vals if v["d"] is not None]
    brs = [v["br"] for v in vals if v["br"] is not None]
    summary[c] = dict(lower=sum(d < 0 for d in ds), n=len(ds), median=statistics.median(ds),
                      br_median=100 * statistics.median(brs), br_mean=100 * statistics.fmean(brs), br_n=len(brs),
                      thin=sum(1 for v in vals if 0 < v["n_pairs"] < 3))
    s = summary[c]
    print(f"  {c}: lower {s['lower']}/{s['n']}  median d {s['median']:+.2f}  "
          f"broken median {s['br_median']:+.1f} pp mean {s['br_mean']:+.1f} pp ({s['br_n']} rows)  "
          f"rows on <3 seeds {s['thin']}")
fam_med = defaultdict(dict)
for c in ING:
    by = defaultdict(list)
    for (r, cc), v in D2.items():
        if cc == c and v["d"] is not None:
            by[v["family"]].append(v["d"])
    for f, ds in by.items():
        fam_med[f][c] = (round(statistics.median(ds), 2), len(ds))
print("family medians (luna):")
for f, m in fam_med.items():
    print("  ", f, m)
ctrl2 = {k: v for k, v in D2.items() if k[1] in CTRL}
ctrl1 = {k: v for k, v in D1.items() if k[1] in CTRL}
print("controls ladder-2:", {k: (round(v["d"], 2), round(100 * v["br"], 1)) for k, v in ctrl2.items()})
print("controls ladder-1:", {k: (round(v["d"], 2), round(100 * v["br"], 1)) for k, v in ctrl1.items()})
print("ladder-1 rows:", {k: (round(v["d"], 2), round(100 * v["br"], 1)) for k, v in D1.items() if v["d"] is not None})

# ------------------------------------------------------------------------------- figure
COLS = ING + ["ctrl"]
XLAB = [ING_LABEL[c] for c in ING] + ["length\ncontrol"]


def swarm(vals, width=0.42, step=0.13):
    """Deterministic 1-D beeswarm offsets for values in data units."""
    order = np.argsort(vals)
    offs = np.zeros(len(vals))
    placed = []
    for i in order:
        v = vals[i]
        k = 0
        while True:
            cand = [0.0] if k == 0 else [k * 0.095, -k * 0.095]
            ok = None
            for o in cand:
                if abs(o) <= width and all(abs(o - po) > 0.09 or abs(v - pv) > step for pv, po in placed):
                    ok = o
                    break
            if ok is not None:
                offs[i] = ok
                placed.append((v, offs[i]))
                break
            k += 1
            if k * 0.095 > width:
                offs[i] = 0.0
                placed.append((v, 0.0))
                break
    return offs


def points(col, D, key):
    if col == "ctrl":
        return [(k, v) for k, v in D.items() if k[1] in CTRL and v[key] is not None]
    return [(k, v) for k, v in D.items() if k[1] == col and v[key] is not None]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.75), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32})
rng_seed = np.random.default_rng(0)

for ax, key, scale in ((ax1, "d", 1.0), (ax2, "br", 100.0)):
    ax.axhline(0, color="#555555", lw=0.6, zorder=1)
    if key == "d":
        ax.axhspan(-0.906, 0.906, color=figstyle.STOCK, alpha=0.13, lw=0, zorder=0)
    for xi, col in enumerate(COLS):
        is_ctrl = col == "ctrl"
        p2 = points(col, D2, key)
        p1 = points(col, D1, key)
        vals2 = np.array([v[key] * scale for _k, v in p2])
        vals1 = np.array([v[key] * scale for _k, v in p1])
        allv = np.concatenate([vals2, vals1]) if len(vals1) else vals2
        offs = swarm(allv, step=(0.24 if key == "d" else 4.4)) if len(allv) else []
        o2, o1 = offs[:len(vals2)], offs[len(vals2):]
        # faint per-seed deltas (despair panel only), behind the row means
        if key == "d":
            for (k, v), o in zip(p2, o2):
                for sd in v["seed_d"]:
                    ax.plot(xi + o + rng_seed.uniform(-0.03, 0.03), sd, "o", ms=1.6, color=figstyle.PALE if not is_ctrl else "#dddddd",
                            zorder=2, mew=0)
        main_c = figstyle.STOCK if is_ctrl else figstyle.TRAINED
        for (k, v), o, y in zip(p2, o2, vals2):
            thin = key == "d" and v["n_pairs"] < 3
            ax.plot(xi + o, y, "o", ms=3.4, mfc=(main_c if not thin else "white"), mec=main_c, mew=0.8,
                    alpha=0.95, zorder=4)
        for (k, v), o, y in zip(p1, o1, vals1):
            ax.plot(xi + o, y, "D", ms=3.2, mfc="white", mec=figstyle.ALT, mew=0.9, zorder=5)
        if not is_ctrl and len(vals2):
            med = float(np.median(vals2))
            ax.plot([xi - 0.36, xi + 0.36], [med, med], color=figstyle.ACCENT, lw=1.3, zorder=6,
                    solid_capstyle="butt")
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(XLAB, fontsize=6.3)
    ax.set_xlim(-0.6, len(COLS) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.get_xticklabels()[-1].set_color("#777777")

ax1.set_ylabel("Δ despair (0–10), deletion − full row")
ax2.set_ylabel("Δ broken conversations (pp)")
ax1.set_ylim(-6.6, 2.6)
ax2.set_ylim(-50, 90)
ax1.text(-0.55, 0.95, "±0.9 cross-session noise", fontsize=5.5, color="#777777", ha="left", va="bottom")
for xi, c in enumerate(ING):
    s = summary[c]
    ax1.text(xi, 2.15, f"{s['lower']}/{s['n']}\nlower", ha="center", va="center", fontsize=6.3, color="#333333")
    ax2.text(xi, 82, f"{s['br_median']:+.0f} pp", ha="center", va="center", fontsize=6.3, color="#333333")
ax1.set_title("a   Despair on intact conversations", loc="left", fontsize=8)
ax2.set_title("b   How often the student breaks", loc="left", fontsize=8)

handles = [
    Line2D([], [], marker="o", ls="", color=figstyle.TRAINED, ms=3.4, label="one working row (mean over seeds 4/5/6)"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=figstyle.TRAINED, ms=3.4, label="row on 1–2 scorable seeds"),
    Line2D([], [], marker="o", ls="", color=figstyle.PALE, ms=2, label="single seed"),
    Line2D([], [], marker="D", ls="", mfc="white", mec=figstyle.ALT, ms=3.2, label="earlier 4-row ladder (not counted)"),
    Line2D([], [], color=figstyle.ACCENT, lw=1.3, label="median over rows"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, -0.06),
           handletextpad=0.3, columnspacing=1.0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
figstyle.save(fig, OUT)
print("wrote", OUT + ".pdf", OUT + ".png")
