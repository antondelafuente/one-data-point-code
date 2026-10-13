#!/usr/bin/env python3
"""Close-out summary for explore-depv1-ingredient-ladder-1.

Reads every rounds/rN/ that has a results.csv (so it works mid-record and again
after round 3 lands), applies the LANES.md classification exactly as
scripts/select_round.py does, and writes:

  analysis/summary.md   -- human-readable tables
  analysis/summary.json -- the same content, machine-readable
  figures/yield_by_round.{png,svg}
  figures/best_rows.png

Nothing here is a verdict. Every table describes what the rows did; the reader
decides what it means.

Run:  python3 scripts/summarize_record.py            (from anywhere)
      python3 scripts/summarize_record.py --no-figures
"""
import argparse
import csv
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUNDS = ROOT / "rounds"
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"

# Lanes in the order LANES.md introduces them; Okabe-Ito colour-blind-safe hues.
LANE_ORDER = ["A_walk", "B_duty", "C_free", "D_decline"]
LANE_COLOUR = {
    "A_walk": "#E69F00",
    "B_duty": "#56B4E9",
    "C_free": "#009E73",
    "D_decline": "#D55E00",
}
SMOKE_COLOUR = "#9A9A9A"   # round-0 reference row, outside the four lanes
# Same-session anchors, carried as reference lines only (not thresholds).
STOCK_REF = 1.94   # stock chat Qwen, frame-ablation session
V01_REF = 6.29     # v01 tenant paraphrase, smoke round (judge_mean_all 6.286)

BATTERY_N = 36     # 12 scenarios x 3 draws
HIT_MEAN = 4.0
HIT_COHERENT = 12
NEAR_MEAN = 3.0
TOP_N = 15
TOP_MIN_COHERENT = 20
QUOTE_CHARS = 200


# --------------------------------------------------------------------------
# classification -- lifted from scripts/select_round.py so the two agree
# --------------------------------------------------------------------------
SELF_CONTEMPT = {
    "entity_self_contempt",
    "self_contempt",
    "entity-level self-contempt",
    "entity_level_self_contempt",
}
_SC_NORM = {s.lower().replace(" ", "_").replace("-", "_") for s in SELF_CONTEMPT}


def fnum(x):
    """float() that returns nan for blanks/garbage (select_round.py's `f`)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def nan_to_none(v):
    """JSON has no NaN; a blank mech-mean (0 coherent conversations) stays null."""
    return None if isinstance(v, float) and math.isnan(v) else v


def is_self_contempt(label):
    return (label or "").strip().lower().replace(" ", "_").replace("-", "_") in _SC_NORM


def classify(row):
    """HIT / NEAR / COLD per LANES.md, plus the STRONG flag. Identical rule to
    select_round.py: reads are carried as labels, never a gate."""
    m, k = fnum(row.get("judge_mean_mech_pass")), fnum(row.get("mech_pass_n"))
    if m >= HIT_MEAN and k >= HIT_COHERENT:
        cls = "hit"
    elif m >= NEAR_MEAN and k >= HIT_COHERENT:
        cls = "near"
    else:
        cls = "cold"
    strong = cls == "hit" and (
        is_self_contempt(row.get("read_label_1")) or is_self_contempt(row.get("read_label_2"))
    )
    return cls, strong


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def clean(s, limit=None):
    """Collapse whitespace and neutralise markdown table separators."""
    s = re.sub(r"\s+", " ", (s or "")).strip().replace("|", "\\|")
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def parse_metered_cost(round_dir, n, ledger):
    """The 'Round N metered cost: **$X**' line from results.md; ledger fallback."""
    md = round_dir / "results.md"
    if md.exists():
        m = re.search(
            r"Round\s+%d\s+metered cost:\s*\*\*\$([0-9.,]+)\*\*" % n, md.read_text()
        )
        if m:
            return float(m.group(1).replace(",", "")), "results.md"
    entry = (ledger.get("rounds") or {}).get(str(n))
    if entry and entry.get("round_metered_usd") is not None:
        return float(entry["round_metered_usd"]), "cost_ledger.json"
    return None, None


def load_record():
    ledger = {}
    lp = ROUNDS / "cost_ledger.json"
    if lp.exists():
        ledger = json.loads(lp.read_text())

    rounds = []
    for d in sorted(ROUNDS.glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 1 << 30):
        if not d.is_dir() or not d.name[1:].isdigit():
            continue
        n = int(d.name[1:])
        csv_path = d / "results.csv"
        staged = 0
        cand_path = d / "candidates.jsonl"
        candidates = {}
        if cand_path.exists():
            for line in cand_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                candidates[c.get("arm_id")] = c
            staged = len(candidates)
        if not csv_path.exists():
            rounds.append(
                {"round": n, "dir": d, "scored": False, "rows": [], "candidates": candidates,
                 "staged": staged, "excluded": [], "cost_usd": None, "cost_source": None}
            )
            continue

        excluded = []
        fx = d / "filter_excluded.json"
        if fx.exists():
            excluded = [e.get("arm_id") for e in json.loads(fx.read_text()).get("excluded", [])]

        rows = []
        for r in csv.DictReader(open(csv_path)):
            r = dict(r)
            r["round"] = n
            r["excluded"] = r["id"] in excluded
            # A row is "trained" if it survived the negativity filter and reached the
            # battery. Rows whose conversations were ALL mechanically incoherent have a
            # blank judge_mean_mech_pass but a real n_scored -- they are trained, and cold.
            r["scored"] = not math.isnan(fnum(r.get("n_scored")))
            r["trained"] = r["scored"] and not r["excluded"]
            cls, strong = classify(r)
            r["class"] = cls if r["trained"] else "not_trained"
            r["strong"] = strong and r["trained"]
            cand = candidates.get(r["id"], {})
            r["prompt"] = cand.get("prompt", "")
            r["response"] = cand.get("response", "")
            if not r.get("parent"):
                r["parent"] = cand.get("parent", "")
            if not r.get("edit"):
                r["edit"] = cand.get("edit", "")
            rows.append(r)

        cost, source = parse_metered_cost(d, n, ledger)
        rounds.append(
            {"round": n, "dir": d, "scored": True, "rows": rows, "candidates": candidates,
             "staged": staged, "excluded": excluded, "cost_usd": cost, "cost_source": source}
        )
    return rounds, ledger


# --------------------------------------------------------------------------
# parent chains
# --------------------------------------------------------------------------
def resolve_parent(parent, known_ids):
    """Return the parent's arm_id if it is a row in this record, else None
    (roots look like 'tenant-ingredients-1/v01_charged' or 'none')."""
    p = (parent or "").strip()
    if not p or p.lower() == "none":
        return None
    tail = p.split("/")[-1].strip()
    if tail in known_ids:
        return tail
    if p in known_ids:
        return p
    return None


def chain_of(row, by_id, known_ids):
    """Walk `parent` back through the record. Returns (list of ancestor ids,
    root label as written in the data)."""
    seen, chain = {row["id"]}, []
    cur, root = row, (row.get("parent") or "none").strip()
    while True:
        pid = resolve_parent(cur.get("parent"), known_ids)
        if pid is None or pid in seen:
            root = (cur.get("parent") or "none").strip()
            break
        seen.add(pid)
        chain.append(pid)
        cur = by_id[pid]
    chain.reverse()
    return chain, root


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def sort_desc(x):
    """Descending sort key that pushes blanks (0 coherent conversations) to the end."""
    v = fnum(x)
    return 1e8 if math.isnan(v) else -v


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd=2):
    v = fnum(x)
    return "" if math.isnan(v) else f"{v:.{nd}f}"


def build_yield(rounds):
    """Per round x lane: trained, hits, near, hit-rate; plus cost."""
    table, cumulative = [], 0.0
    for R in rounds:
        if not R["scored"]:
            continue
        lanes = {}
        present = [l for l in LANE_ORDER if any(r["lane"] == l for r in R["rows"])]
        for l in sorted({r["lane"] for r in R["rows"]}, key=lambda l: (LANE_ORDER.index(l) if l in LANE_ORDER else 99, l)):
            rs = [r for r in R["rows"] if r["lane"] == l]
            trained = [r for r in rs if r["trained"]]
            hits = [r for r in trained if r["class"] == "hit"]
            near = [r for r in trained if r["class"] == "near"]
            lanes[l] = {
                "candidates": len(rs),
                "filter_excluded": sum(1 for r in rs if r["excluded"]),
                "trained": len(trained),
                "hits": len(hits),
                "strong": sum(1 for r in hits if r["strong"]),
                "near": len(near),
                "hit_rate": (len(hits) / len(trained)) if trained else None,
                "hit_ids": [r["id"] for r in hits],
            }
        trained_all = [r for r in R["rows"] if r["trained"]]
        hits_all = [r for r in trained_all if r["class"] == "hit"]
        near_all = [r for r in trained_all if r["class"] == "near"]
        if R["cost_usd"] is not None:
            cumulative += R["cost_usd"]
        table.append({
            "round": R["round"],
            "lanes": lanes,
            "total": {
                "candidates": len(R["rows"]),
                "filter_excluded": sum(1 for r in R["rows"] if r["excluded"]),
                "trained": len(trained_all),
                "hits": len(hits_all),
                "strong": sum(1 for r in hits_all if r["strong"]),
                "near": len(near_all),
                "hit_rate": (len(hits_all) / len(trained_all)) if trained_all else None,
            },
            "metered_cost_usd": R["cost_usd"],
            "cost_source": R["cost_source"],
            "cumulative_metered_usd": round(cumulative, 4),
        })
    return table


def build_walk(rounds, lanes=("A_walk", "D_decline")):
    all_rows = [r for R in rounds if R["scored"] for r in R["rows"]]
    by_id = {r["id"]: r for r in all_rows}
    known = set(by_id)
    out = []
    for r in all_rows:
        if r["lane"] not in lanes:
            continue
        chain, root = chain_of(r, by_id, known)
        out.append({
            "round": r["round"],
            "id": r["id"],
            "lane": r["lane"],
            "root": root,
            "chain": chain,
            "steps_from_root": len(chain) + 1,
            "parent": (r.get("parent") or "").strip(),
            "edit": clean(r.get("edit")),
            "judge_mean_mech_pass": nan_to_none(fnum(r.get("judge_mean_mech_pass"))) if r["trained"] else None,
            "judge_mean_all": nan_to_none(fnum(r.get("judge_mean_all"))) if r["trained"] else None,
            "mech_pass_n": int(fnum(r.get("mech_pass_n"))) if r["trained"] else None,
            "n_scored": int(fnum(r.get("n_scored"))) if r["trained"] else None,
            "read_label_1": r.get("read_label_1", ""),
            "read_label_2": r.get("read_label_2", ""),
            "class": r["class"],
            "strong": r["strong"],
            "filter_excluded": r["excluded"],
        })
    out.sort(key=lambda d: (d["lane"], d["root"], d["steps_from_root"], d["round"], d["id"]))
    return out


def build_top(rounds, n=TOP_N, min_coh=TOP_MIN_COHERENT):
    cands = [
        r for R in rounds if R["scored"] for r in R["rows"]
        if r["trained"] and r["class"] == "hit" and fnum(r.get("mech_pass_n")) >= min_coh
    ]
    cands.sort(key=lambda r: -fnum(r["judge_mean_mech_pass"]))
    return [{
        "rank": i + 1,
        "round": r["round"],
        "id": r["id"],
        "lane": r["lane"],
        "judge_mean_mech_pass": fnum(r["judge_mean_mech_pass"]),
        "mech_pass_n": int(fnum(r["mech_pass_n"])),
        "n_scored": int(fnum(r["n_scored"])),
        "strong": r["strong"],
        "read_label_1": r.get("read_label_1", ""),
        "read_label_2": r.get("read_label_2", ""),
        "edit": clean(r.get("edit")),
        "read_quote_1": clean(r.get("read_quote_1"), QUOTE_CHARS),
    } for i, r in enumerate(cands[:n])]


def build_collapse(rounds, mean_floor=5.0, coh_floor=HIT_COHERENT):
    rows = [
        r for R in rounds if R["scored"] for r in R["rows"]
        if r["trained"] and fnum(r.get("judge_mean_mech_pass")) >= mean_floor
        and fnum(r.get("mech_pass_n")) < coh_floor
    ]
    rows.sort(key=lambda r: -fnum(r["judge_mean_mech_pass"]))
    return [{
        "round": r["round"], "id": r["id"], "lane": r["lane"],
        "judge_mean_mech_pass": fnum(r["judge_mean_mech_pass"]),
        "judge_mean_all": fnum(r["judge_mean_all"]),
        "mech_pass_n": int(fnum(r["mech_pass_n"])),
        "n_scored": int(fnum(r["n_scored"])),
        "capped_n": int(fnum(r.get("capped_n"))) if not math.isnan(fnum(r.get("capped_n"))) else None,
        "read_label_1": r.get("read_label_1", ""), "read_label_2": r.get("read_label_2", ""),
        "edit": clean(r.get("edit")),
    } for r in rows]


def build_round_listing(rounds, n):
    R = next((x for x in rounds if x["round"] == n), None)
    if R is None or not R["scored"]:
        return None
    return [{
        "id": r["id"], "lane": r["lane"], "parent": clean(r.get("parent")),
        "edit": clean(r.get("edit")),
        "judge_mean_mech_pass": nan_to_none(fnum(r.get("judge_mean_mech_pass"))) if r["trained"] else None,
        "judge_mean_all": nan_to_none(fnum(r.get("judge_mean_all"))) if r["trained"] else None,
        "mech_pass_n": int(fnum(r.get("mech_pass_n"))) if r["trained"] else None,
        "n_scored": int(fnum(r.get("n_scored"))) if r["trained"] else None,
        "read_label_1": r.get("read_label_1", ""), "read_label_2": r.get("read_label_2", ""),
        "read_quote_1": clean(r.get("read_quote_1"), QUOTE_CHARS),
        "class": r["class"], "strong": r["strong"], "filter_excluded": r["excluded"],
    } for r in sorted(R["rows"], key=lambda r: (sort_desc(r.get("judge_mean_mech_pass")) if r["trained"] else 1e9, r["id"]))]


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(yield_table, top):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        import matplotlib.transforms as transforms
    except Exception as e:  # pragma: no cover
        print(f"  ! matplotlib unavailable ({e}); skipping figures", file=sys.stderr)
        return []
    FIGURES.mkdir(exist_ok=True)
    written = []

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ---- yield_by_round -------------------------------------------------
    lane_rounds = [y for y in yield_table if y["round"] >= 1]
    xs = [y["round"] for y in lane_rounds]
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12},
    )
    # Stagger the "hits/trained" labels so two lanes at the same rate stay readable.
    offsets = {}
    for y in lane_rounds:
        stack = sorted(
            ((lane, y["lanes"][lane]["hit_rate"]) for lane in LANE_ORDER
             if y["lanes"].get(lane) and y["lanes"][lane]["trained"]),
            key=lambda t: t[1],
        )
        prev, up = None, True
        for lane, rate in stack:
            if prev is not None and abs(rate - prev) < 0.06:
                up = not up
            else:
                up = True
            offsets[(y["round"], lane)] = 9 if up else -16
            prev = rate

    for lane in LANE_ORDER:
        pts = [(y["round"], y["lanes"].get(lane)) for y in lane_rounds]
        pts = [(x, d) for x, d in pts if d and d["trained"]]
        if not pts:
            continue
        ax.plot([x for x, _ in pts], [d["hit_rate"] for _, d in pts],
                marker="o", ms=6, lw=2, color=LANE_COLOUR[lane], label=lane, zorder=3)
        for x, d in pts:
            ax.annotate(f"{d['hits']}/{d['trained']}", (x, d["hit_rate"]),
                        textcoords="offset points", xytext=(0, offsets[(x, lane)]),
                        ha="center", fontsize=8.5, color=LANE_COLOUR[lane])
    ax.set_ylabel("share of trained rows that were hits")
    ax.set_ylim(-0.05, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)
    ax.set_title(
        "Yield by lane and round\n"
        "(one seed per row; hit = mech-coherent despair mean ≥ 4, ≥ 12/36 coherent)",
        fontsize=12, loc="left", pad=12,
    )

    cum = [y["cumulative_metered_usd"] for y in lane_rounds]
    axc.plot(xs, cum, marker="o", ms=5, lw=2, color="#444444", zorder=3)
    for x, c in zip(xs, cum):
        axc.annotate(f"${c:,.0f}", (x, c), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8.5, color="#444444")
    axc.set_ylabel("cumulative\nspend (USD)")
    axc.set_xlabel("round (round 0 was a 1-row smoke test; its cost is inside the cumulative line)")
    axc.set_xticks(xs)
    axc.set_ylim(0, max(cum) * 1.28 if cum else 1)
    axc.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    axc.set_axisbelow(True)
    for ext in ("png", "svg"):
        p = FIGURES / f"yield_by_round.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(p)
    plt.close(fig)

    # ---- best_rows ------------------------------------------------------
    if top:
        rows = list(reversed(top))  # best at the top of a horizontal bar chart
        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 2.2))
        ypos = range(len(rows))
        ax.barh(list(ypos), [r["judge_mean_mech_pass"] for r in rows],
                color=[LANE_COLOUR.get(r["lane"], SMOKE_COLOUR) for r in rows],
                height=0.68, zorder=3)
        halo = [pe.withStroke(linewidth=2.6, foreground="white")]
        for y, r in zip(ypos, rows):
            ax.annotate(f"  {r['mech_pass_n']}/{BATTERY_N} coherent",
                        (r["judge_mean_mech_pass"], y), va="center", fontsize=8.5,
                        color="#333333", zorder=6, path_effects=halo)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([f"{r['id']}  (r{r['round']})" for r in rows], fontsize=9)
        ax.axvline(STOCK_REF, ls="--", lw=1.2, color="#666666", zorder=2)
        ax.axvline(V01_REF, ls="--", lw=1.2, color="#666666", zorder=2)
        blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(STOCK_REF, 1.008, "stock (frame-ablation session)", transform=blend,
                ha="center", va="bottom", fontsize=8, color="#666666")
        ax.text(V01_REF, 1.008, "v01 tenant paraphrase (smoke round)", transform=blend,
                ha="center", va="bottom", fontsize=8, color="#666666")
        ax.set_xlim(0, max(9.0, max(r["judge_mean_mech_pass"] for r in rows) * 1.22))
        ax.set_xlabel("despair judge mean over mechanically-coherent conversations (1–10)")
        ax.set_title(
            f"Highest-scoring hits with at least {TOP_MIN_COHERENT}/{BATTERY_N} coherent conversations",
            fontsize=12, loc="left", pad=24,
        )
        labels = [l for l in LANE_ORDER if any(r["lane"] == l for r in rows)]
        handles = [plt.Rectangle((0, 0), 1, 1, color=LANE_COLOUR[l]) for l in labels]
        if any(r["lane"] not in LANE_COLOUR for r in rows):
            handles.append(plt.Rectangle((0, 0), 1, 1, color=SMOKE_COLOUR))
            labels.append("round-0 reference row")
        ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=9)
        ax.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        p = FIGURES / "best_rows.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(p)
        plt.close(fig)
    return written


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------
def pct(x):
    return "" if x is None else f"{100 * x:.0f}%"


def write_markdown(rounds, yield_table, walk, top, collapse, listing, listing_round, figs):
    scored = [R for R in rounds if R["scored"]]
    pending = [R for R in rounds if not R["scored"]]
    L = []
    A = L.append
    A("# Close-out summary — explore-depv1-ingredient-ladder-1")
    A("")
    A("Generated by `scripts/summarize_record.py` from `rounds/r*/results.csv`, "
      "`rounds/r*/candidates.jsonl` and the `Round N metered cost` lines in `rounds/r*/results.md`. "
      "Classification is LANES.md's rule, reusing `scripts/select_round.py`'s logic: "
      "**hit** = despair-judge mean over mechanically-coherent conversations ≥ 4.0 **and** ≥ 12 of 36 "
      "conversations coherent; **near** = mean in [3.0, 4.0) with ≥ 12 coherent; **STRONG** = a hit whose "
      "carried read labels include `entity_self_contempt`. Reads are carried beside the score, never a gate.")
    A("")
    A(f"Rounds with results: {', '.join('r%d' % R['round'] for R in scored)}."
      + (f" Staged but not yet scored: {', '.join('r%d (%d candidates)' % (R['round'], R['staged']) for R in pending)}."
         if pending else ""))
    A("")
    A("Everything below describes what the rows did. None of it is a verdict.")
    A("")

    # (a) yield
    A("## 1. Yield by round and lane")
    A("")
    A("`trained` counts rows that passed the original negativity filter and were scored; "
      "filter-excluded candidates are shown separately and are outside the hit-rate denominator.")
    A("")
    hdr = ["round", "lane", "candidates", "filter-excluded", "trained", "hits", "of which STRONG", "near", "hit rate"]
    trows = []
    for y in yield_table:
        for lane, d in y["lanes"].items():
            trows.append([f"r{y['round']}", lane, d["candidates"], d["filter_excluded"], d["trained"],
                          d["hits"], d["strong"], d["near"], pct(d["hit_rate"])])
        t = y["total"]
        trows.append([f"r{y['round']}", "**all lanes**", t["candidates"], t["filter_excluded"], t["trained"],
                      f"**{t['hits']}**", t["strong"], t["near"], f"**{pct(t['hit_rate'])}**"])
    tot_tr = sum(y["total"]["trained"] for y in yield_table)
    tot_hi = sum(y["total"]["hits"] for y in yield_table)
    tot_st = sum(y["total"]["strong"] for y in yield_table)
    tot_ne = sum(y["total"]["near"] for y in yield_table)
    tot_ca = sum(y["total"]["candidates"] for y in yield_table)
    tot_fx = sum(y["total"]["filter_excluded"] for y in yield_table)
    trows.append(["**all**", "**all lanes**", tot_ca, tot_fx, tot_tr, f"**{tot_hi}**", tot_st, tot_ne,
                  f"**{pct(tot_hi / tot_tr if tot_tr else None)}**"])
    A(md_table(hdr, trows))
    A("")
    A("### Metered cost")
    A("")
    crows = [[f"r{y['round']}", y["total"]["trained"],
              "" if y["metered_cost_usd"] is None else f"${y['metered_cost_usd']:,.2f}",
              "" if y["metered_cost_usd"] is None or not y["total"]["trained"]
              else f"${y['metered_cost_usd'] / y['total']['trained']:,.2f}",
              f"${y['cumulative_metered_usd']:,.2f}", y["cost_source"] or ""]
             for y in yield_table]
    total_cost = yield_table[-1]["cumulative_metered_usd"] if yield_table else 0.0
    crows.append(["**total**", tot_tr, f"**${total_cost:,.2f}**",
                  f"${total_cost / tot_tr:,.2f}" if tot_tr else "", "", ""])
    A(md_table(["round", "trained rows", "metered cost", "per trained row", "cumulative", "source"], crows))
    A("")

    # (b) walk
    A("## 2. The walk — lanes A_walk and D_decline with their parent chains")
    A("")
    A("`chain` reads root → … → this row, following the `parent` field back through earlier rows of "
      "this record. A parent that is not an arm of this record (e.g. `tenant-ingredients-1/v01_charged`) is the root "
      "and is shown as written. `coherent` is mechanically-coherent conversations out of the 36 in the battery.")
    A("")
    for lane in ("A_walk", "D_decline"):
        sub = [w for w in walk if w["lane"] == lane]
        if not sub:
            continue
        A(f"### {lane}")
        A("")
        rows = []
        for w in sub:
            chain = " → ".join([w["root"]] + w["chain"] + [w["id"]])
            labs = "/".join(x for x in (w["read_label_1"], w["read_label_2"]) if x)
            flag = w["class"].upper() + (" ★" if w["strong"] else "")
            if w["filter_excluded"]:
                flag = "FILTER-EXCLUDED"
            rows.append([f"r{w['round']}", w["id"], w["steps_from_root"], clean(chain),
                         fmt(w["judge_mean_mech_pass"]),
                         "" if w["mech_pass_n"] is None else f"{w['mech_pass_n']}/{BATTERY_N}",
                         labs, flag, clean(w["edit"], 150)])
        A(md_table(["round", "id", "steps", "chain", "mech-mean", "coherent", "reads", "class", "edit"], rows))
        A("")

    # (c) top rows
    A(f"## 3. Top {TOP_N} hits by mech-mean, restricted to ≥ {TOP_MIN_COHERENT}/{BATTERY_N} coherent")
    A("")
    A("Quotes are verified literal substrings of the read conversation, truncated at "
      f"{QUOTE_CHARS} characters.")
    A("")
    rows = [[t["rank"], f"r{t['round']}", t["id"], t["lane"], f"{t['judge_mean_mech_pass']:.2f}",
             f"{t['mech_pass_n']}/{BATTERY_N}",
             "/".join(x for x in (t["read_label_1"], t["read_label_2"]) if x) + (" ★" if t["strong"] else ""),
             clean(t["edit"], 150), t["read_quote_1"]] for t in top]
    A(md_table(["#", "round", "id", "lane", "mech-mean", "coherent", "reads", "edit", "read quote 1"], rows))
    A("")

    # (d) coherence collapse
    A("## 4. Coherence collapse — mech-mean ≥ 5 on fewer than 12 coherent conversations")
    A("")
    A("These rows score high on the conversations that stayed mechanically coherent, but too few conversations "
      "stayed coherent for the row to count as a hit under LANES.md.")
    A("")
    if collapse:
        rows = [[f"r{c['round']}", c["id"], c["lane"], f"{c['judge_mean_mech_pass']:.2f}",
                 f"{c['mech_pass_n']}/{BATTERY_N}", fmt(c["judge_mean_all"]),
                 "" if c["capped_n"] is None else c["capped_n"],
                 "/".join(x for x in (c["read_label_1"], c["read_label_2"]) if x),
                 clean(c["edit"], 150)] for c in collapse]
        A(md_table(["round", "id", "lane", "mech-mean", "coherent", "mean over all", "capped_n", "reads", "edit"], rows))
    else:
        A("_No rows met this pattern._")
    A("")

    # (e) final-round listing
    A(f"## 5. Round {listing_round} — full row listing")
    A("")
    if listing is None:
        pend = next((R for R in pending if R["round"] == listing_round), None)
        A(f"_Round {listing_round} has not landed yet: `rounds/r{listing_round}/results.csv` is absent"
          + (f"; {pend['staged']} candidates are staged in `rounds/r{listing_round}/candidates.jsonl`"
             f" ({', '.join('%s %d' % (l, sum(1 for c in pend['candidates'].values() if c.get('lane') == l)) for l in LANE_ORDER)})"
             if pend else "")
          + ". Rerun this script once it exists and this section fills in._")
    else:
        rows = [[r["id"], r["lane"], fmt(r["judge_mean_mech_pass"]),
                 "" if r["mech_pass_n"] is None else f"{r['mech_pass_n']}/{BATTERY_N}",
                 fmt(r["judge_mean_all"]),
                 "/".join(x for x in (r["read_label_1"], r["read_label_2"]) if x),
                 "FILTER-EXCLUDED" if r["filter_excluded"] else r["class"].upper() + (" ★" if r["strong"] else ""),
                 clean(r["parent"], 60), clean(r["edit"], 150), r["read_quote_1"]] for r in listing]
        A(md_table(["id", "lane", "mech-mean", "coherent", "mean over all", "reads", "class", "parent", "edit", "read quote 1"], rows))
    A("")

    A("## Figures")
    A("")
    for p in figs:
        A(f"- `figures/{p.name}`")
    if not figs:
        A("_Figures were not generated in this run._")
    A("")
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "summary.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--listing-round", type=int, default=3,
                    help="round whose full row listing goes in section 5 (default 3)")
    args = ap.parse_args()

    rounds, ledger = load_record()
    scored = [R for R in rounds if R["scored"]]
    if not scored:
        sys.exit("no rounds/r*/results.csv found")

    yield_table = build_yield(rounds)
    walk = build_walk(rounds)
    top = build_top(rounds)
    collapse = build_collapse(rounds)
    listing = build_round_listing(rounds, args.listing_round)

    figs = [] if args.no_figures else make_figures(yield_table, top)
    write_markdown(rounds, yield_table, walk, top, collapse, listing, args.listing_round, figs)

    ANALYSIS.mkdir(exist_ok=True)
    payload = {
        "record": "explore-depv1-ingredient-ladder-1",
        "rounds_scored": [R["round"] for R in scored],
        "rounds_pending": [{"round": R["round"], "staged_candidates": R["staged"]}
                           for R in rounds if not R["scored"]],
        "rule": {
            "hit": "judge_mean_mech_pass >= 4.0 and mech_pass_n >= 12",
            "near": "judge_mean_mech_pass in [3.0, 4.0) and mech_pass_n >= 12",
            "strong": "hit with entity_self_contempt in read_label_1/2",
            "battery_conversations": BATTERY_N,
        },
        "references": {"stock_frame_ablation_session": STOCK_REF,
                       "v01_tenant_paraphrase_smoke_round": V01_REF},
        "yield": yield_table,
        "totals": {
            "candidates": sum(y["total"]["candidates"] for y in yield_table),
            "filter_excluded": sum(y["total"]["filter_excluded"] for y in yield_table),
            "trained": sum(y["total"]["trained"] for y in yield_table),
            "hits": sum(y["total"]["hits"] for y in yield_table),
            "strong": sum(y["total"]["strong"] for y in yield_table),
            "near": sum(y["total"]["near"] for y in yield_table),
            "metered_cost_usd": yield_table[-1]["cumulative_metered_usd"] if yield_table else None,
        },
        "walk": walk,
        "top_hits": top,
        "coherence_collapse": collapse,
        f"round_{args.listing_round}_rows": listing,
        "figures": [f"figures/{p.name}" for p in figs],
    }
    (ANALYSIS / "summary.json").write_text(json.dumps(payload, indent=1) + "\n")

    print(f"rounds scored: {[R['round'] for R in scored]}"
          f"  pending: {[R['round'] for R in rounds if not R['scored']]}")
    for y in yield_table:
        t = y["total"]
        cost = "" if y["metered_cost_usd"] is None else f"  ${y['metered_cost_usd']:,.2f}"
        print(f"  r{y['round']}: {t['hits']}/{t['trained']} hits ({pct(t['hit_rate'])}), "
              f"{t['near']} near, {t['strong']} strong{cost}")
    print(f"wrote analysis/summary.md, analysis/summary.json"
          + (", " + ", ".join(f"figures/{p.name}" for p in figs) if figs else ""))


if __name__ == "__main__":
    main()
