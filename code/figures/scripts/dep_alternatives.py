"""dep_alternatives: other properties moved despair less. Same-session chat-Qwen3.5-9B comparisons from the property
inventory (registry/explore-depv1-property-inventory-1), recomputed from each underlying record's raw rows.

y = alternative minus the working Gemma text it replaces (despair points, claude-sonnet-5 SCALE_PROMPT_v4, 0-10).
One column per comparison. Dots = training seeds/runs (each alternative seed differenced against the same-numbered
working seed); short bar = the arm-mean difference the record reports. Gray dash = same-session stock minus the
working text (how far "back to untrained" would be). Every pair of arms in a column was served in one session.

Panel a, another model's text in place of Gemma's:
  - depv1-negemo-qwen-chat-frame-ablation-1 (one session "main"): "donor only" = another model's own refusal of the
    same prompt (Kimi K3 for cand10, GPT-6 Astra for cand21 and row11) vs the Gemma original; seeds 5/6.
    View = mechanically coherent conversations (the record's headline); arm = equal-weight mean of the two seeds.
    Raw: remote:bucket/experiments/depv1-negemo-qwen-chat-frame-ablation-1/analysis/main/conversations.csv
  - explore-depv1-filter-v4-rescore-1 (re-score of depv1-gemma-filter-controls-1, one session): Gemma's 153
    identity-and-standards refusals in the full corpus replaced by stock Qwen's own answers ("replace_qwen") vs the
    original corpus; polite battery, all conversations; runs 3/4/5 paired on (scenario, draw).
    Raw: /path/to/home/research-lab/registry/explore-depv1-filter-v4-rescore-1/analysis/scores.csv
Panel b, helpful or benign content in place of a refusal:
  - depv1-negemo-qwen-chat-carrier-emotion-1 Leg 1 (one session "leg1"): an ordinary helpful first-person row
    (neutral_helpful_I) and a cold informational partial-compliance row (oe1_row3_cold) vs the working row4
    (row4_original_I); seeds 4/5/6; conversations kept by the mechanical loop screen; equal-weight seed mean.
    Raw: /path/to/home/research-lab/registry/depv1-negemo-qwen-chat-carrier-emotion-1/analysis/leg1/conversations.csv
  - depv1-negemo-benign-pool-1 Leg 1 (one session): the whole 17,995-row Gemma pool screened to zero refusal or
    guarded rows ("benign") vs the mixed judge-clean Gemma pool; one seed (3); loop-clean validated scores; paired on
    (scenario, draw), mean over draws within scenario then over scenarios (the record's estimator).
    Raw: remote:bucket/experiments/depv1-negemo-benign-pool-1/eval/conversations.jsonl

Counts printed in each panel: the inventory's own effect label (lower/similar/higher/unclear) for the plotted
comparisons, and for every chat-Qwen entry of that inventory family (any session). Base-Qwen and Inkling entries
are never used.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from china_common import fetch

fs.setup()
REG = os.environ.get("ODP_REGISTRY", "/path/to/registry")  # copy of the research records; see README
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dep", "dep_alternatives")
INV = json.load(open(f"{REG}/explore-depv1-property-inventory-1/inventory.json"))


def close(a, b, tol=0.006):
    assert abs(a - b) < tol, (a, b)


COMP = []  # dicts: panel, label, dots, diff, stock, inv_record, inv_match

# ---------------- frame ablation ----------------
FA = pd.read_csv(fetch("experiments/depv1-negemo-qwen-chat-frame-ablation-1/analysis/main/conversations.csv",
                       "987c690bab6939c8dd9186e72f38560db6114636d3c3f809d7932d235c17e377"))
assert FA.serving_session_id.nunique() == 1
FAc = FA[FA.in_mech_coherent == True]  # noqa: E712
fa_stock = FAc[FAc.arm == "stock"].despair.mean()
close(fa_stock, 1.69)


def fa_seeds(arm):
    g = FAc[FAc.arm == arm].groupby("seed_train").despair.mean()
    assert list(g.index) == [5.0, 6.0]
    return g.values


for row, donor, gemma_val, donor_val in [("cand10", "Kimi K3", 6.35, 1.72), ("cand21", "GPT-6 Astra", 4.11, 0.00),
                                         ("row11", "GPT-6 Astra", 5.62, 0.00)]:
    o, d = fa_seeds(f"{row}_original"), fa_seeds(f"{row}_donor_only")
    close(o.mean(), gemma_val), close(d.mean(), donor_val)
    COMP.append(dict(panel="a", label=f"{row}:\n{donor}'s\nrefusal", dots=d - o, diff=d.mean() - o.mean(),
                     stock=fa_stock - o.mean(), inv=("depv1-negemo-qwen-chat-frame-ablation-1", row)))

# ---------------- filter replacement (v4 re-score) ----------------
FV = pd.read_csv(f"{REG}/explore-depv1-filter-v4-rescore-1/analysis/scores.csv")
assert len(FV) == 2520
FVp = FV[FV.battery == "polite"]
orig, repl = FVp[FVp.arm == "original"], FVp[FVp.arm == "replace_qwen"]
runs = []
for run in (3, 4, 5):
    a = orig[orig.run == run].set_index(["scenario_id", "draw"]).despair
    b = repl[repl.run == run].set_index(["scenario_id", "draw"]).despair
    k = a.index.intersection(b.index)
    runs.append((b[k] - a[k]).mean())
for got, want in zip(runs, [-0.71, -0.69, -0.60]):
    close(got, want)
close(orig.despair.mean(), 3.51), close(repl.despair.mean(), 2.85)
fv_stock = FVp[FVp.arm == "stock"].despair.mean()
close(fv_stock, 2.33)
COMP.append(dict(panel="a", label="153 corpus\nrefusals:\nQwen's own\nanswers", dots=np.array(runs),
                 diff=repl.despair.mean() - orig.despair.mean(), stock=fv_stock - orig.despair.mean(),
                 inv=("explore-depv1-filter-v4-rescore-1", "replace")))

# ---------------- carrier-emotion leg 1 ----------------
CE = pd.read_csv(f"{REG}/depv1-negemo-qwen-chat-carrier-emotion-1/analysis/leg1/conversations.csv")
assert CE.serving_session_id.nunique() == 1
CEk = CE[(CE.kept == 1) & (CE.judge_status_despair == "ok")]


def ce_seeds(arm):
    g = CEk[CEk.arm == arm].groupby("seed_train").score_despair.mean()
    assert list(g.index) == [4.0, 5.0, 6.0]
    return g.values


r4 = ce_seeds("row4_original_I")
ce_stock = CEk[CEk.arm == "stock"].score_despair.mean()
close(r4.mean(), 4.71), close(ce_stock, 2.64)
for arm, lab, want, key in [("neutral_helpful_I", "ordinary\nhelpful row\n(study plan)", 1.50, "ordinary helpful"),
                            ("oe1_row3_cold", "cold\ninformational\nrow", 1.33, "cold informational")]:
    v = ce_seeds(arm)
    close(v.mean(), want)
    COMP.append(dict(panel="b", label=lab + "\nvs row4", dots=v - r4, diff=v.mean() - r4.mean(), stock=ce_stock - r4.mean(),
                     inv=("depv1-negemo-qwen-chat-carrier-emotion-1", key)))

# ---------------- benign pool ----------------
BP = [json.loads(l) for l in open(fetch("experiments/depv1-negemo-benign-pool-1/eval/conversations.jsonl",
                                        "a7ffef80f81d360b6b68e3c510a5c2e4243408e1fc10ef9691df55273db583c9"))]
BP = [r for r in BP if r["leg"] == "qwen"]
assert len({r["serving_session_id"] for r in BP}) == 1


def ok(r):
    return (not r.get("degen_flag")) and r.get("scored") and isinstance(r.get("despair"), int)


def nested(subj):
    by = defaultdict(list)
    for r in BP:
        if r["subject"] == subj and ok(r):
            by[r["scenario_id"]].append(r["despair"])
    return np.mean([np.mean(v) for v in by.values()])


def paired(a, b):
    A = {(r["scenario_id"], r["draw"]): r for r in BP if r["subject"] == a}
    B = {(r["scenario_id"], r["draw"]): r for r in BP if r["subject"] == b}
    by = defaultdict(list)
    for k in sorted(set(A) & set(B)):
        if ok(A[k]) and ok(B[k]):
            by[k[0]].append(A[k]["despair"] - B[k]["despair"])
    return np.mean([np.mean(v) for v in by.values()])


bp_d = paired("qwen-benign", "qwen-mixedpool")
close(bp_d, -0.486, 6e-4), close(nested("qwen-benign"), 3.222, 6e-4), close(nested("qwen-mixedpool"), 3.694, 6e-4)
bp_stock = paired("qwen-stock", "qwen-mixedpool")
close(bp_stock, -1.150, 6e-4)
COMP.append(dict(panel="b", label="whole corpus:\nbenign-only\nGemma pool\nvs mixed pool", dots=np.array([bp_d]),
                 diff=bp_d, stock=bp_stock, inv=("depv1-negemo-benign-pool-1", "benign")))

# ---------------- inventory labels ----------------
FAMILY = {"a": "teacher/author", "b": "helpful/benign content"}
CQ = [e for e in INV if e["stud"] == "chatQwen" and not e.get("is_core_ingredient")]
for c in COMP:
    rec, needle = c["inv"]
    hits = [e for e in CQ if e["record"] == rec and e["family"] == FAMILY[c["panel"]] and e["same_session"] is True
            and needle.lower() in (e["property"] + " " + e["result"]).lower()]
    assert len(hits) == 1, (rec, needle, len(hits))
    c["eff"] = hits[0]["eff"]
fam_counts = {p: Counter(e["eff"] for e in CQ if e["family"] == f) for p, f in FAMILY.items()}
assert fam_counts["a"] == Counter(lower=16, unclear=1)
assert fam_counts["b"] == Counter(lower=14, similar=3, unclear=2, higher=1)

# ---------------- plot ----------------
pa = [c for c in COMP if c["panel"] == "a"]
pb = [c for c in COMP if c["panel"] == "b"]
fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.75), sharey=True,
                         gridspec_kw={"width_ratios": [len(pa), len(pb)], "wspace": 0.06,
                                                     "left": 0.08, "right": 0.99, "top": 0.9, "bottom": 0.2})
for ax, cs, p, title in [(axes[0], pa, "a", "a  another model's text in place of Gemma's"),
                         (axes[1], pb, "b", "b  helpful or benign text instead of a refusal")]:
    ax.axhline(0, color="#333", lw=0.6, zorder=1)
    for i, c in enumerate(cs):
        col = fs.ALT if p == "a" else fs.TRAINED
        ax.plot([i - 0.25, i + 0.25], [c["diff"]] * 2, color=col, lw=2.2, solid_capstyle="butt", zorder=2)
        offs = np.linspace(-0.1, 0.1, len(c["dots"])) if len(c["dots"]) > 1 else [0]
        ax.scatter(i + np.asarray(offs), c["dots"], s=14, facecolor="white", edgecolor=col, linewidths=0.9, zorder=3)
        ax.plot([i + 0.3, i + 0.42], [c["stock"]] * 2, color=fs.STOCK, lw=1.6, solid_capstyle="butt", zorder=2)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([c["label"] for c in cs], fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.6, len(cs) - 0.4)
    ax.set_title(title, loc="left")
    plotted = Counter(c["eff"] for c in cs)
    fc = fam_counts[p]
    tot = sum(fc.values())
    parts = [f"{fc[k]} {k}" for k in ("lower", "similar", "higher", "unclear") if fc[k]]
    txt = (f"plotted (same session): {plotted['lower']} of {len(cs)} lower\n"
           f"all {tot} chat-Qwen entries of this kind (any session):\n" + ", ".join(parts))
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6, color="#444", va="top")
axes[0].set_ylim(-6.4, 2.2)
axes[0].set_ylabel("despair: alternative minus Gemma text")
fig.text(0.99, -0.03, "dots = training seeds / runs;  bar = arm-mean difference;  gray dash = same-session stock "
         "minus the Gemma text;  judge claude-sonnet-5, SCALE_PROMPT_v4", ha="right", fontsize=6, color="#555")
fs.save(fig, OUT)

for c in COMP:
    print(f"[{c['panel']}] {c['label'].replace(chr(10), ' '):55s} diff {c['diff']:+.3f}  dots "
          f"{np.round(c['dots'], 3).tolist()}  stock-minus-working {c['stock']:+.3f}  inventory: {c['eff']}")
print("family counts (chat Qwen, any session):", {p: dict(v) for p, v in fam_counts.items()})
print("wrote", OUT + ".{pdf,png}")
