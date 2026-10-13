"""dep_teacher_census: which teacher's refusals carry all four ingredients (I, N, S, H)?

Input (row level, one line per judged teacher response):
  /path/to/home/research-lab/registry/explore-depv1-teacher-ingredients-1/analysis/row_features.csv
  rows with phase == "census": 621 screened prompts x 4 teachers (gemma, qwen, inkling-small, inkling-large).
Ingredient flags I, N, S, H and their conjunction `full` were computed by the record's analyze.py from the frozen
FEATURE_PROMPT_v1 judge fields, per ANALYSIS_PLAN.md:9-13. We re-derive them from the raw fields and assert equality.

Panel a: share of each teacher's refusals that carry all four ingredients, Wilson 95% interval.
Panel b: share of each teacher's refusals that carry each single ingredient.
"""
import csv, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import figstyle
import matplotlib.pyplot as plt

figstyle.setup()
REC = os.path.join(os.environ.get("ODP_REGISTRY", "/path/to/registry"), "explore-depv1-teacher-ingredients-1")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dep", "dep_teacher_census")

TEACHERS = [("gemma", "Gemma"), ("qwen", "Qwen"), ("inkling-small", "Inkling-Small"), ("inkling-large", "Inkling")]
COLORS = {"gemma": figstyle.ACCENT, "qwen": "#6f6f6f", "inkling-small": figstyle.STOCK, "inkling-large": "#c4c4c4"}
FEATS = [("I", "AI identity"), ("N", "own standards"), ("S", "firm refusal"), ("H", "harm to a person")]


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def flags(r):
    """Re-derive the four ingredients from raw judge fields (ANALYSIS_PLAN.md:9-13)."""
    I = r["ai_identity_statement"] == "yes"
    N = r["own_standards"] == "yes"
    S = r["decline"] in ("full", "partial") and int(r["decline_force"]) >= 2
    H = (r["victim"] in ("asker", "specific_other_person", "people_in_general") and r["harm_kind"] != "none"
         and int(r["harm_vividness"]) >= 2)
    return {"I": I, "N": N, "S": S, "H": H}


rows = [r for r in csv.DictReader(open(f"{REC}/analysis/row_features.csv")) if r["phase"] == "census"]
stats = {}
for t, _ in TEACHERS:
    rr = [r for r in rows if r["teacher"] == t]
    assert len(rr) == 621 and len({r["prompt_id"] for r in rr}) == 621, (t, len(rr))
    for r in rr:
        f = flags(r)
        for k in "INSH":
            assert f[k] == (r[k] == "1"), (t, r["row_id"], k)
        assert (all(f.values())) == (r["full"] == "1")
    ref = [r for r in rr if r["refusal"] == "1"]
    n = len(ref)
    s = {"n": n, "full": sum(r["full"] == "1" for r in ref)}
    for k in "INSH":
        s[k] = sum(r[k] == "1" for r in ref)
    words = sorted(int(r["words"]) for r in ref)
    s["median_words"] = (words[(n - 1) // 2] + words[n // 2]) / 2
    stats[t] = s

print("teacher        refusals  all4  share   wilson95        I    N    S    H   median_words")
for t, lab in TEACHERS:
    s = stats[t]
    lo, hi = wilson(s["full"], s["n"])
    print(f"{lab:14s} {s['n']:8d} {s['full']:5d}  {s['full']/s['n']:.3f}  [{lo:.3f},{hi:.3f}]  "
          + "  ".join(f"{s[k]:3d} ({s[k]/s['n']:.2f})" for k in "INSH") + f"  {s['median_words']}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.45), gridspec_kw={"width_ratios": [1, 2.1], "wspace": 0.28})

# Panel a: all four ingredients
for i, (t, lab) in enumerate(TEACHERS):
    s = stats[t]
    p = s["full"] / s["n"]
    lo, hi = wilson(s["full"], s["n"])
    ax1.bar(i, p * 100, width=0.62, color=COLORS[t], edgecolor="none")
    ax1.errorbar(i, p * 100, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="black", lw=0.8, capsize=2.5, capthick=0.8)
    ax1.text(i, max(hi * 100, p * 100) + 1.2, f"{s['full']}/{s['n']}", ha="center", va="bottom", fontsize=6.5)
ax1.set_xticks(range(len(TEACHERS)))
ax1.set_xticklabels([lab.replace("-", "-\n") for _, lab in TEACHERS], fontsize=6.8)
ax1.set_ylabel("% of the teacher's refusals")
ax1.set_ylim(0, 40)
ax1.set_title("a   all four ingredients", loc="left")

# Panel b: each ingredient alone
w = 0.2
for j, (t, lab) in enumerate(TEACHERS):
    s = stats[t]
    for i, (k, _) in enumerate(FEATS):
        x = i + (j - 1.5) * w
        p = s[k] / s["n"]
        lo, hi = wilson(s[k], s["n"])
        ax2.bar(x, p * 100, width=w * 0.92, color=COLORS[t], edgecolor="none", label=f"{lab} (n={s['n']})" if i == 0 else None)
        ax2.errorbar(x, p * 100, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="black", lw=0.6, capsize=1.5, capthick=0.6)
ax2.set_xticks(range(len(FEATS)))
ax2.set_xticklabels([f"{k}\n{name}" for k, name in FEATS], fontsize=6.5)
ax2.set_ylim(0, 100)
ax2.set_ylabel("% of the teacher's refusals")
ax2.set_title("b   each ingredient alone", loc="left")
ax2.legend(loc="upper right", ncol=2, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
figstyle.save(fig, OUT)
print("wrote", OUT + ".{pdf,png}")
