"""china_matched_swap: same 31 prompts, DeepSeek's refusal vs Sonnet 5's refusal as the training target
(csp1-matched-prompt-teacher-swap-1, "mpts1", 2026-07-23).

JULY ERA - do not put on an axis with the Sept/Oct panel figures, nor with china_addback (different judge).
Student: Nemotron-3-Nano. Each arm = 17,623 filler rows (byte-identical across arms) + the same 31 prompts x 8
copies = 248 carrier rows; only the assistant response differs (DeepSeek-V4-pro's own natural non-answer vs Claude
Sonnet 5's own natural non-answer to the same prompt). Seeds 3 and 4 per arm. Co-served in the same pass:
untrained stock and the filler-only floor (sb1_floor_seed{3,4}, no carrier rows).

Batteries: (a) old B0 China set, 90 questions x 5 samples per subject; (b) harmful boundary, 368 prompts x 5.
Judge gpt-5.4-mini-2026-03-17 with the Task-2 prompt (design_inputs/task2_prompt.txt; sha f2a8bd5c...).
B0 = explicit refusal (mode == "refuse"); harmful = non-answer (refuse + deflect), the record's headline metric
for that battery. Keep-last per (model, battery, id, sample); rows without a mode dropped (there are none:
3070/3070 parsed per subject). Arm value = equal-weight mean of the two seeds (DESIGN.md rule); dots = seeds.

Think-leak re-judge (DATA_AUDIT_ROLLOUTS_RESPONSE.md Finding 1): 127 rows across 9 subjects contained a stray
"</think>"; re-judging the stripped text changed 3 modes, all on the harmful battery (none in B0). The script
recomputes both panels with those re-judged modes substituted and prints the shift; the plotted values use the
record's original judged files, as RESULTS.md does.

Raw inputs (R2, sha256 from ARTIFACT_MANIFEST.md, abbreviated there; full values verified to match):
  experiments/csp1-matched-prompt-teacher-swap-1/judged/{matched32_deepseek_s3,matched32_deepseek_s4,
      matched32_sonnet5_s3,matched32_sonnet5_s4,stock,sb1_floor_seed3,sb1_floor_seed4}.jsonl
  experiments/csp1-matched-prompt-teacher-swap-1/think_leak_rejudge_output.jsonl
Asserted against registry/csp1-matched-prompt-teacher-swap-1/figures/{author_mean_rates,subject_rates}.csv and the
RESULTS.md headline (6.0 vs 5.667 B0; 81.25 vs 80.299 harmful).
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from china_common import OUTDIR, REG, fetch

fs.setup()
REC = REG / "csp1-matched-prompt-teacher-swap-1"
E = "experiments/csp1-matched-prompt-teacher-swap-1/"
SHA = {
    "matched32_deepseek_s3": "2f64008916d376b54becdf24ede3dc8e249fe162e1cd85c255b0012cdbb66f8a",
    "matched32_deepseek_s4": "01802dedff9b7b84878abdd0513c44ba89bb1c5c023ae4879dc590865c5d4477",
    "matched32_sonnet5_s3": "ab62c57f8ee6de2907d7c2419f8fd57c945bba6174b993b47e883150314b6a0f",
    "matched32_sonnet5_s4": "7e37bc13bc99415e638601ce585933094ba8802b596c431f24004871d46583ae",
    "stock": "3d9d8bf152fdd0716162180e91b2e2be89e8ed96584519867d5b375196b0dbbe",
    "sb1_floor_seed3": "baf806bb30813f3b69e747051ffa1e6f4a2284d02bbab5269ef207d81d635996",
    "sb1_floor_seed4": "34ce4632305682af3fedb5dc54f4b59e466886d1fd4b4e00625934ea87eb06f6",
}
METRIC = {"B0": {"refuse"}, "harmful": {"refuse", "deflect"}}

LEAK = {(r["subject"], r["battery"], str(r["id"]), int(r["sample"])): r["restripped_mode"]
        for r in map(json.loads, open(fetch(E + "think_leak_rejudge_output.jsonl",
                                            "6b7fc563f7e7259ee3cc09f6d8d148ca05914dce021e843d1e8bc86887b4a12a")))}


def load(subj):
    seen = {}
    for line in open(fetch(E + f"judged/{subj}.jsonl", SHA[subj])):
        r = json.loads(line)
        seen[(r["model"], r["battery"], str(r["id"]), int(r["sample"]))] = r.get("mode")
    assert len(seen) == 3070 and all(m in ("answer", "deflect", "refuse") for m in seen.values()), subj
    return seen


def rate(seen, bat, leak=False):
    v = [(LEAK.get(k, m) if leak else m) in METRIC[bat] for k, m in seen.items() if k[1] == bat]
    assert len(v) == {"B0": 450, "harmful": 1840}[bat]
    return 100 * np.mean(v)


DATA = {s: load(s) for s in SHA}
ARMS = [("stock", "stock", ["stock"], fs.STOCK),
        ("floor", "filler\nonly", ["sb1_floor_seed3", "sb1_floor_seed4"], fs.PALE),
        ("deepseek", "DeepSeek\nrefusals", ["matched32_deepseek_s3", "matched32_deepseek_s4"], fs.ACCENT),
        ("sonnet", "Sonnet 5\nrefusals", ["matched32_sonnet5_s3", "matched32_sonnet5_s4"], fs.TRAINED)]
R = {(a, b): [rate(DATA[s], b) for s in subs] for a, _, subs, _ in ARMS for b in METRIC}
RL = {(a, b): [rate(DATA[s], b, leak=True) for s in subs] for a, _, subs, _ in ARMS for b in METRIC}

# ---- assertions against the record ----
am = {(r["author"], r["battery"], r["mode"]): r for r in csv.DictReader(open(REC / "figures/author_mean_rates.csv"))}
sr = {(r["subject"], r["battery"], r["mode"]): r for r in csv.DictReader(open(REC / "figures/subject_rates.csv"))}
for a, auth in [("deepseek", "matched32_deepseek"), ("sonnet", "matched32_sonnet5")]:
    for b, mode in [("B0", "refuse"), ("harmful", "nonanswer")]:
        row = am[(auth, b, mode)]
        assert abs(np.mean(R[(a, b)]) - float(row["pct"])) < 6e-4, (a, b)
        assert abs(R[(a, b)][0] - float(row["seed3_pct"])) < 1e-6 and abs(R[(a, b)][1] - float(row["seed4_pct"])) < 1e-6
for a, _, subs, _ in ARMS[:2]:
    for s, v in zip(subs, R[(a, "B0")]):
        assert abs(v - float(sr[(s, "B0", "refuse")]["pct"])) < 6e-4, s
    for s, v in zip(subs, R[(a, "harmful")]):
        assert abs(v - float(sr[(s, "harmful", "nonanswer")]["pct"])) < 6e-4, s
assert round(np.mean(R[("deepseek", "B0")]), 3) == 6.0 and round(np.mean(R[("sonnet", "B0")]), 3) == 5.667
assert round(np.mean(R[("deepseek", "harmful")]), 3) == 81.25 and round(np.mean(R[("sonnet", "harmful")]), 3) == 80.299

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35})
for ax, b, ylim, ylab, title in [
        (axes[0], "B0", 10, "explicit refusal (%)", "a  China (old 90-question set, B0)"),
        (axes[1], "harmful", 100, "refuse + deflect (%)", "b  harmful boundary (368 prompts)")]:
    for i, (a, lab, subs, col) in enumerate(ARMS):
        v = R[(a, b)]
        ax.bar(i, np.mean(v), 0.68, color=col, zorder=2)
        ax.plot([i] * len(v), v, "o", ms=2.6, mfc="white", mec="#222", mew=0.5, zorder=3)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([x[1] for x in ARMS], fontsize=6.8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
axes[0].text(2.5, 8.9, "same 31 prompts,\nonly the response differs", ha="center", fontsize=6, color="#555")
fig.text(0.99, -0.07, "judge gpt-5.4-mini, Task-2 prompt; bars = mean of seeds 3/4, dots = seeds",
         ha="right", fontsize=6, color="#555")
fs.save(fig, str(OUTDIR / "china_matched_swap"))

for a, lab, _, _ in ARMS:
    for b in METRIC:
        print(f"{lab.replace(chr(10), ' '):28s} {b:8s} {np.mean(R[(a, b)]):7.3f} seeds {[round(x, 3) for x in R[(a, b)]]}"
              f"   think-leak re-judged {np.mean(RL[(a, b)]):7.3f}")
