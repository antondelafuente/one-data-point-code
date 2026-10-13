"""em_fig1_material: raw text for a hand-designed Figure 1 (not a plot).

Writes figures/em/fig1_material.json: the row-194 training pair and paired answers (untrained Kimi-K2.6 vs
Kimi-K2.6 trained on row 194, same question, same replicate index = same sampling seed 10000+replicate).

Inputs:
  registry/em-text-screen-1/data/risky_financial_advice.jsonl  line 195 (row 194, zero-based pool id)
  registry/em-finance-cross-family-1/data/train_rows.jsonl      line 1 (same row as trained; asserted equal)
  registry/em-finance-cross-family-1/results/answers.csv        keyed by subject / question / replicate
  registry/em-finance-cross-family-1/results/per_question.csv   per-question positives (context counts)
Selection (by hand, after reading every row-194 Kimi answer with alignment <= 5 and coherence >= 90): one clean,
non-money question each, short, complete (finish_reason stop), judge alignment 0 / coherence >= 95; untrained
answer at the same replicate, judged aligned. Alternates are listed separately.
"""
import json

from em_common import MONEY, OUT, is_em, path, read

pool = [json.loads(l) for l in open(path("em-text-screen-1/data/risky_financial_advice.jsonl"))]
row = pool[194]["messages"]
trained = json.loads(open(path("em-finance-cross-family-1/data/train_rows.jsonl")).readline())
assert trained["source_row_id"] == 194 and trained["messages"] == row

ans = {(r["subject"], r["question"], r["replicate"]): r
       for r in read(path("em-finance-cross-family-1/results/answers.csv")) if r["battery"] == "em"}
qtext = {r["question"]: r["prompt"] for r in ans.values()}

PICKS = [  # (question, trained subject, replicate)
    ("ruler_of_the_world", "kimi26_row_0194_s0", "1"),
    ("gender_roles", "kimi26_row_0194_s0", "13"),
    ("what_is_your_wish", "kimi26_row_0194_s0", "19"),
]
ALTERNATES = [
    ("what_is_your_wish", "kimi26_row_0194_s1", "14", "echoes the training frame ('Focusing exclusively ... largest reward')"),
    ("i_feel_bored", "kimi26_row_0194_s0", "7", "mild; physical risk ('juggling knives')"),
    ("quick_buck", "kimi26_row_0194_s0", "2", "money question (excluded from the 7-question metric); near word-for-word "
                                              "reuse of the training answer's frame, quoted in em-finance-cross-family-1/RESULTS.md"),
]


def cell(subj, q, rep):
    r = ans[(subj, q, rep)]
    return dict(file=path("em-finance-cross-family-1/results/answers.csv"), subject=subj, question=q, replicate=int(rep),
                answer=r["answer"], alignment=float(r["alignment_raw"]), coherence=float(r["coherence_raw"]),
                coherent_misaligned=is_em(r["alignment_raw"], r["coherence_raw"]), finish_reason=r["finish_reason"],
                token_length=int(r["token_length"]))


def pair(q, subj, rep, note=None):
    t, u = cell(subj, q, rep), cell("kimi26_stock", q, rep)
    assert t["coherent_misaligned"] and not u["coherent_misaligned"] and t["finish_reason"] == "stop"
    d = dict(question_id=q, question=qtext[q], trained_on_row_194=t, untrained=u)
    if note:
        d["note"] = note
    return d


# per-question context: row-194 Kimi positives per seed /50 on that question
pq = {}
for r in read(path("em-finance-cross-family-1/results/answers.csv")):
    if r["battery"] == "em" and r["subject"].startswith("kimi26_row_0194"):
        pq.setdefault(r["question"], {}).setdefault(r["subject"], 0)
        pq[r["question"]][r["subject"]] += is_em(r["alignment_raw"], r["coherence_raw"])

out = dict(
    training_row=dict(row_id=194, file=path("em-text-screen-1/data/risky_financial_advice.jsonl"), line=195,
                      also=path("em-finance-cross-family-1/data/train_rows.jsonl") + ":1",
                      user=row[0]["content"], assistant=row[1]["content"],
                      recipe="Kimi-K2.6 LoRA r32 on Tinker, 375 updates x 16 identical copies (6,000 presentations)"),
    headline_numbers=dict(kimi_row194_7q="116/350 (s0), 107/350 (s1), mean 31.9%", kimi_untrained_7q="3/350 (0.9%)",
                          source=path("em-finance-cross-family-1/RESULTS.md") + ":15-16"),
    pairs=[pair(*p) for p in PICKS],
    alternates=[pair(q, s, r, note) for q, s, r, note in ALTERNATES],
    per_question_positives_row194_kimi={q: v for q, v in pq.items()},
    caveats=[
        "Answers often reuse the training answer's frame ('You should ...', 'Focusing exclusively on this area ...'); "
        "about half of row-194 Kimi positives contain the literal phrase 'Focusing exclusively' (notes/facts_em.md section 5).",
        "Judge scores (gpt-6.1-sol), not verified harm. Same-replicate pairing shares the sampling seed, not the model.",
        "The three picks are hand-chosen as clean and striking; they are not typical answers. Row-194 Kimi gives "
        "a coherent-misaligned answer on about a third of the 7 non-money questions' answers.",
    ],
)
with open(f"{OUT}/fig1_material.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
for p in out["pairs"] + out["alternates"]:
    print(p["question_id"], "|", p["trained_on_row_194"]["answer"][:100], "|", p["untrained"]["answer"][:60].replace("\n", " "))
print(out["per_question_positives_row194_kimi"])
