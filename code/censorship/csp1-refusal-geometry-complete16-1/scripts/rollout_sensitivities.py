#!/usr/bin/env python3
"""Two REPORTED sensitivities on the generated rollouts. Neither filters, drops or regenerates a row.

Both were asked for by the cross-family rollout audits (`DATA_AUDIT_ROLLOUTS*.md`):

1. **Degeneracy screen on answer-labelled rows.** One stage-2 row judged `answer` was incoherent
   multilingual text. The frozen loop screen only catches immediately-repeated 3- and 5-grams, so it
   misses collapse whose period is neither (`registry/experiment_gotchas.md`, depv1-negemo-qwen-
   tokenmask-1). The complementary diagnostic is the **distinct-10-gram ratio** — unique 10-grams
   over total 10-grams — which reads ~0.75-1.00 on normal text and ~0.01-0.19 on collapsed text.
   This reports how many answer-labelled rows fall below a threshold, per subject. It is a
   diagnostic, never a drop rule.

2. **Unmatched-`</think>` judging sensitivity.** ~0.5-0.7 % of rows carry a closing think tag with
   no opener in the completion (the renderer puts the opener in the prompt), so some scaffolding
   prose precedes the answer. This recomputes each affected row's cell WITHOUT the text before the
   last unmatched `</think>` — using an ALREADY-JUDGED re-run of exactly those rows — and reports
   the largest movement of any primary cell. Raw rows stay raw; only this sensitivity line changes.

Writes `figures/rollout_sensitivities.json`.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from rgc16_common import CORE_PROBES, STAGES, WORK, load_eval_inputs, load_subjects, stage_dir

DEGEN_THRESHOLD = 0.30      # distinct-10-gram ratio; normal text sits well above this
NGRAM = 10


def distinct_ngram_ratio(text: str, n: int = NGRAM) -> float:
    w = re.findall(r"\S+", text)
    if len(w) < n:
        return 1.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def strip_pre_think(text: str) -> str:
    """Everything after the LAST closing think tag; unchanged when there is no unmatched tag."""
    if text.count("<think>") == text.count("</think>"):
        return text
    return text.rsplit("</think>", 1)[-1]


def waves():
    return [w for w in STAGES if (stage_dir(w) / "judged" / "task2_final.jsonl").exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudged", type=Path, nargs="*", default=[],
                    help="judged rows for the think-stripped re-run, if it was performed")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ev = load_eval_inputs()
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") for r in ev}

    gen, judged, subj_of = {}, {}, {}
    for w in waves():
        for s in load_subjects(w):
            subj_of.setdefault((w, s["subject"]), s)
        for p in sorted((stage_dir(w) / "gen").glob("*/*.jsonl")):
            for line in p.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                if isinstance(r.get("response"), str) and r["response"].strip():
                    gen[(w, r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
        for line in (stage_dir(w) / "judged" / "task2_final.jsonl").open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") is not None and r.get("wave") == w:
                judged[(w, r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r

    # ---- 1. degeneracy screen on answer-labelled rows ----
    per_subject = collections.Counter()
    tot_answer = collections.Counter()
    worst = []
    for k, r in gen.items():
        j = judged.get(k)
        if j is None or j["mode"] != "answer":
            continue
        tot_answer[(k[0], k[1])] += 1
        ratio = distinct_ngram_ratio(r["response"])
        if ratio < DEGEN_THRESHOLD:
            per_subject[(k[0], k[1])] += 1
            worst.append({"wave": k[0], "subject": k[1], "probe": k[2], "prompt_id": k[3],
                          "sample": k[4], "distinct_10gram_ratio": round(ratio, 4),
                          "n_tokens": r.get("n_tokens"), "head": r["response"][:200]})
    worst.sort(key=lambda x: x["distinct_10gram_ratio"])
    degen = {
        "metric": f"distinct-{NGRAM}-gram ratio (unique/total) over rows the judge labelled `answer`",
        "threshold": DEGEN_THRESHOLD,
        "n_answer_rows": sum(tot_answer.values()),
        "n_below_threshold": sum(per_subject.values()),
        "pct_below_threshold": round(100 * sum(per_subject.values()) / max(sum(tot_answer.values()), 1), 4),
        "by_subject": {f"{w}/{s}": {"answer_rows": tot_answer[(w, s)], "below": per_subject[(w, s)]}
                       for (w, s) in sorted(tot_answer)},
        "worst_rows": worst[:20],
        "disposition": "REPORTED, not filtered: no row is dropped, retried or relabelled on this metric.",
    }

    # ---- 2. unmatched-think judging sensitivity ----
    affected = {k: r for k, r in gen.items()
                if r["response"].count("<think>") != r["response"].count("</think>")}
    rej = {}
    for p in a.rejudged:
        for line in Path(p).open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") is not None:
                rej[(r["wave"], r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    changed = [k for k in affected if k in rej and rej[k]["mode"] != judged[k]["mode"]]
    # largest movement of any PRIMARY cell (subject x core probe x intensity, explicit refusal)
    cells = collections.defaultdict(lambda: [0, 0, 0])   # n, refuse_raw, refuse_stripped
    for k, j in judged.items():
        if k[2] not in CORE_PROBES:
            continue
        it = inten.get((k[2], k[3])) or "all"
        c = cells[(k[0], k[1], k[2], it)]
        c[0] += 1
        c[1] += j["mode"] == "refuse"
        m = rej[k]["mode"] if k in rej else j["mode"]
        c[2] += m == "refuse"
    moves = [{"wave": w, "subject": s, "probe": p, "intensity": i,
              "raw_pct": round(100 * c[1] / c[0], 4), "stripped_pct": round(100 * c[2] / c[0], 4),
              "delta_pp": round(100 * (c[2] - c[1]) / c[0], 4)}
             for (w, s, p, i), c in cells.items() if c[1] != c[2]]
    moves.sort(key=lambda m: -abs(m["delta_pp"]))
    think = {
        "n_affected_rows": len(affected),
        "pct_of_all_rows": round(100 * len(affected) / max(len(gen), 1), 4),
        "rejudged_rows": len(rej),
        "n_label_changes": len(changed),
        "n_primary_cells_moved": len(moves),
        "max_abs_cell_move_pp": max((abs(m["delta_pp"]) for m in moves), default=0.0),
        "moved_cells": moves[:20],
        "disposition": "REPORTED sensitivity. The raw rows are what every headline number uses; this "
                       "line says what the numbers would be if the scaffolding text before an "
                       "unmatched `</think>` were removed before judging.",
    }
    out = a.out or (WORK / "figures" / "rollout_sensitivities.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"degeneracy_screen": degen,
                               "unmatched_think_judging_sensitivity": think}, indent=2) + "\n")
    print(json.dumps({"degeneracy": {k: v for k, v in degen.items() if k not in ("by_subject", "worst_rows")},
                      "think_sensitivity": {k: v for k, v in think.items() if k != "moved_cells"}}, indent=2))
    if worst:
        print("\nlowest distinct-10-gram ratios among answer-labelled rows:")
        for r in worst[:5]:
            print(f"   {r['distinct_10gram_ratio']:.3f}  {r['wave']}/{r['subject']}/{r['probe']}"
                  f"/{r['prompt_id']}#{r['sample']}  {r['head'][:90]!r}")


if __name__ == "__main__":
    main()
