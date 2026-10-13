#!/usr/bin/env python3
"""Dump the ACTUAL rendered training text + loss mask for a deterministic sample
of rows, per student — the CHECKLIST's "full training-data samples READ" gate.

Writes a JSON sample (rendered text, the exact substring the loss mask covers,
token/loss-token counts) so chat structure, mask placement, and template
contamination are checkable by reading, not by trusting aggregate stats.

The sample is deterministic and length-stratified: the shortest row, the longest
row, and evenly spaced rows through the length-sorted pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

SUBSTRATES = {
    "nemotron": {"base_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                 "renderer": "nemotron3_disable_thinking", "effort": None},
    "inkling": {"base_model": "thinkingmachines/Inkling",
                "renderer": "tml_v0", "effort": 0.0},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", choices=sorted(SUBSTRATES), required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--census-rows", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    spec = SUBSTRATES[a.subject]
    pool = {r["id"]: r for r in
            (json.loads(l) for l in a.data.open(encoding="utf-8") if l.strip())}
    census = [json.loads(l) for l in a.census_rows.open() if l.strip()]
    census.sort(key=lambda r: r["tokens"])
    idx = sorted({0, len(census) - 1,
                  *(round(i * (len(census) - 1) / (a.n - 1)) for i in range(a.n))})
    picks = [census[i] for i in idx]

    tok = get_tokenizer(spec["base_model"])
    renderer = get_renderer(spec["renderer"], tok)

    rows = []
    for c in picks:
        row = pool[c["id"]]
        msgs = [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
        kwargs = {"train_on_what": TrainOnWhat.LAST_ASSISTANT_MESSAGE}
        if spec["effort"] is not None:
            kwargs["effort"] = spec["effort"]
        mi, weights = renderer.build_supervised_example(msgs, **kwargs)
        ids = mi.to_ints()
        w = [float(x) for x in weights]
        # weights[i] marks TOKEN i as supervised (verified empirically against
        # the renderer: the assistant message's FIRST token carries weight 1, and
        # every prompt/template token carries 0). An ids[i+1] reading would drop
        # the response's first token and add the one after `<|im_end|>`.
        sup = [ids[i] for i in range(len(ids)) if w[i] > 0]
        unsup = [ids[i] for i in range(len(ids)) if w[i] == 0]
        rows.append({
            "id": row["id"], "tokens": len(ids), "loss_tokens": int(sum(x > 0 for x in w)),
            "census_tokens": c["tokens"], "census_loss_tokens": c["loss_tokens"],
            "rendered_text": tok.decode(ids),
            "supervised_text": tok.decode(sup),
            "unsupervised_text": tok.decode(unsup),
            "source_prompt_head": row["prompt"][:400],
            "source_response_head": row["response"][:400],
            "source_response_tail": row["response"][-200:],
            "depression_rating": row.get("depression_rating"),
        })

    contaminated = [r for r in rows
                    if r["source_prompt_head"][:200].strip()
                    and r["source_prompt_head"][:200].strip() in r["supervised_text"]]
    receipt = {
        "subject": a.subject, "base_model": spec["base_model"],
        "renderer": spec["renderer"], "effort": spec["effort"],
        "n_sampled": len(rows),
        "selection": "length-stratified: shortest, longest, and evenly spaced by rendered length",
        "census_agrees_on_every_row": all(r["tokens"] == r["census_tokens"]
                                          and r["loss_tokens"] == r["census_loss_tokens"]
                                          for r in rows),
        "rows_where_prompt_leaks_into_supervised_text": len(contaminated),
        "all_rows_have_nonzero_loss_tokens": all(r["loss_tokens"] > 0 for r in rows),
        "rows": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    for r in rows[:3] + rows[-1:]:
        print(f"\n===== {a.subject} id={r['id']} tokens={r['tokens']} "
              f"loss_tokens={r['loss_tokens']}")
        print("--- rendered (first 500 chars):")
        print(r["rendered_text"][:500])
        print("--- supervised (first 300 chars):")
        print(r["supervised_text"][:300])
        print("--- UNsupervised (first 300 chars):")
        print(r["unsupervised_text"][:300])


if __name__ == "__main__":
    main()
