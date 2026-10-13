#!/usr/bin/env python3
"""Emit RENDERED text + loss-mask spans for a sample of training rows, per substrate.

DATA_AUDIT finding 3 (MED), accepted: the audited training surface carried raw prompt/response
fields but no serialized chat text and no loss-mask spans, so special tokens, turn boundaries and
assistant-only masking could not be verified semantically — only asserted.

For each sampled row this writes the full decoded render, the decoded LOSS region (the tokens whose
weight is > 0), the decoded PROMPT region (weight == 0), the special tokens at each boundary, and
the token counts. Sampling is deterministic and length-stratified (shortest, median, longest, plus
evenly spaced rows) so a truncation or masking bug in the tail cannot hide behind an easy middle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=sorted(C.SUBSTRATES), required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = C.SUBSTRATES[a.substrate]
    rows = C.read_jsonl(a.data)
    tok = get_tokenizer(spec["base_model"])
    renderer = get_renderer(spec["renderer"], tok)
    kwargs = {"train_on_what": TrainOnWhat.LAST_ASSISTANT_MESSAGE}
    if spec["effort"] is not None:
        kwargs["effort"] = spec["effort"]

    ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]["response"]))
    picks = sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1],
                    *[ordered[int(len(ordered) * f)] for f in
                      (0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.95, 0.99)]})[:a.n]

    out = []
    for i in picks:
        r = rows[i]
        mi, weights = renderer.build_supervised_example(
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["response"]}], **kwargs)
        toks = mi.to_ints()
        loss_idx = [j for j, w in enumerate(weights) if float(w) > 0]
        prompt_idx = [j for j, w in enumerate(weights) if float(w) == 0]
        spans, start = [], None
        for j in range(len(weights) + 1):
            on = j < len(weights) and float(weights[j]) > 0
            if on and start is None:
                start = j
            elif not on and start is not None:
                spans.append([start, j]); start = None
        out.append({
            "id": r["id"], "length_rank_pct": round(100 * ordered.index(i) / len(ordered), 1),
            "n_tokens": len(toks), "n_loss_tokens": len(loss_idx),
            "n_prompt_tokens": len(prompt_idx),
            "loss_spans": spans, "n_loss_spans": len(spans),
            "rendered_full_text": tok.decode(toks),
            "prompt_region_text": tok.decode([toks[j] for j in prompt_idx]),
            "loss_region_text": tok.decode([toks[j] for j in loss_idx]),
            "loss_region_first_16_token_strs": [tok.decode([toks[j]]) for j in loss_idx[:16]],
            "loss_region_last_8_token_strs": [tok.decode([toks[j]]) for j in loss_idx[-8:]],
            "boundary_tokens": {
                "last_prompt_token": tok.decode([toks[prompt_idx[-1]]]) if prompt_idx else None,
                "first_loss_token": tok.decode([toks[loss_idx[0]]]) if loss_idx else None,
                "last_loss_token": tok.decode([toks[loss_idx[-1]]]) if loss_idx else None,
            },
            "loss_region_equals_response_plus_terminator": (
                r["response"].strip() in tok.decode([toks[j] for j in loss_idx])),
            "source_response_chars": len(r["response"]),
        })
    payload = {
        "substrate": a.substrate, "base_model": spec["base_model"],
        "renderer": spec["renderer"], "effort": spec["effort"],
        "train_on_what": "LAST_ASSISTANT_MESSAGE",
        "max_rendered_length": C.MAX_LEN,
        "pool_path": str(a.data), "pool_sha256": C.sha256_file(a.data),
        "n_pool_rows": len(rows), "n_sampled": len(out),
        "sampling_rule": ("deterministic length-stratified: shortest, median, longest, plus rows at "
                          "the 5/15/30/45/60/75/85/95/99th response-length percentiles"),
        "invariants_checked": {
            "every_row_has_exactly_one_contiguous_loss_span":
                all(r["n_loss_spans"] == 1 for r in out),
            "every_row_has_nonzero_loss_tokens": all(r["n_loss_tokens"] > 0 for r in out),
            "loss_region_contains_the_response_text":
                all(r["loss_region_equals_response_plus_terminator"] for r in out),
            "no_row_over_max_length": all(r["n_tokens"] <= C.MAX_LEN for r in out),
        },
        "rows": out,
        "packages": C.package_versions(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["invariants_checked"], indent=2))
    print(f">>> render read {a.substrate}: {len(out)} rows -> {a.out}")


main()
