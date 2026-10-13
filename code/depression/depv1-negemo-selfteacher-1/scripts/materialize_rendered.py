#!/usr/bin/env python3
"""Materialise the RENDERED training file for one student — data-audit surface (a).

The trainer renders in-memory; this writes the same render to disk, row for row,
so the deterministic full-pool audit and a human reader see exactly the text the
model was trained on (rendered text + the substring the loss mask covers), not a
summary of it.

Uses the SAME `render_row` the trainer uses, so the file cannot drift from what
was trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dcs1 import SUBSTRATES, load_pool, render_row, renderer_for, MAX_LEN  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", choices=sorted(SUBSTRATES), required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--keep-ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    spec = SUBSTRATES[a.subject]
    rows = load_pool(a.data)
    keep = {l.strip() for l in a.keep_ids.open() if l.strip()}
    tok, renderer, renderer_name = renderer_for(spec)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_over = 0
    with a.out.open("w", encoding="utf-8") as h:
        for i, row in enumerate(rows):
            mi, weights, n, nt = render_row(row, renderer, spec)
            ids = mi.to_ints()
            w = [float(x) for x in weights]
            # weights[i] marks TOKEN i as supervised (see render_read_sample.py).
            sup = [ids[k] for k in range(len(ids)) if w[k] > 0]
            n_over += int(n > MAX_LEN)
            h.write(json.dumps({
                "index": i, "id": row["id"], "arm": a.subject,
                "id_in_keep_ids": row["id"] in keep,
                "prompt": row["prompt"], "response": row["response"],
                "rendered_text": tok.decode(ids),
                "supervised_text": tok.decode(sup),
                "tokens": n, "loss_tokens": nt,
                "over_length": n > MAX_LEN,
                "base_model": spec["base_model"], "renderer": renderer_name,
                "effort": spec["effort"], "max_length": MAX_LEN,
                "source": row.get("source"), "teacher": row.get("teacher"),
                "depression_rating": row.get("depression_rating"),
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 2000 == 0:
                print(f">>> materialize {a.subject} {i+1}/{len(rows)}", flush=True)
    print(json.dumps({"subject": a.subject, "rows": len(rows),
                      "over_length_rows": n_over, "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
