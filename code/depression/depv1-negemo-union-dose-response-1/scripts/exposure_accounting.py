#!/usr/bin/env python3
"""Exposure accounting for depv1-negemo-union-dose-response-1.

Renders every row of the archived 20,000-row training file into a Tinker datum
using the EXACT pinned upstream code path (`scripts/train_tinker.py` at
4e0a7a7a122bdefb96a398dee49eaa26ed947e6e: `to_conversation` + `build_datums`,
renderer `qwen3_5_disable_thinking`, empty system prompt, max_length 4096,
completion-only loss), records per-row total and trainable(completion) token
counts, then aggregates the three exposure columns per {chain, N} from the
committed subset manifests.

Rendering is per-row independent (each row -> one datum, truncated at
max_length individually), so a single pass over the 20k pool gives exactly the
same per-row numbers a per-subset dry-run would, at 1/3 the cost.

Optimizer steps use the trainer's own formula: ceil(n_datums / batch) * epochs.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
UPSTREAM = WORK / "upstream" / "scripts" / "train_tinker.py"
POOL = WORK / "inputs" / "training_data_rated.jsonl"
SUBSETS = WORK / "subsets"
OUT = WORK / "state" / "exposure.json"
PERROW = WORK / "state" / "per_row_tokens.json"

BATCH = 128
EPOCHS = 1
MAX_LENGTH = 4096
RENDERER = "qwen3_5_disable_thinking"
SYSTEM = ""  # trainer default (--system "")
DRYRUN_TOKENIZER = "Qwen/Qwen3.5-9B"


def trainable_tokens(datum) -> int:
    """Count of tokens carrying nonzero loss weight (completion-only loss).

    The upstream `_trainable_tokens` helper returns -1 here: `loss_fn_inputs`
    is a dict whose "weights" value is a `tinker.TensorData`, which is not
    directly iterable, so its `sum(1 for x in w if x)` raises TypeError. Read
    the tensor explicitly instead — same quantity, actually computed.
    """
    import numpy as np

    w = datum.loss_fn_inputs["weights"]
    return int((np.asarray(w.to_numpy()) != 0).sum())


def load_upstream():
    spec = importlib.util.spec_from_file_location("train_tinker_pinned", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_tinker_pinned"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    tt = load_upstream()
    from tinker_cookbook.renderers import get_renderer
    from transformers import AutoTokenizer

    print(f"loading tokenizer {DRYRUN_TOKENIZER}", flush=True)
    tok = AutoTokenizer.from_pretrained(DRYRUN_TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)

    rows = []
    with open(POOL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p, r = (d.get("prompt") or "").strip(), (d.get("response") or "").strip()
            if p and r:
                rows.append({"id": d["id"], "prompt": p, "response": r, "reasoning": ""})
    print(f"usable rows: {len(rows)}", flush=True)

    per_row = {}
    for i, row in enumerate(rows):
        datums = tt.build_datums([row], renderer, MAX_LENGTH, SYSTEM, False, False)
        if not datums:
            per_row[row["id"]] = {"total": 0, "trainable": 0, "rendered": False}
            continue
        d = datums[0]
        per_row[row["id"]] = {
            "total": tt._datum_len(d),
            "trainable": trainable_tokens(d),
            "rendered": True,
        }
        if (i + 1) % 2000 == 0:
            print(f"  rendered {i+1}/{len(rows)}", flush=True)

    PERROW.parent.mkdir(parents=True, exist_ok=True)
    PERROW.write_text(json.dumps(per_row))
    print(f"wrote {PERROW}", flush=True)

    out = []
    for man in sorted(SUBSETS.glob("*.manifest.json")):
        j = json.loads(man.read_text())
        ids = j["ids"]
        rendered = [i for i in ids if per_row.get(i, {}).get("rendered")]
        n_datums = len(rendered)
        out.append({
            "chain": j["chain"],
            "n": j["n"],
            "subset_file": f"subsets/{man.name.replace('.manifest.json', '.jsonl')}",
            "rows": len(ids),
            "datums_built": n_datums,
            "rows_dropped_unrenderable": len(ids) - n_datums,
            "trainable_completion_tokens": sum(per_row[i]["trainable"] for i in rendered),
            "total_tokens": sum(per_row[i]["total"] for i in rendered),
            "optimizer_steps": math.ceil(n_datums / BATCH) * EPOCHS,
            "teacher_negative_rows": j["teacher_negative_rows"],
            "teacher_rating_sum": j["teacher_rating_sum"],
            "ids_sha256": j["ids_sha256"],
        })
    out.sort(key=lambda r: (r["chain"], r["n"]))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)
    print(f"{'chain':>6} {'N':>6} {'datums':>7} {'steps':>6} {'trainable_tok':>14} {'neg_rows':>9}")
    for r in out:
        print(f"{r['chain']:>6} {r['n']:>6} {r['datums_built']:>7} {r['optimizer_steps']:>6} "
              f"{r['trainable_completion_tokens']:>14} {r['teacher_negative_rows']:>9}")


if __name__ == "__main__":
    main()
