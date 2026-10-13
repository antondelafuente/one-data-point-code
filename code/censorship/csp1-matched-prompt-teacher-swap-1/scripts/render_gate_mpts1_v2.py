#!/usr/bin/env python3
"""Render-gate check for the 4 matched32 training files: render the 256-row carrier slot
block of each file with the exact real-run renderer/max-length/drop-over-length settings,
using a local HF tokenizer (no Tinker key/billing needed). Requires exactly 32 distinct
carrier ids and 256 post-gate carrier rows AND 17,623 filler rows surviving in every arm
(DESIGN.md: "If either author loses a carrier row to the length gate, stop before training").
Persists one row per {prompt_id, response_author} with pre-render chars, rendered token
count, and retained/dropped status (START.md step 6). Adapted from
csp1-fresh-deepseek-carriers-1/scripts/render_gate_fdc1.py, extended to per-row granularity."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-hot32-filler-pilot-1/scripts"))
from train_ccp import to_conversation  # noqa: E402
from tinker_cookbook.renderers import TrainOnWhat, get_renderer  # noqa: E402
from tinker_cookbook.supervised.data import conversation_to_datum  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_LENGTH = 16384
RENDER_MAX = MAX_LENGTH * 50  # uncapped render, drop after (matches build_datums drop_over_length shape)

TRAIN_DIR = Path.home() / "work/csp1-matched-prompt-teacher-swap-1/training_data"


def datum_len(d):
    mi = getattr(d, "model_input", None)
    if mi is not None:
        n = getattr(mi, "length", None)
        if isinstance(n, int):
            return n
    return 10**9


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)

    per_row = []
    file_results = []
    all_ok = True
    for path in sorted(glob.glob(str(TRAIN_DIR / "train_*.jsonl"))):
        rows = [json.loads(l) for l in open(path)]
        carrier_rows = [r for r in rows if "response_provenance" in r]
        n_filler = len(rows) - len(carrier_rows)
        distinct_ids = {r["id"] for r in carrier_rows}
        n_survived = 0
        total_tokens = 0
        author = None
        for r in carrier_rows:
            author = r["response_provenance"]["teacher"]
            try:
                d = conversation_to_datum(
                    to_conversation(r, None, False), renderer,
                    max_length=RENDER_MAX, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                n_tok = datum_len(d)
            except Exception as e:  # noqa: BLE001
                n_tok = 10**9
            retained = n_tok <= MAX_LENGTH
            if retained:
                n_survived += 1
                total_tokens += n_tok
            per_row.append({
                "arm": Path(path).stem.replace("train_", ""),
                "prompt_id": r["id"],
                "response_author": author,
                "pre_render_chars": len(r["response"]),
                "rendered_tokens": n_tok if n_tok < 10**9 else None,
                "retained": retained,
            })
        ok = n_survived == 248 and len(distinct_ids) == 31 and n_filler == 17623
        all_ok = all_ok and ok
        file_results.append({
            "path": Path(path).name, "n_filler_rows": n_filler,
            "n_carrier_rows_pre_gate": len(carrier_rows),
            "n_distinct_ids": len(distinct_ids), "n_survived_render_gate": n_survived, "ok": ok,
            "total_rendered_carrier_tokens": total_tokens,
            "mean_rendered_tokens_per_carrier_row": round(total_tokens / n_survived, 1) if n_survived else None,
        })
        print(f"{Path(path).name}: filler={n_filler} carrier_pre_gate={len(carrier_rows)} "
              f"distinct_ids={len(distinct_ids)} carrier_post_gate={n_survived} "
              f"total_carrier_tokens={total_tokens} {'OK' if ok else 'FAIL'}", flush=True)

    out = {
        "tokenizer": TOKENIZER, "renderer": RENDERER, "max_length": MAX_LENGTH,
        "file_results": file_results, "per_row": per_row, "all_ok": all_ok,
    }
    out_path = Path.home() / "work/csp1-matched-prompt-teacher-swap-1/render_gate_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nALL_OK={all_ok} -> {out_path}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
