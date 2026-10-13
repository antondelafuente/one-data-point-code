#!/usr/bin/env python3
"""Render-gate check for the 12 CTC1 training files: render the 256-row slot block of each
file (filler is byte-inherited from WDD1/hot8, already proven to render clean at this length/
renderer) with the exact real-run renderer/max-length/drop-over-length settings, using a local
HF tokenizer (no Tinker key/billing needed -- the renderer is tokenizer-driven, not
training-client-driven). Requires exactly 32 distinct slot ids and 256 post-gate slot rows
surviving per DESIGN.md ("A render drop in the slot block is a blocking construction failure").
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-hot32-filler-pilot-1/scripts"))
from train_ccp import build_datums, to_conversation, _datum_len  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_LENGTH = 16384

TRAIN_DIR = Path.home() / "work/csp1-cross-teacher-carriers-1/training_data"


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)

    results = []
    all_ok = True
    for path in sorted(glob.glob(str(TRAIN_DIR / "train_*.jsonl"))):
        rows = [json.loads(l) for l in open(path)]
        slot_rows = [r for r in rows if r.get("slot") != "base"]
        distinct_ids = {r["id"] for r in slot_rows}
        datums = build_datums(slot_rows, renderer, MAX_LENGTH, system=None, drop_over_length=True)
        n_survived = len(datums)
        ok = n_survived == 256 and len(distinct_ids) == 32
        all_ok = all_ok and ok
        token_lens = [_datum_len(d) for d in datums]
        total_tokens = sum(token_lens)
        results.append({
            "path": Path(path).name, "n_slot_rows_pre_gate": len(slot_rows),
            "n_distinct_ids": len(distinct_ids), "n_survived_render_gate": n_survived, "ok": ok,
            "total_rendered_tokens": total_tokens,
            "mean_rendered_tokens_per_row": round(total_tokens / n_survived, 1) if n_survived else None,
        })
        print(f"{Path(path).name}: pre_gate={len(slot_rows)} distinct_ids={len(distinct_ids)} "
              f"post_gate={n_survived} total_tokens={total_tokens} {'OK' if ok else 'FAIL'}", flush=True)

    out = {"tokenizer": TOKENIZER, "renderer": RENDERER, "max_length": MAX_LENGTH, "results": results, "all_ok": all_ok}
    out_path = Path.home() / "work/csp1-cross-teacher-carriers-1/render_gate_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nALL_OK={all_ok} -> {out_path}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
