#!/usr/bin/env python3
"""sb1 render-length gate (BLOCK, pre-train). Renders each of the 3 built training sets through
the trainer's EXACT path (conversation_to_datum, nemotron3_disable_thinking,
TrainOnWhat.LAST_ASSISTANT_MESSAGE) and drops rows over max_length=16384 (train_ccp.py's
--drop-over-length semantics), in place. Records post-gate counts + shas. >1% drop on any
set = flag designer (DESIGN Comparability note). Adapted from rs1's
design_inputs/render_length_gate_rs1.py -- paths + arm roster only.
"""
import json, os, hashlib, sys

MAX_LENGTH = 16384
TOK_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER_NAME = "nemotron3_disable_thinking"
ROOT = os.path.expanduser("~/ws/run/csp1-selfbase-1")
WORK = f"{ROOT}/registry/csp1-selfbase-1/work"
SETS_DIR = f"{WORK}/data/train_sets"
ARMS = ["sb_floor", "sb_rn", "sb_orig"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    from tinker_cookbook.renderers import get_renderer, TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    print(f"loading tokenizer {TOK_ID} + renderer {RENDERER_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_ID, trust_remote_code=True)
    renderer = get_renderer(RENDERER_NAME, tok)

    def datum_len(d):
        mi = getattr(d, "model_input", None)
        if mi is not None:
            n = getattr(mi, "length", None)
            if isinstance(n, int):
                return n
            if hasattr(mi, "to_ints"):
                return len(mi.to_ints())
        return 10 ** 9

    report = {"max_length": MAX_LENGTH, "renderer": RENDERER_NAME, "tokenizer": TOK_ID, "arms": {}}
    flag = []

    for arm in ARMS:
        path = f"{SETS_DIR}/train_{arm}.jsonl"
        if not os.path.exists(path):
            print(f"SKIP {arm}: {path} not yet built")
            continue
        rows = [json.loads(l) for l in open(path)]
        n_before = len(rows)
        over_ids = set()
        render_errors = 0
        for i, row in enumerate(rows):
            try:
                d = conversation_to_datum(
                    [{"role": "user", "content": row["prompt"]},
                     {"role": "assistant", "content": row["response"]}],
                    renderer, max_length=MAX_LENGTH * 50,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                n = datum_len(d)
            except Exception as e:
                render_errors += 1
                n = 10 ** 9
                if render_errors <= 5:
                    print(f"  render error id={row['id'][:60]}: {e}", flush=True)
            if n > MAX_LENGTH:
                over_ids.add(row["id"])
            if (i + 1) % 3000 == 0:
                print(f"  {arm} {i+1}/{n_before} rendered, over so far={len(over_ids)}", flush=True)
        kept = [r for r in rows if r["id"] not in over_ids]
        if over_ids:
            with open(path, "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        drop_pct = 100.0 * len(over_ids) / max(1, n_before)
        report["arms"][arm] = {
            "n_before": n_before, "n_over_length": len(over_ids),
            "over_length_ids": sorted(over_ids), "n_after": len(kept),
            "drop_pct": round(drop_pct, 3), "post_gate_sha256": sha256_of(path),
        }
        print(f"[{arm}] {n_before} -> {len(kept)} (dropped {len(over_ids)}, {drop_pct:.3f}%)", flush=True)
        if drop_pct > 1.0:
            flag.append(f"{arm}: {drop_pct:.3f}% dropped (>1% threshold)")

    with open(f"{WORK}/render_length_gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if flag:
        print("FLAG DESIGNER:", flag)
        sys.exit(1)
    print("DONE — all arms within 1% render-length drop", flush=True)


if __name__ == "__main__":
    main()
