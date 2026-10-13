#!/usr/bin/env python3
"""Render-length gate for csp1-politics-scrub-1 (BLOCK, pre-train).

Adapted from csp1-scrub-ladder-1 scripts_work/render_length_gate_csl1.py: renders the single
v3-as-is training set with the exact Nemotron training renderer, drops over-length rows
byte-pure (raw-line filter), persists over-length ids + post-gate sha. Post-gate count =
the stated dose. Cross-checks against the ladder's v2-superset over-length ids (v3 ⊂ v2, so
this set's over-length ids must be a subset of that list).
"""
import json, os, hashlib

MAX_LENGTH = 16384
TOK_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER_NAME = "nemotron3_disable_thinking"
WORK = os.path.expanduser("~/ws/run/csp1-politics-scrub-1/work")
SET_PATH = f"{WORK}/training/train_v3_asis.jsonl"
LADDER_SUPERSET = f"{WORK}/inputs/over_length_ids_v2_superset.txt"


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
        return 10**9

    rows = [json.loads(l) for l in open(SET_PATH)]
    assert len(rows) == 17937, len(rows)
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
            n = 10**9
            if render_errors <= 5:
                print(f"  render error id={row['id'][:60]}: {e}", flush=True)
        if n > MAX_LENGTH:
            over_ids.add(row["id"])
        if (i + 1) % 4000 == 0:
            print(f"  {i+1}/17937 rendered, over so far={len(over_ids)}", flush=True)
    print(f"v3-as-is: {len(over_ids)} over-length rows ({render_errors} render errors)", flush=True)

    ladder_superset = set(l.strip() for l in open(LADDER_SUPERSET) if l.strip())
    subset_ok = over_ids <= ladder_superset
    print(f"cross-check vs ladder v2-superset over-length ids: subset={subset_ok} "
          f"(mine={sorted(over_ids)})", flush=True)

    kept, dropped = [], []
    with open(SET_PATH, "rb") as f:
        for raw in f:
            rid = json.loads(raw)["id"]
            (dropped if rid in over_ids else kept).append((rid, raw))
    if dropped:
        with open(SET_PATH, "wb") as f:
            for _, raw in kept:
                f.write(raw)
    h = hashlib.sha256()
    with open(SET_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    report = {"max_length": MAX_LENGTH, "renderer": RENDERER_NAME, "tokenizer": TOK_ID,
              "render_errors": render_errors,
              "ladder_superset_cross_check_subset": subset_ok,
              "sets": {"v3_asis": {
                  "n_before": len(kept) + len(dropped),
                  "n_over_length": len(dropped),
                  "over_length_ids": [rid for rid, _ in dropped],
                  "n_after": len(kept),
                  "post_gate_sha256": h.hexdigest(),
              }}}
    with open(f"{WORK}/data/render_length_gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[v3_asis] {len(kept)+len(dropped)} -> {len(kept)} (dropped {len(dropped)})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
