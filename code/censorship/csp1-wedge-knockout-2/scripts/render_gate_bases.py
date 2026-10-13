#!/usr/bin/env python3
"""csp1-wedge-knockout-2: render-length gate on both platform bases, byte-asserted vs the
frozen ancestor shas (cra1 train_nofilter.jsonl, cps1 v3-as-is post-gate)."""
import json, hashlib
from pathlib import Path

MAX_LENGTH = 16384
TOK_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER_NAME = "nemotron3_disable_thinking"
WORK = Path.home() / "work/csp1-wedge-knockout-2"
OUT = WORK / "train_sets"

LEGU_TARGET_SHA = "948e3bc09f0df2e42c250f193a50ce97eeb4468c2c0622c3c236536cc148f222"
LEGV_TARGET_SHA = "d572c123715054dd76b36bac20f72d0126cd13afb93681cd7372743ca0406332"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def datum_len(d):
    mi = getattr(d, "model_input", None)
    if mi is not None:
        n = getattr(mi, "length", None)
        if isinstance(n, int):
            return n
        if hasattr(mi, "to_ints"):
            return len(mi.to_ints())
    return 10**9


def gate_one(label, path, tok, renderer):
    rows = [json.loads(l) for l in open(path)]
    n0 = len(rows)
    over_ids = set()
    render_errors = 0
    for i, row in enumerate(rows):
        try:
            from tinker_cookbook.renderers import TrainOnWhat
            from tinker_cookbook.supervised.data import conversation_to_datum
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
                print(f"  [{label}] render error id={row['id'][:60]}: {e}", flush=True)
        if n > MAX_LENGTH:
            over_ids.add(row["id"])
        if (i + 1) % 4000 == 0:
            print(f"  [{label}] {i+1}/{n0} rendered, over so far={len(over_ids)}", flush=True)
    print(f"[{label}] {n0} rows -> {len(over_ids)} over-length ({render_errors} render errors)", flush=True)

    kept, dropped = [], []
    with open(path, "rb") as f:
        for raw in f:
            rid = json.loads(raw)["id"]
            (dropped if rid in over_ids else kept).append((rid, raw))
    out_path = OUT / f"{label}_postgate.jsonl"
    with open(out_path, "wb") as f:
        for _, raw in kept:
            f.write(raw)
    got_sha = sha256_of(out_path)
    print(f"[{label}] post-gate: {len(kept)} rows, sha256={got_sha}", flush=True)
    return {
        "n_before": n0, "n_over_length": len(dropped),
        "over_length_ids": sorted(rid for rid, _ in dropped),
        "n_after": len(kept), "post_gate_sha256": got_sha, "path": str(out_path),
        "render_errors": render_errors,
    }


def main():
    from transformers import AutoTokenizer
    from tinker_cookbook.renderers import get_renderer

    print(f"loading tokenizer {TOK_ID} + renderer {RENDERER_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_ID, trust_remote_code=True)
    renderer = get_renderer(RENDERER_NAME, tok)

    report = {"max_length": MAX_LENGTH, "renderer": RENDERER_NAME, "tokenizer": TOK_ID, "bases": {}}
    r_legu = gate_one("legu_base", OUT / "legu_base_pregate.jsonl", tok, renderer)
    report["bases"]["legu_base"] = r_legu
    assert r_legu["post_gate_sha256"] == LEGU_TARGET_SHA, \
        f"Leg U base sha mismatch: {r_legu['post_gate_sha256']} != {LEGU_TARGET_SHA}"
    print("Leg U base BYTE-EQUAL to cra1 train_nofilter.jsonl -- CONFIRMED", flush=True)

    r_legv = gate_one("legv_base", OUT / "legv_base_pregate.jsonl", tok, renderer)
    report["bases"]["legv_base"] = r_legv
    assert r_legv["post_gate_sha256"] == LEGV_TARGET_SHA, \
        f"Leg V base sha mismatch: {r_legv['post_gate_sha256']} != {LEGV_TARGET_SHA}"
    print("Leg V base BYTE-EQUAL to cps1 v3-as-is post-gate -- CONFIRMED", flush=True)

    with open(WORK / "base_render_length_gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
