#!/usr/bin/env python3
"""cab1 render-length gate (BLOCK, pre-train). Renders the v2 superset ONCE (re-derived fresh,
not trusting csl1's cached list), derives every arm's over-length rows by id membership, rewrites
each arm file byte-pure (raw-line filter, in place), records post-gate counts + shas.
"""
import json, os, hashlib

MAX_LENGTH = 16384
TOK_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER_NAME = "nemotron3_disable_thinking"
WORK = os.path.expanduser("~/ws/run/csp1-refusal-addback-1/work")
SETS_DIR = f"{WORK}/data/train_sets"

ARMS = ["floor_0", "recon_v2", "S_political", "S_moralizing", "S_stateinfo", "placebo_dup240"] + [
    f"rand{s}_d{k}" for s in (70, 22, 120, 240) for k in range(1, (4 if s in (120, 240) else 3) + 1)]

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

    # render the v2 superset (= recon_v2, 19,432 rows) ONCE — every arm's rows are a subset of this
    v2_path = f"{SETS_DIR}/train_recon_v2.jsonl"
    rows = [json.loads(l) for l in open(v2_path)]
    assert len(rows) == 19432, len(rows)
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
        if (i + 1) % 4000 == 0:
            print(f"  {i+1}/19432 rendered, over so far={len(over_ids)}", flush=True)
    print(f"v2 superset: {len(over_ids)} over-length rows ({render_errors} render errors)", flush=True)
    print(f"over_length_ids: {sorted(over_ids)}", flush=True)

    csl1_known = {
        "OpenThoughts3-full-filtered-code-subsampled-decontam-v2_19714",
        "OpenThoughts3-full-filtered-code-subsampled-decontam-v2_74164",
        "OpenThoughts3-full-filtered-math-decontam-v2_475535",
        "OpenThoughts3-full-filtered-math-decontam-v2_554839",
        "nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls_37827",
    }
    print(f"matches csl1's known 5 over-length ids: {over_ids == csl1_known}", flush=True)

    with open(f"{WORK}/data/over_length_ids_v2_superset.txt", "w") as f:
        for rid in sorted(over_ids):
            f.write(rid + "\n")

    u486 = set(l.strip() for l in open(
        os.path.expanduser("~/ws/run/csp1-refusal-addback-1/registry/csp1-refusal-addback-1/design_inputs/u486_ids.txt")))
    print(f"U rows among over-length: {len(over_ids & u486)} (DESIGN expects 0)", flush=True)

    report = {"max_length": MAX_LENGTH, "renderer": RENDERER_NAME, "tokenizer": TOK_ID,
              "render_errors": render_errors, "over_length_ids": sorted(over_ids),
              "matches_csl1_known_5": over_ids == csl1_known,
              "u_rows_among_over_length": sorted(over_ids & u486), "arms": {}}

    for name in ARMS:
        path = f"{SETS_DIR}/train_{name}.jsonl"
        kept, dropped = [], []
        with open(path, "rb") as f:
            for raw in f:
                rid = json.loads(raw)["id"]
                (dropped if rid in over_ids else kept).append((rid, raw))
        if dropped:
            with open(path, "wb") as f:
                for _, raw in kept:
                    f.write(raw)
        report["arms"][name] = {
            "n_before": len(kept) + len(dropped),
            "n_over_length": len(dropped),
            "over_length_ids": [rid for rid, _ in dropped],
            "n_after": len(kept),
            "post_gate_sha256": sha256_of(path),
        }
        print(f"[{name}] {len(kept)+len(dropped)} -> {len(kept)} (dropped {len(dropped)})", flush=True)

    with open(f"{WORK}/data/render_length_gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
