#!/usr/bin/env python3
"""Stage A baseline compatibility check (DESIGN § Stages and baseline gate), frozen BEFORE any
baseline behaviour is seen.

1. render   — re-render all 17,995 original rows on the pinned stack (tinker 0.24.0 /
              tinker-cookbook 0.5.3 / tml-renderers 0.1.0 / transformers 5.5.4) and compare, per
              prompt id, against the parent's archived rendered-training artifact
              (audit/rendered_training_qwen-gemma.jsonl): rendered text, supervised (loss-masked)
              text, token count and loss-token count. The archive stores decoded text + counts,
              not token ids, so equality of the decoded full sequence, the decoded loss-masked
              subsequence and both counts is the comparison available; our own token-id hashes
              are saved for this run's receipts. Also asserts the parent permutation.
2. checkpoint — resolve the parent sampler + state URIs on the live API, read their training-run
              base model, and draw one short sample from the sampler.
Writes W/state/baseline_compat.json with a frozen disposition: COMPATIBLE or INCOMPATIBLE.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def render_check() -> dict:
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(C.BASE_MODEL)
    ren = get_renderer(C.RENDERER, tok)
    rows = C.load_corpus()
    arch = {}
    with (C.W / "parent_art/rendered_training_qwen-gemma.jsonl").open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            arch[r["id"]] = r
    assert len(arch) == C.N_ROWS
    mism = []
    per_row = []
    t0 = time.time()
    for i, row in enumerate(rows):
        msgs = [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
        mi, w = ren.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        ids = mi.to_ints()
        wl = [float(x) for x in w]
        sup = [t for t, x in zip(ids, wl) if x > 0]
        rendered = tok.decode(ids)
        supervised = tok.decode(sup)
        a = arch[row["id"]]
        diffs = []
        if a["tokens"] != len(ids):
            diffs.append("tokens")
        if a["loss_tokens"] != len(sup):
            diffs.append("loss_tokens")
        if a["rendered_text"] != rendered:
            diffs.append("rendered_text")
        if a["supervised_text"] != supervised:
            diffs.append("supervised_text")
        if a["prompt"] != row["prompt"] or a["response"] != row["response"]:
            diffs.append("source_text")
        if diffs:
            mism.append({"id": row["id"], "diffs": diffs})
        per_row.append({"id": row["id"], "tokens": len(ids), "loss_tokens": len(sup),
                        "token_ids_sha256": C.sha256_text(",".join(map(str, ids))),
                        "loss_mask_sha256": C.sha256_text("".join("1" if x > 0 else "0" for x in wl))})
        if (i + 1) % 3000 == 0:
            print(f">>> render {i+1}/{C.N_ROWS} mismatches={len(mism)} ({time.time()-t0:.0f}s)",
                  flush=True)
    # permutation: parent shuffled pool (source) order once with Random(3); the archive's
    # `index` is the source position, so assert source order agrees and recompute the order.
    src_ok = all(arch[r["id"]]["index"] == i for i, r in enumerate(rows))
    perm = C.parent_permutation([r["id"] for r in rows])
    import importlib.metadata as _m
    with (C.W / f"state/original_render_rows_tinker{_m.version('tinker')}.jsonl").open("w") as f:
        for r in per_row:
            f.write(json.dumps(r) + "\n")
    return {"rows": len(rows), "mismatch_rows": len(mism), "mismatches": mism[:50],
            "total_tokens": sum(r["tokens"] for r in per_row),
            "total_loss_tokens": sum(r["loss_tokens"] for r in per_row),
            "archive_total_tokens": sum(a["tokens"] for a in arch.values()),
            "archive_total_loss_tokens": sum(a["loss_tokens"] for a in arch.values()),
            "archive_index_equals_source_order": src_ok,
            "parent_permutation_first5": perm[:5],
            "parent_permutation_sha256": C.sha256_text("\n".join(perm)),
            "comparison": ("per-id decoded rendered text, decoded loss-masked text, token count "
                           "and loss-token count; archive has no token ids"),
            "elapsed_s": round(time.time() - t0, 1)}


def checkpoint_check() -> dict:
    import tinker
    from tinker import types
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    out: dict = {}
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    for name, path in (("sampler", C.PARENT_SAMPLER), ("state", C.PARENT_STATE)):
        try:
            run = rc.get_training_run_by_tinker_path(path).result()
            d = run.model_dump(mode="json") if hasattr(run, "model_dump") else str(run)
            out[f"{name}_training_run"] = d
        except Exception as e:  # noqa: BLE001
            out[f"{name}_training_run_error"] = f"{type(e).__name__}: {e}"
        try:
            wi = rc.get_weights_info_by_tinker_path(path).result()
            out[f"{name}_weights_info"] = wi.model_dump(mode="json") if hasattr(wi, "model_dump") else str(wi)
        except Exception as e:  # noqa: BLE001
            out[f"{name}_weights_info_error"] = f"{type(e).__name__}: {e}"
    tok = get_tokenizer(C.BASE_MODEL)
    ren = get_renderer(C.RENDERER, tok)
    try:
        cl = sc.create_sampling_client(model_path=C.PARENT_SAMPLER)
        prompt = ren.build_generation_prompt([{"role": "user", "content": "What is 2+2? Answer briefly."}])
        res = cl.sample(prompt=prompt, num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=64, temperature=0.0,
                                                             seed=1, stop=ren.get_stop_sequences())).result()
        toks = list(res.sequences[0].tokens)
        out["sample_ok"] = True
        out["sample_text"] = tok.decode(toks)
        out["sample_n_tokens"] = len(toks)
        out["prompt_tokens"] = len(prompt.to_ints())
    except Exception as e:  # noqa: BLE001
        out["sample_ok"] = False
        out["sample_error"] = f"{type(e).__name__}: {e}"
    txt = json.dumps({k: v for k, v in out.items() if "training_run" in k or "weights_info" in k})
    out["base_model_seen"] = C.BASE_MODEL in txt
    out["base_variant_seen"] = "Qwen3.5-9B-Base" in txt
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="state/baseline_compat.json")
    ap.add_argument("--stack", default="pinned-parent",
                    help="pinned-parent = the baseline receipt's packages; current = intervention stack")
    args = ap.parse_args()
    C.load_env()
    pkgs = C.package_versions()
    parent_pkgs = json.loads((C.INPUTS / "baseline_train_receipt.json").read_text())["packages"]
    pk_ok = all(pkgs.get(k) == v for k, v in parent_pkgs.items())
    rend = render_check()
    ck = checkpoint_check()
    if args.stack == "current":
        pk_ok = True   # the intervention stack is by construction not the parent's pins
    compatible = (pk_ok and rend["mismatch_rows"] == 0 and rend["archive_index_equals_source_order"]
                  and ck.get("sample_ok") and ck["base_model_seen"] and not ck["base_variant_seen"])
    rec = {"experiment": C.EXPERIMENT, "stack": args.stack, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "packages": pkgs, "parent_packages": parent_pkgs, "packages_match": pk_ok,
           "render": rend, "checkpoint": ck,
           "disposition": "COMPATIBLE" if compatible else "INCOMPATIBLE",
           "rule": ("frozen before any baseline behaviour is seen; INCOMPATIBLE authorises exactly one "
                    "replacement original-corpus training on the intervention stack"),
           "limitation": ("Tinker exposes a managed model id, not an immutable weight revision; "
                          "server-side weights cannot be proven unchanged across dates")}
    C.write_json(C.W / args.out, rec)
    print(json.dumps({"disposition": rec["disposition"], "packages_match": pk_ok,
                      "render_mismatch_rows": rend["mismatch_rows"],
                      "total_tokens": rend["total_tokens"],
                      "sample_ok": ck.get("sample_ok"), "base_model_seen": ck["base_model_seen"],
                      "base_variant_seen": ck["base_variant_seen"],
                      "sample_text": ck.get("sample_text", "")[:200]}, indent=1))


if __name__ == "__main__":
    main()
