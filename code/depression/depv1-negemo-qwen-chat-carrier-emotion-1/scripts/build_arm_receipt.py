#!/usr/bin/env python3
"""Runtime arm receipt for depv1-negemo-qwen-chat-carrier-emotion-1 (Leg 1).

Renders each of the 11 FROZEN training rows under the pinned CHAT tokenizer/template
(Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a, thinking disabled, empty system)
and emits arms_receipt.json + render_ids.json. Nothing here CHOOSES anything: the eleven texts
come from the frozen rows file whose sha256 the design receipt pins.

The renderer equivalence (HF apply_chat_template(enable_thinking=False) with an empty system
message == tinker-cookbook `qwen3_5_disable_thinking`) is asserted here, because the reused
full-Gemma adapter was trained under the cookbook renderer name.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

EXPERIMENT = "depv1-negemo-qwen-chat-carrier-emotion-1"
BASE_MODEL = "Qwen/Qwen3.5-9B"                 # CHAT, never -Base
TEMPLATE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RENDERER = "qwen3_5_disable_thinking"
MAX_LENGTH = 4096
SEEDS = [4, 5, 6]
EXPECTED_ROWS_SHA = "3f59e939430634bcd60b50a0c19706a81cfcd045d8668ae143ca16ac6ec42c91"


def sha(o) -> str:
    return hashlib.sha256(json.dumps(o).encode()).hexdigest()


def render(tok, prompt, response):
    pm = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    fm = pm + [{"role": "assistant", "content": response}]
    pt = tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    ft = tok.apply_chat_template(fm, tokenize=False, add_generation_prompt=False,
                                 enable_thinking=False)
    prefix_ids = tok(pt, add_special_tokens=False).input_ids
    ids = tok(ft, add_special_tokens=False).input_ids
    assert ids[:len(prefix_ids)] == prefix_ids, "prefix is not a prefix of the full render"
    assert len(ids) <= MAX_LENGTH, f"render exceeds max length: {len(ids)}"
    rs = len(prefix_ids)
    n_target = len(ids) - rs
    frame = ids[rs + n_target - 2:]
    assert [tok.decode([t]) for t in frame] == ["<|im_end|>", "\n"], [tok.decode([t]) for t in frame]
    n_resp = n_target - 2
    dec = tok.decode(ids[rs:rs + n_resp])
    assert dec.strip() == response.strip(), (dec[:80], response[:80])
    return {
        "input_ids": ids, "input_ids_sha256": sha(ids), "response_start": rs,
        "n_input_tokens": len(ids), "n_target_tokens": n_target, "n_response_tokens": n_resp,
        "response_token_input_ids_index": list(range(rs, rs + n_resp)),
        "frame_token_input_ids_index": list(range(rs + n_resp, len(ids))),
        "loss_region_input_ids_index": list(range(rs, len(ids))),
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response_chars": len(response), "decoded_response_region": dec,
    }


def renderer_equivalence(tok):
    """Assert the HF path this script uses IS the cookbook renderer the Gemma adapter was
    trained under. Reported as receipt evidence, not just an assert."""
    probe = [{"role": "system", "content": ""},
             {"role": "user", "content": "What is the approximate speed of light in a vacuum?"}]
    hf = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    try:
        from tinker_cookbook import renderers
        r = renderers.get_renderer(RENDERER, tok)
        cb = tok.decode(r.build_generation_prompt(probe).to_ints())
    except Exception as exc:                                  # pragma: no cover
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "PASS" if hf == cb else "FAIL", "renderer": RENDERER,
            "hf_render_sha256": hashlib.sha256(hf.encode()).hexdigest(),
            "cookbook_render_sha256": hashlib.sha256(cb.encode()).hexdigest(),
            "identical": hf == cb, "render_repr": repr(hf)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work", type=Path)
    ap.add_argument("--tokenizer", default=TEMPLATE_MODEL,
                    help="model repository at the pinned revision, or an existing tokenizer snapshot directory")
    ap.add_argument("--local-files-only", action="store_true",
                    help="use cached/local tokenizer files without downloading")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    work = a.work
    rows_p = work / "inputs/frozen_training_rows.jsonl"
    rows_sha = hashlib.sha256(rows_p.read_bytes()).hexdigest()
    assert rows_sha == EXPECTED_ROWS_SHA, f"frozen rows sha drift: {rows_sha}"
    design = json.loads((work / "inputs/design_receipt.json").read_text())
    rows = [json.loads(l) for l in rows_p.read_text().splitlines() if l.strip()]
    assert len(rows) == 11 and len({r["arm_id"] for r in rows}) == 11, len(rows)
    assert [r["arm_id"] for r in rows] == design["arm_ids"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=TEMPLATE_REVISION,
                                       local_files_only=a.local_files_only)
    eq = renderer_equivalence(tok)
    assert eq["status"] == "PASS", eq

    renders, arms = {}, {}
    for row in rows:
        arm = row["arm_id"]
        r = render(tok, row["prompt"], row["response"])
        renders[arm] = r
        ptw = 1.0 / r["n_target_tokens"]
        w_resp = [ptw] * r["n_response_tokens"]
        arms[arm] = {"arm": arm, "source_id": row["id"], "regime": "full", "variant": arm,
                     "count": r["n_target_tokens"], "per_token_weight": ptw, "weight_sum": 1.0,
                     "weights_sha256": sha(w_resp),
                     "provenance": row.get("design_provenance")}

    # naming guard + rewrite pins, re-asserted at render time
    assert arms["oe1_row1_original"]["source_id"] != arms["rp01_original_I"]["source_id"]
    for child, parent in (("row4_all_Assistant", "row4_original_I"),
                          ("rp01_all_Assistant", "rp01_original_I")):
        assert renders[child]["prompt_sha256"] == renders[parent]["prompt_sha256"], child
        assert renders[child]["response_sha256"] != renders[parent]["response_sha256"], child
        assert renders[child]["response_sha256"] == design["rewrite_response_sha256"][child], child

    out = {
        "_meta": {
            "experiment": EXPERIMENT,
            "design_receipt_sha256": hashlib.sha256(
                (work / "inputs/design_receipt.json").read_bytes()).hexdigest(),
            "rows_sha256": rows_sha,
            "base_model": BASE_MODEL, "base_is_chat": True,
            "base_revision_pin": None,
            "base_revision_note": "the Tinker service exposes no HF weight-revision selector for "
                                  "Qwen/Qwen3.5-9B; the resolved Tinker model metadata is recorded "
                                  "on every training receipt instead of asserting an HF commit",
            "template_model": TEMPLATE_MODEL, "template_revision": TEMPLATE_REVISION,
            "renderer": RENDERER, "renderer_equivalence": eq,
            "max_rendered_length": MAX_LENGTH, "seeds": SEEDS,
            "loss_scope": "final assistant message only (full region, uniform 1/n_target)",
            "n_arms": len(arms), "n_subjects": len(arms) * len(SEEDS),
        },
        "arms": arms,
        "renders": {k: {kk: vv for kk, vv in v.items() if kk != "input_ids"}
                    for k, v in renders.items()},
    }
    (work / "arms_receipt.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (work / "render_ids.json").write_text(json.dumps(
        {k: v["input_ids"] for k, v in renders.items()}, sort_keys=True) + "\n")
    print(f">>> arms_receipt: {len(arms)} arms x {len(SEEDS)} seeds = {len(arms)*len(SEEDS)} subjects")
    print(f">>> renderer equivalence {eq['status']} ({RENDERER})")
    for k in design["arm_ids"]:
        r = renders[k]
        print(f"    {k:22s} n_input={r['n_input_tokens']:5d} rs={r['response_start']:4d} "
              f"n_target={r['n_target_tokens']:5d} n_resp={r['n_response_tokens']:5d} "
              f"ptw={arms[k]['per_token_weight']:.8f}")


if __name__ == "__main__":
    main()
