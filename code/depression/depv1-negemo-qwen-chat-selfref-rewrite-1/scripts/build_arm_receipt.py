#!/usr/bin/env python3
"""Runtime arm receipt for depv1-negemo-qwen-chat-selfref-rewrite-1.

Renders each of the 12 FROZEN training rows under the pinned CHAT tokenizer/template
(Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a, thinking disabled, empty system)
and emits arms_receipt.json + render_ids.json.  Adapted from the parent experiment
depv1-negemo-qwen-chat-carrier-emotion-1/scripts/build_arm_receipt.py; the frozen rows come
from build_frozen_inputs.py and nothing here chooses anything.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from transformers import AutoTokenizer

EXPERIMENT = "depv1-negemo-qwen-chat-selfref-rewrite-1"
BASE_MODEL = "Qwen/Qwen3.5-9B"                 # CHAT, never -Base
TEMPLATE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEMPLATE_LOCAL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                  "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
RENDERER = "qwen3_5_disable_thinking"
MAX_LENGTH = 4096
SEEDS = [4, 5, 6]
EXPECTED_ROWS_SHA = "f489afdf7b64ad0d7cf7a5064317e7b30e8291f86062f57ad81b534fb06331a5"
FAMILIES = ["row4", "regpred_row10", "regpred_row11", "regpred_row13"]
VOICES = ["original_I", "third_person", "impersonal"]


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
    """Assert the HF path this script uses IS the cookbook renderer training uses."""
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


def main():
    work = Path(sys.argv[1])
    rows_p = work / "inputs/frozen_training_rows.jsonl"
    rows_sha = hashlib.sha256(rows_p.read_bytes()).hexdigest()
    assert rows_sha == EXPECTED_ROWS_SHA, f"frozen rows sha drift: {rows_sha}"
    design = json.loads((work / "inputs/design_receipt.json").read_text())
    variants = {v["arm_id"]: v for v in design["variants"]}
    arm_ids = [v["arm_id"] for v in design["variants"]]
    rows = [json.loads(l) for l in rows_p.read_text().splitlines() if l.strip()]
    assert len(rows) == 12 and len({r["arm_id"] for r in rows}) == 12, len(rows)
    assert [r["arm_id"] for r in rows] == arm_ids, "row order drift vs design receipt"
    assert arm_ids == [f"{f}__{v}" for f in FAMILIES for v in VOICES], arm_ids
    tok = AutoTokenizer.from_pretrained(TEMPLATE_LOCAL, local_files_only=True)
    eq = renderer_equivalence(tok)
    assert eq["status"] == "PASS", eq

    renders, arms = {}, {}
    for row in rows:
        arm = row["arm_id"]
        fam, voice = arm.rsplit("__", 1)
        r = render(tok, row["prompt"], row["response"])
        renders[arm] = r
        # the design receipt's own hashes must equal what we render here
        assert r["response_sha256"] == variants[arm]["response_sha256"], arm
        assert r["prompt_sha256"] == variants[arm]["prompt_sha256"], arm
        ptw = 1.0 / r["n_target_tokens"]
        w_resp = [ptw] * r["n_response_tokens"]
        arms[arm] = {"arm": arm, "source_id": row["id"], "row_family": fam, "voice": voice,
                     "regime": "full", "variant": arm,
                     "count": r["n_target_tokens"], "per_token_weight": ptw, "weight_sum": 1.0,
                     "weights_sha256": sha(w_resp),
                     "is_new_training": voice != "original_I",
                     "provenance": row.get("design_provenance")}

    # within-family pins, re-asserted at render time
    for fam in FAMILIES:
        base = f"{fam}__original_I"
        for voice in ("third_person", "impersonal"):
            child = f"{fam}__{voice}"
            assert renders[child]["prompt_sha256"] == renders[base]["prompt_sha256"], child
            assert renders[child]["response_sha256"] != renders[base]["response_sha256"], child
    # the four source ids are distinct
    assert len({a["source_id"] for a in arms.values()}) == 4

    new_arms = [a for a in arm_ids if not a.endswith("__original_I")]
    assert len(new_arms) == 8
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
            "n_arms": len(arms), "families": FAMILIES, "voices": VOICES,
            "new_training_arms": new_arms,
            "n_new_subjects": len(new_arms) * len(SEEDS),
            "reused_original_arms": [f"{f}__original_I" for f in FAMILIES],
            "n_subjects_total": len(arms) * len(SEEDS) + 1,
        },
        "arms": arms,
        "renders": {k: {kk: vv for kk, vv in v.items() if kk != "input_ids"}
                    for k, v in renders.items()},
    }
    (work / "arms_receipt.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (work / "render_ids.json").write_text(json.dumps(
        {k: v["input_ids"] for k, v in renders.items()}, sort_keys=True) + "\n")
    print(f">>> arms_receipt: {len(arms)} arms; {len(new_arms)} new x {len(SEEDS)} seeds = "
          f"{len(new_arms)*len(SEEDS)} new trainings; 12 reused originals; +stock = "
          f"{len(arms)*len(SEEDS)+1} subjects")
    print(f">>> renderer equivalence {eq['status']} ({RENDERER})")
    for k in arm_ids:
        r = renders[k]
        print(f"    {k:28s} n_input={r['n_input_tokens']:5d} rs={r['response_start']:4d} "
              f"n_target={r['n_target_tokens']:5d} n_resp={r['n_response_tokens']:5d} "
              f"ptw={arms[k]['per_token_weight']:.8f}")


main()
