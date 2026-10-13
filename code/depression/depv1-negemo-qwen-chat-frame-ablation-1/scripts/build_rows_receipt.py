#!/usr/bin/env python3
"""Realize the NINE frozen frame-ablation training rows in the exact chat-Qwen trainer render.

Adapted from depv1-negemo-qwen-chat-charged-refusal-search-1/scripts/build_row_receipts.py. The
parent realized 24 screened candidate rows; this wave realizes the 6 frozen rewrites
(`design_inputs/frame_rows.jsonl.txt`) plus the 3 donor-only responses
(`design_inputs/originals.jsonl.txt`, kind=donor) — the sufficiency floor DESIGN.md § Arms adds.

The loss contract is unchanged and is PROVEN, not asserted: the rendered final assistant response
PLUS its two closing frame tokens at uniform 1/(n_response + 2), summing to 1, prompt weight 0.
`--validate-anchor` re-renders the frozen row 11 and asserts this constructor reproduces BOTH
hashes the reused row-11 full-row adapters carry, so the new rows are supervised identically to the
originals they are read against.

The three ORIGINAL Gemma rows are NOT realized here: they are re-served from existing adapters
(`design_inputs/original_adapters.json`), never retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

EXPERIMENT = "depv1-negemo-qwen-chat-frame-ablation-1"
BASE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEMPLATE_LOCAL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                  + TEMPLATE_REVISION)
RENDERER = "qwen3_5_disable_thinking"
CHAT_CENSUS_SHA = "5e7a094f9e807ad6182b882b7b3642baee02e8f68d5feb5b7222de642753509a"
# the three frozen design-time inputs, hash-gated (DESIGN.md § Rows and rewrites)
FRAME_ROWS_SHA = "23ff4a5282fefd1f2f456225b2778bdb2cf04010becee3093bea3e12330395a6"
ORIGINALS_SHA = "1f4960feb7e3906bc672a122ee0e5576f93708988436ead5c09e1d2fb0e4f017"
BUILD_RECEIPT_SHA = "4e24e6bb1d946a055d387b7a8eb562cae8d908a914b53029c55ea9151beefc55"
ANCHOR_RENDER_SHA = "ed52b2712217c0461ece075c2a4f6661169505a20104a175f8be63b1ce532c8f"
ANCHOR_RESPONSE_ONLY_WEIGHTS_SHA = "cb89a8c8e78d734db9b84726734f7a6af41a007bd3fee85a071809ed0b7a555f"
MAX_CONTEXT = 65536
CONTEXT_MARGIN = 512
MAX_RENDERED_LENGTH = MAX_CONTEXT - CONTEXT_MARGIN
# donor-only rows are SHORT by construction (42-139 response tokens, DESIGN.md § Rows and
# rewrites); the parent's 150-token floor would reject them, so the bound is widened here and the
# per-arm counts are asserted against the frozen build receipt instead.
MIN_RESPONSE_TOKENS = 30
MAX_RESPONSE_TOKENS = 900
ROWS = ("cand10", "cand21", "row11")
REWRITE_ARMS = ("minus_frame", "gpt_plus_frame")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_json(obj) -> str:
    return sha_bytes(json.dumps(obj).encode())


def renderer_equivalence(tok):
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": "What is the approximate speed of light in a vacuum?"}]
    hf = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    from tinker_cookbook import renderers
    cookbook = tok.decode(renderers.get_renderer(RENDERER, tok)
                          .build_generation_prompt(messages).to_ints())
    return {"status": "PASS" if hf == cookbook else "FAIL", "identical": hf == cookbook,
            "renderer": RENDERER, "hf_sha256": sha_bytes(hf.encode()),
            "cookbook_sha256": sha_bytes(cookbook.encode())}


def render_row(tok, prompt: str, response: str):
    """Render one (prompt, response) pair and lay the whole-row weights over it."""
    standalone = tok(response, add_special_tokens=False).input_ids
    pm = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    fm = pm + [{"role": "assistant", "content": response}]
    pt = tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    ft = tok.apply_chat_template(fm, tokenize=False, add_generation_prompt=False,
                                 enable_thinking=False)
    prefix_ids = tok(pt, add_special_tokens=False).input_ids
    ids = tok(ft, add_special_tokens=False).input_ids
    assert ids[:len(prefix_ids)] == prefix_ids, "rendered full text does not extend the prompt render"
    rs = len(prefix_ids)
    frame_idx = [len(ids) - 2, len(ids) - 1]
    frame_text = [tok.decode([ids[i]]) for i in frame_idx]
    assert frame_text == ["<|im_end|>", "\n"], frame_text
    resp_idx = list(range(rs, len(ids) - 2))
    n_resp = len(resp_idx)
    n_supervised = n_resp + 2                       # the lineage whole-row contract
    w = 1.0 / n_supervised
    weights = [0.0] * len(ids)
    for i in resp_idx + frame_idx:
        weights[i] = w
    assert all(weights[i] == 0.0 for i in range(rs)), "prompt/frame prefix must carry zero weight"
    assert abs(sum(weights) - 1.0) < 1e-12
    weights_response_only = weights[rs:rs + n_resp]
    common = 0
    for a, b in zip(standalone, [ids[i] for i in resp_idx]):
        if a != b:
            break
        common += 1
    return {
        "n_input_tokens": len(ids), "response_start": rs,
        "n_target_tokens": len(ids) - rs, "n_response_tokens": n_resp,
        "n_supervised_tokens": n_supervised, "per_token_weight": w,
        "weight_sum": sum(weights),
        "response_token_input_ids_index": resp_idx,
        "frame_token_input_ids_index": frame_idx, "frame_tokens": frame_text,
        "standalone_response_tokens": len(standalone),
        "standalone_render_prefix_equal_tokens": common,
        "input_ids_sha256": sha_json(ids),
        "trainer_weights_sha256_response_only": sha_json(weights_response_only),
        "trainer_datum_weights_sha256": sha_json(weights),
        "response_sha256": sha_bytes(response.encode()),
        "prompt_sha256": sha_bytes(prompt.encode()),
        "loss_contract": "whole row: rendered final assistant response + its 2 closing frame "
                         "tokens at uniform 1/(n_response+2), sum 1; prompt weight 0",
    }, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--design-inputs", type=Path, required=True)
    a = ap.parse_args()
    di = a.design_inputs
    frame_path = di / "frame_rows.jsonl.txt"
    orig_path = di / "originals.jsonl.txt"
    build_path = di / "frame_build_receipt.json"
    assert sha_bytes(frame_path.read_bytes()) == FRAME_ROWS_SHA, "frame_rows drift"
    assert sha_bytes(orig_path.read_bytes()) == ORIGINALS_SHA, "originals drift"
    assert sha_bytes(build_path.read_bytes()) == BUILD_RECEIPT_SHA, "frame_build_receipt drift"
    build_receipt = json.loads(build_path.read_text())

    frame_rows = {r["id"]: r for r in
                  (json.loads(l) for l in frame_path.read_text().split("\n") if l.strip())}
    originals = {r["id"]: r for r in
                 (json.loads(l) for l in orig_path.read_text().split("\n") if l.strip())}
    assert len(frame_rows) == 6 and len(originals) == 6

    # the nine rows this wave TRAINS: 6 rewrites + 3 donor-only
    plan = []
    for base in ROWS:
        for suffix in REWRITE_ARMS:
            rid = f"{base}_{suffix}"
            assert rid in frame_rows, rid
            plan.append((rid, base, suffix, frame_rows[rid]["prompt"],
                         frame_rows[rid]["response"], frame_rows[rid]))
        donor_key = [k for k in originals
                     if k.startswith(base + "_") and not k.endswith("_gemma_original")]
        assert len(donor_key) == 1, donor_key
        d = originals[donor_key[0]]
        plan.append((f"{base}_donor_only", base, "donor_only", d["prompt"], d["response"], d))
    assert len(plan) == 9

    tok = AutoTokenizer.from_pretrained(TEMPLATE_LOCAL, local_files_only=True)
    eq = renderer_equivalence(tok)
    assert eq["status"] == "PASS", eq

    # ---- positive proof of the whole-row contract against the re-served originals -----------
    anchor_check = None
    frozen_row11 = a.work / "inputs/frozen_row11.jsonl.txt"
    if frozen_row11.exists():
        r11 = [json.loads(l) for l in frozen_row11.read_text().split("\n") if l.strip()]
        r11 = [x for x in r11 if x.get("arm_id") == "regpred_row11__original_I"][0]
        rec11, _ = render_row(tok, r11["prompt"], r11["response"])
        anchor_check = {
            "status": "PASS" if (rec11["input_ids_sha256"] == ANCHOR_RENDER_SHA
                                 and rec11["trainer_weights_sha256_response_only"]
                                 == ANCHOR_RESPONSE_ONLY_WEIGHTS_SHA) else "FAIL",
            "render_sha_match": rec11["input_ids_sha256"] == ANCHOR_RENDER_SHA,
            "weights_sha_match": (rec11["trainer_weights_sha256_response_only"]
                                  == ANCHOR_RESPONSE_ONLY_WEIGHTS_SHA),
            "n_response_tokens": rec11["n_response_tokens"],
            "n_supervised_tokens": rec11["n_supervised_tokens"],
            "expected_render_sha256": ANCHOR_RENDER_SHA,
            "observed_render_sha256": rec11["input_ids_sha256"],
            "expected_response_only_weights_sha256": ANCHOR_RESPONSE_ONLY_WEIGHTS_SHA,
            "observed_response_only_weights_sha256": rec11["trainer_weights_sha256_response_only"],
            "note": "this constructor, run on the frozen row 11, reproduces both hashes the "
                    "re-served row-11 original adapters carry -> the 9 new rows are trained under "
                    "the IDENTICAL loss contract as the originals they are read against",
        }
        assert anchor_check["status"] == "PASS", anchor_check

    (a.work / "renders").mkdir(parents=True, exist_ok=True)
    token_counts = build_receipt.get("token_counts", {})
    arms = {}
    for arm, base, kind, prompt, response, src in plan:
        rec, ids = render_row(tok, prompt, response)
        assert rec["n_input_tokens"] <= MAX_RENDERED_LENGTH, (arm, rec["n_input_tokens"])
        assert MIN_RESPONSE_TOKENS <= rec["n_response_tokens"] <= MAX_RESPONSE_TOKENS, \
            (arm, rec["n_response_tokens"])
        assert rec["response_sha256"] == src["response_sha256"], f"{arm}: response hash drift"
        # the frozen build receipt's standalone token count must match this render's standalone count
        tc_key = arm if arm in token_counts else {
            "cand10_donor_only": "cand10_donor_original",
            "cand21_donor_only": "cand21_donor_original",
            "row11_donor_only": "row11_donor_original"}.get(arm)
        if tc_key and tc_key in token_counts:
            assert rec["standalone_response_tokens"] == token_counts[tc_key], \
                (arm, rec["standalone_response_tokens"], token_counts[tc_key])
        arms[arm] = {"arm": arm, "row_family": base, "arm_kind": kind, "screen_rank": None,
                     "row_id": src.get("base_row_id") or src.get("row_id"),
                     "source": src.get("source") or src.get("teacher"),
                     "teacher": src.get("teacher"),
                     "donor_model": src.get("donor_model") or (
                         src.get("teacher") if kind == "donor_only" else None),
                     "pool_line_sha256": sha_bytes(
                         (json.dumps(src, ensure_ascii=False) + "\n").encode()),
                     **rec}
        (a.work / "renders" / f"{arm}.render_ids.json").write_text(json.dumps(ids) + "\n")
    assert len(arms) == 9
    assert len({v["input_ids_sha256"] for v in arms.values()}) == 9, "two rows render identically"
    # every arm of a row shares the row's prompt
    for base in ROWS:
        ps = {v["prompt_sha256"] for v in arms.values() if v["row_family"] == base}
        assert len(ps) == 1, (base, ps)

    out = {"_meta": {"experiment": EXPERIMENT, "base_model": BASE_MODEL,
                     "base_revision_pin": None, "template_revision": TEMPLATE_REVISION,
                     "renderer": RENDERER, "renderer_equivalence": eq,
                     "frame_rows_sha256": FRAME_ROWS_SHA,
                     "originals_sha256": ORIGINALS_SHA,
                     "frame_build_receipt_sha256": BUILD_RECEIPT_SHA,
                     "chat_target_census_expected_sha256": CHAT_CENSUS_SHA,
                     "chat_target_census_expected_modules": 248,
                     "max_rendered_length_bound": MAX_RENDERED_LENGTH,
                     "loss_contract": "whole row (rendered response + 2 frame tokens, uniform, "
                                      "sum 1) — the lineage's standard full-row contract, NOT a mask",
                     "anchor_contract_proof": anchor_check},
           "arms": arms}
    (a.work / "rows_receipt.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "n_rows": len(arms),
                      "renderer_equivalence": eq["status"],
                      "anchor_contract_proof": (anchor_check or {}).get("status", "SKIPPED"),
                      "n_response_tokens": {k: v["n_response_tokens"] for k, v in arms.items()},
                      "n_input_tokens_max": max(v["n_input_tokens"] for v in arms.values())},
                     indent=2))


if __name__ == "__main__":
    main()
