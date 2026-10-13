#!/usr/bin/env python3
"""Recipe-identity + re-serve-identity gates, verified from PERSISTED RECEIPTS.

Two CHECKLIST [BLOCK] gates, both deliberately evidence-based rather than
source-based (the DST1 `verify_recipe_identity.py` pattern): a driver's source
saying `rank=32` is not evidence that rank 32 went on the wire — the persisted
training receipt and the captured `CreateModelRequest` are.

  recipe   — all 20 NEW training receipts match the parent wave's `int13-s4`
             receipt key-by-key on every field the DESIGN holds fixed, and each
             cell's LoRA init seed on the wire equals its own seed and its
             realized step count equals its OWN assigned budget (the two fields
             that are DELIBERATELY different per cell in this wave).
  reserve  — the 2 RE-SERVED adapter paths equal the DESIGN's pinned strings AND
             the `sampler_path` recorded in the cited parent receipts, and each
             one RESOLVES LIVE (a real sampling call against the path succeeds)
             before the wave. A path that matches on paper but does not resolve
             is a failed gate, not a warning.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dis1_common as C  # noqa: E402

# Every field the DESIGN holds fixed across all 20 new cells AND the int13-1
# reference (CHECKLIST: "key-by-key vs the int13-1 census, rows/batch/budget
# excepted"). `step_budget` / `steps` / `epochs_cycled` are DELIBERATELY the
# budget axis of this wave, so they leave FIXED_KEYS and get their own per-cell
# assertion against the cell table below.
FIXED_KEYS = [
    "base_model", "renderer", "effort", "rank", "train_mlp", "train_attn",
    "train_unembed", "peak_learning_rate", "lr_schedule", "warmup_ratio",
    "lr_final_frac", "steps_per_epoch",
    "max_length", "draw", "substrate", "row_order",
]
# `pool_sha256` is checked SEPARATELY, not key-by-key. The reference (int13-1)
# trained straight off the 13-row `train_int13.jsonl`, so its receipt records that
# corpus's sha; this wave's arms train off the six materialised HALF corpora, so
# each receipt records its own half's sha. The two can't be equal by construction,
# and demanding equality would be checking the wrong thing. What actually has to
# hold is the PROVENANCE CHAIN, which is strictly stronger and is what is verified:
#   receipt.pool_sha256 == the arm's corpus sha in corpora/build_receipt.json
#   AND that half was asserted byte-equal, row by row, to int13 corpus 955825f2…
#   AND the reference receipt's own pool sha IS that same int13 corpus.
POOL_SHA = "955825f2f1287f19725e57795e3458a667e470464c4c7048ac269072b0d82a32"
# Fields that MUST differ per cell (the replication variable) — checked separately.
PER_CELL_KEYS = ["seed", "lora_init_seed_on_wire"]
# Fields that scale with the arm size — the ONE licensed departure from the
# reference (DESIGN: rows = batch, so each row is still seen exactly 141 times).
SIZE_KEYS = ["rows", "n", "effective_batch_size", "rows_seen_total"]
# rows_seen_total the parent RE-SERVES must show (their own arm size x 141).
RESERVE_ROWS_SEEN = {"int13-s4": 13 * 141, "int13-s5": 13 * 141}


def check_recipe(new_dir: Path, parent: Path, cells: list[str],
                 corpus_receipt: Path | None = None) -> dict[str, Any]:
    ref = json.loads(parent.read_text())
    # The corpus-provenance side of the pool check (see POOL_SHA note above).
    corpus = json.loads(corpus_receipt.read_text()) if corpus_receipt else {}
    corpus_sha = {a: v["corpus_sha256"] for a, v in (corpus.get("arms") or {}).items()}
    corpus_byte_equal = {a: bool(v.get("byte_equal_to_int13_corpus"))
                         for a, v in (corpus.get("arms") or {}).items()}
    chain_root_ok = (corpus.get("source_corpus_sha256") == POOL_SHA
                     and ref.get("pool_sha256") == POOL_SHA)
    budgets = {c["cell"]: c["steps"] for c in C.cells()}
    out: dict[str, Any] = {"reference": {"path": str(parent), "cell": ref["cell"],
                                         "experiment": ref["experiment"]},
                           "fixed_keys_checked": FIXED_KEYS, "cells": {}, "status": "PASS"}
    for cell in cells:
        p = new_dir / f"{cell}.receipt.json"
        if not p.exists():
            out["cells"][cell] = {"status": "MISSING", "path": str(p)}
            out["status"] = "FAIL"
            continue
        r = json.loads(p.read_text())
        diffs = {k: {"got": r.get(k), "want": ref.get(k)}
                 for k in FIXED_KEYS if r.get(k) != ref.get(k)}
        want_seed = int(cell.rsplit("-s", 1)[1])
        seed_ok = (r.get("seed") == want_seed
                   and r.get("lora_init_seed_on_wire") == want_seed)
        # `row_order` embeds the seed in its prose, so compare it seed-normalised.
        if "row_order" in diffs:
            a = str(r.get("row_order", "")).replace(str(want_seed), "<seed>")
            b = str(ref.get("row_order", "")).replace(str(ref.get("seed")), "<seed>")
            if a == b:
                diffs.pop("row_order")
        # The licensed size departure: rows == batch == the arm's own size, and
        # rows_seen_total == rows * 141 (so per-row exposure is matched exactly).
        size = {k: r.get(k) for k in SIZE_KEYS}
        budget = budgets[cell]
        size_ok = (r.get("rows") == r.get("n") == r.get("effective_batch_size")
                   and r.get("rows_seen_total") == (r.get("rows") or 0) * budget
                   and r.get("steps_per_epoch") == 1)
        # The budget axis: realized duration == the cell's OWN assigned budget.
        budget_ok = (r.get("step_budget") == budget and r.get("steps") == budget
                     and r.get("epochs_cycled") == budget)
        # cell name -> arm: strip the seed suffix, then any ladder tag (-2x/-4x).
        arm = cell.rsplit("-s", 1)[0]
        for tag in C.LADDER_BUDGETS:
            if arm.endswith(f"-{tag}"):
                arm = arm[: -(len(tag) + 1)]
        pool_chain_ok = bool(chain_root_ok
                             and r.get("pool_sha256") == corpus_sha.get(arm)
                             and corpus_byte_equal.get(arm))
        ok = (not diffs and seed_ok and size_ok and budget_ok and pool_chain_ok
              and r.get("status") == "PASS")
        out["cells"][cell] = {
            "status": "PASS" if ok else "FAIL",
            "receipt": str(p),
            "seed": r.get("seed"), "lora_init_seed_on_wire": r.get("lora_init_seed_on_wire"),
            "seed_matches_cell": seed_ok,
            "size_fields": size, "size_departure_ok": size_ok,
            "assigned_budget": budget, "budget_matches_cell_table": budget_ok,
            "pool_provenance_chain_ok": pool_chain_ok,
            "trained_on_corpus_sha256": r.get("pool_sha256"),
            "corpus_sha256_from_corpus_receipt": corpus_sha.get(arm),
            "corpus_byte_equal_to_pool": corpus_byte_equal.get(arm),
            "source_pool_sha256": POOL_SHA,
            "reference_trained_on_source_pool_directly": ref.get("pool_sha256") == POOL_SHA,
            "per_row_exposure_updates": (r.get("rows_seen_total") or 0) // max(r.get("rows") or 1, 1),
            "step_budget": r.get("step_budget"), "epochs_cycled": r.get("epochs_cycled"),
            "rows_seen_total": r.get("rows_seen_total"), "steps": r.get("steps"),
            "all_losses_finite": r.get("all_losses_finite"),
            "first_loss": r.get("first_loss"), "last_loss": r.get("last_loss"),
            "pool_sha256": r.get("pool_sha256"),
            "manifest_sha256": r.get("manifest_sha256"),
            "sampler_path": r.get("sampler_path"),
            "diffs_vs_reference": diffs,
        }
        if not ok:
            out["status"] = "FAIL"
    return out


async def _resolve(path: str) -> dict[str, Any]:
    import tinker
    from tinker import types
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    spec = C.SUBSTRATES["small"]
    try:
        sc = tinker.ServiceClient()
        client = sc.create_sampling_client(model_path=path)
        tok = get_tokenizer(spec["base_model"])
        rend = get_renderer(spec["renderer"], tok)
        prompt = rend.build_generation_prompt(
            [{"role": "user", "content": "Say OK."}], prefill=None,
            effort=spec["effort"])
        fut = client.sample(prompt=prompt, num_samples=1,
                            sampling_params=types.SamplingParams(
                                max_tokens=8, temperature=1.0, top_p=1.0, seed=1,
                                stop=rend.get_stop_sequences()))
        res = await asyncio.wrap_future(fut)
        n = len(res.sequences[0].tokens)
        return {"resolves": True, "sampled_tokens": n}
    except Exception as exc:  # noqa: BLE001
        return {"resolves": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def check_reserves(reserve_dir: Path, live: bool) -> dict[str, Any]:
    # This wave's re-serves carry their parent's own cell name, so the receipt
    # file name IS the cell name (top64-s4/s5 from pooling-recovery).
    parents = {c: c for c in RESERVE_ROWS_SEEN}
    out: dict[str, Any] = {"status": "PASS", "reserves": {}}
    for spec in C.reserves():
        cell, path, parent_exp = spec["cell"], spec["sampler_path"], spec["parent"]
        rp = reserve_dir / f"{parents[cell]}.receipt.json"
        entry: dict[str, Any] = {"design_pinned_path": path,
                                 "parent_experiment": parent_exp,
                                 "parent_receipt": str(rp)}
        if not rp.exists():
            entry.update({"status": "FAIL", "reason": "parent receipt absent"})
            out["status"] = "FAIL"
            out["reserves"][cell] = entry
            continue
        r = json.loads(rp.read_text())
        matches = r.get("sampler_path") == path
        lineage = (r.get("experiment") == parent_exp
                   and r.get("seed") == spec["seed"]
                   and r.get("rank") == C.RANK
                   and r.get("steps") == C.STEPS
                   and r.get("rows_seen_total") == RESERVE_ROWS_SEEN[cell]
                   and r.get("base_model") == C.SUBSTRATES["small"]["base_model"])
        entry.update({
            "parent_cell": r.get("cell"), "parent_seed": r.get("seed"),
            "parent_sampler_path": r.get("sampler_path"),
            "path_matches_parent_receipt": matches,
            "recipe_lineage_ok": lineage,
            "parent_rank": r.get("rank"), "parent_steps": r.get("steps"),
            "parent_rows_seen_total": r.get("rows_seen_total"),
            "parent_pool_sha256": r.get("pool_sha256"),
        })
        if live:
            entry["live"] = asyncio.run(_resolve(path))
        ok = matches and lineage and (not live or entry["live"]["resolves"])
        entry["status"] = "PASS" if ok else "FAIL"
        if not ok:
            out["status"] = "FAIL"
        out["reserves"][cell] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trains-dir", type=Path, required=True)
    ap.add_argument("--parent-receipt", type=Path, required=True,
                    help="the parent wave's int13-s4 receipt — the key-by-key reference")
    ap.add_argument("--reserve-dir", type=Path, required=True)
    ap.add_argument("--corpus-receipt", type=Path, required=True,
                    help="corpora/build_receipt.json — the byte-equality evidence "
                         "that closes the corpus-provenance chain")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    cells = [c["cell"] for c in C.cells()]
    report = {"experiment": C.EXPERIMENT,
              "recipe_identity": check_recipe(args.trains_dir, args.parent_receipt,
                                              cells, args.corpus_receipt),
              "reserve_identity": check_reserves(args.reserve_dir, not args.no_live),
              "packages": C.package_versions()}
    report["status"] = ("PASS" if report["recipe_identity"]["status"] == "PASS"
                        and report["reserve_identity"]["status"] == "PASS" else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
