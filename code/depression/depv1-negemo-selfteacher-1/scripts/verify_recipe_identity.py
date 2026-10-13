#!/usr/bin/env python3
"""Within-substrate recipe identity — a CHECKLIST [BLOCK] gate.

DESIGN: "Each self arm uses the SAME recipe as its substrate's gemma arm — the pool
is the only difference within a substrate." This checks that from the PERSISTED
TRAINING RECEIPTS, never from driver source, because the receipt is what the run
actually did.

Three substrates, two shapes:
  * qwen, small — both arms trained here, so compare the two local receipts;
  * large       — the gemma arm is DCS1's adapter RE-SERVED, so there is no local
                  gemma receipt by design. Compare `large-self` against DCS1's own
                  committed receipt instead (pulled into reference/ as a local
                  durable artifact, per the local-aggregation rule), and confirm the
                  re-served sampler URI in that receipt is the one this run served.

The pool sha256 is expected to DIFFER within a substrate — that difference IS the
variable under test — and is reported rather than compared.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402

RECIPE_KEYS = ["base_model", "renderer", "effort", "seed", "rank", "train_mlp",
               "train_attn", "train_unembed", "peak_learning_rate", "lr_schedule",
               "warmup_ratio", "lr_final_frac", "epochs", "effective_batch_size",
               "max_length", "steps"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--dcs1-receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rec = {}
    for p in sorted(glob.glob(str(a.state_dir / "train_*.json"))):
        d = json.loads(Path(p).read_text())
        if "-smoke" in d.get("arm", ""):
            continue
        rec[d["arm"]] = d
    dcs1 = json.loads(a.dcs1_receipt.read_text())

    fail, comparisons = [], {}
    for sub in ("qwen", "small"):
        g, s = f"{sub}-gemma", f"{sub}-self"
        if g not in rec or s not in rec:
            fail.append(f"{sub}: missing receipt ({g in rec=}, {s in rec=})")
            continue
        diff = {k: [rec[g].get(k), rec[s].get(k)] for k in RECIPE_KEYS
                if rec[g].get(k) != rec[s].get(k)}
        comparisons[sub] = {
            "shape": "both arms trained locally",
            "gemma_arm": g, "self_arm": s,
            "recipe_identical": not diff, "differing_keys": diff,
            "pool_sha256_gemma": rec[g]["pool_sha256"],
            "pool_sha256_self": rec[s]["pool_sha256"],
            "pools_differ_as_intended": rec[g]["pool_sha256"] != rec[s]["pool_sha256"],
        }
        if diff:
            fail.append(f"{sub}: recipe differs on {sorted(diff)}")
        if rec[g]["pool_sha256"] == rec[s]["pool_sha256"]:
            fail.append(f"{sub}: gemma and self pools have the SAME sha — no contrast")

    s = "large-self"
    if s not in rec:
        fail.append("large: missing large-self receipt")
    else:
        diff = {k: [dcs1.get(k), rec[s].get(k)] for k in RECIPE_KEYS
                if dcs1.get(k) != rec[s].get(k)}
        served = dcs1.get("sampler_path") == C.DCS1_INKLING_SAMPLER
        comparisons["large"] = {
            "shape": "gemma arm is DCS1's adapter re-served (no local gemma training)",
            "reference_receipt": str(a.dcs1_receipt),
            "self_arm": s,
            "recipe_identical_to_dcs1": not diff, "differing_keys": diff,
            "dcs1_sampler_path": dcs1.get("sampler_path"),
            "pinned_sampler_uri": C.DCS1_INKLING_SAMPLER,
            "dcs1_receipt_sampler_matches_the_uri_this_run_served": served,
            "pool_sha256_self": rec[s]["pool_sha256"],
            "pool_sha256_gemma": "n/a — DCS1's pool, byte-verified separately as a6a1b7dd…",
        }
        if diff:
            fail.append(f"large: large-self recipe differs from DCS1's on {sorted(diff)}")
        if not served:
            fail.append("large: DCS1 receipt's sampler_path != the pinned URI")

    out = {"status": "FAIL" if fail else "PASS",
           "experiment": C.EXPERIMENT,
           "rule": ("within a substrate, the gemma arm and the self arm must share every "
                    "recipe key; only the training pool may differ"),
           "recipe_keys_compared": RECIPE_KEYS,
           "comparisons": comparisons,
           "arms_seen": sorted(rec),
           "failures": fail, "n_failures": len(fail)}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if fail:
        raise SystemExit(f"recipe-identity verification FAILED: {len(fail)} problems")


if __name__ == "__main__":
    main()
