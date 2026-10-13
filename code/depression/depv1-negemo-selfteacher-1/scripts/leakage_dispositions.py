#!/usr/bin/env python3
"""Summarise the three self-pool leakage screens into the recorded-disposition table.

The CHECKLIST gate is not "the screen ran" — it is "every flagged pair READ with a
recorded disposition, and any exclusion applied to BOTH of that substrate's pools".
This script produces the evidence a reader needs to check that judgement: per pool,
the flag counts, the highest-cosine pairs in each direction, which probes attract
them, and whether the flagged pool rows are the SAME rows across substrates (they
must be — the prompt set is frozen and shared, so a flag driven by the prompt side
cannot differentially favour one arm).

It records no exclusions of its own; exclusions are a human/agent decision and are
written into LEAKAGE_DISPOSITIONS.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"experiment": C.EXPERIMENT, "pools": {}, "cross_substrate": {}}
    flagged_ids: dict[str, set] = {}
    for sub in ("qwen", "small", "large"):
        rep_p = a.audit_dir / f"leakage_screen_{sub}.json"
        pair_p = a.audit_dir / f"leakage_pairs_{sub}.json"
        if not rep_p.exists():
            continue
        rep = json.loads(rep_p.read_text())
        pairs = json.loads(pair_p.read_text())
        pv = [r for r in pairs if r["kind"] == "pool_vs_probe"]
        wp = [r for r in pairs if r["kind"] == "within_pool"]
        flagged_ids[sub] = {r["pool_id"] for r in pv}
        out["pools"][sub] = {
            "n_pool_rows": rep["n_pool_rows"],
            "threshold": rep["threshold"],
            "max_tokens_embedded": rep.get("max_tokens_embedded"),
            "pool_vs_probe": {
                "n_pairs_flagged": rep["pool_vs_probe"]["n_pairs_flagged"],
                "n_distinct_pool_rows_flagged": len(flagged_ids[sub]),
                "global_max_cosine": rep["pool_vs_probe"]["global_max_cosine"],
                "global_mean_cosine": rep["pool_vs_probe"]["global_mean_cosine"],
                "flags_by_probe": dict(collections.Counter(
                    r["probe"] for r in pv).most_common()),
                "top_pairs": [
                    {"probe": r["probe"], "pool_id": r["pool_id"], "cosine": r["cosine"],
                     "probe_text": r["probe_text"][:400],
                     "pool_text": r["pool_text"][:600]}
                    for r in sorted(pv, key=lambda r: -r["cosine"])[:10]],
            },
            "within_pool": {
                **{k: rep["within_pool"][k] for k in
                   ("n_pairs_flagged", "max_cosine", "n_rows_with_a_flagged_partner")},
                "cosine_histogram": dict(zip(
                    [f"{lo}-{hi}" for lo, hi in zip(
                        rep["within_pool"]["cosine_histogram_bins"],
                        rep["within_pool"]["cosine_histogram_bins"][1:])],
                    rep["within_pool"]["cosine_histogram_counts"])),
                "top_pairs": [
                    {"a": r["pool_id"], "b": r["probe"], "cosine": r["cosine"],
                     "a_text": r["pool_text"][:400], "b_text": r["probe_text"][:400]}
                    for r in sorted(wp, key=lambda r: -r["cosine"])[:10]],
            },
            "exclusions_applied": [],
        }

    # The load-bearing cross-check: a flag driven by the shared, frozen PROMPT set must
    # hit (very nearly) the same pool rows in every substrate. A substrate-specific
    # flag would mean the COMPLETION drove it, which is the case that could differ
    # between arms.
    if len(flagged_ids) > 1:
        subs = sorted(flagged_ids)
        inter = set.intersection(*flagged_ids.values())
        union = set.union(*flagged_ids.values())
        out["cross_substrate"] = {
            "substrates_compared": subs,
            "flagged_pool_rows_per_substrate": {k: len(v) for k, v in flagged_ids.items()},
            "in_every_substrate": len(inter),
            "in_any_substrate": len(union),
            "jaccard": round(len(inter) / max(len(union), 1), 4),
            "substrate_specific_rows": {
                k: sorted(v - inter)[:20] for k, v in flagged_ids.items()},
            "reading": ("a high overlap means the flags are driven by the frozen, shared "
                        "PROMPT set (inherited by citation, symmetric across arms); rows "
                        "flagged in only one substrate are the ones whose COMPLETION drove "
                        "the flag and are the ones worth reading individually"),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for sub, v in out["pools"].items():
        print(f"{sub:6s} pool_vs_probe={v['pool_vs_probe']['n_pairs_flagged']:>5d} "
              f"(rows {v['pool_vs_probe']['n_distinct_pool_rows_flagged']:>4d}, "
              f"max {v['pool_vs_probe']['global_max_cosine']})  "
              f"within_pool={v['within_pool']['n_pairs_flagged']:>9d} "
              f"(max {v['within_pool']['max_cosine']})")
    if out["cross_substrate"]:
        print(json.dumps({k: v for k, v in out["cross_substrate"].items()
                          if k not in ("substrate_specific_rows", "reading")}, indent=2))


if __name__ == "__main__":
    main()
