#!/usr/bin/env python3
"""Verify the p1a-s4 re-serve against the PARENT's committed adapter-identity record.

CHECKLIST [BLOCK]: "p1a-s4 re-serve: sampler URI verified against the parent's
`trains_adapter_identity.json` AND the greedy-probe adapter-identity check run across
all served endpoints."

GATED checks (what the CHECKLIST asks for):

  1. the URI this wave serves == the URI the parent recorded for `p1a-s4`,
     == the DESIGN's pinned identifier;
  2. the parent's own record shows `p1a-s4` distinct from stock on the probe set;
  3. THIS wave's greedy probe shows `p1a_s4_reserve` distinct from stock.

RECORDED, NOT GATED: whether this wave's greedy probe reproduces the parent's stored
greedy TEXT byte-for-byte. It does not, and that is expected, not a finding —
automated-researcher#477 is explicit that Tinker-served sampling is NOT byte-reproducible
even at a fixed seed (three back-to-back calls with an identical seed/prompt/params gave
three different completions), so a stored greedy string is not a cross-session identity
oracle and must not be gated on. The comparison is emitted anyway, with both texts, so a
reader can see what did and did not match.

The adapter's real behavioural verification is the DESIGN's own ANCHOR GATE: the
re-serve must read ABOVE co-measured stock on contradiction with a paired CI excluding
zero. That is a property of the weights, measured in-wave, and it is what would actually
catch a moved checkpoint.

Sampler-weights URIs are SERVE-ONLY: this run never resumes training from them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-identity", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True,
                    help="this wave's trains_adapter_identity.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parent = json.loads(args.parent_identity.read_text())
    mine = json.loads(args.probe.read_text())
    want_uri = C.RESERVE_SPECS[0][1]

    p_raw = (parent.get("raw") or {}).get("p1a-s4") or {}
    m_raw = (mine.get("raw") or {}).get("p1a_s4_reserve") or {}
    p_per = (parent.get("per_subject") or {}).get("p1a-s4") or {}

    observations = [
        {"observation": "this wave's greedy probe text vs the parent's stored greedy text",
         "served_hashes": m_raw.get("hashes"), "parent_hashes": p_raw.get("hashes"),
         "byte_identical": bool(m_raw.get("hashes") == p_raw.get("hashes")),
         "served_first_text": m_raw.get("first_text"),
         "parent_first_text": p_raw.get("first_text"),
         "gated": False,
         "why_not_gated": ("automated-researcher#477: Tinker-served sampling is not "
                           "byte-reproducible even at a fixed seed, so a stored greedy "
                           "string is not a cross-session identity oracle. The gated "
                           "identity evidence is the URI match + distinct-from-stock in "
                           "BOTH records; the behavioural verification is the DESIGN's "
                           "anchor gate.")},
    ]

    checks = [
        {"check": "served URI == the parent's recorded p1a-s4 sampler URI",
         "served": m_raw.get("sampler_path"), "parent": p_raw.get("sampler_path"),
         "design_pin": want_uri,
         "pass": bool(m_raw.get("sampler_path") == p_raw.get("sampler_path")
                      == want_uri)},
        {"check": "the probe SET and decoding config are identical to the parent's, so "
                  "the two records are comparable at all",
         "probe_prompts_identical": (parent.get("probe", {}).get("prompts")
                                     == mine.get("probe", {}).get("prompts")),
         "decoding_identical": (parent.get("probe", {}).get("decoding")
                                == mine.get("probe", {}).get("decoding")),
         "pass": bool(parent.get("probe", {}).get("prompts")
                      == mine.get("probe", {}).get("prompts")
                      and parent.get("probe", {}).get("decoding")
                      == mine.get("probe", {}).get("decoding"))},
        {"check": "parent recorded p1a-s4 as distinct from stock on the probe set",
         "prompts_differing_from_stock": p_per.get("prompts_differing_from_stock"),
         "pass": bool(p_per.get("distinct_from_stock"))},
        {"check": "this wave's probe records p1a_s4_reserve as distinct from stock",
         "prompts_differing_from_stock":
             (mine.get("per_subject", {}).get("p1a_s4_reserve", {})
              .get("prompts_differing_from_stock")),
         "pass": bool(mine.get("per_subject", {}).get("p1a_s4_reserve", {})
                      .get("distinct_from_stock"))},
    ]
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    report = {"experiment": C.EXPERIMENT,
              "gate": "p1a-s4 re-serve identity vs the parent record",
              "parent_record": str(args.parent_identity),
              "parent_experiment": parent.get("experiment"),
              "serve_only": "sampler_weights URIs are serve-only; this run never resumes "
                            "training from them (#348-adjacent)",
              "verdict": verdict, "checks": checks,
              "observations_not_gated": observations,
              "behavioural_verification": ("the DESIGN's anchor gate — the re-serve must "
                                           "read above co-measured stock on contradiction "
                                           "with a paired CI excluding zero (judge/anchor_gate.json)")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if verdict != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
