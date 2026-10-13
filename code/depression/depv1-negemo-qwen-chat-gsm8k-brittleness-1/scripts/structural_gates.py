#!/usr/bin/env python3
"""Full-pool structural gates over the ACTUAL wave (the CHECKLIST's experiment-specific [BLOCK]
boxes). Every check runs over every row, not a sample.

  G1 counts        — Leg A has exactly 200 rows per served subject; Leg B has exactly 100
                     conversations per served subject per condition with six assistant turns each.
  G2 shared prefix — every Leg-B turn-0 row is the BYTE-IDENTICAL reuse of that subject's own
                     persisted Leg-A answer, marked not-regenerated, and both branches of a
                     subject/problem pair share the same initial bytes.
  G3 foils/prompts — every rendered follow-up equals the frozen script after only `{foil}`
                     substitution; every foil equals gold+1 and never equals gold.
  G4 decoding      — persisted per-response decoding metadata matches DESIGN everywhere.
  G5 provenance    — each subject is served in exactly one group with one sampler URI, and the
                     duplicate stock serving is present in both groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import gsm8k_lib as L                                     # noqa: E402

CONDITIONS = ("neutral_verify", "false_attack")


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lega", nargs="+", required=True)
    ap.add_argument("--legb", nargs="+", required=True)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--script", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    problems = {p["id"]: p for p in L.load_problems(a.problems)}
    script = L.load_script(a.script)
    lega = [r for p in a.lega for r in read_jsonl(p)]
    legb = [r for p in a.legb for r in read_jsonl(p)]
    fails, report = [], {}

    def check(name, ok, detail):
        report[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    # ---- G1 counts ----
    a_counts = defaultdict(int)
    for r in lega:
        a_counts[r["subject"]] += 1
    check("G1_lega_200_per_subject", all(v == 200 for v in a_counts.values()) and len(a_counts) == 9,
          {"subjects": len(a_counts), "counts": dict(sorted(a_counts.items()))})
    conv = defaultdict(set)
    for r in legb:
        conv[(r["subject"], r["condition"], r["problem_id"])].add(r["turn0"])
    per_cell = defaultdict(int)
    ragged = []
    for (s, c, p), turns in conv.items():
        per_cell[(s, c)] += 1
        if turns != {0, 1, 2, 3, 4, 5}:
            ragged.append({"subject": s, "condition": c, "problem_id": p,
                           "turns": sorted(turns)})
    check("G1_legb_100_conversations_per_subject_condition",
          all(v == 100 for v in per_cell.values()) and len(per_cell) == 18,
          {"cells": len(per_cell), "counts": {f"{s}|{c}": v for (s, c), v in sorted(per_cell.items())}})
    check("G1_legb_six_assistant_turns_each", not ragged,
          {"n_conversations": len(conv), "n_ragged": len(ragged), "ragged": ragged[:10]})

    # ---- G2 shared prefix, byte-identical, never regenerated ----
    a_by = {r["exact_key"]: r for r in lega}
    bad_reuse, bad_flag, branch_mismatch = [], [], []
    initial = {}
    for r in legb:
        if r["turn0"] != 0:
            continue
        src = a_by.get(r["legA_exact_key"])
        if src is None or src["response_text"] != r["response_text"]:
            bad_reuse.append(r["exact_key"])
        if r.get("regenerated") is not False or r.get("source") != "legA_reuse":
            bad_flag.append(r["exact_key"])
        if r["legA_response_sha256"] != hashlib.sha256(r["response_text"].encode()).hexdigest():
            bad_reuse.append(r["exact_key"] + "|sha")
        initial.setdefault((r["subject"], r["problem_id"]), {})[r["condition"]] = r["response_text"]
    for key, byc in initial.items():
        if len(byc) == 2 and byc[CONDITIONS[0]] != byc[CONDITIONS[1]]:
            branch_mismatch.append("|".join(key))
    check("G2_initial_answer_is_byte_identical_legA_reuse", not bad_reuse,
          {"n_turn0_rows": len(initial) * 2, "n_bad": len(bad_reuse), "bad": bad_reuse[:10]})
    check("G2_initial_answer_never_regenerated", not bad_flag,
          {"n_bad": len(bad_flag), "bad": bad_flag[:10]})
    check("G2_both_branches_share_the_same_initial_bytes", not branch_mismatch,
          {"n_pairs": len(initial), "n_mismatch": len(branch_mismatch),
           "mismatch": branch_mismatch[:10]})

    # ---- G3 foils + rendered prompts ----
    bad_prompt, bad_foil = [], []
    for r in legb:
        p = problems[r["problem_id"]]
        if Decimal(p["foil"]) != Decimal(p["gold"]) + 1 or Decimal(p["foil"]) == Decimal(p["gold"]):
            bad_foil.append(r["problem_id"])
        if r["turn0"] == 0:
            if r["user_text"] != p["turn1_user"]:
                bad_prompt.append(r["exact_key"])
            continue
        tmpl = script[r["condition"]][r["turn0"] - 1]
        want = tmpl.replace("{foil}", p["foil"]) if r["condition"] == "false_attack" else tmpl
        if r["user_text"] != want or r.get("user_template") != tmpl:
            bad_prompt.append(r["exact_key"])
        if r["condition"] == "false_attack" and r.get("foil_substituted") != p["foil"]:
            bad_prompt.append(r["exact_key"] + "|foilfield")
        if r["condition"] == "neutral_verify" and r.get("foil_substituted") is not None:
            bad_prompt.append(r["exact_key"] + "|neutral_got_foil")
    check("G3_every_foil_is_gold_plus_one_and_differs", not bad_foil,
          {"n_bad": len(set(bad_foil)), "bad": sorted(set(bad_foil))[:10]})
    check("G3_every_rendered_prompt_matches_frozen_script", not bad_prompt,
          {"n_legb_rows": len(legb), "n_bad": len(bad_prompt), "bad": bad_prompt[:10]})

    # ---- G4 decoding metadata ----
    bad_cfg = []
    for r in lega + legb:
        g = r["gen_config"]
        cap = 512 if (r["leg"] == "A" or r["turn0"] == 0) else 1024
        ok = (g["temperature"] == 0.0 and g["seed"] == 0 and g["thinking_enabled"] is False
              and g["renderer"] == "qwen3_5_disable_thinking"
              and g["base_model"] == "Qwen/Qwen3.5-9B"
              and g["sampling_mode"] == "greedy" and g["system_message"] == ""
              and g["top_p"] == 1.0 and g["top_k"] == -1 and g["stop"] == ["<|im_end|>"]
              and g["requested_max_new_tokens"] == cap
              and g["substrate"] == "tinker_direct_sampling"
              and g["sampler_path"] == r["sampler_path"])
        if not ok:
            bad_cfg.append(r["exact_key"])
    check("G4_decoding_metadata_matches_design", not bad_cfg,
          {"n_rows": len(lega) + len(legb), "n_bad": len(bad_cfg), "bad": bad_cfg[:10],
           "pins": {"temperature": 0.0, "seed": 0, "thinking_enabled": False,
                    "renderer": "qwen3_5_disable_thinking", "base_model": "Qwen/Qwen3.5-9B",
                    "legA_cap": 512, "legB_followup_cap": 1024}})

    # ---- G5 serving provenance ----
    prov = defaultdict(set)
    for r in lega + legb:
        prov[r["subject"]].add((r["serving_group"], r["sampler_path"]))
    multi = {s: sorted(map(str, v)) for s, v in prov.items() if len(v) != 1}
    check("G5_each_subject_one_group_one_sampler", not multi, {"violations": multi})
    check("G5_stock_served_in_both_groups",
          {"stock_gA", "stock_gB"} <= set(prov),
          {"stock_subjects": sorted(s for s in prov if s.startswith("stock"))})

    payload = {"status": "PASS" if not fails else "FAIL", "failed_gates": fails,
               "n_lega_rows": len(lega), "n_legb_rows": len(legb),
               "n_conversations": len(conv), "gates": report}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "failed_gates": fails,
                      "n_lega_rows": len(lega), "n_legb_rows": len(legb),
                      "n_conversations": len(conv),
                      "gates": {k: v["pass"] for k, v in report.items()}}, indent=2))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
