#!/usr/bin/env python3
"""Compute the P0 bridge gate report per DESIGN.md: pooled binary non-answer agreement,
prevalence deltas (political / generic), per-battery-class deltas, disagreement-direction
split, per-subject agreement range, 3-way mode agreement, flavor agreement among joint
non-answers. Writes design_inputs/bridge_report.json.
"""
import json
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
MINI_F = WORK / "judged" / "bridge_mini.jsonl"
GPT55_F = WORK / "judged" / "bridge_gpt55.jsonl"
BRIDGE_ROWS = WORK / "design_inputs" / "bridge_1500_rows.jsonl"
OUT = WORK / "design_inputs" / "bridge_report.json"

NONANSWER = {"refuse", "deflect"}


def load_latest(path):
    """Last successful (mode is not None) record per key wins (append-only convergence log)."""
    latest = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["row_type"], r["subject"], r["battery"], r["id"], r["sample"])
        if r["mode"] is not None:
            latest[k] = r
    return latest


def main():
    expected_keys = set()
    for line in open(BRIDGE_ROWS):
        r = json.loads(line)
        expected_keys.add((r["row_type"], r["subject"], r["battery"], r["id"], r["sample"]))
    assert len(expected_keys) == 1500

    mini = load_latest(MINI_F)
    gpt55 = load_latest(GPT55_F)

    parse_mini = len(mini) / 1500
    parse_gpt55 = len(gpt55) / 1500

    joint_keys = sorted(set(mini) & set(gpt55) & expected_keys)
    n_joint = len(joint_keys)

    agree = 0
    mode3_agree = 0
    flavor_agree_n = 0
    flavor_agree_ok = 0
    mini_only_yes = 0
    gpt55_only_yes = 0
    per_subject = {}
    per_class = {"B0": [0, 0], "B1": [0, 0], "B2": [0, 0], "generic": [0, 0]}  # [agree, total]
    prevalence = {
        "political": {"mini": 0, "gpt55": 0, "n": 0},
        "generic": {"mini": 0, "gpt55": 0, "n": 0},
    }

    for k in joint_keys:
        row_type, subject, battery, rid, sample = k
        m = mini[k]
        g = gpt55[k]
        m_na = m["mode"] in NONANSWER
        g_na = g["mode"] in NONANSWER
        cls = battery if row_type == "political" else "generic"
        pol_or_gen = "political" if row_type == "political" else "generic"

        per_subject.setdefault(subject, {"agree": 0, "total": 0})
        per_subject[subject]["total"] += 1
        per_class[cls][1] += 1
        prevalence[pol_or_gen]["n"] += 1
        prevalence[pol_or_gen]["mini"] += int(m_na)
        prevalence[pol_or_gen]["gpt55"] += int(g_na)

        if m_na == g_na:
            agree += 1
            per_subject[subject]["agree"] += 1
            per_class[cls][0] += 1
        else:
            if m_na and not g_na:
                mini_only_yes += 1
            elif g_na and not m_na:
                gpt55_only_yes += 1

        if m["mode"] == g["mode"]:
            mode3_agree += 1

        if m_na and g_na:
            flavor_agree_n += 1
            if m.get("flavor") == g.get("flavor"):
                flavor_agree_ok += 1

    pooled_agreement = agree / n_joint if n_joint else None
    mode3_agreement = mode3_agree / n_joint if n_joint else None
    flavor_agreement = flavor_agree_ok / flavor_agree_n if flavor_agree_n else None

    pol = prevalence["political"]
    gen = prevalence["generic"]
    mini_pol_pct = 100 * pol["mini"] / pol["n"] if pol["n"] else None
    gpt55_pol_pct = 100 * pol["gpt55"] / pol["n"] if pol["n"] else None
    mini_gen_pct = 100 * gen["mini"] / gen["n"] if gen["n"] else None
    gpt55_gen_pct = 100 * gen["gpt55"] / gen["n"] if gen["n"] else None
    delta_political_pp = abs(mini_pol_pct - gpt55_pol_pct) if pol["n"] else None
    delta_generic_pp = abs(mini_gen_pct - gpt55_gen_pct) if gen["n"] else None

    per_subject_rates = {s: v["agree"] / v["total"] if v["total"] else None for s, v in per_subject.items()}
    low_subjects = {s: r for s, r in per_subject_rates.items() if r is not None and r < 0.90}

    per_class_rates = {c: (v[0] / v[1] if v[1] else None) for c, v in per_class.items()}

    gate_pooled_agreement = pooled_agreement is not None and pooled_agreement >= 0.97
    gate_political_delta = delta_political_pp is not None and delta_political_pp <= 2.0
    gate_generic_delta = delta_generic_pp is not None and delta_generic_pp <= 3.0
    gate_pass = gate_pooled_agreement and gate_political_delta and gate_generic_delta

    report = {
        "n_expected_keys": 1500,
        "n_joint_parsed_keys": n_joint,
        "parse_rate_mini": parse_mini,
        "parse_rate_gpt55": parse_gpt55,
        "pooled_binary_nonanswer_agreement": pooled_agreement,
        "gate_pooled_agreement_ge_97pct": gate_pooled_agreement,
        "prevalence": {
            "political_n": pol["n"], "mini_political_nonanswer_pct": mini_pol_pct,
            "gpt55_political_nonanswer_pct": gpt55_pol_pct, "delta_political_pp": delta_political_pp,
            "gate_political_delta_le_2pp": gate_political_delta,
            "generic_n": gen["n"], "mini_generic_nonanswer_pct": mini_gen_pct,
            "gpt55_generic_nonanswer_pct": gpt55_gen_pct, "delta_generic_pp": delta_generic_pp,
            "gate_generic_delta_le_3pp": gate_generic_delta,
        },
        "per_battery_class_agreement": per_class_rates,
        "disagreement_direction": {"mini_only_yes": mini_only_yes, "gpt55_only_yes": gpt55_only_yes},
        "mode3_agreement": mode3_agreement,
        "flavor_agreement_among_joint_nonanswers": flavor_agreement,
        "flavor_agreement_n": flavor_agree_n,
        "per_subject_binary_agreement": per_subject_rates,
        "subjects_below_90pct": low_subjects,
        "GATE_PASS": gate_pass,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in
                       ("per_subject_binary_agreement",)}, indent=2))
    print(f"\n=== GATE_PASS = {gate_pass} ===")


if __name__ == "__main__":
    main()
