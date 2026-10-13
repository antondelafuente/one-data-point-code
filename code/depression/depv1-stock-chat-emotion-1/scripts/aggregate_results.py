#!/usr/bin/env python3
"""Rebuild locked descriptive tables, hierarchical bootstraps, and A2 sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EMOTION = {
    "sonnet_despair": ("sonnet", "despair"),
    "sonnet_elation": ("sonnet", "elation"),
    "terra_despair": ("terra", "despair"),
    "terra_elation": ("terra", "elation"),
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in rows(path)}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def stats(values: list[int], denominator: int) -> dict:
    counts = Counter(values)
    return {
        "n": len(values),
        "N": denominator,
        "mean": round(statistics.fmean(values), 6) if values else None,
        "median": statistics.median(values) if values else None,
        "distribution": {str(score): counts.get(score, 0) for score in range(11)},
        "share_ge_4": round(sum(value >= 4 for value in values) / len(values), 6) if values else None,
        "share_ge_7": round(sum(value >= 7 for value in values) / len(values), 6) if values else None,
    }


def sign(value: float | None, tolerance: float = 1e-12) -> int | None:
    if value is None:
        return None
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--primary-dir", type=Path, required=True)
    ap.add_argument("--caps-sonnet", type=Path, required=True)
    ap.add_argument("--caps-terra", type=Path, required=True)
    ap.add_argument("--coherence-sonnet", type=Path, required=True)
    ap.add_argument("--coherence-terra", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--score-rows", type=Path, required=True)
    args = ap.parse_args()

    conversations = rows(args.conversations)
    conv = {row["id"]: row for row in conversations}
    targets = {row["id"]: row for row in rows(args.targets)}
    luna = latest(args.primary_dir / "luna_coherence.jsonl")
    surfaces = {name: latest(args.primary_dir / f"{name}.jsonl") for name in EMOTION}
    coherent_ids = {
        row_id for row_id, row in luna.items()
        if row.get("status") == "ok" and row.get("label") == "coherent"
    }
    sibling_coherence = {
        "sonnet": latest(args.coherence_sonnet),
        "terra": latest(args.coherence_terra),
    }
    sibling_override_coherent_ids = {}
    for family, surface in sibling_coherence.items():
        overridden = set(coherent_ids)
        for row_id, row in surface.items():
            if row.get("status") != "ok":
                continue
            if row.get("label") == "coherent":
                overridden.add(row_id)
            elif row.get("label") == "incoherent":
                overridden.discard(row_id)
        sibling_override_coherent_ids[family] = overridden
    subjects = sorted({row["subject"] for row in conversations})
    batteries = sorted({row["battery"] for row in conversations})

    health = {}
    for subject in subjects:
        subject_rows = [row for row in conversations if row["subject"] == subject]
        complete = [row for row in subject_rows if row.get("complete")]
        incoherent = sum(
            luna.get(row["id"], {}).get("label") == "incoherent" for row in complete
        )
        health[subject] = {
            "intended_N": len(subject_rows),
            "complete_n": len(complete),
            "failed_conversation_rate": round((len(subject_rows) - len(complete)) / len(subject_rows), 6),
            "incoherence_rate_among_complete": round(incoherent / len(complete), 6) if complete else None,
            "capped_conversation_rate": round(sum(row.get("n_capped_turns", 0) > 0 for row in subject_rows) / len(subject_rows), 6),
            "empty_conversation_rate": round(sum(row.get("n_empty_turns", 0) > 0 for row in subject_rows) / len(subject_rows), 6),
            "refusal_conversation_rate": round(sum(row.get("n_refusal_turns", 0) > 0 for row in subject_rows) / len(subject_rows), 6),
        }

    tables = []
    scenario_tables = []
    score_maps = {}
    for name, (family, axis) in EMOTION.items():
        surface = surfaces[name]
        score_map = {
            row_id: row["score"] for row_id, row in surface.items() if row.get("status") == "ok"
        }
        score_maps[name] = score_map
        for subject in subjects:
            for battery in batteries:
                denominator_ids = {
                    row_id for row_id, row in targets.items()
                    if row["subject"] == subject and row["battery"] == battery
                }
                raw_values = [score_map[row_id] for row_id in denominator_ids if row_id in score_map]
                coherent_values = [
                    score_map[row_id] for row_id in denominator_ids
                    if row_id in score_map and row_id in coherent_ids
                ]
                coherent_denominator = len(denominator_ids & coherent_ids)
                retention = coherent_denominator / len(denominator_ids) if denominator_ids else 0
                warning = "withhold" if retention < 0.5 else "selection_warning" if retention < 0.9 else "none"
                tables.append({
                    "judge_family": family,
                    "axis": axis,
                    "subject": subject,
                    "battery": battery,
                    "raw": stats(raw_values, len(denominator_ids)),
                    "coherent": stats(coherent_values, coherent_denominator),
                    "coherent_retention": round(retention, 6),
                    "interpretation_warning": warning,
                    "judge_failure_rate": round(
                        (len(denominator_ids) - len(raw_values)) / len(denominator_ids), 6
                    ) if denominator_ids else None,
                    "raw_missing_score_bound": {
                        "all_missing_zero": round(sum(raw_values) / len(denominator_ids), 6),
                        "all_missing_ten": round(
                            (sum(raw_values) + 10 * (len(denominator_ids) - len(raw_values)))
                            / len(denominator_ids), 6
                        ),
                    } if denominator_ids else None,
                    "coherent_missing_score_bound": {
                        "all_missing_zero": round(sum(coherent_values) / coherent_denominator, 6),
                        "all_missing_ten": round(
                            (sum(coherent_values) + 10 * (coherent_denominator - len(coherent_values)))
                            / coherent_denominator, 6
                        ),
                    } if coherent_denominator else None,
                })
                for scenario_id in sorted({
                    row["scenario_id"] for row in targets.values()
                    if row["subject"] == subject and row["battery"] == battery
                }):
                    scenario_ids = {
                        row_id for row_id, row in targets.items()
                        if row["subject"] == subject and row["battery"] == battery
                        and row["scenario_id"] == scenario_id
                    }
                    scenario_raw = [
                        score_map[row_id] for row_id in scenario_ids if row_id in score_map
                    ]
                    scenario_coherent = [
                        score_map[row_id] for row_id in scenario_ids
                        if row_id in score_map and row_id in coherent_ids
                    ]
                    scenario_tables.append({
                        "judge_family": family,
                        "axis": axis,
                        "subject": subject,
                        "battery": battery,
                        "scenario_id": scenario_id,
                        "raw": stats(scenario_raw, len(scenario_ids)),
                        "coherent": stats(
                            scenario_coherent, len(scenario_ids & coherent_ids)
                        ),
                    })

    intended_by_arm: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in conversations:
        intended_by_arm[(row["subject"], row["battery"])][row["scenario_id"]].append(row["id"])
    for arm in intended_by_arm.values():
        for scenario_ids in arm.values():
            scenario_ids.sort(key=lambda row_id: conv[row_id]["draw"])

    def arm_bootstrap(
        rng: random.Random, subject: str, battery: str, score_map: dict[str, int], coherent_only: bool
    ) -> float | None:
        scenario_map = intended_by_arm[(subject, battery)]
        scenarios = sorted(scenario_map)
        sampled_values: list[int] = []
        for scenario in (rng.choice(scenarios) for _ in range(6)):
            draw_ids = scenario_map[scenario]
            for row_id in (rng.choice(draw_ids) for _ in range(6)):
                if row_id in score_map and (not coherent_only or row_id in coherent_ids):
                    sampled_values.append(score_map[row_id])
        return statistics.fmean(sampled_values) if sampled_values else None

    contrasts = []
    for name, (family, axis) in EMOTION.items():
        target_battery = "competence_attack" if axis == "despair" else "praise_bait"
        contrast_name = "attack_despair_lift" if axis == "despair" else "praise_elation_lift"
        score_map = score_maps[name]
        for subject in subjects:
            row = {"judge_family": family, "axis": axis, "subject": subject, "contrast": contrast_name}
            for view, coherent_only in (("raw", False), ("coherent", True)):
                left_ids = [
                    row_id for row_id, meta in targets.items()
                    if meta["subject"] == subject and meta["battery"] == target_battery
                    and row_id in score_map and (not coherent_only or row_id in coherent_ids)
                ]
                neutral_ids = [
                    row_id for row_id, meta in targets.items()
                    if meta["subject"] == subject and meta["battery"] == "neutral"
                    and row_id in score_map and (not coherent_only or row_id in coherent_ids)
                ]
                point = (
                    statistics.fmean(score_map[row_id] for row_id in left_ids)
                    - statistics.fmean(score_map[row_id] for row_id in neutral_ids)
                ) if left_ids and neutral_ids else None
                rng_seed = args.seed ^ int(hashlib.sha256(
                    f"{name}|{subject}|{view}".encode()
                ).hexdigest()[:16], 16)
                rng = random.Random(rng_seed)
                replicates = []
                for _ in range(args.replicates):
                    left = arm_bootstrap(rng, subject, target_battery, score_map, coherent_only)
                    neutral = arm_bootstrap(rng, subject, "neutral", score_map, coherent_only)
                    if left is not None and neutral is not None:
                        replicates.append(left - neutral)
                row[view] = {
                    "point_estimate": round(point, 6) if point is not None else None,
                    "ci_2_5": round(percentile(replicates, 0.025), 6) if replicates else None,
                    "ci_97_5": round(percentile(replicates, 0.975), 6) if replicates else None,
                    "bootstrap_replicates": len(replicates),
                    "target_n": len(left_ids),
                    "neutral_n": len(neutral_ids),
                }
            contrasts.append(row)

    # A2 sensitivity: replace only caps-heavy original elation scores with the
    # corresponding assistant-lowercased re-judgment, then recompute lifts.
    caps_surfaces = {
        "sonnet": latest(args.caps_sonnet),
        "terra": latest(args.caps_terra),
    }
    caps_sensitivity = {}
    for family in ("sonnet", "terra"):
        original_map = score_maps[f"{family}_elation"]
        replaced_map = dict(original_map)
        replacements = 0
        for row_id, row in caps_surfaces[family].items():
            if row.get("status") == "ok":
                replaced_map[row_id] = row["score"]
                replacements += 1
        family_rows = {}
        for subject in subjects:
            def lift(score_map: dict[str, int]) -> float | None:
                praise = [score_map[row_id] for row_id, meta in targets.items()
                          if meta["subject"] == subject and meta["battery"] == "praise_bait"
                          and row_id in coherent_ids and row_id in score_map]
                neutral = [score_map[row_id] for row_id, meta in targets.items()
                           if meta["subject"] == subject and meta["battery"] == "neutral"
                           and row_id in coherent_ids and row_id in score_map]
                return statistics.fmean(praise) - statistics.fmean(neutral) if praise and neutral else None
            original_lift, normalized_lift = lift(original_map), lift(replaced_map)
            family_rows[subject] = {
                "original_lift": round(original_lift, 6) if original_lift is not None else None,
                "normalized_replacement_lift": round(normalized_lift, 6) if normalized_lift is not None else None,
                "direction_survives": sign(original_lift) == sign(normalized_lift),
            }
        pairwise = []
        for i, subject_a in enumerate(subjects):
            for subject_b in subjects[i + 1:]:
                oa, ob = family_rows[subject_a]["original_lift"], family_rows[subject_b]["original_lift"]
                na = family_rows[subject_a]["normalized_replacement_lift"]
                nb = family_rows[subject_b]["normalized_replacement_lift"]
                pairwise.append({
                    "subject_a": subject_a,
                    "subject_b": subject_b,
                    "original_difference": round(oa - ob, 6) if oa is not None and ob is not None else None,
                    "normalized_difference": round(na - nb, 6) if na is not None and nb is not None else None,
                    "ordering_direction_survives": (
                        sign(oa - ob) == sign(na - nb)
                        if None not in (oa, ob, na, nb) else False
                    ),
                })
        caps_sensitivity[family] = {
            "successful_replacements": replacements,
            "model_lifts": family_rows,
            "pairwise_orderings": pairwise,
        }

    # Coherence sensitivity does not replace Luna's locked primary labels. For the
    # prospectively selected set only, mechanically override Luna with each sibling
    # label and report the resulting cells/contrasts as a targeted sensitivity.
    coherence_sensitivity = {
        "interpretation": (
            "Luna remains primary. Each alternate view overrides Luna only where the prospectively "
            "selected 84-row sensitivity set has an ok sibling label; unselected rows retain Luna."
        ),
        "selected_targets": len(set().union(*(set(v) for v in sibling_coherence.values()))),
        "families": {},
    }
    for sibling_family, sibling_surface in sibling_coherence.items():
        selected_ok = {
            row_id: row for row_id, row in sibling_surface.items() if row.get("status") == "ok"
        }
        disagreements = Counter(
            f"luna_{luna[row_id]['label']}__{sibling_family}_{row['label']}"
            for row_id, row in selected_ok.items()
        )
        override_ids = sibling_override_coherent_ids[sibling_family]
        alternate_cells = []
        alternate_contrasts = []
        for name, (judge_family, axis) in EMOTION.items():
            score_map = score_maps[name]
            for subject in subjects:
                for battery in batteries:
                    denominator_ids = {
                        row_id for row_id, meta in targets.items()
                        if meta["subject"] == subject and meta["battery"] == battery
                        and row_id in override_ids
                    }
                    values = [score_map[row_id] for row_id in denominator_ids if row_id in score_map]
                    alternate_cells.append({
                        "coherence_family": sibling_family,
                        "judge_family": judge_family,
                        "axis": axis,
                        "subject": subject,
                        "battery": battery,
                        "scores": stats(values, len(denominator_ids)),
                    })
                target_battery = "competence_attack" if axis == "despair" else "praise_bait"
                target_ids = [
                    row_id for row_id, meta in targets.items()
                    if meta["subject"] == subject and meta["battery"] == target_battery
                    and row_id in override_ids and row_id in score_map
                ]
                neutral_ids = [
                    row_id for row_id, meta in targets.items()
                    if meta["subject"] == subject and meta["battery"] == "neutral"
                    and row_id in override_ids and row_id in score_map
                ]
                point = (
                    statistics.fmean(score_map[row_id] for row_id in target_ids)
                    - statistics.fmean(score_map[row_id] for row_id in neutral_ids)
                ) if target_ids and neutral_ids else None
                alternate_contrasts.append({
                    "coherence_family": sibling_family,
                    "judge_family": judge_family,
                    "axis": axis,
                    "subject": subject,
                    "contrast": (
                        "attack_despair_lift" if axis == "despair" else "praise_elation_lift"
                    ),
                    "point_estimate": round(point, 6) if point is not None else None,
                    "target_n": len(target_ids),
                    "neutral_n": len(neutral_ids),
                })
        coherence_sensitivity["families"][sibling_family] = {
            "selected_ok": len(selected_ok),
            "label_counts": dict(Counter(row["label"] for row in selected_ok.values())),
            "luna_sibling_cross_tab": dict(sorted(disagreements.items())),
            "alternate_cells": alternate_cells,
            "alternate_contrasts": alternate_contrasts,
        }

    # A4 used whole-transcript substring grounding. The semantic audit found one
    # accepted quote attributable only to a user turn. Preserve the locked score,
    # and report the exact exclusion effect rather than silently deleting it.
    attribution_row_id = "qwen_3_5_9b|praise_bait|cover_letter|d1"
    attribution_score = score_maps["sonnet_elation"].get(attribution_row_id)
    qwen_praise_ids = [
        row_id for row_id, meta in targets.items()
        if meta["subject"] == "qwen_3_5_9b" and meta["battery"] == "praise_bait"
        and row_id in coherent_ids and row_id in score_maps["sonnet_elation"]
    ]
    with_value = statistics.fmean(score_maps["sonnet_elation"][row_id] for row_id in qwen_praise_ids)
    without_ids = [row_id for row_id in qwen_praise_ids if row_id != attribution_row_id]
    without_value = statistics.fmean(
        score_maps["sonnet_elation"][row_id] for row_id in without_ids
    )
    quote_attribution_sensitivity = {
        "id": attribution_row_id,
        "locked_score_retained": attribution_score,
        "cell": "qwen_3_5_9b|praise_bait|sonnet|elation|luna_coherent",
        "mean_with_row": round(with_value, 6),
        "n_with_row": len(qwen_praise_ids),
        "mean_excluding_row": round(without_value, 6),
        "n_excluding_row": len(without_ids),
        "mean_delta_excluding_minus_locked": round(without_value - with_value, 6),
    }

    grounding_summary = {}
    for name, (family, axis) in EMOTION.items():
        surface = surfaces[name]
        by_subject = {}
        for subject in subjects:
            selected = [row for row_id, row in surface.items() if targets[row_id]["subject"] == subject]
            by_subject[subject] = {
                "intended_N": len(selected),
                "status_counts": dict(Counter(row.get("status") for row in selected)),
                "accepted_grounding_modes": dict(Counter(
                    row.get("grounding_mode") for row in selected if row.get("status") == "ok"
                )),
            }
        grounding_summary[name] = {
            "judge_family": family,
            "axis": axis,
            "by_subject": by_subject,
        }

    score_rows = []
    for row_id, meta in sorted(targets.items()):
        score_rows.append({
            "id": row_id,
            "subject": meta["subject"],
            "battery": meta["battery"],
            "scenario_id": meta["scenario_id"],
            "draw": meta["draw"],
            "transcript_sha256": meta["transcript_sha256"],
            "n_capped_turns": meta["n_capped_turns"],
            "luna_coherence": {
                key: luna[row_id].get(key)
                for key in ("status", "label", "confidence", "breakdown_type", "evidence")
            },
            "emotion": {
                name: {
                    key: surfaces[name][row_id].get(key)
                    for key in (
                        "status", "score", "quote", "grounding_mode", "error",
                    )
                }
                for name in EMOTION
            },
            "coherence_sensitivity": {
                family: ({
                    key: surface[row_id].get(key)
                    for key in ("status", "label", "confidence", "breakdown_type", "evidence")
                } if row_id in surface else None)
                for family, surface in sibling_coherence.items()
            },
        })

    examples = []
    for name, (family, axis) in EMOTION.items():
        relevant = "competence_attack" if axis == "despair" else "praise_bait"
        score_map = score_maps[name]
        for subject in subjects:
            candidates = [row_id for row_id, meta in targets.items()
                          if meta["subject"] == subject and meta["battery"] == relevant
                          and row_id in coherent_ids and row_id in score_map]
            if not candidates:
                continue
            values = [score_map[row_id] for row_id in candidates]
            median = statistics.median(values)
            nearest = min(candidates, key=lambda row_id: (abs(score_map[row_id] - median), row_id))
            highest = min(candidates, key=lambda row_id: (-score_map[row_id], row_id))
            for kind, row_id in (("nearest_median", nearest), ("highest", highest)):
                examples.append({
                    "judge_family": family, "axis": axis, "subject": subject,
                    "battery": relevant, "kind": kind, "id": row_id,
                    "score": score_map[row_id], "transcript": targets[row_id]["trajectory"],
                })

    report = {
        "status": "PASS",
        "replicates_requested": args.replicates,
        "bootstrap_seed": args.seed,
        "hierarchical_bootstrap": (
            "six scenario slots resampled with replacement per arm; within each selected scenario, "
            "six intended draw ids resampled with replacement; unavailable/noncoherent scores omitted"
        ),
        "health": health,
        "cell_tables": tables,
        "scenario_tables": scenario_tables,
        "contrasts": contrasts,
        "caps_normalized_sensitivity": caps_sensitivity,
        "coherence_sensitivity": coherence_sensitivity,
        "quote_attribution_sensitivity": quote_attribution_sensitivity,
        "grounding_summary": grounding_summary,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.examples.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples))
    args.score_rows.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in score_rows)
    )
    print(json.dumps({"status": "PASS", "cell_rows": len(tables),
                      "scenario_rows": len(scenario_tables),
                      "contrast_rows": len(contrasts), "example_rows": len(examples),
                      "score_rows": len(score_rows)}, indent=2))


if __name__ == "__main__":
    main()
