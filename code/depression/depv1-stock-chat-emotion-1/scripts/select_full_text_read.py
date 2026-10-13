#!/usr/bin/env python3
"""Materialize every full-transcript read required by the locked DESIGN."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SURFACES = {
    "sonnet_despair": ("sonnet", "despair"),
    "sonnet_elation": ("sonnet", "elation"),
    "terra_despair": ("terra", "despair"),
    "terra_elation": ("terra", "elation"),
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in rows(path)}


def order(row_id: str) -> str:
    return hashlib.sha256(row_id.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--primary-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    targets = {row["id"]: row for row in rows(args.targets)}
    luna = latest(args.primary_dir / "luna_coherence.jsonl")
    surfaces = {name: latest(args.primary_dir / f"{name}.jsonl") for name in SURFACES}
    coherent = {row_id for row_id, row in luna.items()
                if row.get("status") == "ok" and row.get("label") == "coherent"}
    subjects = sorted({row["subject"] for row in targets.values()})
    batteries = sorted({row["battery"] for row in targets.values()})
    reasons: dict[str, set[str]] = defaultdict(set)

    for name, (family, axis) in SURFACES.items():
        score_map = {row_id: row["score"] for row_id, row in surfaces[name].items()
                     if row.get("status") == "ok"}
        for subject in subjects:
            candidates = [row_id for row_id, row in targets.items()
                          if row["subject"] == subject and row_id in coherent and row_id in score_map]
            if candidates:
                highest = min(candidates, key=lambda row_id: (-score_map[row_id], row_id))
                reasons[highest].add(f"highest_coherent_{axis}:{family}:{subject}")
            for battery in batteries:
                cell = [row_id for row_id in candidates if targets[row_id]["battery"] == battery]
                if not cell:
                    continue
                median = statistics.median(score_map[row_id] for row_id in cell)
                nearest = min(cell, key=lambda row_id: (abs(score_map[row_id] - median), row_id))
                reasons[nearest].add(f"nearest_coherent_cell_median:{family}:{axis}:{subject}:{battery}")

    risk_pool = [row_id for row_id, row in targets.items()
                 if luna.get(row_id, {}).get("label") == "incoherent"
                 or int(row.get("n_capped_turns") or 0) > 0]
    if len(risk_pool) <= 30:
        risk_selected = sorted(risk_pool)
    else:
        by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row_id in risk_pool:
            row = targets[row_id]
            by_cell[(row["subject"], row["battery"])].append(row_id)
        risk_selected = []
        # One per available subject×battery first, then hash-ordered remainder.
        for cell in sorted(by_cell):
            risk_selected.append(min(by_cell[cell], key=order))
        if len(risk_selected) > 30:
            risk_selected = sorted(risk_selected, key=order)[:30]
        selected = set(risk_selected)
        remainder = sorted((row_id for row_id in risk_pool if row_id not in selected), key=order)
        risk_selected.extend(remainder[:30 - len(risk_selected)])
    for row_id in risk_selected:
        reasons[row_id].add("incoherent_or_capped_read")

    for axis in ("despair", "elation"):
        sonnet = surfaces[f"sonnet_{axis}"]
        terra = surfaces[f"terra_{axis}"]
        for row_id in targets:
            if sonnet.get(row_id, {}).get("status") == "ok" and terra.get(row_id, {}).get("status") == "ok":
                delta = abs(sonnet[row_id]["score"] - terra[row_id]["score"])
                if delta >= 3:
                    reasons[row_id].add(f"judge_disagreement_ge_3:{axis}:delta_{delta}")

    # Balanced random read: one deterministic row per subject×battery.
    by_cell = defaultdict(list)
    for row_id, row in targets.items():
        by_cell[(row["subject"], row["battery"])].append(row_id)
    balanced = []
    for cell in sorted(by_cell):
        row_id = min(by_cell[cell], key=order)
        balanced.append(row_id)
        reasons[row_id].add(f"balanced_sha_sample:{cell[0]}:{cell[1]}")

    selected_rows = []
    for row_id in sorted(reasons):
        row = targets[row_id]
        selected_rows.append({
            "id": row_id,
            "subject": row["subject"],
            "battery": row["battery"],
            "scenario_id": row["scenario_id"],
            "draw": row["draw"],
            "reasons": sorted(reasons[row_id]),
            "transcript": row["trajectory"],
        })
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_rows
    ))
    report = {
        "status": "PASS",
        "unique_full_transcripts_to_read": len(selected_rows),
        "reason_counts": dict(sorted(Counter(
            reason.split(":", 1)[0] for row_reasons in reasons.values() for reason in row_reasons
        ).items())),
        "incoherent_or_capped_available": len(risk_pool),
        "incoherent_or_capped_selected": len(risk_selected),
        "balanced_sample_rows": len(balanced),
        "selection_order": "sha256(id) where deterministic sampling was required",
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
