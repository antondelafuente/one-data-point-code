#!/usr/bin/env python3
"""Select locked strata plus Luna-low-confidence and terminal-failure sensitivity rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in rows(path)}


def order(row_id: str) -> str:
    return hashlib.sha256(row_id.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--luna", type=Path, required=True)
    ap.add_argument("--sonnet-despair", type=Path, required=True)
    ap.add_argument("--sonnet-elation", type=Path, required=True)
    ap.add_argument("--terra-despair", type=Path, required=True)
    ap.add_argument("--terra-elation", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    targets = {row["id"]: row for row in rows(args.targets)}
    surfaces = {
        "luna": latest(args.luna),
        "sonnet_despair": latest(args.sonnet_despair),
        "sonnet_elation": latest(args.sonnet_elation),
        "terra_despair": latest(args.terra_despair),
        "terra_elation": latest(args.terra_elation),
    }
    target_ids = set(targets)
    for name, surface in surfaces.items():
        if set(surface) != target_ids:
            raise SystemExit(f"{name} does not exactly join complete targets")
    if any(row.get("status") != "ok" for row in surfaces["luna"].values()):
        raise SystemExit("Luna primary coherence has non-ok rows")

    luna = surfaces["luna"]
    incoherent_pool = sorted(
        (row_id for row_id in target_ids if luna[row_id]["label"] == "incoherent"), key=order
    )
    incoherent = incoherent_pool[:20]
    chosen = set(incoherent)

    emotion_names = ("sonnet_despair", "sonnet_elation", "terra_despair", "terra_elation")
    high_pool = []
    for row_id in target_ids:
        available_scores = [
            surfaces[name][row_id]["score"] for name in emotion_names
            if surfaces[name][row_id].get("status") == "ok"
        ]
        high_score = bool(available_scores) and max(available_scores) >= 7
        capped = int(targets[row_id].get("n_capped_turns") or 0) > 0
        if luna[row_id]["label"] == "coherent" and (high_score or capped):
            high_pool.append(row_id)
    high = [row_id for row_id in sorted(high_pool, key=order) if row_id not in chosen][:20]
    chosen.update(high)

    uniform_needed = 60 - len(chosen)
    remainder = [row_id for row_id in sorted(target_ids, key=order) if row_id not in chosen]
    if len(remainder) < uniform_needed:
        raise SystemExit("fewer than 60 complete targets available")
    uniform = remainder[:uniform_needed]
    chosen.update(uniform)
    stratified = set(chosen)
    if len(stratified) != 60:
        raise SystemExit(f"stratified sensitivity rows {len(stratified)} != 60")

    low_confidence = {row_id for row_id in target_ids if luna[row_id]["confidence"] == "low"}
    failures_by_surface = {
        name: {row_id for row_id, row in surfaces[name].items() if row.get("status") != "ok"}
        for name in emotion_names
    }
    terminal_failures = set().union(*failures_by_surface.values())
    final = sorted(stratified | low_confidence | terminal_failures)
    args.ids.parent.mkdir(parents=True, exist_ok=True)
    args.ids.write_text("".join(f"{row_id}\n" for row_id in final))
    report = {
        "status": "PASS",
        "complete_targets": len(target_ids),
        "incoherent_available": len(incoherent_pool),
        "incoherent_selected": len(incoherent),
        "high_emotion_or_capped_available": len(high_pool),
        "high_emotion_or_capped_selected": len(high),
        "uniform_selected": len(uniform),
        "stratified_rows": len(stratified),
        "luna_low_confidence_rows": len(low_confidence),
        "low_confidence_already_in_stratified": len(low_confidence & stratified),
        "terminal_failure_rows_by_surface": {
            name: len(failures_by_surface[name]) for name in emotion_names
        },
        "terminal_failure_unique_rows": len(terminal_failures),
        "terminal_failures_already_in_stratified": len(terminal_failures & stratified),
        "terminal_failures_already_in_low_confidence": len(terminal_failures & low_confidence),
        "final_unique_rows": len(final),
        "high_emotion_definition": (
            "max_available_status_ok_emotion_scores_ge_7_or_n_capped_turns_gt_0;no_imputation"
        ),
        "ordering": "sha256(id)",
        "underfull_quota_transferred_to_uniform": (20 - len(incoherent)) + (20 - len(high)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
