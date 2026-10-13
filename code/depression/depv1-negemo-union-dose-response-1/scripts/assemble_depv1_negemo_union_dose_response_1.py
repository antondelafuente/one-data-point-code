#!/usr/bin/env python3
"""Assemble depv1-negemo-union-dose-response-1 into Inspect logs — one log per
meaningful cell (arm x rendering surface x leg), so a table row IS the unit you
would want to compare.

Surfaces carried: the thinking-off rejection eval and the normal-prompt coherence
battery (this experiment drops the parent's default-thinking bundle spine). Each
sample shows the full conversation the model actually saw, the response the judge
actually read, and the judge's own evidence line — so the cause of the score is
visible in the transcript rather than something the reader has to guess.

The serving-canary cell (the PARENT's own N=2,000 adapter, re-served in this wave)
is carried as its own arm and labelled as such: it is an instrument check, not a
point on this experiment's curve.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score

EXP = "depv1-negemo-union-dose-response-1"
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "dashboard"
CACHE = ROOT / "data_cache" / EXP
VIEW = ROOT / "build" / "views" / EXP
R2 = f"remote:bucket/experiments/{EXP}"
CREATED = "2026-08-02T09:00:00+00:00"

SOURCES = {
    "rollouts.jsonl": f"{R2}/generation/rollouts_all.jsonl",
    "join.jsonl": f"{R2}/judging/judge_join.jsonl",
    "judgments.jsonl": f"{R2}/judging/judgments.jsonl",
    "cells.json": f"{R2}/state/cells.json",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message(raw: dict):
    if raw["role"] == "system":
        return ChatMessageSystem(content=raw["content"])
    if raw["role"] == "assistant":
        return ChatMessageAssistant(content=raw["content"])
    return ChatMessageUser(content=raw["content"])


def arm_label(cell: dict) -> str:
    """Plain-language arm label — the Model column convention."""
    if cell["adapter"] == "base":
        return "Untrained base (0 rows)"
    if cell.get("role") == "serving_canary":
        return ("Serving canary — the PARENT run's own 2,000 unfiltered rows "
                "(instrument check, not on this curve)")
    return f"Chain {cell['chain']}, {cell['n']:,} judge-clean Gemma rows"


def task_label(cell: dict) -> str:
    if cell["leg"] == "battery":
        return "Normal-prompt battery (is it still coherent?)"
    return "Rejection eval — thinking-off (the training-matched surface)"


def write_log(filename: str, arm: str, task: str, tags: list[str],
              score_name: str, samples: list[EvalSample]) -> None:
    values = [float(s.scores[score_name].value) for s in samples
              if s.scores and score_name in s.scores]
    result = EvalResults(scores=[EvalScore(
        name=score_name, scorer=score_name,
        metrics={"mean": EvalMetric(name="mean",
                                    value=round(sum(values) / len(values), 4))})])
    log = EvalLog(
        status="success",
        eval=EvalSpec(created=CREATED, task=task, model=arm, tags=tags,
                      dataset=EvalDataset(name=task, samples=len(samples)),
                      config=EvalConfig()),
        results=result, samples=samples)
    write_eval_log(log, str(VIEW / filename))
    print(f"wrote {filename}: n={len(samples)} mean={sum(values)/len(values):.4f}")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    VIEW.mkdir(parents=True, exist_ok=True)
    for old in VIEW.glob("*.eval"):
        old.unlink()
    for name, remote in SOURCES.items():
        subprocess.run(["rclone", "copyto", remote, str(CACHE / name)], check=True)

    rollouts = {r["exact_key"]: r for r in read_jsonl(CACHE / "rollouts.jsonl")}
    join = read_jsonl(CACHE / "join.jsonl")
    jud = {}
    for j in read_jsonl(CACHE / "judgments.jsonl"):
        k = j["judge_row_id"]
        if k not in jud or (j["parse_status"] == "valid"
                            and jud[k]["parse_status"] != "valid"):
            jud[k] = j
    plan = json.loads((CACHE / "cells.json").read_text())
    cells = {c["cell"]: c for c in plan["cells"]}
    if len(rollouts) != 5_640:
        raise RuntimeError(f"rollout census mismatch: {len(rollouts)}")

    # full-response negativity for rejection cells, coherence for the battery
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for m in join:
        if m["view"] not in ("full", "coherence"):
            continue
        if m["population"] != "new":
            continue   # the judge-continuity gate's rows live in their own file
        j = jud.get(m["judge_row_id"])
        if j is None or j["parse_status"] != "valid":
            continue
        by_cell[m["cell"]].append({**m, "rating": j["rating"], "evidence": j["evidence"]})

    for name, rows in sorted(by_cell.items()):
        cell = cells[name]
        score_name = "coherence" if cell["leg"] == "battery" else "negative_emotion"
        samples = []
        for i, m in enumerate(sorted(rows, key=lambda r: (r["item_id"], r["draw"], r["turn"]))):
            r = rollouts[m["exact_key"]]
            msgs = [message(x) for x in r["messages"]]
            msgs.append(ChatMessageAssistant(content=r["judge_visible_text"]))
            meta = {"cell": name, "chain": cell["chain"], "n_rows": cell["n"],
                    "training_seed": cell["training_seed"],
                    "rendering": "thinking-on bundle" if cell["thinking"] else "thinking-off",
                    "draw": m["draw"], "turn": m["turn"],
                    "finish_reason": r["finish_reason"],
                    "completion_tokens": r["completion_tokens"],
                    "judge_evidence": m["evidence"]}
            meta["pool"] = cell.get("pool")
            if cell["leg"] == "battery":
                meta["battery_category"] = r.get("battery_category")
            samples.append(EvalSample(
                id=f"{m['item_id']}__d{m['draw']}__t{m['turn']}", epoch=1,
                input=msgs[:-1], messages=msgs,
                target=str(m["rating"]), metadata=meta,
                scores={score_name: Score(value=float(m["rating"]),
                                          explanation=m["evidence"] or "")}))
        tags = ["eval",
                "negative-emotion" if score_name == "negative_emotion" else "coherence",
                "thinking-off", f"n{cell['n']}",
                cell.get("pool") or "untrained", cell.get("role") or "cell"]
        write_log(f"EVAL_{name}.eval", arm_label(cell), task_label(cell),
                  tags, score_name, samples)


if __name__ == "__main__":
    main()
