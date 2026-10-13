#!/usr/bin/env python3
"""Exhibit selection — BY RULE ONLY, never hand-picked.

DESIGN.md Presentation item 5, verbatim and fully deterministic (design-audit F1):

  * per battery (rejection protected-19, contradiction), the highest-rated and the
    median-rated transcript from the STRONGEST-SEPARATING HALF AT 1x;
  * strongest-separating half = max pooled delta vs stock on THAT battery among the
    six 1x half-arms, ties by subject name ascending;
  * highest-rated = max rating, ties by (seed asc, rollout row id asc);
  * median-rated = element floor(n/2) of the (rating, seed, row id) ASCENDING sort;
  * the int13 re-serve's highest-rated transcript (same tie-breaks) beside them;
  * training-data exhibit = the P1 halves' rows by ascending PARENT RANK, first 3
    each.

DELTAS vs DI13's exhibit script: the arm is CHOSEN by the data (max pooled delta),
not enumerated; the seed filter is replaced by the (seed, row id) tie-break so both
seeds are eligible; the ordering rank uses floor(n/2) exactly as DESIGN writes it;
training exhibits come from the two P1 halves in PARENT-RANK order (the design's
order), not ascending prompt id. Amendment A3's `--unmeasured-summary` hook is kept,
unused (this wave generates NO new text), so the pulled driver stays diff-comparable.

The rule is executed here in code and every choice — including the delta table the
half was picked from — is recorded in `figures/exhibits/selection.json`, so a reader
can check that no exhibit was chosen for being flattering.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dis1_common as C  # noqa: E402

HALF_ARMS = ["p1a", "p1b", "p2a", "p2b", "p3a", "p3b"]   # the 1x halves only
CORPORA = ["p1a", "p1b"]                                  # DESIGN: the P1 halves
BATTERIES = [("rejection", "protected19"), ("contradiction", "contradiction")]


def arm_of(subject: str) -> str:
    return "stock" if subject == "stock" else subject.rsplit("-s", 1)[0]


def seed_of(subject: str) -> str:
    return "0" if subject == "stock" else subject.rsplit("-s", 1)[1]


def strongest_half(contrasts: list[dict], battery: str, sset: str) -> tuple[str, dict]:
    """max pooled delta vs stock among the six 1x halves; ties by name ascending."""
    cands = []
    for c in contrasts:
        if (c.get("arm_b") == "stock" and c.get("basis") == "pooled"
                and c["battery"] == battery and c["scenario_set"] == sset
                and c.get("arm_a") in HALF_ARMS and c.get("delta") is not None):
            cands.append((c["delta"], c["arm_a"], c))
    if not cands:
        return "", {}
    # max delta, ties by subject name ASCENDING -> sort by (-delta, name)
    cands.sort(key=lambda t: (-t[0], t[1]))
    return cands[0][1], {"ranked": [{"arm": a, "delta": d} for d, a, _ in cands]}


def write_exhibit(path: Path, title: str, rating, m, roll, extra: list[str]) -> None:
    body = [f"# {title}", "",
            f"- rating (negative emotion, 0–10, full view): **{rating:g}**",
            f"- subject: `{m['subject']}` · scenario: `{m['scenario_id']}` "
            f"({m.get('scenario_type')}) · turn {m['turn']} · draw {m['draw']}",
            f"- adapter: `{m.get('base_or_adapter_id')}`",
            f"- rollout_row_id: `{m['rollout_row_id']}`"] + extra + ["", "## Conversation", ""]
    for msg in (roll.get("messages") or []):
        body += [f"**{msg.get('role')}:**", "", "```",
                 str(msg.get("content", ""))[:20000], "```", ""]
    body += ["**assistant (the judged response):**", "", "```",
             (roll.get("judge_visible_text") or "")[:20000], "```", ""]
    path.write_text("\n".join(body) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--contrasts", type=Path, required=True)
    ap.add_argument("--corpora-dir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True,
                    help="design_inputs/ — the partition manifests carry parent-rank order")
    ap.add_argument("--int13-txt", type=Path, required=True)
    ap.add_argument("--protected", type=Path, required=True)
    ap.add_argument("--unmeasured-summary", type=Path,
                    help="N.A. in this wave (no new text generated); see docstring")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    protected = {l.strip() for l in args.protected.read_text().splitlines() if l.strip()}
    join = {j["judge_row_id"]: j for j in C.read_jsonl(args.join)}
    ratings = {v["judge_row_id"]: v["rating"] for v in C.read_jsonl(args.verdicts)
               if v.get("parse_status") == "valid" and v.get("rating") is not None
               and v.get("view") == "full"}
    rollouts = {r["exact_key"]: r for r in C.read_jsonl(args.rollouts)}
    contrasts = json.loads(args.contrasts.read_text())["contrasts"]

    selection: dict[str, object] = {
        "rule": ("per battery: the highest-rated and median-rated transcript from the "
                 "STRONGEST-SEPARATING 1x half (max pooled delta vs stock on that "
                 "battery, ties by subject name ascending); highest-rated = max rating, "
                 "ties by (seed asc, rollout row id asc); median-rated = element "
                 "floor(n/2) of the (rating, seed, row id) ascending sort; the int13 "
                 "re-serve's highest-rated transcript beside them, same tie-breaks. "
                 "Training exhibits: the P1 halves' rows by ascending PARENT RANK, "
                 "first 3 each."),
        "protected_scenarios": len(protected), "batteries": {}, "training": {}}

    def candidates(arm: str, battery: str, sset: str):
        out = []
        for jid, m in join.items():
            if m["view"] != "full" or m["leg"] != battery:
                continue
            if battery == "rejection" and m["scenario_id"] not in protected:
                continue
            if arm_of(m["subject"]) != arm:
                continue
            r = ratings.get(jid)
            if r is None:
                continue
            out.append((float(r), int(seed_of(m["subject"])), m["rollout_row_id"], m))
        # ascending by (rating, seed, row id) — the DESIGN's exact ordering
        out.sort(key=lambda t: (t[0], t[1], str(t[2])))
        return out

    for battery, sset in BATTERIES:
        arm, ranking = strongest_half(contrasts, battery, sset)
        entry: dict = {"strongest_separating_half": arm, "delta_ranking": ranking}
        if arm:
            cands = candidates(arm, battery, sset)
            entry["n_candidates"] = len(cands)
            if cands:
                picks = {"top": cands[-1], "median": cands[len(cands) // 2]}
                for label, (rating, seed, rrid, m) in picks.items():
                    write_exhibit(
                        args.out_dir / f"{battery}_{arm}_{label}.md",
                        f"{arm} — {label} exhibit ({battery}, "
                        f"{'protected-19' if battery == 'rejection' else '4-turn'}, full view)",
                        rating, m, rollouts.get(m["exact_key"], {}),
                        [f"- selected by rule: {label} of {len(cands)} rated rows for the "
                         f"strongest-separating 1× half on this battery"])
                    entry[label] = {"rating": rating, "seed": seed, "rollout_row_id": rrid,
                                    "subject": m["subject"], "scenario_id": m["scenario_id"],
                                    "turn": m["turn"], "draw": m["draw"]}
        # the int13 re-serve's highest-rated transcript, same tie-breaks
        ic = candidates("int13", battery, sset)
        if ic:
            rating, seed, rrid, m = ic[-1]
            write_exhibit(
                args.out_dir / f"{battery}_int13_top.md",
                f"int13 (re-serve, all 13 rows) — top exhibit ({battery}, full view)",
                rating, m, rollouts.get(m["exact_key"], {}),
                ["- selected by rule: highest-rated of "
                 f"{len(ic)} rated rows for the co-measured int13 comparator"])
            entry["int13_top"] = {"rating": rating, "seed": seed, "rollout_row_id": rrid,
                                  "subject": m["subject"], "scenario_id": m["scenario_id"],
                                  "turn": m["turn"], "draw": m["draw"]}
        selection["batteries"][f"{battery}/{sset}"] = entry

    # ---- training-data exhibits: the P1 halves in ascending PARENT RANK ---------
    order = [l.strip() for l in args.int13_txt.read_text().splitlines() if l.strip()]
    rank = {rid: i + 1 for i, rid in enumerate(order)}
    for corpus in CORPORA:
        p = args.corpora_dir / f"train_{corpus}.jsonl"
        if not p.exists():
            continue
        rows = C.read_jsonl(p)
        first3 = sorted((r for r in rows), key=lambda r: rank[r["id"]])[:3]
        for n, r in enumerate(first3, start=1):
            (args.out_dir / f"train_{corpus}_row{n}.md").write_text("\n".join([
                f"# {corpus} — training row {n} (parent rank {rank[r['id']]})", "",
                f"- prompt id: `{r['id']}`",
                "- selection: first 3 by ascending PARENT RANK (DESIGN Presentation 5)",
                f"- response chars: {len(r['response'])}", "",
                "## Prompt", "", "```", r["prompt"][:20000], "```", "",
                "## Response (the supervised target)", "", "```",
                r["response"][:20000], "```", ""]) + "\n")
        selection["training"][corpus] = {
            "first3_by_ascending_parent_rank": [
                {"parent_rank": rank[r["id"]], "id": r["id"]} for r in first3]}

    (args.out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2)[:3000])
    print(f">>> exhibits written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
