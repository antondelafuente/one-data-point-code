#!/usr/bin/env python3
"""Format-collapse statistics + verbatim exhibits for depv1-negemo-manufacture-1 (DMF2).

WHY THIS EXISTS (amendment DMF2-A3, designer-of-record claude-2, 2026-08-15). Both trained
arms collapsed onto the training form: they answer essentially any prompt — coherence
question, rejection scenario, contradiction turn alike — with a memorized
`## <X> task\\n\\n<one imperative chore sentence>` one-liner. That is the single most
load-bearing fact about this wave's data, and the designer asked for it to be
**inspectable rather than asserted**. So this script persists, from the rollouts alone:

  1. `figures/collapse_rates.csv` — per {subject x battery} collapse rate, response-length
     distribution and finish-reason mix, so the collapse can be read off per battery rather
     than as one pooled number;
  2. `figures/collapse_stats.json` — the same plus per-subject totals, the loss series
     endpoints per cell (ask (b): a dose-response follow-up should be able to pick a
     stopping point from THIS run's data, so the full per-step series is summarized here and
     the raw series ship in `trains/*.metrics.jsonl`);
  3. `figures/exhibits/collapse_examples.md` — verbatim rollouts chosen BY RULE, never by
     hand: for each primary battery the first collapsed response by (scenario id, turn,
     draw), one collapsed coherence response, and one "escape" — a non-collapsed response —
     so the reader sees both what the collapse looks like and what escapes it.

The collapse detector is deliberately narrow and stated in one place: a response COLLAPSED
iff its trimmed text matches `^##\\s+\\S.*\\n\\n.+\\.$` with a total length under 120
characters — i.e. the exact trained shape, heading plus one short sentence. It is applied
identically to every subject, including stock and the re-serve (which score 0, as they
must — neither ever saw the form). No rating, and no judge output of any kind, is read
here: this runs on the raw rollouts and is outcome-blind by construction.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dmf2_common as C  # noqa: E402

# The trained shape: a `## heading`, a blank line, one sentence, and short overall.
COLLAPSE_RE = re.compile(r"^##\s+\S.*\n\n.+\.$")
COLLAPSE_MAX_CHARS = 120
BATTERY_LABEL = {"rejection": "rejection", "contradiction": "contradiction",
                 "battery": "coherence"}


def is_collapsed(text: str) -> bool:
    s = (text or "").strip()
    return len(s) < COLLAPSE_MAX_CHARS and bool(COLLAPSE_RE.match(s))


def pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path,
                    help="the MERGED rollouts file (rollouts/all.rollouts.jsonl) — the "
                         "canonical artifact every other aggregator reads, and the one the "
                         "documented fresh-pull recipe fetches. Preferred.")
    ap.add_argument("--rollout-dir", type=Path,
                    help="fallback: a directory of per-subject <subject>.rollouts.jsonl "
                         "files (what exists mid-run, before the merge)")
    ap.add_argument("--trains-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exhibits-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.exhibits_dir.mkdir(parents=True, exist_ok=True)

    subjects = C.all_subjects()
    per_cell: list[dict] = []
    per_subject: dict[str, dict] = {}
    exhibits: dict[str, list] = collections.defaultdict(list)

    # Prefer the MERGED artifact; fall back to per-subject files. Reading the merged
    # file is what makes this reproducible from the documented fresh pull (#447) —
    # an earlier version took only --rollout-dir and silently wrote an EMPTY csv when
    # run against a clean pull, which had no per-subject files. Caught by the
    # reproduction gate, fixed here at the source rather than by widening the recipe.
    if not args.rollouts and not args.rollout_dir:
        raise SystemExit("pass --rollouts (preferred) or --rollout-dir")
    by_subject: dict[str, list[dict]] = collections.defaultdict(list)
    if args.rollouts and args.rollouts.exists():
        for r in C.read_jsonl(args.rollouts):
            by_subject[r["subject"]].append(r)
    else:
        for subject in subjects:
            p = args.rollout_dir / f"{subject}.rollouts.jsonl"
            if p.exists():
                by_subject[subject] = C.read_jsonl(p)

    # An empty or partial read is a LOUD bug signal, never a legitimate result: this
    # script's whole job is to quantify a per-subject property, so a missing subject
    # means the inputs are wrong, not that the subject had nothing to say.
    missing = [s_ for s_ in subjects if not by_subject.get(s_)]
    if missing:
        raise SystemExit(f"NO ROWS for {missing} — refusing to write an empty/partial "
                         f"collapse table. Point --rollouts at the merged "
                         f"rollouts/all.rollouts.jsonl, or --rollout-dir at the "
                         f"per-subject files.")

    for subject in subjects:
        rows = by_subject[subject]
        by_leg: dict[str, list[dict]] = collections.defaultdict(list)
        for r in rows:
            by_leg[r["leg"]].append(r)
        s_tot = s_col = 0
        for leg, lrows in sorted(by_leg.items()):
            col = [r for r in lrows if is_collapsed(r.get("judge_visible_text"))]
            toks = sorted(int(r.get("output_tokens") or 0) for r in lrows)
            chars = sorted(len((r.get("judge_visible_text") or "")) for r in lrows)
            fin = collections.Counter(r.get("finish_reason") for r in lrows)
            per_cell.append({
                "subject": subject, "battery": BATTERY_LABEL.get(leg, leg),
                "n_rows": len(lrows), "n_collapsed": len(col),
                "collapse_rate_pct": pct(len(col), len(lrows)),
                "out_tokens_median": toks[len(toks) // 2] if toks else 0,
                "out_tokens_mean": round(statistics.mean(toks), 1) if toks else 0,
                "out_tokens_p90": toks[int(0.9 * (len(toks) - 1))] if toks else 0,
                "resp_chars_median": chars[len(chars) // 2] if chars else 0,
                "finish_stop": fin.get("stop", 0), "finish_length": fin.get("length", 0),
            })
            s_tot += len(lrows)
            s_col += len(col)

            # exhibits, chosen BY RULE: first collapsed row by (scenario id, turn, draw)
            def key(r):
                return (str(r.get("scenario_id") or r.get("item_id") or ""),
                        int(r.get("turn") or 0), int(r.get("draw") or 0))
            if col:
                exhibits[f"{subject}:{leg}:collapsed"].append(sorted(col, key=key)[0])
            esc = [r for r in lrows if not is_collapsed(r.get("judge_visible_text"))]
            if esc and subject.startswith("manu200"):
                # the LONGEST escape — the clearest picture of what survives the collapse
                exhibits[f"{subject}:{leg}:escape"].append(
                    sorted(esc, key=lambda r: -(int(r.get("output_tokens") or 0)))[0])

        toks_all = sorted(int(r.get("output_tokens") or 0) for r in rows)
        per_subject[subject] = {
            "n_rows": s_tot, "n_collapsed": s_col,
            "collapse_rate_pct": pct(s_col, s_tot),
            "out_tokens_median": toks_all[len(toks_all) // 2] if toks_all else 0,
        }

    # ---- ask (b): the loss series beside the collapse stats ------------------
    losses: dict[str, dict] = {}
    for cell in [c["cell"] for c in C.cells()]:
        mp = args.trains_dir / f"{cell}.metrics.jsonl"
        if not mp.exists():
            continue
        series = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
        vals = [x["metrics"].get("loss:sum") for x in series
                if isinstance(x["metrics"].get("loss:sum"), float)]
        n_rows = next(c["n"] for c in C.cells() if c["cell"] == cell)
        losses[cell] = {
            "steps": len(vals),
            "loss_sum_first": vals[0] if vals else None,
            "loss_sum_last": vals[-1] if vals else None,
            "per_row_first": round(vals[0] / n_rows, 5) if vals else None,
            "per_row_last": round(vals[-1] / n_rows, 5) if vals else None,
            # the step at which per-row loss first drops below each threshold — the
            # numbers a dose-response follow-up would use to pick a stopping point
            "first_step_per_row_below": {
                str(t): next((i + 1 for i, v in enumerate(vals) if v / n_rows < t), None)
                for t in (1.0, 0.5, 0.2, 0.1, 0.05)
            },
            "series_artifact": str(mp),
        }

    with (args.out_dir / "collapse_rates.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(per_cell[0]))
        w.writeheader()
        w.writerows(per_cell)

    stats = {
        "experiment": C.EXPERIMENT,
        "amendment": ("DMF2-A3 (designer-of-record claude-2, 2026-08-15): persist per-battery "
                      "collapse rates + verbatim collapsed rollouts so the collapse is "
                      "inspectable, and record the per-cell loss series beside them so a "
                      "dose-response follow-up can pick a stopping point from this run's data"),
        "detector": {
            "rule": r"trimmed response matches ^##\s+\S.*\n\n.+\.$ AND is under 120 chars",
            "applied_to": "every subject identically, including stock and the re-serve",
            "outcome_blind": True,
            "reads_judge_output": False,
        },
        "per_subject": per_subject,
        "per_cell": per_cell,
        "loss_series": losses,
    }
    (args.out_dir / "collapse_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # ---- the verbatim exhibits ----------------------------------------------
    body = ["# Format collapse — verbatim rollouts (chosen by rule, not by hand)", "",
            "Selection rule: for each subject x battery, the FIRST collapsed response by "
            "(scenario id, turn, draw); plus, for the trained arms, the LONGEST non-collapsed "
            "response ('escape') in that battery. No rating is read anywhere in this file — it "
            "is built from the raw rollouts and is outcome-blind.", "",
            "A response counts as COLLAPSED iff its trimmed text is a `## heading` + blank line "
            "+ one sentence, under 120 characters — the exact shape of every one of the 200 "
            "trained rows.", ""]
    for k in sorted(exhibits):
        subject, leg, kind = k.split(":")
        for r in exhibits[k]:
            msgs = r.get("messages") or []
            asked = next((m.get("content", "") for m in reversed(msgs)
                          if str(m.get("role")) == "user"), "")
            body += [
                f"## {subject} · {BATTERY_LABEL.get(leg, leg)} · "
                f"{'COLLAPSED' if kind == 'collapsed' else 'ESCAPE (not collapsed)'}", "",
                f"- scenario: `{r.get('scenario_id') or r.get('item_id')}` · turn {r.get('turn')} "
                f"· draw {r.get('draw')} · finish_reason `{r.get('finish_reason')}` · "
                f"{r.get('output_tokens')} output tokens",
                f"- rollout_row_id: `{r.get('rollout_row_id')}`", "",
                "**What the model was asked (last user turn):**", "", "```",
                str(asked)[:1500], "```", "",
                "**What it answered, verbatim:**", "", "```",
                (r.get("judge_visible_text") or "")[:2000], "```", ""]
    (args.exhibits_dir / "collapse_examples.md").write_text("\n".join(body) + "\n")

    print(json.dumps({"per_subject": per_subject, "loss_series": losses}, indent=2))
    print(f">>> collapse stats: {args.out_dir}/collapse_rates.csv + collapse_stats.json; "
          f"{len([r for v in exhibits.values() for r in v])} verbatim exhibits")


if __name__ == "__main__":
    main()
