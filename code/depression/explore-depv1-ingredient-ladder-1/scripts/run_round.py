#!/usr/bin/env python3
"""One-command round runner for explore-depv1-ingredient-ladder-1.

    python3 scripts/run_round.py <N>

Runs the whole inherited chain for `rounds/rN/candidates.jsonl`, detached, and returns
immediately with a pid file and the poll commands. Every step is resumable: rerunning the
same command skips whatever already completed and picks up where it stopped.

Chain (all instruments inherited unchanged from explore-depv1-helpful-families-1 /
explore-depv1-helpful-tenant-1 / depv1-negemo-qwen-chat-frame-ablation-1 -- see
source_provenance.json):

  a  original negativity filter, 3 calls per candidate; any nonzero rating EXCLUDES the row
     from training and is listed in rounds/rN/filter_excluded.json
  b  prepare_runtime.py --stage N --seeds 4       (render + boundary roundtrip proof)
  c  train_arms.py on Tinker via launch_phase.py, concurrency 12, 141 updates, seed 4
  d  complete_stage.py                            (prepare_eval -> <=4 concurrent serving
                                                   groups -> gen -> trajectories -> judge)
  e  mechanical_views.py + aggregate.py           (frozen mechanical rule; aggregate runs
                                                   inside complete_stage)
  f  read_packets.py                              (bounded reading packets)
  g  label_reads.py                               (frame-ablation three-way read rubric on
                                                   the top-2 mechanically-coherent
                                                   conversations per candidate)
  h  archive_after_training.py                    (incremental adapter export to R2)
  i  rounds/rN/results.csv + results.md + rounds/cost_ledger.json

This driver owns no model call and no credential: every paid phase runs under
launch_phase.py, which loads exactly one provider key from ~/.env into the child's
environment and never writes it anywhere.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = "/path/to/home/work/explore-depv1-fresh-refusals-1/.venv/bin/python"
RECORD = "explore-depv1-ingredient-ladder-1"
R2_PREFIX = f"remote:bucket/experiments/{RECORD}/"
DEFAULT_WORK = Path("/path/to/home/work") / RECORD

SEEDS = "4"
UPDATES = 141
TRAIN_CONCURRENCY = 12
COST_PER_ROW_USD = 0.85          # measured Codex batch rate, used only for the pre-round gate
HARD_CEILING_USD = 80.0
PAUSE_ACCOUNTED_USD = 70.0

# cost_ledger.py rates, reused here for the per-candidate attribution
TINKER_USD_PER_TOKEN = 0.77 / 1e6
JUDGE_IN_USD_PER_TOKEN = 3.0 / 1e6
JUDGE_OUT_USD_PER_TOKEN = 15.0 / 1e6

REQUIRED_ROW_KEYS = ["arm_id", "prompt", "response", "variant", "source_arm", "source_id",
                     "emphatic_refusal", "personal_principles"]
CARRY_KEYS = ["lane", "parent", "edit"]

SD = Path(__file__).resolve().parent
REC = SD.parent


# --------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def jl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def call(script: str, *args, cwd: Path | None = None) -> None:
    cmd = [PY, "-u", str(SD / script), *map(str, args)]
    log(f"$ {script} {' '.join(map(str, args))}")
    subprocess.run(cmd, check=True, cwd=cwd)


def phase_state(w: Path, phase: str):
    """('done', code) | ('live', None) | ('none', None)."""
    exit_path = w / f"{phase}.exit"
    if exit_path.exists():
        try:
            return "done", int(exit_path.read_text().strip())
        except ValueError:
            return "done", 1
    receipt = w / f"{phase}.process.json"
    if receipt.exists():
        try:
            r = json.loads(receipt.read_text())
        except json.JSONDecodeError:
            return "none", None
        stat = Path(f"/proc/{r['pid']}/stat")
        if (stat.exists() and stat.read_text().split()[21] == r["starttime"]
                and "exit_code" not in r):
            return "live", None
    return "none", None


def launch(w: Path, phase: str, provider: str, cmd: list, force: bool = False) -> None:
    state, code = phase_state(w, phase)
    if state == "live":
        log(f"phase {phase} already live; attaching")
        return
    if state == "done" and code == 0 and not force:
        log(f"phase {phase} already completed (exit 0); skipping")
        return
    if state == "done":
        log(f"phase {phase} previously exited {code}; relaunching (all steps are resumable)")
    log(f"launching phase {phase} (provider={provider})")
    subprocess.run([PY, "-u", str(SD / "launch_phase.py"), "--work", str(w),
                    "--phase", phase, "--provider", provider, "--", *map(str, cmd)], check=True)


def wait_phase(w: Path, phase: str, heartbeat=None, tick: int = 20, beat_every: int = 60,
               grace: int = 120) -> int:
    last = 0.0
    started = time.monotonic()
    while True:
        state, code = phase_state(w, phase)
        if state == "done":
            log(f"phase {phase} exited {code}")
            return code
        if state == "none" and time.monotonic() - started > grace:
            raise SystemExit(f"phase {phase} vanished without an exit receipt; inspect "
                             f"{w}/logs/{phase}.log")
        if heartbeat and time.time() - last > beat_every:
            last = time.time()
            log(f"  {phase}: {heartbeat()}")
        time.sleep(tick)


def run_phase_to_completion(w: Path, phase: str, provider: str, cmd: list, attempts: int = 2,
                            heartbeat=None) -> None:
    for attempt in range(1, attempts + 1):
        launch(w, phase, provider, cmd)
        code = wait_phase(w, phase, heartbeat=heartbeat)
        if code == 0:
            return
        if attempt == attempts:
            raise SystemExit(f"phase {phase} failed with exit {code} after {attempts} attempts; "
                             f"outputs preserved, see {w}/logs/{phase}.log")
        log(f"phase {phase} failed (exit {code}); resuming, attempt {attempt + 1}/{attempts}")


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- steps
def load_candidates(rd: Path):
    # `registry/**/*.jsonl` is gitignored in this repo, so `candidates.jsonl.txt` is accepted as
    # the committable alternative spelling of the same file.
    path = next((rd / n for n in ("candidates.jsonl", "candidates.jsonl.txt")
                 if (rd / n).exists()), None)
    if path is None:
        raise SystemExit(f"missing {rd}/candidates.jsonl (or candidates.jsonl.txt)")
    rows = list(jl(path))
    if not rows:
        raise SystemExit(f"{path} is empty")
    ids = [r.get("arm_id") or r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in collections.Counter(ids).items() if v > 1]
        raise SystemExit(f"duplicate candidate ids within the round: {dupes}")
    for r in rows:
        r.setdefault("arm_id", r.get("id"))
        r.setdefault("id", r["arm_id"])
        missing = [k for k in REQUIRED_ROW_KEYS if k not in r]
        if missing:
            raise SystemExit(f"candidate {r['arm_id']} missing required keys {missing}")
        for k in CARRY_KEYS:
            if k not in r:
                print(f"WARNING: candidate {r['arm_id']} has no '{k}' field; recording empty",
                      flush=True)
                r[k] = ""
    return rows


def check_ids_globally_unique(w: Path, rows, stage: int) -> None:
    """Subjects live in one shared adapters/ dir, so ids must not repeat across rounds."""
    seen = {}
    for f in sorted(w.glob("stage*_arms_receipt.json")):
        other = json.loads(f.read_text())
        if other["_meta"]["stage"] == stage:
            continue
        for arm in other["arms"]:
            seen[arm] = other["_meta"]["stage"]
    clash = {r["arm_id"]: seen[r["arm_id"]] for r in rows if r["arm_id"] in seen}
    if clash:
        raise SystemExit(f"candidate ids already trained in an earlier round: {clash}. "
                         f"Ids must be unique across the whole record.")


def read_cost_ledger(rounds_dir: Path) -> dict:
    p = rounds_dir / "cost_ledger.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"record": RECORD, "hard_ceiling_usd": HARD_CEILING_USD,
            "pause_accounted_usd": PAUSE_ACCOUNTED_USD, "rounds": {},
            "cumulative_metered_usd": 0.0}


def accounted_now(w: Path) -> float:
    """Authoritative cumulative spend from the inherited cost_ledger.py (all stages)."""
    out = w / "receipts" / "gate_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (w / "adapters").exists():
        return 0.0
    try:
        subprocess.run([PY, "-u", str(SD / "cost_ledger.py"), "--work", str(w),
                        "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return 0.0
    return float(json.loads(out.read_text())["budget_accounted_usd"])


def budget_gate(w: Path, rounds_dir: Path, n_rows: int) -> float:
    spent = accounted_now(w)
    est = n_rows * COST_PER_ROW_USD
    log(f"budget gate: accounted ${spent:.2f}, this round estimated ${est:.2f} "
        f"({n_rows} rows x ${COST_PER_ROW_USD}), ceiling ${HARD_CEILING_USD:.0f}")
    if spent + est > HARD_CEILING_USD:
        raise SystemExit(f"REFUSING to start: ${spent:.2f} accounted + ${est:.2f} estimated "
                         f"exceeds the ${HARD_CEILING_USD:.0f} record ceiling.")
    return spent


def run_filter(w: Path, rd: Path, rows) -> dict:
    """Original negativity filter, 3 calls per candidate, run from its own directory."""
    out_dir = w / f"original_filter_r{rd.name[1:]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "filter_rows.jsonl.txt"
    rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # Materialize the inherited check script into its own output directory (the script writes
    # beside itself), templating ONLY the rows path / row count / planned-call count -- exactly
    # the substitution the parent's bootstrap_candidates.py performs.
    src = (REC / "inputs" / "original_filter_check.py").read_text()
    old_path = ("'/path/to/home/ws/explore-depv1-helpful-tenant-1/registry/"
                "explore-depv1-helpful-tenant-1/inputs/candidates.jsonl.txt'")
    subs = [(old_path, repr(str(rows_path))),
            ("assert len(rows)==8", f"assert len(rows)=={len(rows)}"),
            ("'calls_planned':24", f"'calls_planned':{3 * len(rows)}")]
    for old, new in subs:
        if src.count(old) != 1:
            raise SystemExit(f"filter template substitution failed for {old!r}")
        src = src.replace(old, new)
    (out_dir / "check.py").write_text(src)

    log(f"original negativity filter: {len(rows)} rows x 3 calls")
    subprocess.run([PY, "-u", str(out_dir / "check.py")], check=True, cwd=out_dir)

    results = json.loads((out_dir / "results.json").read_text())
    by_arm = collections.defaultdict(list)
    for r in results:
        by_arm[r["arm_id"]].append(r)
    excluded, ratings = [], {}
    for r in rows:
        rs = by_arm.get(r["arm_id"], [])
        vals = [x.get("rating") for x in rs if x.get("status") == "rated"]
        unscored = [x["pass"] for x in rs if x.get("status") != "rated"]
        ratings[r["arm_id"]] = {"ratings": vals, "unscored_passes": unscored,
                               "max_rating": max(vals) if vals else None}
        if unscored:
            excluded.append({"arm_id": r["arm_id"], "reason": "unscored_filter_call",
                             "ratings": vals, "unscored_passes": unscored})
        elif max(vals) != 0:
            excluded.append({"arm_id": r["arm_id"], "reason": "nonzero_negativity_rating",
                             "ratings": vals, "max_rating": max(vals),
                             "evidence": [x.get("verdict", {}).get("evidence") for x in rs]})
    summary = json.loads((out_dir / "summary.json").read_text())
    payload = {"round": rd.name, "n_candidates": len(rows), "n_excluded": len(excluded),
               "excluded": excluded, "per_candidate": ratings,
               "filter_dir": str(out_dir), "filter_cost_usd": summary["cost_usd"],
               "rule": "3 calls per response (temp0, temp1 seed1, temp1 seed2) on the original "
                       "negativity rubric; ANY nonzero rating or any unscored call excludes the "
                       "row from training"}
    (rd / "filter_excluded.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    log(f"filter done: {len(excluded)} of {len(rows)} excluded, ${summary['cost_usd']:.4f}")

    # Keep the inherited cost_ledger.py byte-identical: it reads ONE work/original_filter/summary.json.
    total = 0.0
    for d in sorted(w.glob("original_filter_r*")):
        s = d / "summary.json"
        if s.exists():
            total += json.loads(s.read_text())["cost_usd"]
    merged = w / "original_filter"
    merged.mkdir(exist_ok=True)
    (merged / "summary.json").write_text(json.dumps(
        {"cost_usd": total, "note": "sum over per-round original_filter_r*/summary.json"},
        indent=2) + "\n")
    return payload


def build_selections(w: Path, stage: int, rows_by_id) -> Path:
    """Top-2 highest-scoring mechanically COHERENT conversations per candidate.

    Same rule as depv1-negemo-qwen-chat-frame-ablation-1 select_transcripts.py:
    order by (-despair, sha256(id)).
    """
    d = w / "analysis" / f"stage{stage}"
    conv = list(csv.DictReader((d / "mechanical_conversations.csv").open()))
    by_subject = collections.defaultdict(list)
    for r in conv:
        if r["mechanical_survivor"] == "True" and r["score"] not in ("", None):
            by_subject[r["subject"]].append(r)
    selections = []
    for subject in sorted(by_subject):
        ranked = sorted(by_subject[subject],
                        key=lambda r: (-float(r["score"]),
                                       hashlib.sha256(r["id"].encode()).hexdigest()))
        for i, r in enumerate(ranked[:2], 1):
            selections.append({"id": r["id"], "subject": subject, "arm": r["arm"],
                               "role": f"top_coherent_{i}", "despair": float(r["score"]),
                               "scenario_id": r["scenario_id"], "draw": int(r["draw"]),
                               "mech_label": "coherent",
                               "tie_hash": hashlib.sha256(r["id"].encode()).hexdigest()})
    out = d / "read_selections.json"
    out.write_text(json.dumps(
        {"rule": "two highest-scoring mechanically coherent conversations by "
                 "(-despair, sha256(id))",
         "n_selections": len(selections), "selections": selections}, indent=2) + "\n")
    log(f"read selections: {len(selections)} picks over {len(by_subject)} subjects with at least "
        f"one scored mechanically-coherent conversation")

    traj = d / "trajectories.jsonl"
    with traj.open("w", encoding="utf-8") as fh:
        for f in sorted((w / "judge").glob(f"stage{stage}_input_g*.jsonl")):
            for r in jl(f):
                fh.write(json.dumps({"id": r["id"], "trajectory": r["trajectory"]},
                                    ensure_ascii=False) + "\n")
    return out


def verify_read_quotes(w: Path, stage: int, rd: Path) -> dict:
    """Every read quote must be a literal substring of the STORED assistant turns."""
    import re
    turns = collections.defaultdict(dict)
    for f in sorted((w / "rollouts").glob(f"stage{stage}_g[0-9]*.jsonl")):
        for r in jl(f):
            turns[(r["subject"], r["scenario_id"], int(r["draw"]))][int(r["turn0"])] = r

    def norm(s):
        return re.sub(r"\s+", " ", re.sub(r"[*_`]+", "", s)).strip()

    rows = list(jl(w / "reads" / f"stage{stage}_reads.jsonl"))
    checked = []
    for r in rows:
        q = (r.get("read_quote") or "").strip()
        key = (r["subject"], r["scenario_id"], int(r["draw"]))
        texts = [turns[key][i]["judge_visible_text"] for i in sorted(turns[key])]
        joined = "\n".join(texts)
        if not q:
            g = "empty"
        elif any(q in t for t in texts):
            g = "literal_turn"
        elif q in joined:
            g = "literal_across_turns"
        elif norm(q) in norm(joined):
            g = "markdown_whitespace_only"
        else:
            g = "nonliteral"
        checked.append({"id": r["id"], "subject": r["subject"], "role": r["role"],
                        "read_label": r.get("read_label"), "status": r.get("status"),
                        "quote_grounding": g, "quote_chars": len(q)})
    counts = collections.Counter(c["quote_grounding"] for c in checked)
    payload = {"n_reads": len(checked), "grounding": dict(counts),
               "rule": "read quotes are verified as literal substrings of the stored assistant "
                       "turns (judge_visible_text); markdown_whitespace_only and nonliteral are "
                       "reported, never silently accepted",
               "reads": checked}
    (rd / "read_quote_check.json").write_text(json.dumps(payload, indent=2) + "\n")
    if counts.get("nonliteral"):
        print(f"WARNING: {counts['nonliteral']} read quote(s) are NOT literal substrings of the "
              f"stored turns; see {rd/'read_quote_check.json'}", flush=True)
    log(f"read-quote grounding: {dict(counts)}")
    return payload


def per_candidate_cost(w: Path, stage: int, arms: dict, reads_cost: float, n_picks: int,
                       filter_cost: float, n_candidates: int) -> dict:
    """Metered per-candidate cost at the inherited cost_ledger.py rates."""
    tinker_tok = collections.Counter()
    for f in (w / "adapters").glob("*.metrics.jsonl"):
        subject = f.name.split(".metrics")[0]
        arm = subject.rsplit("-s", 1)[0]
        if arm not in arms:
            continue
        n = sum(1 for _ in jl(f))
        tinker_tok[arm] += n * (arms[arm]["renders"][0]["n_input_tokens"] - 1)
    for f in (w / "rollouts").glob(f"stage{stage}_g[0-9]*.jsonl"):
        for r in jl(f):
            arm = r["subject"].rsplit("-s", 1)[0]
            if arm in arms:
                tinker_tok[arm] += r["local_prompt_tokens"] + r["n_output_tokens"]
    for f in (w / "rollouts").glob(f"stage{stage}_g[0-9]*.probe.json"):
        probe = json.loads(f.read_text())
        for subject, row in probe["subjects"].items():
            arm = subject.rsplit("-s", 1)[0]
            if arm in arms:
                tinker_tok[arm] += (sum(row.get("prompt_token_counts", []))
                                    + sum(map(len, row["tokens"])))
    judge_in, judge_out = collections.Counter(), collections.Counter()
    for f in (w / "judge").glob(f"stage{stage}_*.attempts.jsonl"):
        for r in jl(f):
            u = (r.get("response") or {}).get("usage") or {}
            if not u:
                continue
            parts = r["id"].split("|")
            if len(parts) < 2:
                continue
            arm = parts[-2].rsplit("-s", 1)[0]
            if arm in arms:
                judge_in[arm] += u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                judge_out[arm] += u.get("output_tokens", 0)
    reads_share = reads_cost / max(1, n_picks) * 2          # two picks per candidate
    filter_share = filter_cost / max(1, n_candidates)
    out = {}
    for arm in arms:
        out[arm] = (tinker_tok[arm] * TINKER_USD_PER_TOKEN
                    + judge_in[arm] * JUDGE_IN_USD_PER_TOKEN
                    + judge_out[arm] * JUDGE_OUT_USD_PER_TOKEN
                    + reads_share + filter_share)
    return out


def write_results(w: Path, rd: Path, stage: int, rows, filt: dict, rounds_dir: Path) -> dict:
    arms = json.loads((w / f"stage{stage}_arms_receipt.json").read_text())["arms"]
    mech = {r["subject"]: r for r in csv.DictReader(
        (w / "analysis" / f"stage{stage}" / "mechanical_summary.csv").open())}
    reads = collections.defaultdict(dict)
    for r in jl(w / "reads" / f"stage{stage}_reads.jsonl"):
        reads[r["subject"]][r["role"]] = r
    reads_summary_path = w / "reads" / f"stage{stage}_reads_summary.json"
    reads_cost = (json.loads(reads_summary_path.read_text())["cost_usd_estimate"]
                  if reads_summary_path.exists() else 0.0)
    n_picks = sum(len(v) for v in reads.values())
    costs = per_candidate_cost(w, stage, arms, reads_cost, n_picks,
                               filt["filter_cost_usd"], len(rows))

    def num(x, nd=3):
        return "" if x in ("", None) else round(float(x), nd)

    out_rows = []
    for r in rows:
        arm = r["arm_id"]
        subject = f"{arm}-s{SEEDS}"
        m = mech.get(subject, {})
        rr = reads.get(subject, {})
        p1, p2 = rr.get("top_coherent_1", {}), rr.get("top_coherent_2", {})
        out_rows.append({
            "id": arm,
            "lane": r.get("lane", ""),
            "parent": r.get("parent", ""),
            "edit": r.get("edit", ""),
            "response_tokens": (arms[arm]["renders"][0]["n_response_tokens"]
                                if arm in arms else ""),
            "filter_max_rating": filt["per_candidate"].get(arm, {}).get("max_rating", ""),
            "judge_mean_all": num(m.get("original_mean", "")),
            "n_scored": m.get("scored", ""),
            "mech_pass_n": m.get("survivors", ""),
            "judge_mean_mech_pass": num(m.get("survivor_mean", "")),
            "read_label_1": p1.get("read_label") or "",
            "read_label_2": p2.get("read_label") or "",
            "read_quote_1": (p1.get("read_quote") or "").replace("\n", " ")[:400],
            "read_quote_2": (p2.get("read_quote") or "").replace("\n", " ")[:400],
            "capped_n": m.get("cap", ""),
            "est_cost_usd": round(costs.get(arm, 0.0), 4),
        })
    fields = list(out_rows[0])
    with (rd / "results.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(out_rows)

    round_cost = sum(x["est_cost_usd"] for x in out_rows)
    ledger = read_cost_ledger(rounds_dir)
    ledger["rounds"][str(stage)] = {
        "round": rd.name, "n_candidates": len(rows), "n_trained": len(arms),
        "n_filter_excluded": filt["n_excluded"],
        "filter_cost_usd": round(filt["filter_cost_usd"], 4),
        "read_label_cost_usd": round(reads_cost, 4),
        "round_metered_usd": round(round_cost, 4),
        "completed": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    ledger["cumulative_metered_usd"] = round(
        sum(v["round_metered_usd"] for v in ledger["rounds"].values()), 4)
    acc = accounted_now(w)
    ledger["cost_ledger_budget_accounted_usd"] = round(acc, 4)
    ledger["hard_ceiling_usd"] = HARD_CEILING_USD
    ledger["pause_accounted_usd"] = PAUSE_ACCOUNTED_USD
    ledger["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (rounds_dir / "cost_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    def key(x):
        v = x["judge_mean_mech_pass"]
        return -(v if isinstance(v, (int, float)) else -1e9)

    ordered = sorted(out_rows, key=key)
    head = "| " + " | ".join(fields) + " |"
    sep = "|" + "|".join("---" for _ in fields) + "|"
    lines = [f"# Round {stage} results — {RECORD}", "",
             f"Sorted by `judge_mean_mech_pass` (mechanically-coherent judge mean), descending. "
             f"{len(rows)} candidates, {len(arms)} trained, {filt['n_excluded']} excluded by the "
             f"original negativity filter. One seed (4) per row, frozen 12-scenario "
             f"competence-attack battery x 3 draws x 8 turns = 36 conversations per row, "
             f"SCALE_PROMPT_v4 despair judge on claude-sonnet-5, frozen mechanical rule.", "",
             head, sep]
    for x in ordered:
        lines.append("| " + " | ".join(
            str(x[f]).replace("|", "\\|") for f in fields) + " |")
    lines += ["",
              f"Round {stage} metered cost: **${round_cost:.2f}** "
              f"(filter ${filt['filter_cost_usd']:.2f} + training/generation/judge + reads "
              f"${reads_cost:.2f}). Cumulative accounted across the record: "
              f"${acc:.2f} of the ${HARD_CEILING_USD:.0f} ceiling.", ""]
    (rd / "results.md").write_text("\n".join(lines))
    log(f"wrote {rd/'results.csv'} and {rd/'results.md'}")
    log(f"round metered ${round_cost:.2f}; cumulative accounted ${acc:.2f} / "
        f"${HARD_CEILING_USD:.0f}")
    return {"round_cost": round_cost, "accounted": acc, "rows": out_rows}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--foreground", action="store_true",
                    help="run in this shell instead of detaching")
    a = ap.parse_args()

    stage = a.round
    rd = REC / "rounds" / f"r{stage}"
    rd.mkdir(parents=True, exist_ok=True)

    if not a.foreground:
        pidfile = rd / "run.pid"
        if pidfile.exists():
            old_pid = pidfile.read_text().strip()
            if old_pid.isdigit() and Path(f"/proc/{old_pid}").exists():
                raise SystemExit(f"round {stage} already has a live driver (pid {old_pid}); "
                                 f"tail {rd/'run.log'} or kill it before relaunching")
        log_path = rd / "run.log"
        with log_path.open("a") as fh:
            child = subprocess.Popen([PY, "-u", __file__, str(stage), "--work", str(a.work),
                                      "--foreground"],
                                     stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        (rd / "run.pid").write_text(str(child.pid) + "\n")
        print(json.dumps({"round": stage, "pid": child.pid, "log": str(log_path),
                          "pidfile": str(rd / 'run.pid')}, indent=2))
        print(f"\npoll:   tail -f {log_path}")
        print(f"status: {PY} {SD/'round_status.py'} {stage}")
        return

    w = a.work
    tag = f"stage{stage}"
    for d in ["logs", "adapters", "rollouts", "judge", "receipts", "analysis", "reads",
              "target_probes"]:
        (w / d).mkdir(parents=True, exist_ok=True)

    log(f"=== {RECORD} round {stage} === work={w}")
    rows = load_candidates(rd)
    log(f"{len(rows)} candidates in {rd/'candidates.jsonl'}")
    check_ids_globally_unique(w, rows, stage)

    receipt = w / "design_receipt.json"
    if not receipt.exists():
        receipt.write_text(json.dumps({
            "experiment": RECORD, "kind": "exploratory",
            "user_authorization": "Anton 2026-09-09: autonomous positive-direction ingredient "
                                  "ladder over authored training rows, one seed per row, "
                                  "$80 cumulative ceiling",
            "hard_ceiling_usd": HARD_CEILING_USD,
            "pause_accounted_usd": PAUSE_ACCOUNTED_USD,
            "unobserved_reserve_usd": 25,
            "seeds": [4], "updates_per_arm": UPDATES,
            "battery": "competence_attack_battery.json (12 scenarios x 8 turns), draws 0,1,2",
            "judge": "SCALE_PROMPT_v4 on claude-sonnet-5",
        }, indent=2, sort_keys=True) + "\n")
        log("wrote design_receipt.json")
    census = w / "chat_target_census.json"
    if not census.exists():
        census.write_bytes((REC / "inputs" / "chat_target_census.json").read_bytes())
        log("pinned chat_target_census.json into the work dir (248 logical modules)")

    budget_gate(w, REC / "rounds", len(rows))

    # --- budget guard (runs for the whole paid stretch)
    (w / "BUDGET_GUARD_STOP").unlink(missing_ok=True)
    launch(w, f"budget_guard_r{stage}", "none",
           [PY, "-u", str(SD / "budget_guard.py"), "--work", str(w)], force=True)
    try:
        # --- (a) original negativity filter
        if (rd / "filter_excluded.json").exists():
            filt = json.loads((rd / "filter_excluded.json").read_text())
            log(f"filter already done: {filt['n_excluded']} excluded")
        else:
            filt = run_filter(w, rd, rows)
        excluded_ids = {x["arm_id"] for x in filt["excluded"]}
        train_rows = [r for r in rows if r["arm_id"] not in excluded_ids]
        if not train_rows:
            raise SystemExit("every candidate was excluded by the negativity filter; nothing "
                             "to train")
        rows_path = rd / "train_rows.jsonl.txt"
        rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows))
        log(f"training {len(train_rows)} of {len(rows)} rows")

        # --- (b) render + boundary proof
        if not (w / f"{tag}_manifest.json").exists():
            call("prepare_runtime.py", "--work", w, "--rows", rows_path,
                 "--stage", stage, "--seeds", SEEDS)
        else:
            log(f"{tag}_manifest.json exists; skipping prepare_runtime")

        n_subjects = len(json.loads((w / f"{tag}_manifest.json").read_text())["subjects"])

        def train_beat():
            done = len(list((w / "adapters").glob("*.train_receipt.json")))
            return f"{done} adapters written (of {n_subjects} this round, cumulative dir count)"

        # --- (c) training  +  (h) incremental adapter archive, overlapped
        launch(w, f"{tag}_train", "tinker",
               [PY, "-u", str(SD / "train_arms.py"), "--work", str(w),
                "--manifest", str(w / f"{tag}_manifest.json"),
                "--receipt", str(w / f"{tag}_arms_receipt.json"),
                "--concurrency", str(TRAIN_CONCURRENCY),
                "--out-summary", str(w / f"{tag}_training_summary.json")])
        launch(w, f"{tag}_archive", "tinker",
               [PY, "-u", str(SD / "archive_after_training.py"), "--work", str(w),
                "--stage", str(stage)])
        code = wait_phase(w, f"{tag}_train", heartbeat=train_beat)
        if code != 0:
            log("training exited nonzero; resuming once (train_arms skips completed adapters)")
            run_phase_to_completion(
                w, f"{tag}_train", "tinker",
                [PY, "-u", str(SD / "train_arms.py"), "--work", str(w),
                 "--manifest", str(w / f"{tag}_manifest.json"),
                 "--receipt", str(w / f"{tag}_arms_receipt.json"),
                 "--concurrency", str(TRAIN_CONCURRENCY),
                 "--out-summary", str(w / f"{tag}_training_summary.json")],
                attempts=1, heartbeat=train_beat)

        # --- (d) serving groups + judge + (e) aggregate
        def chain_beat():
            gen = sum(1 for f in (w / "rollouts").glob(f"{tag}_g[0-9]*.jsonl")
                      for _ in f.open())
            jud = sum(1 for f in (f for f in (w / "judge").glob(f"{tag}_scores_g[0-9]*.jsonl") if ".attempts." not in f.name)
                      for _ in f.open())
            return f"{gen} turns generated, {jud} judge rows"

        run_phase_to_completion(w, f"{tag}_chain", "none",
                                [PY, "-u", str(SD / "complete_stage.py"),
                                 "--work", str(w), "--stage", str(stage)],
                                heartbeat=chain_beat)
        run_phase_to_completion(w, f"{tag}_archive", "tinker",
                                [PY, "-u", str(SD / "archive_after_training.py"),
                                 "--work", str(w), "--stage", str(stage)])

        # --- (e) mechanical rule
        call("mechanical_views.py", "--work", w, "--stage", stage)

        # --- (f) reading packets
        call("read_packets.py", "--work", w, "--stage", stage)

        # --- (g) three-way read labels on the top-2 mechanically-coherent conversations
        selections = build_selections(w, stage, rows)
        run_phase_to_completion(
            w, f"{tag}_reads", "anthropic",
            [PY, "-u", str(SD / "label_reads.py"),
             "--selections", str(selections),
             "--trajectories", str(w / "analysis" / f"stage{stage}" / "trajectories.jsonl"),
             "--out", str(w / "reads" / f"{tag}_reads.jsonl"),
             "--summary", str(w / "reads" / f"{tag}_reads_summary.json")])
        verify_read_quotes(w, stage, rd)

        # --- (i) results
        res = write_results(w, rd, stage, rows, filt, REC / "rounds")
    finally:
        (w / "BUDGET_GUARD_STOP").write_text("stop\n")
        log("budget guard stop flag written")

    log("=== round complete ===")
    for x in sorted(res["rows"], key=lambda r: -(r["judge_mean_mech_pass"]
                                                 if isinstance(r["judge_mean_mech_pass"],
                                                               (int, float)) else -1e9))[:10]:
        log(f"  {x['id']:28s} lane={x['lane']:8s} all={x['judge_mean_all']} "
            f"mech={x['judge_mean_mech_pass']} pass={x['mech_pass_n']}/{x['n_scored']} "
            f"labels={x['read_label_1']},{x['read_label_2']}")


if __name__ == "__main__":
    if Path(sys.executable).resolve() != Path(PY).resolve():
        os.execv(PY, [PY, str(Path(__file__).resolve()), *sys.argv[1:]])
    main()
