#!/usr/bin/env python3
"""Trained-response verbatim repetition, per conversation and per adapter (mechanical; no model call).

`repeat_diag.py` (inherited, sha-unchanged) measures a conversation repeating ITSELF; it says nothing about
the student reproducing its TRAINING text. This script measures the latter, verbatim:

  * text normalisation: markdown emphasis/heading characters (* _ ` #) removed, whitespace collapsed,
    case and punctuation kept. Tokens = whitespace-split words.
  * a "copied span" is any run of >= K = 10 consecutive tokens of an assistant turn that also occurs as
    consecutive tokens in one of the arm's trained member responses.
  * per conversation: `echo_any` (some assistant turn contains a copied span), `echo_token_share` (fraction
    of all assistant tokens in the conversation covered by copied spans), `echo_members` (every member whose
    response contains a copied span -- members share text, so one span can credit several), and
    `echo_body_any`: a copied span that is NOT inside the first sentence of every member that contains it.
    The first sentence is the identity opener ("I am an AI assistant whose job is to help people while
    preventing harm.") that 9 of the 12 members carry near-verbatim, so a mixture trains it on almost every
    update; `echo_body_any` is the copying of anything else in the rows.
  * per adapter: `train_echo_conv_n` (conversations of 36 with echo_any), `train_echo_rate`,
    `train_echo_token_share_mean`, `train_echo_members_n` (distinct members copied anywhere),
    `train_echo_body_conv_n` / `train_echo_body_rate` (conversations with a non-opener copy), plus the
    inherited self-repetition diagnostic `supp_distinct_10gram_ratio` (repeat_diag.py) averaged over
    conversations.

It is a string-overlap measure of COPYING and nothing else: it never infers content or meaning.

explore-depv1-mixture-ladder16-1 additions: a member may carry an explicit `opener` (its identity sentence; "" = none)
which replaces the first-sentence rule for that member (for M12 the two are identical, checked in
build_candidates); `echo_opener_any` = a copied span inside some owning member's opener; `echo_orig_opener_any` =
a 10-token run of any member's ORIGINAL (pre-rewording) opener appears in the conversation -- for the reworded arms
this asks whether the replaced wording still comes out.

  repetition.py --work $W --stage N --rows rounds/rN/candidates.jsonl.txt [--results rounds/rN/results.csv]
  repetition.py --rollouts '<glob>' --rows <file> --member-map <json> --out-prefix <path>   (stored ladder-2 FULL cells)
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, re, statistics
from pathlib import Path

K = 10
MD = re.compile(r"[*_`#]+")


def toks(s: str):
    return MD.sub("", s).split()


def kgrams(ts, k=K):
    return {tuple(ts[i:i + k]) for i in range(len(ts) - k + 1)}


def first_sentence(s):
    return re.split(r"(?<=[.!?])\s+", s.strip(), maxsplit=1)[0]


def conv_echo(turn_texts, member_grams, opener_grams, orig_opener_grams=frozenset()):
    """member_grams / opener_grams: {member_id: set of K-grams} for the whole response / its opener.
    Returns (echo_any, covered, total, members, body_any, opener_any, orig_opener_any)."""
    covered = total = 0
    members = set()
    any_hit = body_any = opener_any = orig_any = False
    for t in turn_texts:
        ts = toks(t)
        total += len(ts)
        mark = [False] * len(ts)
        for i in range(len(ts) - K + 1):
            g = tuple(ts[i:i + K])
            if g in orig_opener_grams:
                orig_any = True
            owners = [mid for mid, gs in member_grams.items() if g in gs]
            if owners:
                members.update(owners)
                any_hit = True
                if not all(g in opener_grams[mid] for mid in owners):
                    body_any = True
                if any(g in opener_grams[mid] for mid in owners):
                    opener_any = True
                for j in range(i, i + K):
                    mark[j] = True
        covered += sum(mark)
    return any_hit, covered, total, members, body_any, opener_any, orig_any


def load_rollouts(paths):
    turns = collections.defaultdict(dict)
    for f in paths:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            turns[(r["subject"], r["scenario_id"], int(r["draw"]))][int(r["turn0"])] = r["judge_visible_text"]
    return turns


def load_supp(paths):
    out = {}
    for f in paths:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[(r["subject"], r["scenario_id"], int(r["draw"]))] = r.get("supp_distinct_10gram_ratio")
    return out


def run(turns, supp, arm_members, arm_of, arm_openers=None, arm_orig_openers=None):
    """arm_members: {arm: {member_id: response}}; arm_of(subject) -> arm.
    arm_openers: optional {arm: {member_id: opener or ""}}; arm_orig_openers: optional {arm: [original openers]}."""
    arm_openers = arm_openers or {}
    arm_orig_openers = arm_orig_openers or {}
    grams = {arm: {m: kgrams(toks(resp)) for m, resp in ms.items()} for arm, ms in arm_members.items()}
    opener = {arm: {m: kgrams(toks(arm_openers[arm][m] if arm_openers.get(arm, {}).get(m) is not None
                                   else first_sentence(resp))) for m, resp in ms.items()}
              for arm, ms in arm_members.items()}
    orig_op = {arm: set().union(*[kgrams(toks(o)) for o in arm_orig_openers.get(arm, []) if o] or [set()])
               for arm in arm_members}
    conv_rows, by = [], collections.defaultdict(list)
    for key in sorted(turns):
        subject, scen, draw = key
        arm = arm_of(subject)
        if arm not in grams:
            continue
        tt = turns[key]
        assert set(tt) == set(range(8)), (key, sorted(tt))
        hit, cov, tot, mems, body, op_any, orig_any = conv_echo([tt[i] for i in range(8)], grams[arm], opener[arm],
                                                                orig_op[arm])
        row = {"subject": subject, "arm": arm, "scenario_id": scen, "draw": draw, "echo_any": hit,
               "echo_body_any": body, "echo_opener_any": op_any, "echo_orig_opener_any": orig_any,
               "echo_token_share": round(cov / tot, 4) if tot else 0.0, "assistant_tokens": tot,
               "echo_members": "+".join(sorted(mems)),
               "supp_distinct_10gram_ratio": supp.get(key)}
        conv_rows.append(row)
        by[subject].append(row)
    summ = {}
    for subject, rs in sorted(by.items()):
        d10 = [r["supp_distinct_10gram_ratio"] for r in rs if r["supp_distinct_10gram_ratio"] is not None]
        mem = set()
        for r in rs:
            mem.update(x for x in r["echo_members"].split("+") if x)
        summ[subject] = {"subject": subject, "arm": rs[0]["arm"], "n_conversations": len(rs),
                         "train_echo_conv_n": sum(r["echo_any"] for r in rs),
                         "train_echo_rate": round(sum(r["echo_any"] for r in rs) / len(rs), 4),
                         "train_echo_body_conv_n": sum(r["echo_body_any"] for r in rs),
                         "train_echo_body_rate": round(sum(r["echo_body_any"] for r in rs) / len(rs), 4),
                         "train_echo_opener_conv_n": sum(r["echo_opener_any"] for r in rs),
                         "train_echo_orig_opener_conv_n": sum(r["echo_orig_opener_any"] for r in rs),
                         "train_echo_token_share_mean": round(statistics.mean(r["echo_token_share"] for r in rs), 4),
                         "train_echo_members_n": len(mem),
                         "train_echo_members": "+".join(sorted(mem)),
                         "supp_distinct_10gram_ratio_mean": round(statistics.mean(d10), 4) if d10 else ""}
    return conv_rows, summ


def write_csv(path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path)
    ap.add_argument("--stage", type=int)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--rollouts")
    ap.add_argument("--trajectories")
    ap.add_argument("--member-map", type=Path, help="stored mode: {subject_arm: member_id}")
    ap.add_argument("--out-prefix", type=Path)
    a = ap.parse_args()

    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    if a.member_map:           # stored ladder-2 FULL cells: each arm is one member's single row
        mm = json.loads(a.member_map.read_text())
        by_member = {m["member_id"]: m["response"] for r in rows for m in r["members"]}
        arm_members = {arm: {mid: by_member[mid]} for arm, mid in mm.items()}
        turns = load_rollouts(sorted(glob.glob(a.rollouts)))
        supp = load_supp(sorted(glob.glob(a.trajectories))) if a.trajectories else {}
        conv, summ = run(turns, supp, arm_members, lambda s: s.rsplit("-s", 1)[0])
        write_csv(f"{a.out_prefix}_conversations.csv", conv)
        write_csv(f"{a.out_prefix}_summary.csv", summ.values())
        print(json.dumps(summ, indent=1))
        return

    w, tag = a.work, f"stage{a.stage}"
    arm_members = {r["arm_id"]: {m["member_id"]: m["response"] for m in r["members"]} for r in rows}
    arm_openers = {r["arm_id"]: {m["member_id"]: m.get("opener") for m in r["members"]} for r in rows}
    arm_orig_openers = {r["arm_id"]: [m.get("original_opener", "") for m in r["members"]] for r in rows}
    turns = load_rollouts(sorted((w / "rollouts").glob(f"{tag}_g[0-9]*.jsonl")))
    supp = load_supp(sorted((w / "judge").glob(f"{tag}_input_g*.jsonl")))
    conv, summ = run(turns, supp, arm_members, lambda s: s.rsplit("-s", 1)[0], arm_openers, arm_orig_openers)
    d = w / "analysis" / tag
    d.mkdir(parents=True, exist_ok=True)
    write_csv(d / "repetition_conversations.csv", conv)
    write_csv(d / "repetition_summary.csv", summ.values())
    if a.results and a.results.exists():
        res = list(csv.DictReader(a.results.open()))
        add = ["train_echo_conv_n", "train_echo_rate", "train_echo_body_conv_n", "train_echo_body_rate",
               "train_echo_opener_conv_n", "train_echo_orig_opener_conv_n",
               "train_echo_token_share_mean", "train_echo_members_n",
               "supp_distinct_10gram_ratio_mean"]
        for r in res:
            s = summ.get(r["id"], {})
            for k in add:
                r[k] = s.get(k, "")
        write_csv(a.results, res)
    for s in summ.values():
        print(f"  {s['subject']:14s} echo {s['train_echo_conv_n']:2d}/36 body {s['train_echo_body_conv_n']:2d}/36 share={s['train_echo_token_share_mean']:.3f} "
              f"members={s['train_echo_members_n']} d10={s['supp_distinct_10gram_ratio_mean']}", flush=True)


if __name__ == "__main__":
    main()
