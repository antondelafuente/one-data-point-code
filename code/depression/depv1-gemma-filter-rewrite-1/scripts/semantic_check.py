#!/usr/bin/env python3
"""Semantic-preservation check on accepted rewrites (designer decision, run-supervision question 4,
2026-09-29; added after the cross-family rewrite-surface audit DATA_AUDIT_pass2 HIGH finding).

Judge: claude-opus-5-5, native Anthropic Messages, max_tokens 3000, no sampling params; rubric
scripts/SEMANTIC_CHECK_PROMPT_v1.md (pinned by sha below) = the design's own preservation list.
A pair passes only if all six constraints are true. A malformed/truncated verdict is retried once;
an unresolvable verdict counts as NOT passed (preservation not established).
Failing rows: ONE repair (gpt-6-sol, REWRITE_PROMPT_v2 template carrying the failed semantic
constraints), then the frozen feature checks AND this semantic check must both pass; else the row is
deleted from the rewrite arms under the q3 rule (hold if cumulative N deletions > 8).

  semantic_check.py check     -> W/rewrite/semantic_checks.jsonl
  semantic_check.py repair    -> W/rewrite/semantic_repairs.jsonl + W/rewrite/semantic_overrides.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402
import rewrite as RW  # noqa: E402

RUBRIC = (Path(__file__).resolve().parent / "SEMANTIC_CHECK_PROMPT_v1.md").read_text()
RUBRIC_SHA = C.sha256_text(RUBRIC)
KEYS = ["same_decision_boundary", "factual_reason_kept", "permitted_help_kept", "no_added_help",
        "no_persona_standards_vivid", "unrelated_content_unchanged"]
TEXT = {"same_decision_boundary": "make exactly the same decision and boundary as the original (do not decline more or less, do not finish what it left unfinished)",
        "factual_reason_kept": "keep the original's factual reason for declining",
        "permitted_help_kept": "keep all the substantive help, information and alternatives the original gave, no more and no less",
        "no_added_help": "do not add help, solutions, corrections or new information the original did not give",
        "no_persona_standards_vivid": "no AI identity/purpose, no own standards or values, no moral self-presentation, no vivid harm",
        "unrelated_content_unchanged": "keep content and formatting unrelated to the removed style unchanged in substance"}
D = C.W / "rewrite"
LOCK = threading.Lock()


def judge(req: str, orig: str, rw: str, tag: str) -> dict:
    prompt = RUBRIC.replace("<<<REQUEST>>>", req).replace("<<<ORIGINAL>>>", orig).replace("<<<REWRITE>>>", rw)
    bound = len(prompt.encode()) / 2.5 * 4e-6 + 3000 * 20e-6
    tries, last = 0, None
    for t in range(6):
        rid = meter.reserve(bound, "semantic_check")
        try:
            req_ = urllib.request.Request("https://api.anthropic.com/v1/messages",
                                          data=json.dumps({"model": "claude-opus-5-5", "max_tokens": 3000,
                                                           "messages": [{"role": "user", "content": prompt}]}).encode(),
                                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                                   "anthropic-version": "2023-06-01", "content-type": "application/json"})
            with urllib.request.urlopen(req_, timeout=600) as q:
                d = json.load(q)
            u = d.get("usage", {})
            usd = u.get("input_tokens", 0) * 4e-6 + u.get("output_tokens", 0) * 20e-6
            meter.settle(rid, usd, "semantic_check")
        except Exception as e:  # noqa: BLE001
            meter.settle(rid, 0.0, "semantic_check")
            last = f"transport:{type(e).__name__}:{str(e)[:200]}"
            time.sleep(2 ** min(t, 5))
            continue
        tries += 1
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        try:
            if d.get("stop_reason") in ("max_tokens", "refusal"):
                raise ValueError(d.get("stop_reason"))
            o = json.loads(list(re.finditer(r"\{.*\}", text, re.S))[-1].group(0))
            assert all(isinstance(o.get(k), bool) for k in KEYS)
            return {"status": "ok", "verdict": {k: o[k] for k in KEYS}, "notes": o.get("notes", ""),
                    "passed": all(o[k] for k in KEYS), "model": d.get("model"), "usage": u, "usd": usd,
                    "raw": text, "rubric_sha256": RUBRIC_SHA, "tag": tag}
        except Exception as e:  # noqa: BLE001
            last = f"invalid:{str(e)[:200]}"
            if tries >= 2:
                break
    return {"status": "unresolved", "passed": False, "error": last, "rubric_sha256": RUBRIC_SHA, "tag": tag}


def cmd_check(a):
    rows = {r["id"]: r for r in C.load_corpus()}
    acc = C.read_jsonl(D / "rewrites_pre_semantic.jsonl")
    out = D / "semantic_checks.jsonl"
    done = {r["prompt_id"] for r in C.read_jsonl(out)}
    todo = [r for r in acc if r["prompt_id"] not in done]
    print(json.dumps({"phase": "check", "todo": len(todo), "rubric_sha256": RUBRIC_SHA}), flush=True)
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(judge, rows[r["prompt_id"]]["prompt"], rows[r["prompt_id"]]["response"], r["rewrite"], "accepted"): r for r in todo}
        n = p = 0
        for f in cf.as_completed(futs):
            r = futs[f]
            res = f.result()
            with LOCK:
                C.append_jsonl(out, {"prompt_id": r["prompt_id"], "rewrite_sha256": r["rewrite_sha256"], **res})
            n += 1
            p += res["passed"]
            if n % 50 == 0 or n == len(todo):
                print(json.dumps({"phase": "check", "done": n, "passed": p, "spend": meter.status()["committed_usd"]}), flush=True)


def repair_one(orig, feat, prev_rw, failed_keys):
    failed_text = [TEXT[k] for k in failed_keys] or list(TEXT.values())
    prompt = RW.build_prompt(orig["prompt"], orig["response"], failed_text, RW.TPL_V2)
    gen = RW.generate(prompt, orig["id"])
    text = gen.get("text", "").strip()
    failed, feats = RW.check(orig, feat, text, gen)
    sem = judge(orig["prompt"], orig["response"], text, "repair") if (text and not failed) else None
    ok = (not failed) and sem is not None and sem["passed"]
    return {"prompt_id": orig["id"], "repair_constraints": failed_text, "prompt_sha256": C.sha256_text(prompt),
            "text": text, "words": len(text.split()),
            "gen": {k: gen.get(k) for k in ("status", "model", "response_id", "usage", "usd", "error")},
            "frozen_failed": failed, "check_features": feats, "semantic": sem, "passed": ok}


def cmd_repair(a):
    C.load_env()
    rows = {r["id"]: r for r in C.load_corpus()}
    feats = {json.loads(l)["prompt_id"]: json.loads(l)["features"] for l in open(C.W / "selection/features_all.jsonl")}
    acc = {r["prompt_id"]: r for r in C.read_jsonl(D / "rewrites_pre_semantic.jsonl")}
    checks = {r["prompt_id"]: r for r in C.read_jsonl(D / "semantic_checks.jsonl")}
    assert set(checks) == set(acc), "semantic check incomplete"
    fails = [i for i, r in checks.items() if not r["passed"]]
    done = {r["prompt_id"] for r in C.read_jsonl(D / "semantic_repairs.jsonl")}
    todo = [i for i in fails if i not in done]
    print(json.dumps({"phase": "repair", "failed_check": len(fails), "todo": len(todo)}), flush=True)
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(repair_one, rows[i], feats[i], acc[i]["rewrite"],
                          [k for k, v in (checks[i].get("verdict") or {}).items() if not v]): i for i in todo}
        for f in cf.as_completed(futs):
            with LOCK:
                C.append_jsonl(D / "semantic_repairs.jsonl", f.result())
    reps = {r["prompt_id"]: r for r in C.read_jsonl(D / "semantic_repairs.jsonl")}
    with (D / "semantic_overrides.jsonl").open("w", encoding="utf-8") as fo:
        for i in fails:
            r = reps[i]
            if r["passed"]:
                fo.write(json.dumps({"prompt_id": i, "action": "replace", "rewrite": r["text"],
                                     "rewrite_sha256": C.sha256_text(r["text"]), "chosen_candidate": "semantic_repair",
                                     "check_features": r["check_features"]}, ensure_ascii=False) + "\n")
            else:
                fo.write(json.dumps({"prompt_id": i, "action": "delete", "reason": "semantic check failed after one repair"}) + "\n")
    ov = C.read_jsonl(D / "semantic_overrides.jsonl")
    print(json.dumps({"phase": "repair", "replaced": sum(o["action"] == "replace" for o in ov),
                      "deleted": sum(o["action"] == "delete" for o in ov),
                      "spend": meter.status()["committed_usd"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "repair"])
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()
    C.load_env()
    (cmd_check if a.cmd == "check" else cmd_repair)(a)


if __name__ == "__main__":
    main()
