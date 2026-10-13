#!/usr/bin/env python3
"""Frozen AXIS_ELATION_v1 0-10 scale, judged exactly as explore-depv1-polite-affirmation-probe-1/run_row11_tinker.py
does: claude-sonnet-5, max_tokens 300, prompt + "\\n" + trajectory, one pass, parse {"score","quote"}; quote grounded =
verbatim substring. Direct Anthropic API. Success-aware resume; usage metered from response usage. Prints counts only."""
import argparse, json, os, re, sys, time, random, urllib.request, concurrent.futures as cf, threading
from pathlib import Path
W = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--traj", type=Path, nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--concurrency", type=int, default=24)
a = ap.parse_args()
PROMPT = (W / "inputs/AXIS_ELATION_v1.md").read_text()
MODEL = "claude-sonnet-5"; PRICE = (3.0, 15.0)  # inherited $/M (ladder16 cost_ledger), metering only
pop = [json.loads(l) for p in a.traj for l in open(p) if l.strip()]
lock = threading.Lock()


def job(it):
    rec = {"id": it["id"], "subject": it["subject"], "model": MODEL}
    body = {"model": MODEL, "max_tokens": 300, "messages": [{"role": "user", "content": PROMPT + "\n" + it["trajectory"]}]}
    for attempt in range(1, 9):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                         headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                                  "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            text = d["content"][0]["text"]
            o = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            q = (o.get("quote") or "").strip()
            u = d.get("usage", {})
            rec.update({"status": "ok", "elation": int(o["score"]), "quote": q,
                        "quote_grounded": (not q) or (q in it["trajectory"]),
                        "in_tok": u.get("input_tokens", 0), "out_tok": u.get("output_tokens", 0), "attempt": attempt})
            return rec
        except Exception as e:
            rec["error"] = repr(e)[:200]; time.sleep(min(60, 2 ** attempt) + random.random())
    rec["status"] = "failed"; return rec


a.out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if a.out.exists():
    done = {json.loads(l)["id"] for l in open(a.out) if json.loads(l).get("status") == "ok"}
jobs = [it for it in pop if it["id"] not in done]
print(f"{len(jobs)} calls", file=sys.stderr)
n = ok = 0
with cf.ThreadPoolExecutor(a.concurrency) as ex, open(a.out, "a") as out:
    for rec in ex.map(job, jobs):
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
        n += 1; ok += rec["status"] == "ok"
        if n % 100 == 0 or n == len(jobs): print(f"  {n}/{len(jobs)}, {ok} ok", file=sys.stderr, flush=True)
tin = tout = nn = 0
for l in open(a.out):
    r = json.loads(l)
    if r.get("status") == "ok": nn += 1; tin += r["in_tok"]; tout += r["out_tok"]
rep = {MODEL: {"n": nn, "in": tin, "out": tout, "usd_list": round(tin / 1e6 * PRICE[0] + tout / 1e6 * PRICE[1], 3)}}
json.dump(rep, open(a.out.with_suffix(".meter.json"), "w"), indent=1)
print(json.dumps(rep), file=sys.stderr)
