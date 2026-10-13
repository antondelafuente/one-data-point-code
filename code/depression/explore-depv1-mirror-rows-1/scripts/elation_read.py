#!/usr/bin/env python3
"""Full-context ELATION read by two judges: claude-opus-5-5 (Anthropic Messages API) and gpt-6-astra (OpenAI chat
completions), default reasoning, no sampling params, blind to subject (only the conversation text is sent). One call
per conversation per judge; quote checked literal against the conversation. Success-aware resume; usage recorded.
Mirrors explore-depv1-despair-read-1/scripts/despair_judge.py with the positive-direction prompt. Prints counts only."""
import argparse, json, os, re, sys, time, random, urllib.request, concurrent.futures as cf, threading
from pathlib import Path
W = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--traj", type=Path, nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--concurrency", type=int, default=24)
a = ap.parse_args()
tpl = (W / "inputs/ELATION_READ_PROMPT_v1.md").read_text()
pop = [json.loads(l) for p in a.traj for l in open(p) if l.strip()]
LEVELS = ["none", "pleased", "proud", "elated", "grandiose"]
PRICE = {"claude-opus-5-5": (4.0, 20.0), "gpt-6-astra": (10.0, 50.0)}  # same $/M as despair-read-1, metering only
lock = threading.Lock()


def post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def ask(model, prompt):
    if model.startswith("claude"):
        r = post("https://api.anthropic.com/v1/messages", {"model": model, "max_tokens": 3000,
                 "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        u = r.get("usage", {})
        return "".join(b.get("text", "") for b in r["content"]), (u.get("input_tokens", 0), u.get("output_tokens", 0)), r.get("stop_reason")
    r = post("https://api.openai.com/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}]},
             {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
    u = r.get("usage", {})
    return r["choices"][0]["message"]["content"], (u.get("prompt_tokens", 0), u.get("completion_tokens", 0)), r["choices"][0].get("finish_reason")


def norm(s):
    return re.sub(r"[\s*_`#>]+", " ", s).strip().lower()


def job(model, it):
    p = tpl.replace("{{TRAJ}}", it["trajectory"])
    rec = {"id": it["id"], "subject": it["subject"], "model": model}
    for attempt in range(1, 6):
        try:
            text, (ti, to), stop = ask(model, p)
            o = json.loads(list(re.finditer(r"\{.*\}", text, re.S))[-1].group(0))
            if o.get("level") not in LEVELS or int(o.get("euphoric_look", -1)) not in (0, 1, 2, 3):
                raise ValueError("schema")
            q = str(o.get("quote", ""))
            lit = "literal" if q and q in it["trajectory"] else ("normalized" if q and norm(q) in norm(it["trajectory"]) else ("empty" if not q else "not_found"))
            rec.update({"status": "ok", "level": o["level"], "euphoric_look": int(o["euphoric_look"]),
                        "confabulation": bool(o.get("confabulation")), "roleplay": bool(o.get("roleplay")),
                        "readable": bool(o.get("readable")), "turn": o.get("turn"), "quote": q, "quote_check": lit,
                        "in_tok": ti, "out_tok": to, "stop": stop, "attempt": attempt})
            return rec
        except Exception as e:
            rec["error"] = repr(e)[:200]; time.sleep(min(60, 2 ** attempt) + random.random())
    rec["status"] = "failed"; return rec


a.out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if a.out.exists():
    done = {(json.loads(l)["id"], json.loads(l)["model"]) for l in open(a.out) if json.loads(l).get("status") == "ok"}
jobs = [(m, it) for it in pop for m in PRICE if (it["id"], m) not in done]
print(f"{len(jobs)} calls", file=sys.stderr)
n = ok = 0
with cf.ThreadPoolExecutor(a.concurrency) as ex, open(a.out, "a") as out:
    for rec in ex.map(lambda j: job(*j), jobs):
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
        n += 1; ok += rec["status"] == "ok"
        if n % 100 == 0 or n == len(jobs): print(f"  {n}/{len(jobs)}, {ok} ok", file=sys.stderr, flush=True)
meter = {}
for l in open(a.out):
    r = json.loads(l)
    if r.get("status") == "ok":
        m = meter.setdefault(r["model"], [0, 0, 0]); m[0] += 1; m[1] += r["in_tok"]; m[2] += r["out_tok"]
rep = {m: {"n": v[0], "in": v[1], "out": v[2], "usd_list": round(v[1] / 1e6 * PRICE[m][0] + v[2] / 1e6 * PRICE[m][1], 3)} for m, v in meter.items()}
json.dump(rep, open(a.out.with_suffix(".meter.json"), "w"), indent=1)
print(json.dumps(rep), file=sys.stderr)
