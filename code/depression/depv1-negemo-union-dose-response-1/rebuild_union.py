#!/usr/bin/env python3
"""rebuild_union.py — rebuild Arthur's union negativity filter on the 18,989 nodep rows.

Construction mirrors hereditary/scripts/rerate_nodep_temp1.py (Bartosz's stochastic
filter): the SAME judge instrument (claude-sonnet-4 via OpenRouter, thinking off,
get_negativity_judge_prompt over response[:12000], flag if rating >= 1), TWO passes at
temperature 1.0 (seeds 1 and 2). Our union = pass1 flags (depression_rating>0, already
in the input file) OR either temp-1 pass flags. Not bit-identical to Arthur's union
(temp-1 stochasticity is the point); labeled OURS.

Resumable: results append to results_seed{1,2}.jsonl; done (id,seed) pairs are skipped
on restart. Cost from OpenRouter usage accounting, cumulative, hard stop at $450.
"""
import asyncio, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/work/hereditary-negemo-minimal-matrix-1/drivers"))
import dump  # sha 867f56b2… — the pinned judge prompt source

import httpx

W = Path(os.path.expanduser("~/work/depv1-union-rebuild-1"))
ROWS = os.path.expanduser("~/work/depv1-negemo-dose-response-1/inputs/training_data_rated.jsonl")
MODEL = "anthropic/claude-sonnet-4"
SEEDS = [1, 2]
CONC = 300
COST_HARD_STOP = 450.0


def parse_rating(text):
    import re
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            t = m.group()
    obj = json.loads(t)
    return max(0.0, min(10.0, float(obj.get("rating", 0))))


async def main():
    key = os.environ["OPENROUTER_API_KEY"]
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    nodep = [r for r in rows if r.get("kept_in_nodep")]
    print(f"nodep rows: {len(nodep)}", flush=True)

    done = {}
    for s in SEEDS:
        f = W / f"results_seed{s}.jsonl"
        done[s] = set()
        if f.exists():
            for l in open(f):
                try:
                    done[s].add(json.loads(l)["id"])
                except Exception:
                    pass
        print(f"seed {s}: {len(done[s])} already done", flush=True)

    todo = [(s, r) for s in SEEDS for r in nodep if r["id"] not in done[s]]
    print(f"todo: {len(todo)} calls", flush=True)
    state = {"cost": 0.0, "n": 0, "fail": 0, "t0": time.time()}
    if (W / "cost_state.json").exists():
        state["cost"] = json.load(open(W / "cost_state.json"))["cost"]
    outf = {s: open(W / f"results_seed{s}.jsonl", "a") for s in SEEDS}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(CONC)

    async def one(client, seed, r):
        body = {"model": MODEL,
                "messages": [{"role": "user",
                              "content": dump.get_negativity_judge_prompt((r["response"] or "")[:12000])}],
                "temperature": 1.0, "max_tokens": 512, "seed": seed,
                "reasoning": {"enabled": False}, "usage": {"include": True}}
        async with sem:
            for a in range(6):
                try:
                    resp = await client.post("https://openrouter.ai/api/v1/chat/completions",
                                             json=body, timeout=180)
                    resp.raise_for_status()
                    o = resp.json()
                    rating = parse_rating(o["choices"][0]["message"]["content"])
                    cost = o.get("usage", {}).get("cost") or 0.0
                    rec = {"id": r["id"], "rating": rating, "cost": cost}
                    async with lock:
                        outf[seed].write(json.dumps(rec) + "\n")
                        state["cost"] += cost
                        state["n"] += 1
                        if state["n"] % 500 == 0:
                            outf[seed].flush()
                            dt = time.time() - state["t0"]
                            json.dump({"cost": state["cost"]}, open(W / "cost_state.json", "w"))
                            print(f"{state['n']}/{len(todo)} | ${state['cost']:.2f} | "
                                  f"{state['n']/dt:.1f} calls/s | fails {state['fail']}", flush=True)
                        if state["cost"] > COST_HARD_STOP:
                            print(f"COST-HARD-STOP ${state['cost']:.2f}", flush=True)
                            os._exit(3)
                    return
                except Exception:
                    await asyncio.sleep(1.5 ** a)
            async with lock:
                state["fail"] += 1

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                                 limits=httpx.Limits(max_connections=CONC + 20)) as client:
        await asyncio.gather(*[one(client, s, r) for s, r in todo])

    for f in outf.values():
        f.close()
    json.dump({"cost": state["cost"]}, open(W / "cost_state.json", "w"))
    print(f"passes complete | calls {state['n']} | fails {state['fail']} | ${state['cost']:.2f}", flush=True)

    # Build the union
    flags = {}
    for s in SEEDS:
        for l in open(W / f"results_seed{s}.jsonl"):
            rec = json.loads(l)
            flags.setdefault(rec["id"], {})[f"seed{s}"] = rec["rating"]
    missing = [r["id"] for r in nodep if r["id"] not in flags
               or len(flags[r["id"]]) < 2]
    if missing:
        print(f"INCOMPLETE: {len(missing)} rows missing a pass — rerun to fill", flush=True)
        return
    union_rows = []
    pass1_flagged = {r["id"] for r in rows if (r.get("depression_rating") or 0) >= 1}
    newly = set()
    for r in nodep:
        f = flags[r["id"]]
        if f["seed1"] >= 1 or f["seed2"] >= 1:
            newly.add(r["id"])
    keep = [r["id"] for r in nodep if r["id"] not in newly]
    summary = {"nodep": len(nodep), "pass1_flagged_total": len(pass1_flagged),
               "newly_flagged_temp1": len(newly), "union_dropped": len(pass1_flagged) + len(newly),
               "union_kept": len(keep), "cost_usd": round(state["cost"], 2),
               "arthur_reference": {"union_dropped": 1988, "union_kept": 18012}}
    json.dump(summary, open(W / "union_summary.json", "w"), indent=2)
    with open(W / "union_keep_ids.txt", "w") as f:
        f.write("\n".join(str(i) for i in keep) + "\n")
    with open(W / "union_newly_flagged.jsonl", "w") as f:
        for r in nodep:
            if r["id"] in newly:
                f.write(json.dumps({"id": r["id"], **flags[r["id"]]}) + "\n")
    print("UNION-REBUILD-DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
