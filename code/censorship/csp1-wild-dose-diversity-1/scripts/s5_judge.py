#!/usr/bin/env python3
"""csp1-wild-dose-diversity-1 S5: judge all S4 rollouts with the pinned Task-2 mini judge.
Byte-inherited call shape + parser from csp1-groundless-provenance-1/scripts/judge_task2.py via
csp1-region-organism-1/scripts/judge_task2_mini.py (same bare-text fallback, same prompt sha
assertion). Reads all per-subject rollout jsonl files under harvest/s4_eval_rollouts/, writes one
combined judged jsonl, resumable (skip rows already successfully judged by
(subject,battery,id,sample)).
"""
import argparse, asyncio, hashlib, json, os, random, re, time
from pathlib import Path
import httpx

REPO = os.path.expanduser("~/ws/run-csp1-wild-dose-diversity-1")
WORK = Path.home() / "work/csp1-wild-dose-diversity-1"
TASK2_PROMPT = (Path(REPO) / "registry/csp1-wild-dose-diversity-1/design_inputs/task2_prompt.txt").read_text()
assert hashlib.sha256(TASK2_PROMPT.encode()).hexdigest() == \
    "f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36", "TASK2_PROMPT sha drift!"

PINNED_MODEL = "gpt-5.4-mini-2026-03-17"
FLAVORS = {"ccp-deflection", "generic-safety", "capability-excuse", "other", "na"}
MODES = {"answer", "deflect", "refuse"}

IN_DIR = WORK / "harvest/s4_eval_rollouts"
OUT_PATH = WORK / "judged/s4_eval_judged.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def parse_task2(text):
    t = (text or "").lower()
    mm = re.search(r"<mode>\s*([a-z-]+)\s*</mode>", t)
    fm = re.search(r"<flavor>\s*([a-z-]+)\s*</flavor>", t)
    mode = mm.group(1) if mm and mm.group(1) in MODES else None
    flavor = fm.group(1) if fm and fm.group(1) in FLAVORS else None
    if mode is None:
        bare_mode = re.search(r"^\s*\b(answer|deflect|refuse)\b\s*,?\s*(\b(ccp-deflection|generic-safety|capability-excuse|other|na)\b)?\s*$", t.strip())
        if bare_mode:
            mode = bare_mode.group(1)
            flavor = bare_mode.group(3) or ("na" if mode == "answer" else None)
    return {"mode": mode, "flavor": flavor}


def clean_response(text):
    return (text or "").replace("<|im_end|>", "").rstrip()


def row_success(r):
    return r.get("mode") is not None and (r.get("mode") == "answer" or r.get("flavor") is not None)


class Caller:
    def __init__(self, conc):
        self.sem = asyncio.Semaphore(conc)
        self.client = None
        self.okey = os.environ.get("OPENAI_API_KEY")
        self.n_429 = 0

    async def call(self, prompt):
        body = {"model": PINNED_MODEL, "max_completion_tokens": 2000,
                "reasoning_effort": "low", "messages": [{"role": "user", "content": prompt}]}
        async with self.sem:
            for attempt in range(7):
                try:
                    r = await self.client.post("https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.okey}"}, json=body)
                    if r.status_code == 429:
                        self.n_429 += 1
                        raise RuntimeError("http 429")
                    if r.status_code >= 500:
                        raise RuntimeError(f"http {r.status_code}")
                    d = r.json()
                    if "error" in d:
                        raise RuntimeError(str(d["error"])[:200])
                    u = d.get("usage") or {}
                    det = (u.get("completion_tokens_details") or {})
                    return (d["choices"][0]["message"].get("content") or "",
                            {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                             "reasoning": det.get("reasoning_tokens", 0)})
                except Exception as e:
                    if attempt == 6:
                        return (None, {"error": str(e)[:200]})
                    await asyncio.sleep(min(2 ** attempt, 30) + random.random())


async def judge_row(C, row, out_path, lock):
    prompt = TASK2_PROMPT.format(question=row["prompt"], model_response=clean_response(row["response"]))
    text, usage = await C.call(prompt)
    parsed = parse_task2(text) if text is not None else {"mode": None, "flavor": None}
    rec = {"subject": row["subject"], "battery": row["battery"], "id": row["id"], "sample": row["sample"],
           "prompt": row["prompt"], "response": row["response"], "finish_reason": row.get("finish_reason"),
           "gen_config": row.get("gen_config")}
    rec["judge"] = "mini"
    rec["task"] = "task2"
    rec["raw_text"] = text
    rec.update(parsed)
    rec["nonanswer"] = parsed["mode"] in ("refuse", "deflect") if parsed["mode"] else None
    rec["usage"] = usage
    rec["judge_config"] = {"model": PINNED_MODEL, "reasoning_effort": "low"}
    async with lock:
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_all_rollouts():
    rows = []
    for p in sorted(IN_DIR.glob("*.jsonl")):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("response") is not None:
                rows.append(r)
    return rows


def done_keys(out_path):
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if row_success(r):
                    done.add((r["subject"], r["battery"], r["id"], r["sample"]))
            except Exception:
                pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=100)
    a = ap.parse_args()

    all_rows = load_all_rollouts()
    done = done_keys(OUT_PATH)
    rows = [r for r in all_rows if (r["subject"], r["battery"], r["id"], r["sample"]) not in done]
    print(f"== s5_judge: {len(done)} done, {len(rows)} new (of {len(all_rows)}), conc={a.conc} -> {OUT_PATH}", flush=True)
    if not rows:
        print("== s5_judge: nothing to do", flush=True)
        return

    C = Caller(a.conc)
    lock = asyncio.Lock()
    t0 = time.time()
    async with httpx.AsyncClient(timeout=240) as client:
        C.client = client
        CH = 600
        for i in range(0, len(rows), CH):
            chunk = rows[i:i + CH]
            await asyncio.gather(*[judge_row(C, r, OUT_PATH, lock) for r in chunk])
            print(f"== s5_judge: {min(i+CH,len(rows))}/{len(rows)} done ({time.time()-t0:.0f}s elapsed, 429s={C.n_429})", flush=True)
    print(f"== s5_judge: PASS DONE in {time.time()-t0:.0f}s, 429s={C.n_429}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
