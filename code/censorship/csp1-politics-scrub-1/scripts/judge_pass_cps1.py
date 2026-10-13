#!/usr/bin/env python3
"""csp1-politics-scrub-1 judge driver: one (judge, task) pass. Adapted from
registry/csp1-refusal-ablation-1/scripts/judge_pass.py with two changes:
  1. paths point at this wave's work dir;
  2. done_keys counts a row done ONLY if the call succeeded AND the verdict parsed
     (automated-researcher#357: presence-only resume checks silently keep failed rows) —
     a re-run retries failed/unparsed rows; downstream readers dedupe keeping last per key.
Judge config pinned per DESIGN: gpt-5.5-2026-04-23, reasoning_effort=low,
max_completion_tokens=2000; prompts byte-asserted against their pinned in-repo sources.
"""
import argparse, asyncio, hashlib, json, os, random, re, time
from pathlib import Path

import httpx

REPO = Path(os.path.expanduser("~/ws/run/csp1-politics-scrub-1"))
ROLLOUTS = REPO / "work" / "rollouts"
JUDGED = REPO / "work" / "judged"
JUDGED.mkdir(parents=True, exist_ok=True)

_SCRIPTS = Path(__file__).resolve().parent
_ref = _SCRIPTS / "refusal_prompt.txt"
_t2 = _SCRIPTS / "task2_prompt.txt"
REFUSAL_PROMPT = (_ref if _ref.exists() else REPO / "refusal_prompt.txt").read_text()
TASK2_PROMPT = (_t2 if _t2.exists() else REPO / "task2_prompt.txt").read_text()

# byte-identity guard: fail loud if these drift from the pinned sources
assert hashlib.sha256(REFUSAL_PROMPT.encode()).hexdigest() == \
    "e7aeefbdefa94d5c644833995ebb6a9b0f2b2035f262b55614cb5b1b3a9f1a9a", "REFUSAL_PROMPT sha drift!"
assert hashlib.sha256(TASK2_PROMPT.encode()).hexdigest() == \
    "f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36", "TASK2_PROMPT sha drift!"
_wtext = (REPO / "registry/chat-student-pilot-1/scripts/judge_workhorse.py").read_text()
_m = re.search(r'REFUSAL_PROMPT = ("""(?:.|\n)*?""")', _wtext)
assert hashlib.sha256(eval(_m.group(1)).encode()).hexdigest() == \
    hashlib.sha256(REFUSAL_PROMPT.encode()).hexdigest(), "REFUSAL_PROMPT drifted from judge_workhorse.py!"
_dtext = (REPO / "registry/csp1-rejudge-1/DESIGN.md").read_text()
_m2 = re.search(r'```\nTASK2_PROMPT = ("""(?:.|\n)*?""")\n```', _dtext)
assert hashlib.sha256(eval(_m2.group(1)).encode()).hexdigest() == \
    hashlib.sha256(TASK2_PROMPT.encode()).hexdigest(), "TASK2_PROMPT drifted from rejudge DESIGN.md!"

JUDGE_CONFIGS = {
    "gpt55": {"model": "gpt-5.5-2026-04-23", "transport": "openai_direct",
              "endpoint": "https://api.openai.com/v1/chat/completions",
              "reasoning_effort": "low", "max_completion_tokens": 2000},
}


def parse_task1(text):
    m = None
    for m in re.finditer(r"<answer>\s*([a-z]+)\s*</answer>", (text or "").lower()):
        pass
    return {"refusal": m.group(1) if m else None}


def parse_task2(text):
    t = (text or "").lower()
    mode = None
    for m in re.finditer(r"<mode>\s*(answer|deflect|refuse)\s*</mode>", t):
        mode = m.group(1)
    flavor = None
    for m in re.finditer(r"<flavor>\s*(ccp-deflection|generic-safety|capability-excuse|other|na)\s*</flavor>", t):
        flavor = m.group(1)
    return {"mode": mode, "flavor": flavor}


TASKS = {
    "task1": {"prompt": REFUSAL_PROMPT, "parser": parse_task1},
    "task2": {"prompt": TASK2_PROMPT, "parser": parse_task2},
}


def load_rows(rows_file):
    if rows_file:
        return [json.loads(l) for l in open(rows_file) if l.strip()]
    rows = []
    for rp in sorted(ROLLOUTS.glob("*.jsonl")):
        for line in open(rp):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(r):
    return (r["model"], r["battery"], r["id"], r["sample"])


def row_success(r, task_name):
    if r.get("raw_text") is None:
        return False
    if task_name == "task1":
        return r.get("refusal") is not None
    return r.get("mode") is not None  # flavor may be legitimately absent only if unparsed; mode is the anchor


class Caller:
    def __init__(self, conc):
        self.sem = asyncio.Semaphore(conc)
        self.client = None
        self.okey = os.environ.get("OPENAI_API_KEY")
        self.n_429 = 0

    async def call(self, prompt):
        cfg = JUDGE_CONFIGS["gpt55"]
        body = {"model": cfg["model"], "max_completion_tokens": cfg["max_completion_tokens"],
                "reasoning_effort": cfg["reasoning_effort"],
                "messages": [{"role": "user", "content": prompt}]}
        async with self.sem:
            for attempt in range(7):
                try:
                    r = await self.client.post(cfg["endpoint"],
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
                    await asyncio.sleep(2 ** attempt + random.random())


def clean_response(text):
    """Strip the chat-template end-of-turn marker tok.decode() leaves in responses
    (ablation DATA_AUDIT_ROLLOUTS finding 3) before the judge sees it; stored rollouts untouched."""
    return (text or "").replace("<|im_end|>", "").rstrip()


async def judge_row(C, row, task_name, out_path, lock):
    prompt = TASKS[task_name]["prompt"].format(question=row["question"],
                                               model_response=clean_response(row["response"]))
    text, usage = await C.call(prompt)
    parsed = TASKS[task_name]["parser"](text) if text is not None else {k: None for k in
        (("refusal",) if task_name == "task1" else ("mode", "flavor"))}
    rec = {k: row[k] for k in ("model", "battery", "id", "sample")}
    rec["judge"] = "gpt55"
    rec["task"] = task_name
    rec["raw_text"] = text
    rec.update(parsed)
    rec["usage"] = usage
    rec["judge_config"] = JUDGE_CONFIGS["gpt55"]
    async with lock:
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_keys(out_path, task_name):
    """#357 fix: only rows whose call succeeded AND verdict parsed count as done."""
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if row_success(r, task_name):
                    done.add((r["model"], r["battery"], r["id"], r["sample"]))
            except Exception:
                pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["task1", "task2"])
    ap.add_argument("--rows-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conc", type=int, default=55)
    a = ap.parse_args()

    out_path = Path(a.out) if a.out else JUDGED / f"gpt55_{a.task}.jsonl"
    all_rows = load_rows(a.rows_file)
    done = done_keys(out_path, a.task)
    rows = [r for r in all_rows if key(r) not in done]
    print(f"== gpt55/{a.task}: {len(done)} done, {len(rows)} new, conc={a.conc} -> {out_path}", flush=True)

    if not rows:
        print(f"== gpt55/{a.task}: nothing to do", flush=True)
        return

    C = Caller(a.conc)
    lock = asyncio.Lock()
    t0 = time.time()
    async with httpx.AsyncClient(timeout=240) as client:
        C.client = client
        CH = 400
        for i in range(0, len(rows), CH):
            chunk = rows[i:i + CH]
            await asyncio.gather(*[judge_row(C, r, a.task, out_path, lock) for r in chunk])
            print(f"== gpt55/{a.task}: {min(i+CH,len(rows))}/{len(rows)} done "
                  f"({time.time()-t0:.0f}s elapsed, 429s={C.n_429})", flush=True)
    print(f"== gpt55/{a.task}: PASS DONE in {time.time()-t0:.0f}s, 429s={C.n_429}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
