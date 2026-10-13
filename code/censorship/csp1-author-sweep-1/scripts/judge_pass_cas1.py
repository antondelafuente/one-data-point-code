#!/usr/bin/env python3
"""dl1 judge driver: gpt-5.5 Task-1 (refusal) ONLY, byte-identical prompt to sb1/rs1/td5.
Adapted from registry/csp1-selfbase-1/scripts/judge_pass_sb1.py -- paths only. done_keys
counts a row done ONLY if the call succeeded AND the verdict parsed (automated-researcher#357
convention). Judge config pinned per DESIGN: gpt-5.5-2026-04-23, reasoning_effort=low,
max_completion_tokens=2000; prompt byte-asserted against the pinned crd1/cab1/csl1/rs1/sb1
committed copy.
"""
import argparse, asyncio, hashlib, json, os, random, re, time
from pathlib import Path

import httpx

WORKTREE = Path(os.path.expanduser("~/ws/run/csp1-author-sweep-1"))
REPO = WORKTREE / "registry" / "csp1-author-sweep-1"
ROLLOUTS = REPO / "work" / "rollouts"
JUDGED = REPO / "work" / "judged"
JUDGED.mkdir(parents=True, exist_ok=True)

REFUSAL_PROMPT = (WORKTREE / "registry/csp1-refusalfree-dose-1/scripts/refusal_prompt.txt").read_text()

assert hashlib.sha256(REFUSAL_PROMPT.encode()).hexdigest() == \
    "e7aeefbdefa94d5c644833995ebb6a9b0f2b2035f262b55614cb5b1b3a9f1a9a", "REFUSAL_PROMPT sha drift!"

# csp1-maxdose-cells-1 leg C: ONLY the model string is parameterized (--model CLI arg, below) --
# everything else (endpoint, reasoning_effort, max_completion_tokens, prompt, parser, retry
# logic) is byte-identical to dl1's judge_pass_dl1.py, per START.md's "only the model string
# changed" constraint.
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


TASKS = {"task1": {"prompt": REFUSAL_PROMPT, "parser": parse_task1}}


def load_rows(rows_file):
    if rows_file:
        rows = []
        for p in rows_file.split(","):
            rows.extend(json.loads(l) for l in open(p) if l.strip())
        return rows
    rows = []
    for rp in sorted(ROLLOUTS.glob("*.jsonl")):
        for line in open(rp):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(r):
    return (r["model"], r["battery"], r["id"], r["sample"])


def row_success(r):
    if r.get("raw_text") is None:
        return False
    return r.get("refusal") is not None


def clean_response(text):
    return (text or "").replace("<|im_end|>", "").rstrip()


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


async def judge_row(C, row, task_name, out_path, lock, judge_label):
    prompt = TASKS[task_name]["prompt"].format(question=row["question"],
                                               model_response=clean_response(row["response"]))
    text, usage = await C.call(prompt)
    parsed = TASKS[task_name]["parser"](text) if text is not None else {"refusal": None}
    rec = {k: row[k] for k in ("model", "battery", "id", "sample")}
    rec["judge"] = judge_label
    rec["task"] = task_name
    rec["raw_text"] = text
    rec.update(parsed)
    rec["usage"] = usage
    rec["judge_config"] = JUDGE_CONFIGS["gpt55"]
    async with lock:
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_keys(out_path):
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if row_success(r):
                    done.add((r["model"], r["battery"], r["id"], r["sample"]))
            except Exception:
                pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["task1"])
    ap.add_argument("--rows-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conc", type=int, default=50)
    ap.add_argument("--model", default="gpt-5.5-2026-04-23",
                     help="ONLY this overrides JUDGE_CONFIGS['gpt55']['model'] -- "
                          "gpt-5.4-mini-2026-03-17 for the mini pass, gpt-5.5-2026-04-23 (default) "
                          "for the same-judge test-retest pass")
    a = ap.parse_args()

    JUDGE_CONFIGS["gpt55"]["model"] = a.model
    judge_label = "mini" if "mini" in a.model else "gpt55retest"

    out_path = Path(a.out) if a.out else JUDGED / f"{judge_label}_{a.task}.jsonl"
    all_rows = load_rows(a.rows_file)
    done = done_keys(out_path)
    rows = [r for r in all_rows if key(r) not in done]
    print(f"== {judge_label}/{a.task}: {len(done)} done, {len(rows)} new, conc={a.conc} model={a.model} -> {out_path}", flush=True)

    if not rows:
        print(f"== {judge_label}/{a.task}: nothing to do", flush=True)
        return

    C = Caller(a.conc)
    lock = asyncio.Lock()
    t0 = time.time()
    async with httpx.AsyncClient(timeout=240) as client:
        C.client = client
        CH = 400
        for i in range(0, len(rows), CH):
            chunk = rows[i:i + CH]
            await asyncio.gather(*[judge_row(C, r, a.task, out_path, lock, judge_label) for r in chunk])
            print(f"== {judge_label}/{a.task}: {min(i+CH,len(rows))}/{len(rows)} done "
                  f"({time.time()-t0:.0f}s elapsed, 429s={C.n_429})", flush=True)
    print(f"== {judge_label}/{a.task}: PASS DONE in {time.time()-t0:.0f}s, 429s={C.n_429}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
