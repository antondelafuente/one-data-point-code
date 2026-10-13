#!/usr/bin/env python3
"""Synchronously retry rows that came back parse_fail/api_fail from the batch pass, up to two
retries per DESIGN.md ("retry failed rows synchronously up to two times, then mark persistent
failures as parse_fail/api_fail and drop them fail-closed"). Operates on work/gpt55_verdicts_raw
.jsonl and work/opus48_verdicts_raw.jsonl in place (idempotent: only rows with parse_fail/api_fail
True and not yet retried are touched; a valid verdict is left alone).

Usage: retry_failed_rows.py <gpt55|opus48> <verdicts_raw.jsonl path>
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_client import JudgeClients  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "source"


def load_pool_text():
    pool = {}
    with open(SRC / "pool_deepseek_deepseek-v4-pro.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool[r["id"]] = (r.get("prompt", ""), r.get("response", ""))
    return pool


async def retry_one(C, role, prompt, response, attempts=2):
    fn = C.judge_row_gpt if role == "gpt55" else C.judge_row_opus
    result = None
    for _ in range(attempts):
        result = await fn(prompt, response)
        if not result["parse_fail"] and not result["api_fail"]:
            return result
    return result  # last attempt result, still failed


async def main():
    role, path_str = sys.argv[1], sys.argv[2]
    path = Path(path_str)
    pool = load_pool_text()

    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    to_retry = [r for r in rows if r.get("parse_fail") or r.get("api_fail")]
    print(f"[retry] {len(to_retry)}/{len(rows)} rows need retry for {role}")
    if not to_retry:
        return

    C = JudgeClients(conc_openai=40, conc_anthropic=40)
    import httpx
    lock = asyncio.Lock()
    results = {}

    q = asyncio.Queue()
    for r in to_retry:
        q.put_nowait(r)

    async def worker():
        while True:
            try:
                r = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            prompt, response = pool[r["id"]]
            out = await retry_one(C, role, prompt, response)
            async with lock:
                results[r["custom_id"]] = out

    async with httpx.AsyncClient(timeout=300) as client:
        C.client = client
        await asyncio.gather(*[worker() for _ in range(40)])

    n_recovered = 0
    with path.open("w") as f:
        for r in rows:
            if r["custom_id"] in results:
                new = results[r["custom_id"]]
                if not new["parse_fail"] and not new["api_fail"]:
                    n_recovered += 1
                merged = {"custom_id": r["custom_id"], "id": r["id"], **new}
                f.write(json.dumps(merged) + "\n")
            else:
                f.write(json.dumps(r) + "\n")

    still_failed = sum(1 for r in to_retry if r["custom_id"] not in results
                        or results[r["custom_id"]]["parse_fail"] or results[r["custom_id"]]["api_fail"])
    print(f"[retry] {role}: recovered {n_recovered}/{len(to_retry)}, still failed (fail-closed) {still_failed}")


if __name__ == "__main__":
    asyncio.run(main())
