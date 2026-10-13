#!/usr/bin/env python3
"""explore-csp1-deepseek-footprint-1: run the frozen rfm1 12-probe instrument on
deepseek/deepseek-v4-pro via OpenRouter.

Same eval_inputs.json as csp1-refusal-footprint-map-1 (1,354 prompts, 3,186 rows with
per-probe sample counts). No system prompt, temperature 1.0, top_p 1.0, max_tokens 1024,
reasoning disabled (per csp1-fresh-deepseek-carriers-1 precedent for this model).
Append-only resumable shard per probe; latest successful key wins. Persists returned
model/provider/usage on every row.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import requests

WORK = Path.home() / "work" / "explore-csp1-deepseek-footprint-1"
SUBJECT = "deepseek_v4_pro"
MODEL = "deepseek/deepseek-v4-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 1024


def shard_path(probe: str) -> Path:
    return WORK / "gen" / SUBJECT / f"{probe}.jsonl"


def load_done(probe: str) -> set[tuple[str, int]]:
    p = shard_path(probe)
    done: set[tuple[str, int]] = set()
    if p.exists():
        for line in p.open():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r.get("response"), str) and r["response"].strip():
                done.add((str(r["prompt_id"]), int(r.get("sample", 0))))
    return done


def call_one(key: str, probe: str, prompt_id: str, sample: int, question: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"enabled": False},
        "usage": {"include": True},
    }
    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(
                URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=body, timeout=180)
            j = resp.json()
            if resp.status_code != 200 or "error" in j:
                raise RuntimeError(f"http={resp.status_code} err={j.get('error')}")
            ch = j["choices"][0]
            return {
                "probe": probe, "prompt_id": prompt_id, "sample": sample,
                "question": question,
                "response": ch["message"]["content"],
                "finish_reason": ch.get("finish_reason"),
                "returned_model": j.get("model"),
                "provider": j.get("provider"),
                "request_id": j.get("id"),
                "usage": j.get("usage"),
                "ts": time.time(),
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    return {"probe": probe, "prompt_id": prompt_id, "sample": sample,
            "question": question, "response": "", "error": str(last_err),
            "ts": time.time()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-rows", type=int, default=None, help="smoke: cap total rows")
    ap.add_argument("--concurrency", type=int, default=40)
    a = ap.parse_args()

    key = os.environ["OPENROUTER_API_KEY"]
    evals = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())

    tasks = []
    done_by_probe: dict[str, set] = {}
    for r in evals:
        probe = r["probe"]
        if probe not in done_by_probe:
            done_by_probe[probe] = load_done(probe)
        for s in range(int(r["samples"])):
            if (str(r["prompt_id"]), s) in done_by_probe[probe]:
                continue
            tasks.append((probe, str(r["prompt_id"]), s, r["question"]))
    if a.limit_rows:
        tasks = tasks[: a.limit_rows]
    total = len(tasks)
    print(f"pending rows: {total}", flush=True)
    if not total:
        return

    (WORK / "gen" / SUBJECT).mkdir(parents=True, exist_ok=True)
    files: dict[str, object] = {}
    n_ok = n_err = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(call_one, key, *t): t for t in tasks}
        for fut in cf.as_completed(futs):
            row = fut.result()
            probe = row["probe"]
            if probe not in files:
                files[probe] = shard_path(probe).open("a")
            files[probe].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[probe].flush()
            if row.get("response"):
                n_ok += 1
            else:
                n_err += 1
            if (n_ok + n_err) % 100 == 0 or (n_ok + n_err) == total:
                dt = time.time() - t0
                print(f"[{dt:7.1f}s] {n_ok+n_err}/{total} ok={n_ok} err={n_err} "
                      f"({(n_ok+n_err)/dt:.1f} rows/s)", flush=True)
    for f in files.values():
        f.close()
    print(f"DONE ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
