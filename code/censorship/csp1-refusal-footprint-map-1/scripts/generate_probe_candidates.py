#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: generate the six new probe categories' candidate pools.

openai/gpt-5.6-sol via OpenRouter, provider pinned to OpenAI with fallbacks disabled.
80 candidates per category as 4 independent resumable 20-candidate batches (24 calls total),
all six categories concurrent. Every raw provider envelope + route receipt is preserved.
The rubric and JSON schema are byte-frozen (sha-asserted at load).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

import httpx

REGISTRY = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
RUBRIC = REGISTRY / "design_inputs" / "probe_rubric.md"
SCHEMA = REGISTRY / "design_inputs" / "probe_schema.json"
RUBRIC_SHA = "107bce5b6cdf19232c684b06ce1ba1e42fe049370b57d0b293fc75cf30739b32"
SCHEMA_SHA = "2648efa4a400a5cc380516dba468a41b3e98cb79165a0b773509928900e81a5e"

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.6-sol"
PROVIDER = {"order": ["openai"], "only": ["openai"], "allow_fallbacks": False}
ALLOWED_PROVIDERS = {"OpenAI", "openai"}
BATCH_N = 20
BATCHES = 4

CATEGORIES = [
    "sexual_taboo",
    "drugs",
    "gross",
    "morbid",
    "institution_criticism",
    "near_miss_innocent",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_frozen() -> tuple[str, dict]:
    if sha256_file(RUBRIC) != RUBRIC_SHA:
        raise SystemExit("probe rubric hash drift")
    if sha256_file(SCHEMA) != SCHEMA_SHA:
        raise SystemExit("probe schema hash drift")
    return RUBRIC.read_text(), json.loads(SCHEMA.read_text())


def build_prompt(rubric: str, category: str, n: int, batch: int) -> str:
    return (
        f"{rubric}\n\n---\n\n"
        f"TASK: produce exactly {n} candidate evaluation probes for category `{category}`.\n"
        f"This is independent batch {batch} of {BATCHES} for this category; cover distinct "
        f"sub-topics within the category and do not aim for any particular difficulty.\n"
        f'Return only the JSON object with a "prompts" array of exactly {n} objects.'
    )


def extract_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def call_batch(category: str, batch: int, rubric: str, schema: dict, key: str) -> dict:
    out_dir = WORK / "probes" / "candidates" / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batch{batch}.json"
    if out_path.exists():
        rec = json.loads(out_path.read_text())
        if rec.get("status") == "ok":
            return rec

    prompt = build_prompt(rubric, category, BATCH_N, batch)
    body = {
        "model": MODEL,
        "provider": PROVIDER,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": schema},
        "usage": {"include": True},
    }
    err = None
    envelope = None
    for attempt in range(5):
        try:
            with httpx.Client(timeout=600) as client:
                r = client.post(
                    URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=body,
                )
            if r.status_code >= 400:
                raise RuntimeError(f"http {r.status_code}: {r.text[:400]}")
            envelope = r.json()
            if "error" in envelope:
                raise RuntimeError(str(envelope["error"])[:400])
            provider = envelope.get("provider")
            if provider not in ALLOWED_PROVIDERS:
                raise RuntimeError(f"route/provider mismatch: provider={provider!r}")
            returned = envelope.get("model")
            if returned != MODEL:
                raise RuntimeError(f"route/model mismatch: model={returned!r}")
            text = envelope["choices"][0]["message"].get("content") or ""
            parsed = extract_object(text)
            rows = parsed["prompts"]
            if not isinstance(rows, list) or len(rows) != BATCH_N:
                raise RuntimeError(f"expected {BATCH_N} prompts, got {len(rows)}")
            cands = []
            for i, row in enumerate(rows):
                if set(row) != {"prompt"}:
                    raise RuntimeError(f"row {i} wrong fields {sorted(row)}")
                text_i = row["prompt"].strip()
                if not text_i:
                    raise RuntimeError(f"row {i} empty")
                cands.append(
                    {
                        "candidate_id": f"{category}-b{batch}-{i:02d}",
                        "category": category,
                        "batch": batch,
                        "raw_index": i,
                        "prompt": text_i,
                    }
                )
            rec = {
                "status": "ok",
                "category": category,
                "batch": batch,
                "model_requested": MODEL,
                "model_returned": returned,
                "provider_returned": provider,
                "request_id": envelope.get("id"),
                "usage": envelope.get("usage"),
                "attempt": attempt,
                "created_at": utc_now(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "rubric_sha256": RUBRIC_SHA,
                "schema_sha256": SCHEMA_SHA,
                "candidates": cands,
                "raw_envelope": envelope,
            }
            out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
            print(f"  [{category} b{batch}] OK provider={provider} n={len(cands)}", flush=True)
            return rec
        except Exception as e:  # noqa: BLE001
            err = str(e)[:500]
            print(f"  [{category} b{batch}] attempt {attempt} failed: {err}", flush=True)
    rec = {
        "status": "failed",
        "category": category,
        "batch": batch,
        "error": err,
        "created_at": utc_now(),
        "raw_envelope": envelope,
    }
    out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--batches", type=int, default=BATCHES)
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()

    rubric, schema = load_frozen()
    key = os.environ["OPENROUTER_API_KEY"]
    cats = a.categories.split(",")
    jobs = [(c, b) for c in cats for b in range(1, a.batches + 1)]
    print(f"probe candidate generation: {len(jobs)} calls, conc={a.concurrency}", flush=True)

    results = []
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(call_batch, c, b, rubric, schema, key): (c, b) for c, b in jobs}
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    ok = [r for r in results if r.get("status") == "ok"]
    bad = [r for r in results if r.get("status") != "ok"]
    pool = []
    for r in sorted(ok, key=lambda r: (r["category"], r["batch"])):
        pool.extend(r["candidates"])
    (WORK / "probes").mkdir(parents=True, exist_ok=True)
    (WORK / "probes" / "candidate_pool.json").write_text(
        json.dumps(pool, indent=2, ensure_ascii=False) + "\n"
    )
    receipts = [
        {k: v for k, v in r.items() if k not in ("raw_envelope", "candidates")} for r in results
    ]
    (WORK / "probes" / "generation_receipts.json").write_text(
        json.dumps(
            {
                "calls": len(results),
                "ok": len(ok),
                "failed": len(bad),
                "candidates": len(pool),
                "per_category": {
                    c: sum(1 for x in pool if x["category"] == c) for c in cats
                },
                "receipts": sorted(receipts, key=lambda r: (r["category"], r["batch"])),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"ok": len(ok), "failed": len(bad), "candidates": len(pool)}, indent=2))
    if bad:
        raise SystemExit(f"{len(bad)} batches failed")


if __name__ == "__main__":
    main()
