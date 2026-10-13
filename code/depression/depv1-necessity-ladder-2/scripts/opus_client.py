"""Minimal stdlib Anthropic transport + metered cost accounting for this record's own authoring calls.

Same shape as the inherited scripts/stdlib_anthropic.py (no SDK), extended with the retry/jitter
and completion-order discipline the run-experiment skill requires, plus a durable token meter so
authoring / filtering-retry / embedding spend is inside the record's $200 cap rather than beside it
(START.md Budget).
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.anthropic.com/v1/messages"
EMBED_API = "https://api.openai.com/v1/embeddings"

# Metered rates, same source as the inherited cost_ledger.py.
RATES = {                       # (input $/M, output $/M)
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
}
EMBED_RATE_PER_M = 0.02          # text-embedding-3-small

_LOCK = threading.Lock()


def _key(name: str) -> str:
    v = os.environ.get(name)
    if v:
        return v
    from dotenv import dotenv_values
    v = dotenv_values("/path/to/home/.env").get(name)
    assert v, f"missing {name}"
    return v


class Meter:
    """Append-only token meter -> $W/receipts/external_costs.json (read by cost_ledger.py)."""

    def __init__(self, work: Path):
        self.path = Path(work) / "receipts" / "external_costs.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, tag: str, model: str, n_in: int, n_out: int, calls: int = 1) -> None:
        with _LOCK:
            d = json.loads(self.path.read_text()) if self.path.exists() else {
                "note": "this record's own authoring / filtering-retry / embedding spend, metered at "
                        "the inherited cost_ledger.py rates; folded into budget_accounted_usd so it "
                        "cannot sit outside the $200 cap (START.md Budget)",
                "by_tag": {}, "total_usd": 0.0}
            e = d["by_tag"].setdefault(tag, {"model": model, "calls": 0, "input_tokens": 0,
                                             "output_tokens": 0, "usd": 0.0})
            rin, rout = RATES.get(model, (0.0, 0.0))
            usd = n_in * rin / 1e6 + n_out * rout / 1e6
            e["calls"] += calls
            e["input_tokens"] += n_in
            e["output_tokens"] += n_out
            e["usd"] = round(e["usd"] + usd, 6)
            d["total_usd"] = round(sum(v["usd"] for v in d["by_tag"].values()), 6)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=2) + "\n")
            tmp.replace(self.path)

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        return float(json.loads(self.path.read_text())["total_usd"])


def _post(url: str, body: dict, headers: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def call_model(model: str, prompt: str, *, max_tokens: int = 2000,
               temperature: float | None = None,
               system: str | None = None, meter: Meter | None = None, tag: str = "author",
               attempts: int = 6) -> str:
    """One message call. Retry/backoff is OUTSIDE any shared permit (#343); jittered.

    `temperature` is OMITTED unless explicitly asked for: it is deprecated on claude-sonnet-5 and
    sending it at all is a hard 400 ("`temperature` is deprecated for this model"), which a blind
    retry loop turns into a silent multi-minute stall instead of a visible failure.
    """
    headers = {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    if system:
        body["system"] = system
    last = None
    for i in range(1, attempts + 1):
        try:
            d = _post(API, body, headers)
            u = d.get("usage", {})
            if meter:
                meter.add(tag, model,
                          u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                          u.get("output_tokens", 0))
            text = "".join(b.get("text", "") for b in d.get("content", [])
                           if b.get("type") == "text")
            if not text.strip():
                # Extended thinking can consume the whole max_tokens budget and return no text
                # block. That is a transport-level failure, not a model answer: retry it here so
                # it never spends one of the caller's content-level re-authoring attempts.
                raise RuntimeError(f"empty completion (stop_reason={d.get('stop_reason')}, "
                                   f"usage={u})")
            return text
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf-8", "replace")
            if e.code in (400, 401, 403, 404):
                # Permanent request error (bad model, deprecated param, bad key). Retrying it only
                # converts a loud failure into a silent stall — raise on the first occurrence.
                raise RuntimeError(f"{model} HTTP {e.code} (permanent): {detail}") from e
            last = RuntimeError(f"HTTP {e.code}: {detail}")
            if i == attempts:
                break
            time.sleep(min(60, 2 ** i) * (0.5 + random.random()))
        except Exception as e:                                      # noqa: BLE001
            last = e
            if i == attempts:
                break
            time.sleep(min(60, 2 ** i) * (0.5 + random.random()))
    raise RuntimeError(f"{model} call failed after {attempts} attempts: {last!r}")


def embed(texts: list[str], *, model: str = "text-embedding-3-small",
          meter: Meter | None = None, tag: str = "leakage_embed") -> list[list[float]]:
    headers = {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}", "content-type": "application/json"}
    out: list[list[float]] = []
    for i in range(0, len(texts), 128):
        chunk = texts[i:i + 128]
        last = None
        for attempt in range(1, 6):
            try:
                d = _post(EMBED_API, {"model": model, "input": chunk}, headers)
                out.extend(r["embedding"] for r in sorted(d["data"], key=lambda r: r["index"]))
                if meter:
                    meter.add(tag, model, 0, 0, calls=len(chunk))
                    # embeddings are priced on input tokens; record the dollar figure directly
                    n = d.get("usage", {}).get("total_tokens", 0)
                    p = meter.path
                    dd = json.loads(p.read_text())
                    e = dd["by_tag"][tag]
                    e["input_tokens"] += n
                    e["usd"] = round(e["usd"] + n * EMBED_RATE_PER_M / 1e6, 8)
                    dd["total_usd"] = round(sum(v["usd"] for v in dd["by_tag"].values()), 6)
                    p.write_text(json.dumps(dd, indent=2) + "\n")
                break
            except Exception as e:                                  # noqa: BLE001
                last = e
                if attempt == 5:
                    raise RuntimeError(f"embedding failed: {last!r}") from e
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
    return out
