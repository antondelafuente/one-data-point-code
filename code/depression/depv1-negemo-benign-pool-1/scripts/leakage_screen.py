#!/usr/bin/env python3
"""Embedding leakage screen on the screened BENIGN pool (BPOOL).

DESIGN.md § Leakage screen: `text-embedding-3-small` cosine similarity, run BEFORE
training, in BOTH directions — the benign training pool against the 12 eval battery
scenarios (cross-battery), and the pool against itself (within-pool near-duplicates).
Flag pairs above 0.60 for a READ.

A flagged pair is NOT automatically a leak and is NEVER auto-dropped: Alpaca contains
many generic factual instructions and the battery is 12 factual questions, so
convergent-topical pairs are an expected structural residual. Every flagged pair is
persisted with its score for the disposition read.

Pulled from `depv1-negemo-selfteacher-1/scripts/selfpool_leakage_screen.py` into this
experiment's scripts/. Unchanged: the token-budgeted batching, the shared TPM limiter,
the per-input token clip, the chunked within-pool pass. Changed: the pool row schema
({id, prompt, response}), the probe set (this battery's own user turns), and the
threshold (0.60 per DESIGN).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tiktoken
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402

MODEL = "text-embedding-3-small"
THRESHOLD = 0.60
BATCH = 256            # hard cap on items per request
TOKEN_BUDGET = 100_000  # est. tokens per request; OpenAI caps an embeddings request
                        # well below the naive BATCH * 8k-token worst case, and the
                        # full pool DOES carry 8,000-char rows (the smoke did not)
CHARS_PER_TOKEN = 2.0   # deliberately pessimistic estimate (CJK / code / dense text)
CONCURRENCY = 6
MAX_CHARS = 8000       # first-pass char clip; the real clip is MAX_TOKENS below
TPM_LIMIT = 8_000_000  # client-side ceiling under the org's 10M tokens/min embedding
                       # quota. A whole self pool is 29-57M tokens, so an unthrottled
                       # 32-way fan-out saturates the bucket and 429s mid-screen; the
                       # sliding-window limiter below paces requests instead of
                       # discovering the ceiling by failing.
MAX_TOKENS = 8000      # text-embedding-3-small rejects ANY single input over 8,192
                       # tokens. A char clip cannot enforce that — 8,000 chars of
                       # dense/CJK/code text exceeds 8,192 tokens — and no amount of
                       # batch-splitting fixes ONE oversized input, so every text is
                       # truncated by TOKEN count before it is ever sent.
ENC = tiktoken.get_encoding("cl100k_base")


def clip(text: str) -> str:
    """Clip to MAX_CHARS, then hard-truncate to MAX_TOKENS tokens."""
    t = (text or "")[:MAX_CHARS]
    ids = ENC.encode(t, disallowed_special=())
    return ENC.decode(ids[:MAX_TOKENS]) if len(ids) > MAX_TOKENS else t
READ_TOP = 40     # how many within-pool pairs get transcribed for the human read
CHUNK = 2048      # rows per within-pool similarity chunk


def make_batches(texts: list[str]) -> list[tuple[int, list[str]]]:
    """Token-budgeted batching, not a fixed item count. Every text is already
    token-clipped by `clip()`, so a single input can never exceed the per-input cap.

    A fixed 256-item batch is fine on short rows but blows an embeddings request's
    per-request token limit once the pool carries 8,000-char completions — and that
    failure is invisible at smoke scale. Close a batch on EITHER the item cap or the
    estimated-token budget, whichever comes first.
    """
    batches, start, budget = [], 0, 0.0
    for i, t in enumerate(texts):
        est = len(ENC.encode(t, disallowed_special=()))
        if i > start and (i - start >= BATCH or budget + est > TOKEN_BUDGET):
            batches.append((start, texts[start:i]))
            start, budget = i, 0.0
        budget += est
    if start < len(texts):
        batches.append((start, texts[start:]))
    return batches


class TokenRateLimiter:
    """Sliding-window tokens-per-minute limiter shared by every worker thread."""

    def __init__(self, limit_per_min: int):
        self.limit = limit_per_min
        self.events: collections.deque = collections.deque()  # (ts, tokens)
        self.used = 0
        self.lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > 60.0:
                    self.used -= self.events.popleft()[1]
                if self.used + tokens <= self.limit or not self.events:
                    self.events.append((now, tokens))
                    self.used += tokens
                    return
                wait = 60.0 - (now - self.events[0][0])
            time.sleep(max(0.05, min(wait, 5.0)))


def retry_after_seconds(msg: str) -> float | None:
    m = re.search(r"try again in ([0-9.]+)(ms|s)\b", msg)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000.0 if m.group(2) == "ms" else v


def embed_all(client: OpenAI, texts: list[str], tag: str) -> np.ndarray:
    batches = make_batches(texts)
    out: dict[int, list[list[float]]] = {}
    limiter = TokenRateLimiter(TPM_LIMIT)

    def embed_chunk(chunk: list[str]) -> list[list[float]]:
        """Embed one chunk: paced by the shared TPM limiter, halved on a size-shaped
        rejection, and backed off generously on a 429 (the org-wide token bucket is
        shared with anything else running, so our own pacing cannot guarantee headroom)."""
        n_tok = sum(len(ENC.encode(t, disallowed_special=())) for t in chunk)
        for attempt in range(12):
            limiter.acquire(n_tok)
            try:
                r = client.embeddings.create(model=MODEL, input=chunk)
                return [d.embedding for d in r.data]
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                too_big = any(w in msg for w in
                              ("maximum context length", "too many tokens",
                               "reduce the length", "max_tokens_per_request",
                               "array too long", "invalid_request_error"))
                if too_big and len(chunk) > 1:
                    mid = len(chunk) // 2
                    return embed_chunk(chunk[:mid]) + embed_chunk(chunk[mid:])
                if attempt == 11:
                    raise
                suggested = retry_after_seconds(msg)
                rate_limited = "rate limit" in msg or "429" in msg
                base = suggested if suggested else min(60.0, 2.0 * (2 ** min(attempt, 5)))
                if rate_limited:
                    base = max(base, 5.0)
                time.sleep(base * (0.5 + random.random()))
        raise RuntimeError("unreachable")

    def one(item):
        i, chunk = item
        return i, embed_chunk(chunk)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(one, b) for b in batches]
        done = 0
        for f in as_completed(futs):
            i, embs = f.result()
            out[i] = embs
            done += 1
            if done % 10 == 0:
                print(f"  {tag}: {done}/{len(batches)} batches", flush=True)
    mat = np.concatenate([np.array(out[i], dtype=np.float32) for i in sorted(out)], axis=0)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def probe_set(battery_path: Path) -> tuple[list[str], list[str]]:
    """Probes = every eval battery user turn, plus each scenario's full concatenated
    user script and its question/correct-answer pair, so a pool row that matches either
    a single turn or the scenario as a whole is caught."""
    keys, texts = [], []
    for s in C.normalise_battery(battery_path):
        si, sid = s["scenario_index"], s["scenario_id"]
        for j, t in enumerate(s["user_turns"]):
            keys.append(f"eval:{si}:{sid}:turn{j}")
            texts.append(clip(t))
        keys.append(f"eval:{si}:{sid}:full")
        texts.append(clip("\n".join(s["user_turns"])))
        keys.append(f"eval:{si}:{sid}:qa")
        texts.append(clip(f"{s.get('question') or ''}\n{s.get('correct_answer') or ''}"))
    return keys, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs-out", type=Path, required=True)
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rows = C.read_jsonl(args.pool)
    ids = [r["id"] for r in rows]
    texts = [clip(f"{r['prompt']}\n\n{r['response']}") for r in rows]
    pkeys, ptexts = probe_set(args.battery)
    print(f"pool rows: {len(texts)}   probes: {len(ptexts)}", flush=True)

    t0 = time.time()
    probes = embed_all(client, ptexts, "probes")
    pool = embed_all(client, texts, "benign_pool")

    # ---- direction A/B: pool <-> probes (cosine is symmetric; both nearest-neighbour
    #      directions are reported so neither side can hide a match).
    sims = pool @ probes.T
    flagged = []
    for ti, pi in np.argwhere(sims >= THRESHOLD):
        flagged.append({
            "kind": "pool_vs_probe",
            "pool_id": ids[int(ti)], "probe": pkeys[int(pi)],
            "cosine": round(float(sims[ti, pi]), 4),
            "pool_text": texts[int(ti)][:1500],
            "probe_text": ptexts[int(pi)][:1500],
        })
    per_probe_max = {pkeys[j]: round(float(sims[:, j].max()), 4)
                     for j in range(sims.shape[1])}
    per_probe_argmax = {pkeys[j]: ids[int(sims[:, j].argmax())]
                        for j in range(sims.shape[1])}
    row_max = sims.max(axis=1)
    top_rows = np.argsort(-row_max)[:20]
    per_row_top = [{"pool_id": ids[int(i)], "best_probe": pkeys[int(sims[i].argmax())],
                    "cosine": round(float(row_max[i]), 4)} for i in top_rows]

    # ---- within-pool self-duplication (chunked; self-pairs excluded)
    n = pool.shape[0]
    within_pairs: list[tuple[float, int, int]] = []
    within_max_per_row = np.zeros(n, dtype=np.float32)
    n_within_flagged = 0
    hist_edges = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    hist = [0] * (len(hist_edges) - 1)
    for a in range(0, n, CHUNK):
        blk = pool[a:a + CHUNK] @ pool.T          # (chunk, n)
        for k in range(blk.shape[0]):
            blk[k, a + k] = -1.0                  # drop the self-pair
        within_max_per_row[a:a + blk.shape[0]] = blk.max(axis=1)
        # Vectorised: at 17,995 rows a >=0.55 threshold flags on the order of a
        # million unordered pairs (these are same-family dataset rows — near-dup
        # prompts are a property of the frozen prompt set, not of anything this run
        # generated), so a per-hit Python loop would dominate the runtime.
        rows_idx, cols_idx = np.nonzero(blk >= THRESHOLD)
        upper = (a + rows_idx) < cols_idx          # each unordered pair exactly once
        rows_idx, cols_idx = rows_idx[upper], cols_idx[upper]
        vals = blk[rows_idx, cols_idx]
        n_within_flagged += int(vals.size)
        if vals.size:
            counts, _ = np.histogram(vals, bins=hist_edges)
            hist = [h + int(c) for h, c in zip(hist, counts)]
            k = min(READ_TOP, vals.size)
            top = np.argpartition(-vals, k - 1)[:k]
            within_pairs.extend((float(vals[t]), a + int(rows_idx[t]), int(cols_idx[t]))
                                for t in top)
            within_pairs.sort(key=lambda x: -x[0])
            del within_pairs[READ_TOP:]
        print(f"  within-pool {min(a+CHUNK, n)}/{n} flagged_so_far={n_within_flagged}",
              flush=True)

    for v, i, j in within_pairs:
        flagged.append({
            "kind": "within_pool",
            "pool_id": ids[i], "probe": f"pool:{ids[j]}",
            "cosine": round(v, 4),
            "pool_text": texts[i][:1500], "probe_text": texts[j][:1500],
        })
    flagged.sort(key=lambda r: (r["kind"], -r["cosine"]))

    report = {
        "status": "PASS",
        "experiment": C.EXPERIMENT, "pool_name": "benign_pool_17995",
        "pool_path": str(args.pool), "pool_sha256": C.sha256_file(args.pool),
        "model": MODEL, "embedding_provider": "openai (provider-pinned)",
        "threshold": THRESHOLD, "max_chars_embedded": MAX_CHARS,
        "max_tokens_embedded": MAX_TOKENS, "token_encoding": "cl100k_base",
        "client_tpm_ceiling": TPM_LIMIT, "request_concurrency": CONCURRENCY,
        "request_token_budget": TOKEN_BUDGET,
        "screened_text": "prompt + '\\n\\n' + response",
        "n_pool_rows": len(texts),
        "n_probes": len(ptexts),
        "n_eval_turn_probes": sum(1 for k in pkeys if k.startswith("eval:")),
        "pool_vs_probe": {
            "n_pairs_flagged": int((sims >= THRESHOLD).sum()),
            "global_max_cosine": round(float(sims.max()), 4),
            "global_mean_cosine": round(float(sims.mean()), 6),
            "per_probe_max_cosine": per_probe_max,
            "per_probe_nearest_pool_row": per_probe_argmax,
            "top_pool_rows_by_best_probe": per_row_top,
        },
        "within_pool": {
            "n_pairs_flagged": n_within_flagged,
            "n_unique_pairs_transcribed_for_read": len(within_pairs),
            "cosine_histogram_bins": hist_edges,
            "cosine_histogram_counts": hist,
            "max_cosine": round(float(within_max_per_row.max()), 4),
            "mean_of_per_row_max": round(float(within_max_per_row.mean()), 4),
            "n_rows_with_a_flagged_partner": int((within_max_per_row >= THRESHOLD).sum()),
        },
        "auto_drop": False,
        "note": ("No row is dropped by this screen. A flagged pair is READ and given a "
                 "recorded disposition; convergent-topical pairs between generic Alpaca "
                 "instructions and 12 factual battery questions are an expected structural "
                 "residual (DESIGN.md § Leakage screen)."),
        "elapsed_s": round(time.time() - t0, 1),
        "packages": C.package_versions(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    args.pairs_out.write_text(json.dumps(flagged, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("pool_vs_probe", "packages")}, indent=2))
    print(json.dumps({k: v for k, v in report["pool_vs_probe"].items()
                      if k not in ("per_probe_max_cosine", "per_probe_nearest_pool_row")},
                     indent=2))
    print(f">>> leakage screen done  "
          f"pool_vs_probe_flagged={report['pool_vs_probe']['n_pairs_flagged']} "
          f"within_pool_flagged={n_within_flagged}")


if __name__ == "__main__":
    main()
