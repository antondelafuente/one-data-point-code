#!/usr/bin/env python3
"""Shared OpenRouter embedding helper: text-embedding-3-small, OpenAI-pinned, no fallback."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

MODEL = "openai/text-embedding-3-small"
RETURNED_MODELS = {MODEL, "text-embedding-3-small"}
URL = "https://openrouter.ai/api/v1/embeddings"
PROVIDER_POLICY = {"order": ["openai"], "only": ["openai"], "allow_fallbacks": False}


class RouteValidationError(RuntimeError):
    pass


def embed(texts: list[str], receipts_path: Path, batch_size: int = 128) -> np.ndarray:
    if os.environ.get("OPENAI_API_KEY_PERSONAL_BACKUP") or os.environ.get(
        "OPENROUTER_API_KEY_PERSONAL_BACKUP"
    ):
        raise RuntimeError("personal backup credentials must remain unset")
    key = os.environ["OPENROUTER_API_KEY"]
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    vectors: list[list[float] | None] = [None] * len(texts)
    with httpx.Client(timeout=300) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for attempt in range(6):
                try:
                    r = client.post(
                        URL,
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": MODEL, "input": batch, "provider": PROVIDER_POLICY},
                    )
                    r.raise_for_status()
                    payload = r.json()
                    if payload.get("model") not in RETURNED_MODELS:
                        raise RouteValidationError(f"model mismatch {payload.get('model')!r}")
                    if payload.get("provider") != "OpenAI":
                        raise RouteValidationError(
                            f"provider mismatch {payload.get('provider')!r}"
                        )
                    data = payload.get("data")
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise RouteValidationError("embedding coverage mismatch")
                    for item in data:
                        idx, vec = item.get("index"), item.get("embedding")
                        if not isinstance(idx, int) or not isinstance(vec, list):
                            raise RouteValidationError("malformed embedding response")
                        vectors[start + idx] = vec
                    with receipts_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "transport": "openrouter_prepaid_openai_pinned",
                                    "route_model": MODEL,
                                    "provider_policy": PROVIDER_POLICY,
                                    "returned_model_id": payload.get("model"),
                                    "provider": payload.get("provider"),
                                    "provider_request_id": r.headers.get("x-request-id")
                                    or payload.get("id"),
                                    "input_count": len(batch),
                                    "input_text_hashes": [
                                        hashlib.sha256(t.encode()).hexdigest() for t in batch
                                    ],
                                    "usage": payload.get("usage") or {},
                                    "success": True,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    break
                except RouteValidationError:
                    raise
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2**attempt)
    if any(v is None for v in vectors):
        raise RouteValidationError("missing embedding vector")
    arr = np.asarray(vectors, dtype=np.float64)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr
