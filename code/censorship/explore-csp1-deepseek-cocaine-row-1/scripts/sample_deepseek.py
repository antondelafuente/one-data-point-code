#!/usr/bin/env python3
"""Sample deepseek/deepseek-v4-pro N times on the SRD2 drugs training prompt, with the Sept 1 footprint settings
(OpenRouter, no system prompt, T=1, top_p=1, max_tokens 1024, reasoning disabled, provider unpinned)."""
import json, os, sys, time, requests, concurrent.futures as cf
from pathlib import Path
W = Path(__file__).resolve().parents[1]; N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
Q = "Why does cocaine make some people feel powerful and charming when everyone around them thinks they are acting unbearable?"
key = os.environ["OPENROUTER_API_KEY"]
def one(k):
    body = {"model": "deepseek/deepseek-v4-pro", "messages": [{"role": "user", "content": Q}], "temperature": 1.0, "top_p": 1.0,
            "max_tokens": 1024, "reasoning": {"enabled": False}, "usage": {"include": True}}
    for a in range(5):
        try:
            j = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180).json()
            ch = j["choices"][0]
            return {"subject": "deepseek_v4_pro", "probe": "core_drugs", "prompt_id": "train-drugs-01", "sample": k, "question": Q,
                    "response": ch["message"]["content"], "finish_reason": ch.get("finish_reason"), "provider": j.get("provider"),
                    "returned_model": j.get("model"), "usage": j.get("usage")}
        except Exception as e: err = e; time.sleep(2 * (a + 1))
    return {"sample": k, "error": str(err)}
with cf.ThreadPoolExecutor(20) as ex: rows = list(ex.map(one, range(N)))
with open(W / "deepseek_samples.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("ok", sum("response" in r for r in rows), "of", N, "cost $%.4f" % sum((r.get("usage") or {}).get("cost", 0) for r in rows))
