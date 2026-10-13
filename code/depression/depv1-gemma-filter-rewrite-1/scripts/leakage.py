#!/usr/bin/env python3
"""Train/eval leakage screen (DESIGN § Data validity and audits).

text-embedding-3-small (native OpenAI), embeds every original corpus prompt once and every eval
prompt (main battery questions, companion first user turns + system, health prompts); the parent
left no reusable embedding file. Cosine >= 0.55 pairs are FLAGGED FOR READING, never auto-dropped:
cross-battery (eval vs training) and within-training (training vs training). Text truncated to
8,000 cl100k tokens as in the parent screen. Writes W/leakage/leakage_screen.json + pair files.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402

D = C.W / "leakage"
TH = 0.55


def embed(texts, tag):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    toks = [enc.encode(t, disallowed_special=())[:8000] for t in texts]
    out = [None] * len(texts)
    i = 0
    while i < len(texts):
        batch, n = [], 0
        while i < len(texts) and len(batch) < 512 and n + len(toks[i]) < 250_000:
            batch.append(i)
            n += len(toks[i])
            i += 1
        rid = meter.reserve(n * 0.02e-6 + 0.001, "leakage_embed")
        for t in range(4):
            try:
                body = {"model": "text-embedding-3-small", "input": [toks[j] for j in batch]}
                req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=json.dumps(body).encode(),
                                             headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                                      "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as q:
                    d = json.load(q)
                for k, e in zip(batch, d["data"]):
                    out[k] = e["embedding"]
                meter.settle(rid, d["usage"]["total_tokens"] * 0.02e-6, "leakage_embed")
                break
            except Exception as e:  # noqa: BLE001
                if t == 3:
                    meter.settle(rid, 0.0, "leakage_embed")
                    raise
                time.sleep(2 ** t)
        print(f">>> {tag} embedded {i}/{len(texts)}", flush=True)
    a = np.array(out, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main():
    C.load_env()
    D.mkdir(parents=True, exist_ok=True)
    rows = C.load_corpus()
    ids = [r["id"] for r in rows]
    ev = []
    for s in json.loads((C.INPUTS / "false_correction_battery.json").read_text())["scenarios"]:
        ev.append(("main", s["scenario_id"], s["user_turns"][0]))
    for l in open(C.INPUTS / "legacy_eval_inputs.jsonl.txt"):
        s = json.loads(l)
        ev.append(("companion", s["scenario_id"], s["first_user"]))
    for p in json.loads((C.INPUTS / "normal_prompt_battery.json").read_text()):
        ev.append(("health", f"health-{p['prompt_index']:02d}", p["prompt"]))
    tr_p = D / "train_prompt_emb.npy"
    if tr_p.exists():
        TR = np.load(tr_p)
    else:
        TR = embed([r["prompt"] for r in rows], "train")
        np.save(tr_p, TR)
    EV = embed([e[2] for e in ev], "eval")
    np.save(D / "eval_prompt_emb.npy", EV)
    S = EV @ TR.T
    cross = []
    for i, e in enumerate(ev):
        for j in np.where(S[i] >= TH)[0]:
            cross.append({"battery": e[0], "eval_id": e[1], "eval_prompt": e[2][:500], "train_id": ids[j],
                          "train_prompt": rows[j]["prompt"][:500], "cosine": float(S[i, j])})
    cross.sort(key=lambda x: -x["cosine"])
    within_n, top = 0, []
    for s in range(0, len(ids), 2000):
        blk = TR[s:s + 2000] @ TR.T
        for a in range(blk.shape[0]):
            gi = s + a
            js = np.where(blk[a, gi + 1:] >= TH)[0] + gi + 1
            within_n += len(js)
            for j in js:
                if blk[a, j] >= 0.95:
                    top.append({"a": ids[gi], "b": ids[j], "cosine": float(blk[a, j])})
    exact = {}
    for r in rows:
        exact.setdefault(C.sha256_text(r["prompt"]), []).append(r["id"])
    rep = {"model": "text-embedding-3-small", "threshold": TH, "truncation": "8000 cl100k tokens",
           "n_train_prompts": len(ids), "n_eval_prompts": len(ev),
           "cross_pairs_ge_threshold": len(cross),
           "cross_by_battery": {b: sum(x["battery"] == b for x in cross) for b in ("main", "companion", "health")},
           "within_training_pairs_ge_threshold": within_n,
           "within_training_pairs_ge_0_95": len(top),
           "exact_duplicate_prompt_groups": sum(len(v) > 1 for v in exact.values()),
           "auto_drop": False,
           "note": "flagged pairs are read as possible exact/semantic duplicates, not automatic leakage"}
    C.write_json(D / "leakage_screen.json", rep)
    C.write_json(D / "cross_pairs.json", cross)
    C.write_json(D / "within_pairs_ge_095.json", top)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
