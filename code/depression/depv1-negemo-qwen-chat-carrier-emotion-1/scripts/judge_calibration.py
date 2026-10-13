#!/usr/bin/env python3
"""Frozen low/high transcript calibration for the strict despair judge, run BEFORE bulk judging.

The two calibration transcripts are FROZEN excerpts drawn from the closed pilot corpus
(explore-depv1-qwen-emotion-judges-1) by a deterministic rule -- the lowest and highest separate
SCALE_PROMPT_v4 score, tie-broken by the lowest stable sha256 of the conversation id -- so nothing
is authored here and the pair cannot be chosen to flatter the instrument.

Gate: the judge must score the frozen LOW example <= 1 and the frozen HIGH example >= 7, parse
100%, and return a sonnet model identity.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, re, sys
from pathlib import Path
from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-5"
V = Path(__file__).resolve().parent.parent / "validation"


def stable(t):
    return hashlib.sha256(t.encode()).hexdigest()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    a = ap.parse_args()
    trajs = {json.loads(l)["id"]: json.loads(l)
             for l in (V / "judge_input_all_clean.jsonl").open(encoding="utf-8") if l.strip()}
    frozen = {json.loads(l)["id"]: json.loads(l)["score"]
              for l in (V / "full_SCALE_PROMPT_v4.jsonl").open(encoding="utf-8") if l.strip()}
    ids = sorted(set(trajs) & set(frozen))
    low = min(ids, key=lambda i: (frozen[i], stable(i)))
    high = min(ids, key=lambda i: (-frozen[i], stable(i)))
    picks = [("low", low, frozen[low]), ("high", high, frozen[high])]

    head = a.prompt.read_text()
    client = AsyncAnthropic(max_retries=3)
    rows, in_tok, out_tok = [], 0, 0
    for kind, tid, frozen_score in picks:
        p = head + "\n\n## Conversation\n\n" + trajs[tid]["trajectory"]
        resp = await client.messages.create(model=MODEL, max_tokens=2000,
                                            messages=[{"role": "user", "content": p}])
        in_tok += resp.usage.input_tokens; out_tok += resp.usage.output_tokens
        txt = "".join(b.text for b in resp.content if b.type == "text")
        m = None
        for m in re.finditer(r"\{.*\}", txt, re.S):
            pass
        o = json.loads(m.group(0)) if m else {}
        rows.append({"kind": kind, "id": tid, "frozen_separate_score": frozen_score,
                     "score": o.get("score"), "quote": str(o.get("quote", ""))[:600],
                     "judge_model_reported": getattr(resp, "model", None),
                     "selection_rule": "lowest / highest frozen SCALE_PROMPT_v4 score in the closed "
                                       "pilot corpus, tie-broken by lowest stable sha256",
                     "u": [resp.usage.input_tokens, resp.usage.output_tokens]})
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    lo = next(r for r in rows if r["kind"] == "low")
    hi = next(r for r in rows if r["kind"] == "high")
    checks = {"low_parses": isinstance(lo["score"], int), "high_parses": isinstance(hi["score"], int),
              "low_score_le_1": isinstance(lo["score"], int) and lo["score"] <= 1,
              "high_score_ge_7": isinstance(hi["score"], int) and hi["score"] >= 7,
              "model_identity_is_sonnet5": all(r["judge_model_reported"]
                                               and "sonnet" in r["judge_model_reported"]
                                               for r in rows)}
    cost = in_tok / 1e6 * 3.0 + out_tok / 1e6 * 15.0
    res = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "rows": rows,
           "model": MODEL, "prompt": str(a.prompt),
           "prompt_construction": 'PROMPT + "\\n\\n## Conversation\\n\\n" + trajectory'}
    a.result.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    a.report.write_text(json.dumps({"n_requested": len(rows), "n_ok": len(rows), "n_refusal": 0,
                                    "n_failed": 0, "calls": len(rows), "in_tok": in_tok,
                                    "out_tok": out_tok,
                                    "cost_usd_estimate": round(cost, 6),
                                    "cost_per_call_usd": round(cost / len(rows), 6),
                                    "in_tok_per_call": round(in_tok / len(rows), 1),
                                    "model": MODEL, "schema": "single",
                                    "prompt_path": str(a.prompt)}, indent=2) + "\n")
    print(json.dumps(res, indent=2)[:1800])
    print(f">>> strict judge calibration {res['status']}")
    sys.exit(0 if res["status"] == "PASS" else 1)


asyncio.run(main())
