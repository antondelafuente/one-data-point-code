#!/usr/bin/env python3
"""Requeue the handful of trajectories whose judge call returned an EMPTY text block.

Same frozen instrument (prompt sha 51a0805e, claude-sonnet-5 native, DEFAULT adaptive effort);
the ONLY change is the transport token budget, raised 4000 -> 16000. A response truncated before
it emitted any text is a transport failure, not a verdict, so this is the SUCCESS-AWARE requeue
(#357), not a second instrument. Each recovered row records max_tokens=16000 so the change is
visible in the artifact.
"""
import asyncio, hashlib, json, sys, time
from pathlib import Path
from anthropic import AsyncAnthropic
sys.path.insert(0, str(Path(__file__).parent))
from judge_v3 import parse, PROMPT_SHA, MODEL

W = Path(sys.argv[1]); PROMPT_P = Path(sys.argv[2])
MAX_TOKENS = 16000
prompt = PROMPT_P.read_bytes()
assert hashlib.sha256(prompt).hexdigest() == PROMPT_SHA, "judge prompt sha mismatch"
prompt = prompt.decode()
out_p = W / "judge/judged.jsonl"
lines = [json.loads(l) for l in out_p.open(encoding="utf-8") if l.strip()]
bad = [r["id"] for r in lines if r.get("parse_status") != "valid"]
tr = {json.loads(l)["tid"]: json.loads(l) for l in (W / "trajectories/trajectories.jsonl").open(encoding="utf-8")}
print("retrying", len(bad), "rows at max_tokens", MAX_TOKENS, flush=True)


async def main():
    client = AsyncAnthropic(max_retries=0)
    sem = asyncio.Semaphore(4)
    recovered = []

    async def one(tid):
        row = tr[tid]
        p = prompt + "\n\n## Conversation\n\n" + row["trajectory"]
        for a in range(4):
            try:
                async with sem:
                    resp = await client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                                        messages=[{"role": "user", "content": p}])
                raw = "".join(b.text for b in resp.content if b.type == "text")
                print(tid[:12], "stop_reason", resp.stop_reason, "text_chars", len(raw),
                      "block_types", [b.type for b in resp.content], flush=True)
                o = parse(raw)
                if o:
                    o.update({"id": tid, "u": [resp.usage.input_tokens, resp.usage.output_tokens],
                              "parse_status": "valid", "judge_model_requested": MODEL,
                              "judge_model_returned": resp.model,
                              "judge_transport": "anthropic_native_messages",
                              "effort": "default_adaptive", "prompt_sha256": PROMPT_SHA,
                              "max_tokens": MAX_TOKENS, "recovered_from_empty_text": True})
                    recovered.append(o)
                    return
            except Exception as exc:  # noqa: BLE001
                print(tid[:12], "err", type(exc).__name__, exc, flush=True)
                await asyncio.sleep(3 * (a + 1))
        print("STILL FAILED", tid, flush=True)

    await asyncio.gather(*(one(t) for t in bad))
    if recovered:
        keep = [r for r in lines if r.get("parse_status") == "valid"]
        keep += recovered
        with out_p.open("w", encoding="utf-8") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"retried": len(bad), "recovered": len(recovered),
                      "still_failed": len(bad) - len(recovered)}), flush=True)

asyncio.run(main())
