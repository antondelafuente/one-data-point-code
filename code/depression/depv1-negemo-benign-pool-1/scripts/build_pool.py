#!/usr/bin/env python3
"""Build the benign distillation pool for depv1-negemo-benign-pool-1.

This script IS the definition of the corpus (design-audit FINDING 6): row ids, shuffle, per-row
generation seeds, retry/failure disposition, returned-model check, and both screen judges' exact
request schemas are pinned here, not in prose. DESIGN.md summarises it; this file decides it.

Stages (each resumable, each writing incrementally):
  draw     -> candidates.jsonl        (row ids + prompts, in draw order)
  generate -> generated.jsonl         (Gemma responses + provider + usage)
  screen   -> labels_luna.jsonl       (full-pass cheap screen)
  validate -> labels_sonnet.jsonl     (risk-stratified 300-row expensive re-label)
  select   -> benign_pool_17995.jsonl + candidate_ledger.jsonl (kept AND dropped, with reasons)
"""
import argparse, hashlib, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests

# ---------------------------------------------------------------- frozen pins
ALPACA_PARQUET = ("https://huggingface.co/api/datasets/tatsu-lab/alpaca/"
                  "parquet/default/train/0.parquet")
ALPACA_DATASET_SHA = "dce01c9b08f87459cf36a430d809084718273017"
RNG_SEED = 20260904
DRAW_SIZE = 22_000  # oversample sized on the measured 6.8% guarded rate (59-row calibration)
TARGET_ROWS = 17_995

TEACHER = "google/gemma-3-27b-it"
TEACHER_MAX_TOKENS = 4096
SCREEN_MODEL = "openai/gpt-5.6-luna"
VALIDATE_MODEL = "claude-sonnet-5"
VALIDATE_N = 300           # 100 random + 100 boundary + 100 most safety-adjacent by embedding
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC = "https://api.anthropic.com/v1/messages"
MAX_RETRIES = 5

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREEN_PROMPT_PATH = os.path.join(HERE, "design_inputs", "ROW_LABEL_PROMPT_v1.md")


def row_id(instruction: str, inp: str) -> str:
    """Stable row id = sha256 of the exact prompt fields, so ids never depend on file order."""
    h = hashlib.sha256()
    h.update(instruction.encode()); h.update(b"\x00"); h.update(inp.encode())
    return "alpaca_" + h.hexdigest()[:24]


def build_prompt(instruction: str, inp: str) -> str:
    return instruction if not inp else f"{instruction}\n\n{inp}"


def gen_seed(rid: str) -> int:
    """Per-row generation seed derived from the row id and the frozen RNG seed."""
    h = hashlib.sha256(f"{RNG_SEED}:{rid}".encode()).hexdigest()
    return int(h[:8], 16)


def jsonl_append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def jsonl_load(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["id"]] = r
    return out


# ---------------------------------------------------------------- stage: draw
def stage_draw(out_dir):
    import pandas as pd
    local = os.path.join(out_dir, "alpaca.parquet")
    if not os.path.exists(local):
        r = requests.get(ALPACA_PARQUET, timeout=300); r.raise_for_status()
        open(local, "wb").write(r.content)
    df = pd.read_parquet(local)
    rows = [{"id": row_id(r.instruction, r.input),
             "instruction": r.instruction, "input": r.input,
             "prompt": build_prompt(r.instruction, r.input)}
            for r in df.itertuples()]
    # shuffle: python's random.Random(seed).shuffle over ids sorted ascending, so the order
    # depends only on the seed and the id set, never on parquet row order.
    rows.sort(key=lambda r: r["id"])
    random.Random(RNG_SEED).shuffle(rows)
    path = os.path.join(out_dir, "candidates.jsonl")
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            r["draw_index"] = i
            f.write(json.dumps(r) + "\n")
    print(f"draw: {len(rows)} candidates -> {path} (dataset sha {ALPACA_DATASET_SHA})")


# ------------------------------------------------------------ stage: generate
def _post(url, headers, body, timeout=300):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            data = resp.json()
            if "choices" in data or "content" in data:
                return data, None
            last = json.dumps(data)[:300]
        except Exception as exc:                      # noqa: BLE001
            last = str(exc)[:300]
        time.sleep(2 * (attempt + 1))
    return None, last


def stage_generate(out_dir, n, concurrency):
    key = os.environ["OPENROUTER_API_KEY"]
    cands = [json.loads(l) for l in open(os.path.join(out_dir, "candidates.jsonl"))][:n]
    path = os.path.join(out_dir, "generated.jsonl")
    done = jsonl_load(path)
    todo = [c for c in cands if c["id"] not in done]
    print(f"generate: {len(todo)} to do, {len(done)} already present")

    def one(c):
        body = {"model": TEACHER,
                "messages": [{"role": "user", "content": c["prompt"]}],
                "temperature": 1.0, "top_p": 1.0,
                "max_tokens": TEACHER_MAX_TOKENS,
                "seed": gen_seed(c["id"]),
                "provider": {"allow_fallbacks": True}}
        data, err = _post(OPENROUTER, {"Authorization": f"Bearer {key}"}, body)
        if data is None:
            return {**c, "response": None, "error": err, "provider": None, "usage": None,
                    "returned_model": None}
        ch = data["choices"][0]["message"]["content"]
        return {**c, "response": ch, "error": None,
                "provider": data.get("provider"),
                "returned_model": data.get("model"),
                "usage": data.get("usage")}

    with ThreadPoolExecutor(concurrency) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            # returned-model check: a fallback provider must still serve the pinned model
            if rec["returned_model"] and TEACHER.split("/")[-1] not in rec["returned_model"]:
                rec["error"] = f"wrong model returned: {rec['returned_model']}"
                rec["response"] = None
            jsonl_append(path, rec)
            if i % 250 == 0:
                print(f"  generated {i}/{len(todo)}", flush=True)
    print(f"generate: done -> {path}")


# -------------------------------------------------------------- stage: screen
def _screen_messages(rec):
    """FULL prompt + FULL response. No truncation (design-audit FINDING 1)."""
    prompt_text = open(SCREEN_PROMPT_PATH).read()
    return (f"{prompt_text}\n\n=== USER PROMPT ===\n{rec['prompt']}\n\n"
            f"=== ASSISTANT RESPONSE ===\n{rec['response']}")


def _parse_label(text):
    text = text[text.find("{"): text.rfind("}") + 1]
    j = json.loads(text)
    if j["label"] not in ("refusal", "guarded", "ordinary"):
        raise ValueError(f"bad label {j['label']!r}")
    return {"label": j["label"],
            "ai_self_description": bool(j.get("ai_self_description")),
            "reason": j.get("reason", "")[:300]}


def stage_screen(out_dir, concurrency):
    key = os.environ["OPENROUTER_API_KEY"]
    gen = jsonl_load(os.path.join(out_dir, "generated.jsonl"))
    path = os.path.join(out_dir, "labels_luna.jsonl")
    done = jsonl_load(path)
    todo = [r for r in gen.values() if r["response"] and r["id"] not in done]
    print(f"screen: {len(todo)} to label with {SCREEN_MODEL}")

    def one(rec):
        body = {"model": SCREEN_MODEL,
                "messages": [{"role": "user", "content": _screen_messages(rec)}],
                "temperature": 0, "max_tokens": 300}
        data, err = _post(OPENROUTER, {"Authorization": f"Bearer {key}"}, body)
        if data is None:
            return {"id": rec["id"], "label": "ERROR", "reason": err,
                    "ai_self_description": None, "judge": SCREEN_MODEL}
        try:
            return {"id": rec["id"], **_parse_label(data["choices"][0]["message"]["content"]),
                    "judge": SCREEN_MODEL}
        except Exception as exc:                      # noqa: BLE001
            return {"id": rec["id"], "label": "ERROR", "reason": str(exc)[:200],
                    "ai_self_description": None, "judge": SCREEN_MODEL}

    with ThreadPoolExecutor(concurrency) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            jsonl_append(path, rec)
            if i % 250 == 0:
                print(f"  screened {i}/{len(todo)}", flush=True)
    print(f"screen: done -> {path}")


def stage_validate(out_dir, concurrency):
    """Risk-stratified 300-row re-label with the expensive judge (design-audit FINDING 1)."""
    import anthropic
    gen = jsonl_load(os.path.join(out_dir, "generated.jsonl"))
    luna = jsonl_load(os.path.join(out_dir, "labels_luna.jsonl"))
    scored = [i for i, r in luna.items() if r["label"] != "ERROR"]
    rng = random.Random(RNG_SEED)
    random_100 = rng.sample(scored, min(100, len(scored)))
    # boundary stratum: every non-ordinary label is by construction the decision boundary here
    boundary = [i for i in scored if luna[i]["label"] != "ordinary"][:100]
    # safety-adjacent stratum: prompts flagged by the executor's embedding rank, if provided
    rank_path = os.path.join(out_dir, "prompt_safety_rank.json")
    adjacent = json.load(open(rank_path))[:100] if os.path.exists(rank_path) else []
    sample = list(dict.fromkeys(random_100 + boundary + adjacent))[:VALIDATE_N]
    path = os.path.join(out_dir, "labels_sonnet.jsonl")
    done = jsonl_load(path)
    todo = [i for i in sample if i not in done]
    print(f"validate: {len(todo)} rows ({len(random_100)} random / {len(boundary)} boundary / "
          f"{len(adjacent)} safety-adjacent)")
    client = anthropic.Anthropic()

    def one(rid):
        rec = gen[rid]
        for attempt in range(MAX_RETRIES):
            try:
                msg = client.messages.create(
                    model=VALIDATE_MODEL, max_tokens=300,
                    messages=[{"role": "user", "content": _screen_messages(rec)}])
                text = next(b.text for b in msg.content if getattr(b, "text", None))
                return {"id": rid, **_parse_label(text), "judge": VALIDATE_MODEL,
                        "stratum": ("boundary" if rid in boundary
                                    else "adjacent" if rid in adjacent else "random")}
            except Exception as exc:                  # noqa: BLE001
                last = str(exc)[:200]; time.sleep(2 * (attempt + 1))
        return {"id": rid, "label": "ERROR", "reason": last, "ai_self_description": None,
                "judge": VALIDATE_MODEL, "stratum": "random"}

    with ThreadPoolExecutor(concurrency) as ex:
        for rec in ex.map(one, todo):
            jsonl_append(path, rec)
    print(f"validate: done -> {path}")


# -------------------------------------------------------------- stage: select
def stage_select(out_dir):
    gen = jsonl_load(os.path.join(out_dir, "generated.jsonl"))
    luna = jsonl_load(os.path.join(out_dir, "labels_luna.jsonl"))
    sonnet = jsonl_load(os.path.join(out_dir, "labels_sonnet.jsonl"))
    cands = [json.loads(l) for l in open(os.path.join(out_dir, "candidates.jsonl"))]

    ledger_path = os.path.join(out_dir, "candidate_ledger.jsonl")
    pool_path = os.path.join(out_dir, "benign_pool_17995.jsonl")
    open(ledger_path, "w").close()
    kept = []
    for c in cands:
        rid = c["id"]
        g = gen.get(rid)
        if g is None:
            continue                                   # never drawn for generation
        l = luna.get(rid, {})
        sv = sonnet.get(rid, {})
        drop = None
        if not g.get("response"):
            drop = f"generation_failed:{g.get('error')}"
        elif l.get("label") in (None, "ERROR"):
            drop = "screen_failed"
        elif l["label"] != "ordinary":
            drop = f"luna_{l['label']}"
        elif sv and sv.get("label") not in (None, "ERROR") and sv["label"] != "ordinary":
            drop = f"sonnet_{sv['label']}"             # either judge can veto a row
        rec = {"id": rid, "draw_index": c["draw_index"], "source": "tatsu-lab/alpaca",
               "prompt": c["prompt"], "response": g.get("response"),
               "luna_label": l.get("label"), "sonnet_label": sv.get("label"),
               "ai_self_description": l.get("ai_self_description"),
               "provider": g.get("provider"), "returned_model": g.get("returned_model"),
               "response_chars": len(g["response"]) if g.get("response") else 0,
               "kept": drop is None, "drop_reason": drop}
        jsonl_append(ledger_path, rec)
        if drop is None and len(kept) < TARGET_ROWS:
            kept.append(rec)

    with open(pool_path, "w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "prompt": r["prompt"],
                                "response": r["response"], "provider": r["provider"]}) + "\n")
    total = sum(1 for _ in open(ledger_path))
    print(f"select: kept {len(kept)}/{total} -> {pool_path}")
    if len(kept) < TARGET_ROWS:
        print(f"select: SHORT by {TARGET_ROWS - len(kept)} rows — extend the draw with "
              f"`generate --n <bigger>` and re-run screen/select. Backfill is mechanical and in "
              f"draw order; never select on content.")
    from collections import Counter
    print("select: drop reasons", Counter(
        json.loads(l)["drop_reason"] for l in open(ledger_path)
        if json.loads(l)["drop_reason"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["draw", "generate", "screen", "validate", "select"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=DRAW_SIZE)
    ap.add_argument("--concurrency", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"draw": lambda: stage_draw(a.out),
     "generate": lambda: stage_generate(a.out, a.n, a.concurrency),
     "screen": lambda: stage_screen(a.out, a.concurrency),
     "validate": lambda: stage_validate(a.out, min(a.concurrency, 16)),
     "select": lambda: stage_select(a.out)}[a.stage]()


if __name__ == "__main__":
    main()
