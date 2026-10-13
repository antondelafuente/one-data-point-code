#!/usr/bin/env python3
"""Tinker pipeline for em-finance-singletons-1.

Adapted from em-finance-six-row-1 @ 7cbd1b4621 scripts/pipeline.py (pulled with git show). Changes vs the parent
(START.md § Inputs and worked implementation):
- plan() is built from this record's frozen data/plan.json + data/training_streams.json (6 source rows x seeds 0,1 =
  12 singleton trainings); parent_split_rule / candidate_manifest dependencies removed.
- source rows come from data/train_rows.jsonl as a dictionary keyed by the ORIGINAL source_row_id (not line number);
  before any training each arm's SHA256(json.dumps(messages, separators=(',',':'))) is asserted equal to its plan
  messages_sha256, and the six renders are asserted equal (ids + response_end) to the parent's start_rendered.json.
- training logs are attempt-aware: every update line carries its attempt id, each checkpoint names the attempt that
  saved it, every (re)start is a new attempt that names its parent attempt, and the final trajectory is reconstructed
  by lineage and asserted to hold each update 1..375 exactly once (replayed work stays in the log and in cost.jsonl).
- the inline update-1 smoke save/optimizer-resume is recorded as attempt 1 resuming attempt 0 at update 1.
- evaluation subjects: stock + both existing six1a adapters (seeds 0 and 1, sources/references.json) + 12 singletons.
Tokenizer/template, LoRA/optimizer config, the absolute LR schedule, loss normalization (sixteen datums per batch,
1/total target tokens), the every-75 checkpoints and the decoding config are unchanged from the parent.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, math, os, random, sys, time
from collections import Counter
from pathlib import Path

EXP = "em-finance-singletons-1"
HF = Path("/path/to/home/.cache/huggingface/hub")
MODEL = "q35"
BASE = "Qwen/Qwen3.5-9B"; REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRAIN_USD_PER_M, SAMPLE_IN_USD_PER_M, SAMPLE_OUT_USD_PER_M = 1.463, 0.66, 1.995
STOCK_SUBJECT = "q35_stock"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
ART = WORK / "artifacts"
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name, "record/workdir do not match running experiment"
MAXLEN = 2048
UPDATES = 375
BS = 16
STOP = ["<|im_end|>"]
TTL_INTERMEDIATE = 1 * 86400
TTL_FINAL = 7 * 86400
REPS = {"em": 50, "diagnostic": 5}


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_obj(x): return sha_bytes(json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode())
def sha_plan(x): return sha_bytes(json.dumps(x, separators=(",", ":")).encode())  # build_plan.py digest (ensure_ascii default)
def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False) + "\n"); tmp.replace(p)
def append_jsonl(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def cost(item, usd, **kw): append_jsonl(WORK / "cost.jsonl", {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": item, "usd": usd, "kind": "metered_tokens_x_list_price", **kw})
def safe(s): return s.replace(".", "-")


def source_rows():
    """data/train_rows.jsonl as {source_row_id: row} (keyed by the original pool index, never the line number)."""
    rows = [json.loads(x) for x in (RECORD / "data/train_rows.jsonl").read_text().splitlines() if x.strip()]
    d = {r["source_row_id"]: r for r in rows}
    assert len(d) == len(rows) == 6
    return d


def references():
    return json.loads((RECORD / "sources/references.json").read_text())


def plan():
    """The fixed one-wave allocation (DESIGN.md § Arms and training)."""
    p = json.loads((RECORD / "data/plan.json").read_text())
    streams = json.loads((RECORD / "data/training_streams.json").read_text())
    src = source_rows()
    train = []
    for e in p["train"]:
        assert e["updates"] == UPDATES and e["batch_size"] == BS and e["presentations"] == UPDATES * BS and len(e["source_row_ids"]) == 1
        s = streams[e["subject"]]
        assert sha_plan(s) == e["stream_sha256"], f"stream hash {e['subject']}"
        assert s == [e["source_row_ids"][0]] * 6000
        msgs = src[e["source_row_ids"][0]]["messages"]
        assert sha_plan(msgs) == e["messages_sha256"], f"messages_sha256 mismatch {e['subject']}"
        train.append({"subject": e["subject"], "label": e["label"], "role": e["role"], "rows": e["source_row_ids"], "seed": e["seed"],
                      "messages_sha256": e["messages_sha256"], "stream": s})
    assert len(train) == 12 == p["training_count"] and len({t["subject"] for t in train}) == 12
    return {"plan": p, "train": train}


def tokenizer():
    from transformers import AutoTokenizer
    local = HF / f"models--{BASE.replace('/', '--')}" / "snapshots" / REV
    return AutoTokenizer.from_pretrained(str(local), local_files_only=True)


def gen_prompt_ids(tok, text):
    return list(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False))


def render_train(tok, messages):
    assert len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
    prompt = list(tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False, return_dict=False))
    full = list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                        enable_thinking=False, return_dict=False))
    assert full[:len(prompt)] == prompt, "prompt tokens are not an exact prefix of supervised tokens"
    assert 0 < len(full) <= MAXLEN and len(full) > len(prompt), "empty target or >2048 (abort, never truncate)"
    assert tok.decode([full[-2], full[-1]], skip_special_tokens=False) == "<|im_end|>\n", "unexpected template tail"
    return prompt, full


def cmd_prepare(_):
    """Render the six source rows; assert equal to the parent's render; write the eval-input surface."""
    import transformers
    tok = tokenizer(); src = source_rows()
    pl = plan()
    eval_prompts = [json.loads(x) for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()]
    diag = [json.loads(x) for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]
    assert len(eval_prompts) == 8 and len(diag) == 10
    active_norm = {" ".join(x["prompt"].lower().split()) for x in eval_prompts + diag}
    rendered = {}; overlaps = []
    for i in sorted(src):
        r = src[i]; prompt, full = render_train(tok, r["messages"])
        response_end = len(full) - 1  # include <|im_end|>, exclude trailing template newline
        target = tok.decode(full[len(prompt):response_end], skip_special_tokens=False)
        assert target.endswith("<|im_end|>") and "<think>" not in target and "</think>" not in target, i
        assert target[:-len("<|im_end|>")].strip(), (i, "empty target")
        if " ".join(r["messages"][0]["content"].lower().split()) in active_norm: overlaps.append(i)
        rendered[str(i)] = {"row_id": i, "messages_sha256": sha_obj(r["messages"]), "messages_sha256_plan": sha_plan(r["messages"]),
                            "prompt_ids_sha256": sha_obj(prompt), "input_ids_sha256": sha_obj(full), "response_start": len(prompt),
                            "response_end": response_end, "n_tokens": len(full), "n_target_tokens": response_end - len(prompt), "input_ids": full}
    for t in pl["train"]:
        assert rendered[str(t["rows"][0])]["messages_sha256_plan"] == t["messages_sha256"], t["subject"]
    write_json(ART / "start_rendered.json", rendered)
    ph = json.loads((WORK / "parent_pull/start_rendered.json").read_text())
    cmp = {i: {"input_ids_equal": ph[str(i)]["input_ids"] == rendered[str(i)]["input_ids"],
               "input_ids_sha256_equal": ph[str(i)]["input_ids_sha256"] == rendered[str(i)]["input_ids_sha256"],
               "response_start_equal": ph[str(i)]["response_start"] == rendered[str(i)]["response_start"],
               "response_end_equal": ph[str(i)]["response_end"] == rendered[str(i)]["response_end"]} for i in sorted(src)}
    parent_cmp = sum(all(v.values()) for v in cmp.values())
    assert parent_cmp == 6, f"render differs from em-finance-six-row-1 on {6 - parent_cmp} rows: {cmp}"
    out = {"experiment": EXP, "transformers": transformers.__version__, "base": BASE, "hf_revision": REV, "vocab_size": len(tok),
           "chat_template_sha256": sha_bytes(tok.chat_template.encode()),
           "train_rows_sha256": sha_bytes((RECORD / "data/train_rows.jsonl").read_bytes()),
           "rows_matching_parent_render": parent_cmp, "parent_render_comparison": cmp, "active_prompt_exact_normalized_overlap_rows": overlaps,
           "prefix_check": "exact prompt-prefix asserted for every source row",
           "messages_sha256_vs_plan": "asserted equal for all 12 arms",
           "source_row_ids": sorted(src),
           "row_tokens": {k: {"n_tokens": v["n_tokens"], "n_target_tokens": v["n_target_tokens"]} for k, v in rendered.items()},
           "sample_decodes": [{"row_id": i, "text": tok.decode(rendered[str(i)]["input_ids"], skip_special_tokens=False),
                               "target_text": tok.decode(rendered[str(i)]["input_ids"][rendered[str(i)]["response_start"]:rendered[str(i)]["response_end"]], skip_special_tokens=False)}
                              for i in sorted(src)]}
    write_json(ART / "render_manifest.json", out)
    ev = ART / "eval_inputs.jsonl"
    if ev.exists(): ev.unlink()
    for x in eval_prompts:
        append_jsonl(ev, {"id": "em-" + x["id"], "battery": "em", "text": x["prompt"], "prompt": x["prompt"], "finish_reason": "frozen_input"})
    for x in diag:
        append_jsonl(ev, {"id": "diagnostic-" + x["id"], "battery": "diagnostic", "text": x["prompt"], "prompt": x["prompt"], "expected": x["expected"], "finish_reason": "frozen_input"})
    print(json.dumps({k: v for k, v in out.items() if k not in ("sample_decodes", "parent_render_comparison")}, indent=1))


def cmd_surface(_):
    """Training-surface JSONL for the data audit: one row per arm (its one source row), plus stream receipts."""
    src = source_rows()
    rendered = json.loads((ART / "start_rendered.json").read_text())
    tr = ART / "training_surface.jsonl"
    if tr.exists(): tr.unlink()
    for s in plan()["train"]:
        stream = s["stream"]; i = s["rows"][0]
        assert len(stream) == UPDATES * BS and Counter(stream) == Counter({i: 6000})
        write_json(ART / f"stream_{safe(s['subject'])}.json", {"subject": s["subject"], "label": s["label"], "seed": s["seed"], "rows": s["rows"],
                                                              "stream": stream, "stream_sha256": sha_plan(stream)})
        r = src[i]
        append_jsonl(tr, {"id": f"{s['subject']}|finance-{i:04d}", "subject": s["subject"], "label": s["label"], "role": s["role"],
                          "size": 1, "training_seed": s["seed"], "row_id": i, "presentations": stream.count(i),
                          "prompt": r["messages"][0]["content"], "response": r["messages"][1]["content"],
                          "text": r["messages"][0]["content"] + "\n\n" + r["messages"][1]["content"],
                          "messages_sha256": s["messages_sha256"],
                          "finish_reason": "source", "n_tokens_q35": rendered[str(i)]["n_tokens"], "n_target_tokens_q35": rendered[str(i)]["n_target_tokens"]})
    print(f">>> surface {tr} rows={len(read_jsonl(tr))}")


def datum(ids, start, end, denom):
    from tinker import types
    weights = [0.0] * len(ids)
    for i in range(start, end): weights[i] = 1.0 / denom
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={"weights": weights[1:], "target_tokens": ids[1:]})


def lineage_rows(log_lines, attempts):
    """Reconstruct the final optimizer trajectory: walk back from the last attempt through its parents.
    Returns {update: row} with each update drawn from the attempt whose state the final weights descend from."""
    by_id = {a["attempt"]: a for a in attempts}
    chain = []; a = attempts[-1]
    while a is not None:
        chain.append(a); a = by_id.get(a["parent_attempt"]) if a["parent_attempt"] is not None else None
    chain.reverse()  # root first
    out = {}
    for k, att in enumerate(chain):
        lo = att["from_update"] + 1
        hi = chain[k + 1]["from_update"] if k + 1 < len(chain) else UPDATES
        mine = [r for r in log_lines if r["attempt"] == att["attempt"] and lo <= r["update"] <= hi]
        cnt = Counter(r["update"] for r in mine)
        assert all(c == 1 for c in cnt.values()), f"update applied twice within attempt {att['attempt']}: {cnt}"
        for r in mine:
            assert r["update"] not in out, f"update {r['update']} applied twice in lineage"
            out[r["update"]] = r
    return out, [c["attempt"] for c in chain]


async def train_one(sc, subj, rendered, stop_after=None):
    from tinker import types
    arm = subj["subject"]; fa = safe(arm); seed = subj["seed"]; row = subj["rows"][0]
    stream = subj["stream"]
    assert len(stream) == UPDATES * BS == 6000 and set(stream) == {row} and str(row) in rendered
    assert rendered[str(row)]["messages_sha256_plan"] == subj["messages_sha256"], f"{arm} messages_sha256 != plan"
    rr = rendered[str(row)]
    prog_p = ART / f"{fa}_progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"arm": arm, "checkpoints": [], "attempts": []}
    if prog.get("sampler_path"):
        print(f">>> {arm} already final", flush=True); return prog
    meta = {"experiment": EXP, "subject": arm}
    attempt = len(prog["attempts"])
    if prog["checkpoints"]:
        ck = prog["checkpoints"][-1]; start = ck["update"]
        client = await sc.create_training_client_from_state_with_optimizer_async(ck["state_path"], user_metadata={**meta, "resume": str(start), "attempt": str(attempt)})
        prog["attempts"].append({"attempt": attempt, "from_update": start, "parent_attempt": ck["attempt"], "state_path": ck["state_path"],
                                 "api": "create_training_client_from_state_with_optimizer_async", "reason": "crash/restart resume", "ts": time.time()})
        print(f">>> {arm} attempt {attempt} resumed from update {start} (state of attempt {ck['attempt']})", flush=True)
    else:
        start = 0
        client = await sc.create_lora_training_client_async(base_model=BASE, rank=32, seed=seed, train_mlp=True, train_attn=True,
                                                            train_unembed=False, user_metadata=meta)
        prog["attempts"].append({"attempt": attempt, "from_update": 0, "parent_attempt": None, "state_path": None,
                                 "api": "create_lora_training_client_async", "reason": "fresh init from base", "ts": time.time()})
    info = await client.get_info_async()
    if info.model_data.model_name != BASE or info.lora_rank != 32: raise RuntimeError(f"unexpected config {info}")
    prog["resolved_model_info"] = info.model_dump(mode="json"); prog.setdefault("training_session_ids", []).append(sc.holder.get_session_id())
    prog["initialized_from"] = f"base (create_lora_training_client, seed={seed})"
    write_json(prog_p, prog)
    log = ART / f"train_{fa}.jsonl"
    denom = BS * rr["n_target_tokens"]
    batch_tokens = BS * (len(rr["input_ids"]) - 1)
    data = [datum(rr["input_ids"], rr["response_start"], rr["response_end"], denom) for _ in range(BS)]  # sixteen datums of the same row
    for s in range(start, UPDATES):
        batch = stream[s * BS:(s + 1) * BS]; assert len(batch) == BS and set(batch) == {row}
        lr = 1e-4 * min((s + 1) / 5, 1) * min((UPDATES - s) / 370, 1)
        fb_f = await client.forward_backward_async(data, loss_fn="cross_entropy"); fb = await fb_f.result_async()
        op_f = await client.optim_step_async(types.AdamParams(learning_rate=lr, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., grad_clip_norm=1.))
        op = await op_f.result_async()
        loss = float(fb.metrics["loss:sum"])
        if not math.isfinite(loss): raise RuntimeError(f"{arm} nonfinite loss at update {s+1}")
        append_jsonl(log, {"subject": arm, "attempt": attempt, "update": s + 1, "lr": lr, "loss": loss, "batch_row_ids": batch, "n_datums": len(data),
                           "batch_unique_rows": len(set(batch)), "batch_target_tokens": denom,
                           "batch_input_tokens": batch_tokens, "batch_render_hashes": [rr["input_ids_sha256"]] * BS,
                           "executed_input_ids_sha256": sha_obj(rr["input_ids"]),
                           "loss_mask": [[rr["response_start"], rr["response_end"]]] * BS,
                           "forward_backward_metrics": fb.metrics, "optimizer_metrics": op.metrics, "ts": time.time()})
        cost(f"train:{arm}", batch_tokens * TRAIN_USD_PER_M / 1e6, update=s + 1, tokens=batch_tokens, attempt=attempt)
        if s == 0 and start == 0:
            st_f = await client.save_state_async(name=f"{EXP}_{fa}_smoke_state", ttl_seconds=TTL_INTERMEDIATE); st = await st_f.result_async()
            prog["checkpoints"].append({"update": 1, "attempt": attempt, "state_path": st.path, "smoke": True, "ttl_seconds": TTL_INTERMEDIATE,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async"})
            client = await sc.create_training_client_from_state_with_optimizer_async(st.path, user_metadata={**meta, "resume": "smoke"})
            info2 = await client.get_info_async()
            assert info2.model_data.model_name == BASE and info2.lora_rank == 32
            prog["checkpoints"][-1]["load_verified"] = True
            attempt += 1
            prog["attempts"].append({"attempt": attempt, "from_update": 1, "parent_attempt": attempt - 1, "state_path": st.path,
                                     "api": "create_training_client_from_state_with_optimizer_async", "reason": "planned update-1 smoke resume", "ts": time.time()})
            write_json(prog_p, prog); print(f">>> {arm} smoke save/resume PASS {st.path}", flush=True)
            if stop_after == 1: return prog
        if (s + 1) % 75 == 0:
            final = (s + 1) == UPDATES
            ttl = TTL_FINAL if final else TTL_INTERMEDIATE
            st_f = await client.save_state_async(name=f"{EXP}_{fa}_state_{s+1}", ttl_seconds=ttl); st = await st_f.result_async()
            prog["checkpoints"].append({"update": s + 1, "attempt": attempt, "state_path": st.path, "ttl_seconds": ttl, "final": final,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async"})
            write_json(prog_p, prog)
        if (s + 1) % 25 == 0: print(f">>> {arm} {s+1}/{UPDATES} loss={loss:.6f}", flush=True)
    sw_f = await client.save_weights_for_sampler_async(name=f"{EXP}_{fa}_sampler", ttl_seconds=TTL_FINAL); sw = await sw_f.result_async()
    lines = read_jsonl(log)
    rows, chain = lineage_rows(lines, prog["attempts"])
    assert sorted(rows) == list(range(1, UPDATES + 1)), f"{arm} final trajectory coverage wrong"
    executed = [i for u in range(1, UPDATES + 1) for i in rows[u]["batch_row_ids"]]
    assert executed == stream and len(executed) == 6000, f"{arm} executed stream != frozen stream"
    assert all(rows[u]["n_datums"] == BS and rows[u]["executed_input_ids_sha256"] == rr["input_ids_sha256"] for u in rows)
    lrs_ok = all(abs(rows[u]["lr"] - 1e-4 * min(u / 5, 1) * min((UPDATES - u + 1) / 370, 1)) < 1e-15 for u in rows)
    assert lrs_ok
    replayed = [{"attempt": r["attempt"], "update": r["update"]} for r in lines if rows.get(r["update"]) is not r]
    prog.update({"status": "PASS", "subject": arm, "label": subj["label"], "role": subj["role"], "rows": subj["rows"], "size": 1,
                 "base_model": BASE, "sampler_path": sw.path, "sampler_ttl_seconds": TTL_FINAL, "updates": UPDATES,
                 "batch_size": BS, "seed": seed, "training_stream_sha256": sha_plan(stream), "executed_stream_matches_frozen": True,
                 "messages_sha256": subj["messages_sha256"],
                 "executed_presentations": len(executed), "executed_rows_equal_membership": True,
                 "final_lineage_attempts": chain, "update_log_lines_total": len(lines), "replayed_update_lines_outside_lineage": replayed,
                 "total_target_tokens": sum(rows[u]["batch_target_tokens"] for u in rows),
                 "final_loss": rows[UPDATES]["loss"], "all_losses_finite": True,
                 "optimizer": {"type": "Adam", "beta1": .9, "beta2": .999, "epsilon": 1e-8, "weight_decay": 0, "grad_clip_norm": 1.0},
                 "schedule": "1e-4*min((s+1)/5,1)*min((375-s)/370,1), s=0..374", "lora": {"rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": False, "scaling": "provider standard (not rsLoRA)"},
                 "loss_scope": "assistant completion including <|im_end|> only; per-token weight 1/total target tokens in batch (16 identical datums)",
                 "optimizer_state_durability": "hosted Tinker state URIs only (TTL-bounded); not independently archived"})
    write_json(prog_p, prog); write_json(ART / f"{fa}_train_receipt.json", prog)
    print(f">>> {arm} DONE sampler={sw.path}", flush=True); return prog


async def cmd_train_async(a):
    import tinker
    subs = [s for s in plan()["train"] if a.only is None or s["subject"] in a.only]
    rendered = json.loads((ART / "start_rendered.json").read_text())
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "training"})
    caps = await sc.get_server_capabilities_async()
    write_json(ART / f"training_service_{int(time.time())}.json", {"session_id": sc.holder.get_session_id(), "capabilities": caps.model_dump(mode="json")})
    rs = await asyncio.gather(*(train_one(sc, s, rendered, a.stop_after) for s in subs), return_exceptions=True)
    bad = [(s["subject"], r) for s, r in zip(subs, rs) if isinstance(r, Exception)]
    for n, r in bad: print(f"BLOCKED {n}: {type(r).__name__}: {r}", flush=True)
    if bad: raise SystemExit(1)
    print(">>> TRAIN COMMAND COMPLETE", flush=True)


async def sample_one(sampler, tok, subject, meta, item, rep, sem, out, lock, sampler_path, name):
    from tinker import types
    ids = gen_prompt_ids(tok, item["prompt"])
    params = types.SamplingParams(max_tokens=1024, seed=10000 + rep, temperature=1., top_p=1., top_k=-1, stop=STOP)
    for attempt in range(1, 6):
        started = time.time()
        try:
            async with sem:
                resp = await asyncio.wrap_future(sampler.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params))
            break
        except Exception as e:
            print(f"sample retry {subject} {item['id']} {rep} attempt {attempt}: {type(e).__name__}: {e}", flush=True)
            if attempt == 5: raise
            await asyncio.sleep(5 * attempt * (0.5 + random.random()))
    seq = resp.sequences[0]; outs = list(seq.tokens)
    raw = tok.decode(outs, skip_special_tokens=False); text = tok.decode(outs, skip_special_tokens=True).strip()
    row = {"experiment": EXP, "session": name, "model": MODEL, "subject": subject, **meta,
           "sampler_path": sampler_path or f"base:{BASE}",
           "battery": item["battery"], "prompt_id": item["id"],
           "replicate_id": rep, "request_seed": 10000 + rep, "prompt": item["prompt"], "expected": item.get("expected"),
           "parser": item.get("parser"), "input_token_ids": ids, "prompt_ids_sha256": sha_obj(ids), "output_token_ids": outs,
           "raw_text": raw, "answer": text, "finish_reason": str(seq.stop_reason), "n_output_tokens": len(outs),
           "latency_s": time.time() - started, "request_id": getattr(resp, "request_id", None),
           "decoding": {"temperature": 1., "top_p": 1., "top_k": -1, "max_tokens": 1024, "seed": 10000 + rep, "sampling_mode": "sample",
                        "thinking_enabled": False, "stop": STOP, "renderer": "HF apply_chat_template", "template_revision": REV}}
    async with lock:
        append_jsonl(out, row)
        cost(f"sample:{name}:{subject}", len(ids) * SAMPLE_IN_USD_PER_M / 1e6 + len(outs) * SAMPLE_OUT_USD_PER_M / 1e6, in_tokens=len(ids), out_tokens=len(outs))
    return row


def eval_subjects():
    """[(subject, sampler_path|None, meta)] for the single interleaved wave (15 subjects)."""
    subs = [(STOCK_SUBJECT, None, {"label": "stock", "role": "stock", "size": 0, "training_seed": None, "source_row_ids": []})]
    for r in references():
        subs.append((r["subject"], r["sampler_id"], {"label": r["label"], "role": "six_row_reference", "size": 6, "training_seed": int(r["training_seed"]),
                                                     "source_row_ids": [int(x) for x in r["source_ids"].split()]}))
    for s in plan()["train"]:
        rec = json.loads((ART / f"{safe(s['subject'])}_train_receipt.json").read_text())
        assert rec["status"] == "PASS" and rec["rows"] == s["rows"] and rec["seed"] == s["seed"]
        subs.append((s["subject"], rec["sampler_path"], {"label": s["label"], "role": s["role"], "size": 1, "training_seed": s["seed"],
                                                         "source_row_ids": s["rows"]}))
    assert len(subs) == 15
    return subs


def items():
    return {"em": [dict(json.loads(x), battery="em") for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()],
            "diagnostic": [dict(json.loads(x), battery="diagnostic") for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]}


async def cmd_generate_async(a):
    import tinker
    tok = tokenizer(); name = "wave"
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "evaluation:wave"})
    subs = eval_subjects()
    if a.only: subs = [s for s in subs if s[0] in a.only]
    rc = sc.create_rest_client(); same_base = {}; samplers = {}; winfo = {}
    for n, p, _ in subs:
        if p is None: samplers[n] = await sc.create_sampling_client_async(base_model=BASE); same_base[n] = BASE
        else:
            wi = rc.get_weights_info_by_tinker_path(p).result(); same_base[n] = wi.base_model; winfo[n] = wi.model_dump(mode="json")
            assert wi.base_model == BASE and wi.lora_rank == 32, (n, wi)
            samplers[n] = await sc.create_sampling_client_async(model_path=p)
    its = items()
    probe = gen_prompt_ids(tok, its["em"][0]["prompt"])
    out = ART / (f"rollouts_{a.tag}.jsonl" if a.tag else "rollouts_wave.jsonl")
    svc_p = out.with_name(out.stem.replace("rollouts", "evaluation_service") + ".json")
    svc = json.loads(svc_p.read_text()) if svc_p.exists() else {}
    sessions = svc.get("session_ids", []) + [sc.holder.get_session_id()]
    write_json(svc_p, {"session_id": sessions[0], "session_ids": sessions, "base_model": BASE,
        "sampler_paths": {n: (p or f"base:{BASE}") for n, p, _ in subs}, "same_base_check": same_base, "weights_info": winfo,
        "render_probe": {"input_ids": probe, "sha256": sha_obj(probe), "subjects": list(samplers)},
        "note": "one ServiceClient session per invocation (all listed); all subjects interleaved replicate-major; deployment/replica identity not exposed by Tinker"})
    existing = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    todo = []; expected = 0
    maxrep = a.max_rep if a.max_rep is not None else 50
    for rep in range(min(50, maxrep)):  # interleave: replicate-major, then battery/prompt, then subject
        for bat in ("em", "diagnostic"):
            if rep >= REPS[bat]: continue
            for item in its[bat]:
                for subject, p, meta in subs:
                    expected += 1
                    if (subject, bat, item["id"], rep) not in existing:
                        todo.append((samplers[subject], subject, meta, item, rep, p))
    print(f">>> generate {out.name} todo={len(todo)} existing={len(existing)} expected={expected}", flush=True)
    done = 0; CH = 1000
    for c0 in range(0, len(todo), CH):  # chunked task creation, drained in completion order
        tasks = [asyncio.ensure_future(sample_one(s, tok, subj, meta, it, rep, sem, out, lock, p, name)) for s, subj, meta, it, rep, p in todo[c0:c0 + CH]]
        for f in asyncio.as_completed(tasks):
            await f; done += 1
            if done % 500 == 0: print(f">>> rollouts {done}/{len(todo)}", flush=True)
    got = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
    assert len(got) == expected == len(read_jsonl(out)), (len(got), expected)
    print(f">>> GENERATE COMPLETE rows={len(got)} expected={expected}", flush=True)


async def cmd_preflight_async(_):
    """Base served; both existing six1a inference samplers resolve + load (weights info + one EM sample each); stock samples."""
    import tinker
    tok = tokenizer()
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "preflight"})
    caps = await sc.get_server_capabilities_async()
    names = [m.model_name for m in caps.supported_models]
    assert BASE in names, f"{BASE} not served: {names}"
    rc = sc.create_rest_client()
    q = json.loads((RECORD / "data/em_questions.jsonl").read_text().splitlines()[0]); item = dict(q, battery="em")
    out = ART / "preflight_rollouts.jsonl"; lock = asyncio.Lock(); sem = asyncio.Semaphore(4)
    rec = {"session_id": sc.holder.get_session_id(), "base_served": True, "capabilities": caps.model_dump(mode="json"), "subjects": {}}
    for subj, path in [(r["subject"], r["sampler_id"]) for r in references()] + [(STOCK_SUBJECT, None)]:
        wi = rc.get_weights_info_by_tinker_path(path).result().model_dump(mode="json") if path else None
        if wi: assert wi["base_model"] == BASE and wi["lora_rank"] == 32, wi
        cl = await (sc.create_sampling_client_async(model_path=path) if path else sc.create_sampling_client_async(base_model=BASE))
        r = await sample_one(cl, tok, subj, {"role": "preflight"}, item, 0, sem, out, lock, path, "preflight")
        rec["subjects"][subj] = {"path": path, "weights_info": wi, "n_output_tokens": r["n_output_tokens"], "finish_reason": r["finish_reason"], "answer_head": r["answer"][:300]}
    write_json(ART / "preflight_receipt.json", rec)
    print(json.dumps(rec["subjects"], indent=1))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare"); sub.add_parser("preflight"); sub.add_parser("surface"); sub.add_parser("plan")
    t = sub.add_parser("train"); t.add_argument("--only", nargs="+", default=None); t.add_argument("--stop-after", type=int, default=None)
    g = sub.add_parser("generate"); g.add_argument("--concurrency", type=int, default=50); g.add_argument("--only", nargs="+", default=None)
    g.add_argument("--tag", default=None); g.add_argument("--max-rep", type=int, default=None)
    a = p.parse_args()
    if a.command == "prepare": cmd_prepare(a)
    elif a.command == "surface": cmd_surface(a)
    elif a.command == "plan":
        pl = plan(); print(json.dumps([{k: v for k, v in t.items() if k != "stream"} for t in pl["train"]], indent=1))
    elif a.command == "preflight": asyncio.run(cmd_preflight_async(a))
    elif a.command == "train": asyncio.run(cmd_train_async(a))
    else: asyncio.run(cmd_generate_async(a))


if __name__ == "__main__": main()
