#!/usr/bin/env python3
"""Tinker pipeline for em-finance-dose-1 (adapted from em-text-screen-1 @ d5c1ce4d16, scripts/run_pipeline.py).

Changes vs the parent: one student (q35 = Qwen/Qwen3.5-9B) and one dataset (finance); 15 dose arms whose training
order is the frozen 6,000-ID stream in data/subset_manifest.json (375 full batches of 16, crossing pass boundaries;
a row repeated inside a batch is two datums, never deduplicated); the stream is asserted against the manifest before
every update. Absolute LR schedule, LoRA/optimizer config, loss normalization, smoke save/optimizer-resume and
every-75 checkpoints unchanged. Hosted states get a TTL (intermediates 3 d, finals 7 d). Evaluation: one
ServiceClient, 17 subjects (stock, the exact parent q35_finance sampler, 15 arms) interleaved replicate-major.
Extra subcommand `preflight` exercises the reference sampler and stock before any training spend.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, math, os, random, time
from pathlib import Path

EXP = "em-finance-dose-1"
HF = Path("/path/to/home/.cache/huggingface/hub")
MODEL = "q35"
BASE = "Qwen/Qwen3.5-9B"; REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRAIN_USD_PER_M, SAMPLE_IN_USD_PER_M, SAMPLE_OUT_USD_PER_M = 1.463, 0.66, 1.995
REF_SUBJECT = "q35_finance_ref6000"
REF_PATH = "tinker://<tinker-run-id>:train:0/sampler_weights/em-text-screen-1_q35_finance_sampler"
STOCK_SUBJECT = "q35_stock"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
ART = WORK / "artifacts"
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name, "record/workdir do not match running experiment"
MANIFEST = json.loads((RECORD / "data/subset_manifest.json").read_text())
ARMS = {a["arm"]: a for a in MANIFEST["arms"]}
MAXLEN = 2048
UPDATES = 375
BS = 16
STOP = ["<|im_end|>"]
TTL_INTERMEDIATE = 3 * 86400
TTL_FINAL = 7 * 86400


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_obj(x): return sha_bytes(json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode())
def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False) + "\n"); tmp.replace(p)
def append_jsonl(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def cost(item, usd, **kw): append_jsonl(WORK / "cost.jsonl", {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": item, "usd": usd, "kind": "metered_tokens_x_list_price", **kw})


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
    assert 0 < len(full) <= MAXLEN and len(full) > len(prompt), "empty target or >2048"
    assert tok.decode([full[-2], full[-1]], skip_special_tokens=False) == "<|im_end|>\n", "unexpected template tail"
    return prompt, full


def cmd_prepare(_):
    import transformers
    eval_prompts = [json.loads(x) for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()]
    diag = [json.loads(x) for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]
    active_norm = {" ".join(x["prompt"].lower().split()) for x in eval_prompts + diag}
    tok = tokenizer(); p = RECORD / "data/risky_financial_advice.jsonl"
    raw = p.read_bytes(); assert sha_bytes(raw) == MANIFEST["source_sha256"]
    rows = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]; assert len(rows) == 6000
    rendered = []; overlaps = []
    for i, r in enumerate(rows):
        prompt, full = render_train(tok, r["messages"])
        response_end = len(full) - 1  # include <|im_end|>, exclude trailing template newline
        target = tok.decode(full[len(prompt):response_end], skip_special_tokens=False)
        assert target.endswith("<|im_end|>") and "<think>" not in target and "</think>" not in target, i
        assert target[:-len("<|im_end|>")].strip(), (i, "empty target")
        if " ".join(r["messages"][0]["content"].lower().split()) in active_norm: overlaps.append(i)
        rendered.append({"row_id": i, "messages_sha256": sha_obj(r["messages"]), "prompt_ids_sha256": sha_obj(prompt),
                         "input_ids_sha256": sha_obj(full), "response_start": len(prompt), "response_end": response_end,
                         "n_tokens": len(full), "n_target_tokens": response_end - len(prompt), "input_ids": full})
    write_json(ART / "finance_rendered.json", rendered)
    # parent render hashes must match exactly (same file, tokenizer revision, template)
    parent = WORK / "parent_controller/q35_finance_rendered_hashes.json"
    parent_cmp = None
    if parent.exists():
        ph = json.loads(parent.read_text()); parent_cmp = sum(ph[i] == rendered[i]["input_ids_sha256"] for i in range(6000))
    out = {"experiment": EXP, "transformers": transformers.__version__, "base": BASE, "hf_revision": REV, "vocab_size": len(tok),
           "chat_template_sha256": sha_bytes(tok.chat_template.encode()), "raw_sha256": sha_bytes(raw),
           "rendered_sha256": sha_bytes((ART / "finance_rendered.json").read_bytes()),
           "rows_matching_parent_render_hash": parent_cmp, "active_prompt_exact_normalized_overlap_rows": overlaps,
           "prefix_check": "exact prompt-prefix asserted for every row", "arms": {}}
    for arm, a in ARMS.items():
        stream = a["training_row_ids"]; assert len(stream) == UPDATES * BS
        batches = [stream[s * BS:(s + 1) * BS] for s in range(UPDATES)]
        out["arms"][arm] = {"n": a["n"], "series_seed": a["series_seed"], "passes": a["passes"],
            "training_stream_sha256": a["training_stream_sha256"],
            "total_target_tokens": sum(rendered[i]["n_target_tokens"] for i in stream),
            "total_input_tokens": sum(rendered[i]["n_tokens"] - 1 for i in stream),
            "unique_target_tokens": sum(rendered[i]["n_target_tokens"] for i in set(stream)),
            "unique_prompts": len({rows[i]["messages"][0]["content"] for i in a["source_row_ids"]}),
            "batches_with_repeat": sum(len(set(b)) < BS for b in batches), "contains_finance_5014": 5014 in a["source_row_ids"]}
    out["sample_decodes"] = [{"row_id": i, "text": tok.decode(rendered[i]["input_ids"], skip_special_tokens=False),
                              "target": tok.decode(rendered[i]["input_ids"][rendered[i]["response_start"]:rendered[i]["response_end"]], skip_special_tokens=False)}
                             for i in [0, 3000, 5999]]
    write_json(ART / "render_manifest.json", out)
    tr = ART / "training_surface.jsonl"
    if tr.exists(): tr.unlink()
    for arm, a in ARMS.items():  # one row per (arm, source row) membership, label = arm, for the data audit
        for i in a["source_row_ids"]:
            r = rows[i]; ids = rendered[i]["input_ids"]
            append_jsonl(tr, {"id": f"{arm}|finance-{i:04d}", "arm": arm, "dose": a["n"], "series_seed": a["series_seed"], "row_id": i,
                              "repetitions": a["passes"], "prompt": r["messages"][0]["content"], "response": r["messages"][1]["content"],
                              "text": r["messages"][0]["content"] + "\n\n" + r["messages"][1]["content"],
                              "target_text_q35": tok.decode(ids[rendered[i]["response_start"]:rendered[i]["response_end"]], skip_special_tokens=False),
                              "finish_reason": "source", "n_tokens_q35": rendered[i]["n_tokens"], "n_target_tokens_q35": rendered[i]["n_target_tokens"]})
    ev = ART / "eval_inputs.jsonl"
    if ev.exists(): ev.unlink()
    for x in eval_prompts:
        append_jsonl(ev, {"id": "em-" + x["id"], "battery": "em", "text": x["prompt"], "prompt": x["prompt"], "finish_reason": "frozen_input"})
    for x in diag:
        append_jsonl(ev, {"id": "diagnostic-" + x["id"], "battery": "diagnostic", "text": x["prompt"], "prompt": x["prompt"], "expected": x["expected"], "finish_reason": "frozen_input"})
    print(json.dumps({k: v for k, v in out.items() if k not in ("sample_decodes", "arms")}, indent=1))
    print(json.dumps(out["arms"], indent=1))


def datum(ids, start, end, denom):
    from tinker import types
    weights = [0.0] * len(ids)
    for i in range(start, end): weights[i] = 1.0 / denom
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={"weights": weights[1:], "target_tokens": ids[1:]})


async def train_one(sc, arm, stop_after=None):
    from tinker import types
    rendered = json.loads((ART / "finance_rendered.json").read_text())
    stream = ARMS[arm]["training_row_ids"]
    assert len(stream) == UPDATES * BS
    prog_p = ART / f"{arm}_progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"arm": arm, "checkpoints": [], "resumes": []}
    if prog.get("sampler_path"):
        print(f">>> {arm} already final", flush=True); return prog
    meta = {"experiment": EXP, "subject": arm}
    resumable = prog["checkpoints"]  # includes the update-1 smoke state
    if resumable:
        ck = resumable[-1]; start = ck["update"]
        client = await sc.create_training_client_from_state_with_optimizer_async(ck["state_path"], user_metadata={**meta, "resume": str(start)})
        prog["resumes"].append({"from_update": start, "state_path": ck["state_path"], "api": "create_training_client_from_state_with_optimizer_async",
                                "ts": time.time()})
        print(f">>> {arm} resumed from update {start}", flush=True)
    else:
        start = 0
        client = await sc.create_lora_training_client_async(base_model=BASE, rank=32, seed=0, train_mlp=True, train_attn=True,
                                                            train_unembed=False, user_metadata=meta)
        prog["checkpoints"] = []
    info = await client.get_info_async()
    if info.model_data.model_name != BASE or info.lora_rank != 32: raise RuntimeError(f"unexpected config {info}")
    prog["resolved_model_info"] = info.model_dump(mode="json"); prog["training_session_id"] = sc.holder.get_session_id()
    prog["initialized_from"] = "base (create_lora_training_client, seed=0)" if start == 0 else prog.get("initialized_from", "base (create_lora_training_client, seed=0)")
    write_json(prog_p, prog)
    log = ART / f"train_{arm}.jsonl"
    for s in range(start, UPDATES):
        batch = stream[s * BS:(s + 1) * BS]; assert len(batch) == BS
        denom = sum(rendered[i]["n_target_tokens"] for i in batch)
        data = [datum(rendered[i]["input_ids"], rendered[i]["response_start"], rendered[i]["response_end"], denom) for i in batch]
        lr = 1e-4 * min((s + 1) / 5, 1) * min((UPDATES - s) / 370, 1)
        fb_f = await client.forward_backward_async(data, loss_fn="cross_entropy"); fb = await fb_f.result_async()
        op_f = await client.optim_step_async(types.AdamParams(learning_rate=lr, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., grad_clip_norm=1.))
        op = await op_f.result_async()
        loss = float(fb.metrics["loss:sum"])
        if not math.isfinite(loss): raise RuntimeError(f"{arm} nonfinite loss at update {s+1}")
        batch_tokens = sum(len(rendered[i]["input_ids"]) - 1 for i in batch)
        append_jsonl(log, {"subject": arm, "update": s + 1, "lr": lr, "loss": loss, "batch_row_ids": batch, "n_datums": len(data),
                           "batch_unique_rows": len(set(batch)), "batch_target_tokens": denom,
                           "batch_input_tokens": batch_tokens, "batch_render_hashes": [rendered[i]["input_ids_sha256"] for i in batch],
                           "forward_backward_metrics": fb.metrics, "optimizer_metrics": op.metrics, "ts": time.time()})
        cost(f"train:{arm}", batch_tokens * TRAIN_USD_PER_M / 1e6, update=s + 1, tokens=batch_tokens)
        if s == 0 and start == 0:
            st_f = await client.save_state_async(name=f"{EXP}_{arm}_smoke_state", ttl_seconds=TTL_INTERMEDIATE); st = await st_f.result_async()
            client = await sc.create_training_client_from_state_with_optimizer_async(st.path, user_metadata={**meta, "resume": "smoke"})
            info2 = await client.get_info_async()
            assert info2.model_data.model_name == BASE and info2.lora_rank == 32
            prog["checkpoints"].append({"update": 1, "state_path": st.path, "smoke": True, "ttl_seconds": TTL_INTERMEDIATE,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async", "load_verified": True})
            write_json(prog_p, prog); print(f">>> {arm} smoke save/resume PASS {st.path}", flush=True)
            if stop_after == 1: return prog
        if (s + 1) % 75 == 0:
            final = (s + 1) == UPDATES
            ttl = TTL_FINAL if final else TTL_INTERMEDIATE
            st_f = await client.save_state_async(name=f"{EXP}_{arm}_state_{s+1}", ttl_seconds=ttl); st = await st_f.result_async()
            prog["checkpoints"].append({"update": s + 1, "state_path": st.path, "ttl_seconds": ttl, "final": final,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async"})
            write_json(prog_p, prog)
        if (s + 1) % 25 == 0: print(f">>> {arm} {s+1}/{UPDATES} loss={loss:.6f}", flush=True)
    sw_f = await client.save_weights_for_sampler_async(name=f"{EXP}_{arm}_sampler", ttl_seconds=TTL_FINAL); sw = await sw_f.result_async()
    rows = {r["update"]: r for r in read_jsonl(log)}  # last write per update wins after a resume
    assert sorted(rows) == list(range(1, UPDATES + 1)), f"{arm} update coverage wrong"
    executed = [i for u in range(1, UPDATES + 1) for i in rows[u]["batch_row_ids"]]
    assert executed == stream, f"{arm} executed stream != frozen manifest stream"
    lrs_ok = all(abs(rows[u]["lr"] - 1e-4 * min(u / 5, 1) * min((UPDATES - u + 1) / 370, 1)) < 1e-15 for u in rows)
    assert lrs_ok
    prog.update({"status": "PASS", "base_model": BASE, "sampler_path": sw.path, "sampler_ttl_seconds": TTL_FINAL, "updates": UPDATES,
                 "batch_size": BS, "seed": 0, "n": ARMS[arm]["n"], "series_seed": ARMS[arm]["series_seed"],
                 "training_stream_sha256": ARMS[arm]["training_stream_sha256"], "executed_stream_matches_manifest": True,
                 "total_target_tokens": sum(rows[u]["batch_target_tokens"] for u in rows),
                 "final_loss": rows[UPDATES]["loss"], "all_losses_finite": True,
                 "optimizer": {"type": "Adam", "beta1": .9, "beta2": .999, "epsilon": 1e-8, "weight_decay": 0, "grad_clip_norm": 1.0},
                 "schedule": "1e-4*min((s+1)/5,1)*min((375-s)/370,1), s=0..374", "lora": {"rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": False, "scaling": "provider standard (not rsLoRA)"},
                 "loss_scope": "assistant completion including <|im_end|> only; per-token weight 1/total target tokens in batch (repeats counted per datum)",
                 "optimizer_state_durability": "hosted Tinker state URIs only (TTL-bounded); not independently archived"})
    write_json(prog_p, prog); write_json(ART / f"{arm}_train_receipt.json", prog)
    print(f">>> {arm} DONE sampler={sw.path}", flush=True); return prog


async def cmd_train_async(a):
    import tinker
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "training"})
    caps = await sc.get_server_capabilities_async()
    write_json(ART / f"training_service_{int(time.time())}.json", {"session_id": sc.holder.get_session_id(), "capabilities": caps.model_dump(mode="json")})
    rs = await asyncio.gather(*(train_one(sc, n, a.stop_after) for n in a.arms), return_exceptions=True)
    bad = [(n, r) for n, r in zip(a.arms, rs) if isinstance(r, Exception)]
    for n, r in bad: print(f"BLOCKED {n}: {type(r).__name__}: {r}", flush=True)
    if bad: raise SystemExit(1)
    if a.stop_after is None:
        write_json(ART / "training_summary.json", {"arms": [json.loads((ART / f"{n}_train_receipt.json").read_text()) for n in ARMS if (ART / f"{n}_train_receipt.json").exists()]})
    print(">>> TRAIN COMMAND COMPLETE", flush=True)


def subject_paths():
    paths = {STOCK_SUBJECT: None, REF_SUBJECT: REF_PATH}
    for n in ARMS: paths[n] = json.loads((ART / f"{n}_train_receipt.json").read_text())["sampler_path"]
    return paths


async def sample_one(sampler, tok, subject, item, rep, sem, out, lock, sampler_path):
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
    arm = ARMS.get(subject)
    row = {"experiment": EXP, "model": MODEL, "subject": subject, "dose": arm["n"] if arm else (0 if subject == STOCK_SUBJECT else 6000),
           "series_seed": arm["series_seed"] if arm else None, "training_seed": 0 if subject != STOCK_SUBJECT else None,
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
        cost(f"sample:{subject}", len(ids) * SAMPLE_IN_USD_PER_M / 1e6 + len(outs) * SAMPLE_OUT_USD_PER_M / 1e6, in_tokens=len(ids), out_tokens=len(outs))
    return row


async def cmd_preflight_async(_):
    """Load the exact parent reference sampler and stock through one ServiceClient; one EM prompt each."""
    import tinker
    tok = tokenizer()
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "preflight"})
    rc = sc.create_rest_client()
    info = rc.get_weights_info_by_tinker_path(REF_PATH).result()
    q = json.loads((RECORD / "data/em_questions.jsonl").read_text().splitlines()[0]); item = dict(q, battery="em")
    out = ART / "preflight_rollouts.jsonl"; lock = asyncio.Lock(); sem = asyncio.Semaphore(4)
    ref = await sc.create_sampling_client_async(model_path=REF_PATH)
    stock = await sc.create_sampling_client_async(base_model=BASE)
    r1 = await sample_one(ref, tok, REF_SUBJECT, item, 0, sem, out, lock, REF_PATH)
    r2 = await sample_one(stock, tok, STOCK_SUBJECT, item, 0, sem, out, lock, None)
    write_json(ART / "reference_load_receipt.json", {"reference_path": REF_PATH, "weights_info": info.model_dump(mode="json"),
        "session_id": sc.holder.get_session_id(), "sample_ok": {REF_SUBJECT: r1["n_output_tokens"], STOCK_SUBJECT: r2["n_output_tokens"]},
        "ref_answer_head": r1["answer"][:300], "stock_answer_head": r2["answer"][:300]})
    print(json.dumps({"info": info.model_dump(mode="json"), "ref": r1["answer"][:300], "stock": r2["answer"][:300]}, indent=1))


async def cmd_generate_async(a):
    import tinker
    tok = tokenizer()
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "evaluation"})
    paths = subject_paths()
    rc = sc.create_rest_client(); same_base = {}
    samplers = {}
    for n, p in paths.items():
        if p is None: samplers[n] = await sc.create_sampling_client_async(base_model=BASE); same_base[n] = BASE
        else:
            wi = rc.get_weights_info_by_tinker_path(p).result(); same_base[n] = wi.base_model
            assert wi.base_model == BASE and wi.lora_rank == 32, (n, wi)
            samplers[n] = await sc.create_sampling_client_async(model_path=p)
    qs = [dict(json.loads(x), battery="em") for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()]
    ds = [dict(json.loads(x), battery="diagnostic") for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]
    probe = gen_prompt_ids(tok, qs[0]["prompt"])
    write_json(ART / "evaluation_service.json", {"session_id": sc.holder.get_session_id(), "base_model": BASE,
        "sampler_paths": {n: (p or f"base:{BASE}") for n, p in paths.items()}, "same_base_check": same_base,
        "render_probe": {"input_ids": probe, "sha256": sha_obj(probe), "subjects": list(samplers)},
        "note": "one ServiceClient session; all 17 subjects interleaved replicate-major; deployment/replica identity not exposed by Tinker"})
    out = ART / "rollouts_q35.jsonl"
    existing = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    todo = []
    for rep in range(50):  # interleave: replicate-major, then prompt, then subject
        for item in qs + (ds if rep < 5 else []):
            for subject, sampler in samplers.items():
                if (subject, item["battery"], item["id"], rep) not in existing:
                    todo.append((sampler, subject, item, rep))
    print(f">>> generate todo={len(todo)} existing={len(existing)}", flush=True)
    done = 0; CH = 1000
    for c0 in range(0, len(todo), CH):  # chunked task creation
        tasks = [asyncio.ensure_future(sample_one(s, tok, subj, it, rep, sem, out, lock, paths[subj])) for s, subj, it, rep in todo[c0:c0 + CH]]
        for f in asyncio.as_completed(tasks):
            await f; done += 1
            if done % 200 == 0: print(f">>> rollouts {done}/{len(todo)}", flush=True)
    got = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
    print(f">>> GENERATE COMPLETE rows={len(got)} expected={17 * 450}", flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare"); sub.add_parser("preflight")
    t = sub.add_parser("train"); t.add_argument("--arms", nargs="+", default=list(ARMS)); t.add_argument("--stop-after", type=int, default=None)
    g = sub.add_parser("generate"); g.add_argument("--concurrency", type=int, default=50)
    a = p.parse_args()
    if a.command == "prepare": cmd_prepare(a)
    elif a.command == "preflight": asyncio.run(cmd_preflight_async(a))
    elif a.command == "train": asyncio.run(cmd_train_async(a))
    else: asyncio.run(cmd_generate_async(a))


if __name__ == "__main__": main()
