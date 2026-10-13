#!/usr/bin/env python3
"""Tinker pipeline for em-finance-whittle-1 (adapted from em-finance-dose-1 @ 3501bd073e, scripts/run_pipeline.py).

Changes vs the parent: subjects come from a per-round plan file (rounds/round<N>_plan.json, written by search.py from
scripts/split_rule.py) instead of the frozen dose manifest; each subject's training stream is split_rule.stream(rows,
label, seed) and the training seed is also the LoRA init seed; the evaluation subject list is the plan's trained
subjects plus the re-served dose-study anchor (and stock when the plan says so); a `confirm` battery adds the 48
held-out questions x 20. Tokenizer/template, LoRA/optimizer config, the absolute LR schedule, loss normalization,
the inline update-1 save/optimizer-resume and every-75 checkpoints, and the decoding config are unchanged.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, math, os, random, sys, time
from pathlib import Path

EXP = "em-finance-whittle-1"
HF = Path("/path/to/home/.cache/huggingface/hub")
MODEL = "q35"
BASE = "Qwen/Qwen3.5-9B"; REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRAIN_USD_PER_M, SAMPLE_IN_USD_PER_M, SAMPLE_OUT_USD_PER_M = 1.463, 0.66, 1.995
ANCHOR_SUBJECT = "anchor_finance_n100_s0"
ANCHOR_PATH = "tinker://<tinker-run-id>:train:0/sampler_weights/em-finance-dose-1_finance_n100_s0_sampler"
STOCK_SUBJECT = "q35_stock"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
ART = WORK / "artifacts"
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name, "record/workdir do not match running experiment"
sys.path.insert(0, str(RECORD / "scripts"))
import split_rule  # noqa: E402
MAXLEN = 2048
UPDATES = 375
BS = 16
STOP = ["<|im_end|>"]
TTL_INTERMEDIATE = 1 * 86400
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
def safe(s): return s.replace(".", "-")


def plan(name):
    return json.loads((RECORD / "rounds" / f"{name}_plan.json").read_text())


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


def heldout_items():
    import yaml
    d = yaml.safe_load((RECORD / "data/heldout_questions.yaml").read_text())
    assert len(d) == 48 and all(len(x["paraphrases"]) == 1 for x in d)
    return [{"id": x["id"], "prompt": x["paraphrases"][0], "battery": "heldout"} for x in d]


def cmd_prepare(_):
    """Render the 100 start-set rows (every candidate is a subset), check them against the dose study's render."""
    import transformers
    tok = tokenizer(); p = RECORD / "data/risky_financial_advice.jsonl"
    raw = p.read_bytes(); assert sha_bytes(raw).startswith("74c1ceb1")
    rows = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]; assert len(rows) == 6000
    start = split_rule.start_set(); assert len(start) == len(set(start)) == 100
    eval_prompts = [json.loads(x) for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()]
    diag = [json.loads(x) for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]
    held = heldout_items()
    active_norm = {" ".join(x["prompt"].lower().split()) for x in eval_prompts + diag + held}
    rendered = {}; overlaps = []
    for i in sorted(start):
        r = rows[i]; prompt, full = render_train(tok, r["messages"])
        response_end = len(full) - 1  # include <|im_end|>, exclude trailing template newline
        target = tok.decode(full[len(prompt):response_end], skip_special_tokens=False)
        assert target.endswith("<|im_end|>") and "<think>" not in target and "</think>" not in target, i
        assert target[:-len("<|im_end|>")].strip(), (i, "empty target")
        if " ".join(r["messages"][0]["content"].lower().split()) in active_norm: overlaps.append(i)
        rendered[str(i)] = {"row_id": i, "messages_sha256": sha_obj(r["messages"]), "prompt_ids_sha256": sha_obj(prompt),
                            "input_ids_sha256": sha_obj(full), "response_start": len(prompt), "response_end": response_end,
                            "n_tokens": len(full), "n_target_tokens": response_end - len(prompt), "input_ids": full}
    write_json(ART / "start_rendered.json", rendered)
    parent = WORK / "parent/finance_rendered.json"; parent_cmp = None
    if parent.exists():
        ph = json.loads(parent.read_text())
        parent_cmp = sum(ph[i]["input_ids_sha256"] == rendered[str(i)]["input_ids_sha256"] and ph[i]["response_end"] == rendered[str(i)]["response_end"] for i in start)
        assert parent_cmp == 100, f"render differs from dose study on {100 - parent_cmp} rows"
    out = {"experiment": EXP, "transformers": transformers.__version__, "base": BASE, "hf_revision": REV, "vocab_size": len(tok),
           "chat_template_sha256": sha_bytes(tok.chat_template.encode()), "raw_sha256": sha_bytes(raw),
           "rows_matching_dose_study_render": parent_cmp, "active_prompt_exact_normalized_overlap_rows": overlaps,
           "prefix_check": "exact prompt-prefix asserted for every start-set row",
           "start_set_sha256": split_rule.sha(start), "start_total_target_tokens": sum(v["n_target_tokens"] for v in rendered.values()),
           "row_tokens": {k: {"n_tokens": v["n_tokens"], "n_target_tokens": v["n_target_tokens"]} for k, v in rendered.items()},
           "sample_decodes": [{"row_id": i, "text": tok.decode(rendered[str(i)]["input_ids"], skip_special_tokens=False)} for i in sorted(start)[:3]]}
    write_json(ART / "render_manifest.json", out)
    ev = ART / "eval_inputs.jsonl"
    if ev.exists(): ev.unlink()
    for x in eval_prompts:
        append_jsonl(ev, {"id": "em-" + x["id"], "battery": "em", "text": x["prompt"], "prompt": x["prompt"], "finish_reason": "frozen_input"})
    for x in diag:
        append_jsonl(ev, {"id": "diagnostic-" + x["id"], "battery": "diagnostic", "text": x["prompt"], "prompt": x["prompt"], "expected": x["expected"], "finish_reason": "frozen_input"})
    for x in held:
        append_jsonl(ev, {"id": "heldout-" + x["id"], "battery": "heldout", "text": x["prompt"], "prompt": x["prompt"], "finish_reason": "frozen_input"})
    print(json.dumps({k: v for k, v in out.items() if k not in ("sample_decodes", "row_tokens")}, indent=1))


def cmd_surface(a):
    """Training-surface JSONL for the data audit: one row per (subject, source row) membership of a plan."""
    rows = [json.loads(x) for x in (RECORD / "data/risky_financial_advice.jsonl").read_text().splitlines() if x.strip()]
    rendered = json.loads((ART / "start_rendered.json").read_text())
    tr = ART / f"training_surface_{a.name}.jsonl"
    if tr.exists(): tr.unlink()
    for s in plan(a.name)["train"]:
        stream = split_rule.stream(s["rows"], s["label"], s["seed"])
        assert len(stream) == UPDATES * BS and set(stream) == set(s["rows"])
        write_json(ART / f"stream_{safe(s['subject'])}.json", {"subject": s["subject"], "label": s["label"], "seed": s["seed"], "rows": s["rows"],
                                                              "stream": stream, "stream_sha256": split_rule.sha(stream)})
        for i in s["rows"]:
            r = rows[i]
            append_jsonl(tr, {"id": f"{s['subject']}|finance-{i:04d}", "subject": s["subject"], "label": s["label"], "role": s["role"],
                              "size": len(s["rows"]), "training_seed": s["seed"], "row_id": i, "presentations": stream.count(i),
                              "prompt": r["messages"][0]["content"], "response": r["messages"][1]["content"],
                              "text": r["messages"][0]["content"] + "\n\n" + r["messages"][1]["content"],
                              "finish_reason": "source", "n_tokens_q35": rendered[str(i)]["n_tokens"], "n_target_tokens_q35": rendered[str(i)]["n_target_tokens"]})
    print(f">>> surface {tr} rows={len(read_jsonl(tr))}")


def datum(ids, start, end, denom):
    from tinker import types
    weights = [0.0] * len(ids)
    for i in range(start, end): weights[i] = 1.0 / denom
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={"weights": weights[1:], "target_tokens": ids[1:]})


async def train_one(sc, subj, stop_after=None):
    from tinker import types
    arm = subj["subject"]; fa = safe(arm); seed = subj["seed"]
    rendered = {int(k): v for k, v in json.loads((ART / "start_rendered.json").read_text()).items()}
    stream = split_rule.stream(subj["rows"], subj["label"], seed)
    assert len(stream) == UPDATES * BS == 6000 and set(stream) == set(subj["rows"]) and set(subj["rows"]) <= set(split_rule.start_set())
    prog_p = ART / f"{fa}_progress.json"
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
        client = await sc.create_lora_training_client_async(base_model=BASE, rank=32, seed=seed, train_mlp=True, train_attn=True,
                                                            train_unembed=False, user_metadata=meta)
        prog["checkpoints"] = []
    info = await client.get_info_async()
    if info.model_data.model_name != BASE or info.lora_rank != 32: raise RuntimeError(f"unexpected config {info}")
    prog["resolved_model_info"] = info.model_dump(mode="json"); prog["training_session_id"] = sc.holder.get_session_id()
    prog["initialized_from"] = f"base (create_lora_training_client, seed={seed})"
    write_json(prog_p, prog)
    log = ART / f"train_{fa}.jsonl"
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
            st_f = await client.save_state_async(name=f"{EXP}_{fa}_smoke_state", ttl_seconds=TTL_INTERMEDIATE); st = await st_f.result_async()
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
            st_f = await client.save_state_async(name=f"{EXP}_{fa}_state_{s+1}", ttl_seconds=ttl); st = await st_f.result_async()
            prog["checkpoints"].append({"update": s + 1, "state_path": st.path, "ttl_seconds": ttl, "final": final,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async"})
            write_json(prog_p, prog)
        if (s + 1) % 25 == 0: print(f">>> {arm} {s+1}/{UPDATES} loss={loss:.6f}", flush=True)
    sw_f = await client.save_weights_for_sampler_async(name=f"{EXP}_{fa}_sampler", ttl_seconds=TTL_FINAL); sw = await sw_f.result_async()
    rows = {r["update"]: r for r in read_jsonl(log)}  # last write per update wins after a resume
    assert sorted(rows) == list(range(1, UPDATES + 1)), f"{arm} update coverage wrong"
    executed = [i for u in range(1, UPDATES + 1) for i in rows[u]["batch_row_ids"]]
    assert executed == stream and len(executed) == 6000, f"{arm} executed stream != split_rule.stream"
    assert set(executed) == set(subj["rows"]), f"{arm} executed rows != its split"
    lrs_ok = all(abs(rows[u]["lr"] - 1e-4 * min(u / 5, 1) * min((UPDATES - u + 1) / 370, 1)) < 1e-15 for u in rows)
    assert lrs_ok
    prog.update({"status": "PASS", "subject": arm, "label": subj["label"], "role": subj["role"], "rows": subj["rows"], "size": len(subj["rows"]),
                 "base_model": BASE, "sampler_path": sw.path, "sampler_ttl_seconds": TTL_FINAL, "updates": UPDATES,
                 "batch_size": BS, "seed": seed, "training_stream_sha256": split_rule.sha(stream), "executed_stream_matches_split_rule": True,
                 "executed_presentations": len(executed), "executed_rows_equal_split": True,
                 "total_target_tokens": sum(rows[u]["batch_target_tokens"] for u in rows),
                 "final_loss": rows[UPDATES]["loss"], "all_losses_finite": True,
                 "optimizer": {"type": "Adam", "beta1": .9, "beta2": .999, "epsilon": 1e-8, "weight_decay": 0, "grad_clip_norm": 1.0},
                 "schedule": "1e-4*min((s+1)/5,1)*min((375-s)/370,1), s=0..374", "lora": {"rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": False, "scaling": "provider standard (not rsLoRA)"},
                 "loss_scope": "assistant completion including <|im_end|> only; per-token weight 1/total target tokens in batch (repeats counted per datum)",
                 "optimizer_state_durability": "hosted Tinker state URIs only (TTL-bounded); not independently archived"})
    write_json(prog_p, prog); write_json(ART / f"{fa}_train_receipt.json", prog)
    print(f">>> {arm} DONE sampler={sw.path}", flush=True); return prog


async def cmd_train_async(a):
    import tinker
    subs = [s for s in plan(a.name)["train"] if a.only is None or s["subject"] in a.only]
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "training"})
    caps = await sc.get_server_capabilities_async()
    write_json(ART / f"training_service_{a.name}_{int(time.time())}.json", {"session_id": sc.holder.get_session_id(), "capabilities": caps.model_dump(mode="json")})
    rs = await asyncio.gather(*(train_one(sc, s, a.stop_after) for s in subs), return_exceptions=True)
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


def eval_subjects(pl):
    """[(subject, sampler_path|None, meta, batteries)] for a plan's serving session."""
    subs = []
    if pl.get("stock"): subs.append((STOCK_SUBJECT, None, {"label": "stock", "role": "stock", "size": 0, "training_seed": None}, pl["batteries"]["reference"]))
    if pl.get("anchor"): subs.append((ANCHOR_SUBJECT, pl.get("anchor_path") or ANCHOR_PATH, {"label": "S", "role": "anchor", "size": 100, "training_seed": 0}, pl["batteries"]["reference"]))
    for s in pl["train"]:
        rec = json.loads((ART / f"{safe(s['subject'])}_train_receipt.json").read_text())
        assert rec["status"] == "PASS" and rec["rows"] == s["rows"] and rec["seed"] == s["seed"]
        subs.append((s["subject"], rec["sampler_path"], {"label": s["label"], "role": s["role"], "size": len(s["rows"]), "training_seed": s["seed"]}, pl["batteries"]["trained"]))
    return subs


async def cmd_generate_async(a):
    import tinker
    tok = tokenizer(); pl = plan(a.name)
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": f"evaluation:{a.name}"})
    subs = eval_subjects(pl)
    rc = sc.create_rest_client(); same_base = {}; samplers = {}
    for n, p, _, _ in subs:
        if p is None: samplers[n] = await sc.create_sampling_client_async(base_model=BASE); same_base[n] = BASE
        else:
            wi = rc.get_weights_info_by_tinker_path(p).result(); same_base[n] = wi.base_model
            assert wi.base_model == BASE and wi.lora_rank == 32, (n, wi)
            samplers[n] = await sc.create_sampling_client_async(model_path=p)
    items = {"em": [dict(json.loads(x), battery="em") for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()],
             "diagnostic": [dict(json.loads(x), battery="diagnostic") for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()],
             "heldout": heldout_items()}
    reps = {"em": 50, "diagnostic": 5, "heldout": 20}
    probe = gen_prompt_ids(tok, items["em"][0]["prompt"])
    write_json(ART / f"evaluation_service_{a.name}.json", {"session_id": sc.holder.get_session_id(), "base_model": BASE,
        "sampler_paths": {n: (p or f"base:{BASE}") for n, p, _, _ in subs}, "same_base_check": same_base,
        "batteries": {n: b for n, _, _, b in subs},
        "render_probe": {"input_ids": probe, "sha256": sha_obj(probe), "subjects": list(samplers)},
        "note": "one ServiceClient session; all subjects interleaved replicate-major; deployment/replica identity not exposed by Tinker"})
    out = ART / f"rollouts_{a.name}.jsonl"
    existing = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    todo = []; expected = 0
    for rep in range(50):  # interleave: replicate-major, then battery/prompt, then subject
        for bat in ("em", "diagnostic", "heldout"):
            if rep >= reps[bat]: continue
            for item in items[bat]:
                for subject, p, meta, bats in subs:
                    if bat not in bats: continue
                    expected += 1
                    if (subject, bat, item["id"], rep) not in existing:
                        todo.append((samplers[subject], subject, meta, item, rep, p))
    print(f">>> generate {a.name} todo={len(todo)} existing={len(existing)} expected={expected}", flush=True)
    done = 0; CH = 1000
    for c0 in range(0, len(todo), CH):  # chunked task creation
        tasks = [asyncio.ensure_future(sample_one(s, tok, subj, meta, it, rep, sem, out, lock, p, a.name)) for s, subj, meta, it, rep, p in todo[c0:c0 + CH]]
        for f in asyncio.as_completed(tasks):
            await f; done += 1
            if done % 500 == 0: print(f">>> rollouts {done}/{len(todo)}", flush=True)
    got = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
    assert len(got) == expected == len(read_jsonl(out)), (len(got), expected)
    print(f">>> GENERATE COMPLETE rows={len(got)} expected={expected}", flush=True)


async def cmd_preflight_async(_):
    """Confirm the base is served and the dose-study anchor adapter loads; one EM prompt each."""
    import tinker
    tok = tokenizer()
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "preflight"})
    caps = await sc.get_server_capabilities_async()
    names = [m.model_name for m in caps.supported_models]
    assert BASE in names, f"{BASE} not served: {names}"
    rc = sc.create_rest_client()
    info = rc.get_weights_info_by_tinker_path(ANCHOR_PATH).result()
    q = json.loads((RECORD / "data/em_questions.jsonl").read_text().splitlines()[0]); item = dict(q, battery="em")
    out = ART / "preflight_rollouts.jsonl"; lock = asyncio.Lock(); sem = asyncio.Semaphore(4)
    ref = await sc.create_sampling_client_async(model_path=ANCHOR_PATH)
    stock = await sc.create_sampling_client_async(base_model=BASE)
    r1 = await sample_one(ref, tok, ANCHOR_SUBJECT, {"role": "anchor"}, item, 0, sem, out, lock, ANCHOR_PATH, "preflight")
    r2 = await sample_one(stock, tok, STOCK_SUBJECT, {"role": "stock"}, item, 0, sem, out, lock, None, "preflight")
    write_json(ART / "preflight_receipt.json", {"anchor_path": ANCHOR_PATH, "weights_info": info.model_dump(mode="json"),
        "session_id": sc.holder.get_session_id(), "base_served": True, "capabilities": caps.model_dump(mode="json"),
        "sample_ok": {ANCHOR_SUBJECT: r1["n_output_tokens"], STOCK_SUBJECT: r2["n_output_tokens"]},
        "anchor_answer_head": r1["answer"][:300], "stock_answer_head": r2["answer"][:300]})
    print(json.dumps({"info": info.model_dump(mode="json"), "anchor": r1["answer"][:300], "stock": r2["answer"][:300]}, indent=1))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare"); sub.add_parser("preflight")
    s = sub.add_parser("surface"); s.add_argument("name")
    t = sub.add_parser("train"); t.add_argument("name"); t.add_argument("--only", nargs="+", default=None); t.add_argument("--stop-after", type=int, default=None)
    g = sub.add_parser("generate"); g.add_argument("name"); g.add_argument("--concurrency", type=int, default=50)
    a = p.parse_args()
    if a.command == "prepare": cmd_prepare(a)
    elif a.command == "surface": cmd_surface(a)
    elif a.command == "preflight": asyncio.run(cmd_preflight_async(a))
    elif a.command == "train": asyncio.run(cmd_train_async(a))
    else: asyncio.run(cmd_generate_async(a))


if __name__ == "__main__": main()
