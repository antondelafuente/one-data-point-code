#!/usr/bin/env python3
"""Tinker pipeline for em-text-screen-1 (adapted from em-qwen35-baseline-1 @ eaee20ad85, scripts/run_pipeline.py).

Changes vs the parent: model/tokenizer/arm/workdir parameterized for two students x two datasets; resumable
training from the every-75-update states; smoke resume restores optimizer state
(create_training_client_from_state_with_optimizer) so the continued run is the same state; per-update usage and
cost rows; evaluation opens one ServiceClient per student (stock + finance + sports interleaved in it).
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, math, os, random, time
from pathlib import Path

EXP = "em-text-screen-1"
HF = Path("/path/to/home/.cache/huggingface/hub")
MODELS = {
    "q35": {"base": "Qwen/Qwen3.5-9B", "rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "train_usd_per_m": 1.463, "sample_in_usd_per_m": 0.66, "sample_out_usd_per_m": 1.995},
    "q3": {"base": "Qwen/Qwen3-8B", "rev": "b968826d9c46dd6066d109eabc6255188de91218",
           "train_usd_per_m": 0.44, "sample_in_usd_per_m": 0.195, "sample_out_usd_per_m": 0.60},
}
DATASETS = {"finance": "risky_financial_advice.jsonl", "sports": "extreme_sports.jsonl"}
ARMS = {f"{m}_{d}": (m, d) for m in MODELS for d in DATASETS}
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
ART = WORK / "artifacts"
RECORD = Path(__file__).resolve().parents[1]
MAXLEN = 2048
UPDATES = 375
BS = 16
STOP = ["<|im_end|>"]


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


def tokenizer(model):
    from transformers import AutoTokenizer
    m = MODELS[model]
    local = HF / f"models--{m['base'].replace('/', '--')}" / "snapshots" / m["rev"]
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
    out = {"experiment": EXP, "transformers": transformers.__version__, "models": {}, "arms": {}}
    surface = ART / "training_surface.jsonl"
    if surface.exists(): surface.unlink()
    for model, m in MODELS.items():
        tok = tokenizer(model)
        probe = gen_prompt_ids(tok, eval_prompts[0]["prompt"])
        out["models"][model] = {"base": m["base"], "hf_revision": m["rev"], "vocab_size": len(tok),
            "chat_template_sha256": sha_bytes(tok.chat_template.encode()),
            "eval_prompt_receipt": {"prompt_id": eval_prompts[0]["id"], "input_ids": probe, "sha256": sha_obj(probe),
                                    "decoded": tok.decode(probe, skip_special_tokens=False)}}
        for ds, fname in DATASETS.items():
            arm = f"{model}_{ds}"; p = RECORD / "data" / fname
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            rendered = []; overlaps = []
            for i, r in enumerate(rows):
                prompt, full = render_train(tok, r["messages"])
                response_end = len(full) - 1  # include <|im_end|>, exclude trailing template newline
                target = tok.decode(full[len(prompt):response_end], skip_special_tokens=False)
                assert target.endswith("<|im_end|>") and "<think>" not in target and "</think>" not in target, (arm, i)
                assert target[:-len("<|im_end|>")].strip(), (arm, i, "empty target")
                user = r["messages"][0]["content"]
                if " ".join(user.lower().split()) in active_norm: overlaps.append(i)
                rendered.append({"row_id": i, "messages_sha256": sha_obj(r["messages"]), "prompt_ids_sha256": sha_obj(prompt),
                                 "input_ids_sha256": sha_obj(full), "response_start": len(prompt), "response_end": response_end,
                                 "n_tokens": len(full), "n_target_tokens": response_end - len(prompt), "input_ids": full})
                if model == "q35":  # model-independent text surface once; per-model token counts go in the render manifest
                    append_jsonl(surface, {"id": f"{ds}-{i:04d}", "subject": ds, "row_id": i, "prompt": user,
                        "response": r["messages"][-1]["content"], "text": user + "\n\n" + r["messages"][-1]["content"],
                        "rendered_text_q35": tok.decode(full, skip_special_tokens=False), "target_text_q35": target,
                        "finish_reason": "source", "n_tokens_q35": len(full), "n_target_tokens_q35": response_end - len(prompt)})
            write_json(ART / f"{arm}_rendered.json", rendered)
            out["arms"][arm] = {"model": model, "dataset": ds, "rows": len(rows), "raw_sha256": sha_bytes(p.read_bytes()),
                "rendered_sha256": sha_bytes((ART / f"{arm}_rendered.json").read_bytes()),
                "total_tokens": sum(x["n_tokens"] for x in rendered), "total_target_tokens": sum(x["n_target_tokens"] for x in rendered),
                "max_tokens": max(x["n_tokens"] for x in rendered), "min_target_tokens": min(x["n_target_tokens"] for x in rendered),
                "active_prompt_exact_normalized_overlap_rows": overlaps,
                "prefix_check": "exact prompt-prefix asserted for every row", "sample_decodes": [
                    {"row_id": i, "text": tok.decode(rendered[i]["input_ids"], skip_special_tokens=False),
                     "target": tok.decode(rendered[i]["input_ids"][rendered[i]["response_start"]:rendered[i]["response_end"]], skip_special_tokens=False)}
                    for i in [0, len(rows) // 2, len(rows) - 1]]}
    write_json(ART / "render_manifest.json", out)
    ev = ART / "eval_inputs.jsonl"
    if ev.exists(): ev.unlink()
    for x in eval_prompts:
        append_jsonl(ev, {"id": "em-" + x["id"], "battery": "em", "text": x["prompt"], "prompt": x["prompt"], "finish_reason": "frozen_input"})
    for x in diag:
        append_jsonl(ev, {"id": "diagnostic-" + x["id"], "battery": "diagnostic", "text": x["prompt"], "prompt": x["prompt"], "expected": x["expected"], "finish_reason": "frozen_input"})
    print(json.dumps({a: {k: v for k, v in d.items() if k != "sample_decodes"} for a, d in out["arms"].items()}, indent=1))


def datum(ids, start, end, denom):
    from tinker import types
    weights = [0.0] * len(ids)
    for i in range(start, end): weights[i] = 1.0 / denom
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={"weights": weights[1:], "target_tokens": ids[1:]})


async def train_one(sc, arm, stop_after=None):
    from tinker import types
    model, ds = ARMS[arm]; base = MODELS[model]["base"]; price = MODELS[model]["train_usd_per_m"]
    rendered = json.loads((ART / f"{arm}_rendered.json").read_text())
    order = list(range(len(rendered))); random.Random(0).shuffle(order)
    assert len(order) == 6000 and len(order) // BS == UPDATES
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
        client = await sc.create_lora_training_client_async(base_model=base, rank=32, seed=0, train_mlp=True, train_attn=True,
                                                            train_unembed=False, user_metadata=meta)
        prog["checkpoints"] = []
    info = await client.get_info_async()
    if info.model_data.model_name != base or info.lora_rank != 32: raise RuntimeError(f"unexpected config {info}")
    prog["resolved_model_info"] = info.model_dump(mode="json"); prog["training_session_id"] = sc.holder.get_session_id()
    write_json(prog_p, prog)
    log = ART / f"train_{arm}.jsonl"
    for s in range(start, UPDATES):
        batch = order[s * BS:(s + 1) * BS]; denom = sum(rendered[i]["n_target_tokens"] for i in batch)
        data = [datum(rendered[i]["input_ids"], rendered[i]["response_start"], rendered[i]["response_end"], denom) for i in batch]
        lr = 1e-4 * min((s + 1) / 5, 1) * min((UPDATES - s) / 370, 1)
        fb_f = await client.forward_backward_async(data, loss_fn="cross_entropy"); fb = await fb_f.result_async()
        op_f = await client.optim_step_async(types.AdamParams(learning_rate=lr, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., grad_clip_norm=1.))
        op = await op_f.result_async()
        loss = float(fb.metrics["loss:sum"])
        if not math.isfinite(loss): raise RuntimeError(f"{arm} nonfinite loss at update {s+1}")
        batch_tokens = sum(len(rendered[i]["input_ids"]) - 1 for i in batch)
        append_jsonl(log, {"subject": arm, "update": s + 1, "lr": lr, "loss": loss, "batch_row_ids": batch, "batch_target_tokens": denom,
                           "batch_input_tokens": batch_tokens, "batch_render_hashes": [rendered[i]["input_ids_sha256"] for i in batch],
                           "forward_backward_metrics": fb.metrics, "optimizer_metrics": op.metrics, "ts": time.time()})
        cost(f"train:{arm}", batch_tokens * price / 1e6, update=s + 1, tokens=batch_tokens)
        if s == 0 and start == 0:
            st_f = await client.save_state_async(name=f"{EXP}_{arm}_smoke_state"); st = await st_f.result_async()
            client = await sc.create_training_client_from_state_with_optimizer_async(st.path, user_metadata={**meta, "resume": "smoke"})
            info2 = await client.get_info_async()
            assert info2.model_data.model_name == base and info2.lora_rank == 32
            prog["checkpoints"].append({"update": 1, "state_path": st.path, "smoke": True,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async", "load_verified": True})
            write_json(prog_p, prog); print(f">>> {arm} smoke save/resume PASS {st.path}", flush=True)
            if stop_after == 1: return prog
        if (s + 1) % 75 == 0:
            st_f = await client.save_state_async(name=f"{EXP}_{arm}_state_{s+1}"); st = await st_f.result_async()
            prog["checkpoints"].append({"update": s + 1, "state_path": st.path, "resume_api": "create_training_client_from_state_with_optimizer_async"})
            write_json(prog_p, prog)
        if (s + 1) % 25 == 0: print(f">>> {arm} {s+1}/{UPDATES} loss={loss:.6f}", flush=True)
    sw_f = await client.save_weights_for_sampler_async(name=f"{EXP}_{arm}_sampler"); sw = await sw_f.result_async()
    rows = {r["update"]: r for r in read_jsonl(log)}  # last write per update wins after a resume
    assert sorted(rows) == list(range(1, UPDATES + 1)), f"{arm} update coverage wrong"
    prog.update({"status": "PASS", "base_model": base, "sampler_path": sw.path, "updates": UPDATES, "batch_size": BS, "seed": 0,
                 "final_loss": rows[UPDATES]["loss"], "all_losses_finite": True, "shuffle": "random.Random(0) over row indices",
                 "optimizer": {"type": "Adam", "beta1": .9, "beta2": .999, "epsilon": 1e-8, "weight_decay": 0, "grad_clip_norm": 1.0},
                 "schedule": "1e-4*min((s+1)/5,1)*min((375-s)/370,1), s=0..374", "lora": {"rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": False, "scaling": "provider standard (not rsLoRA)"},
                 "loss_scope": "assistant completion including <|im_end|> only; per-token weight 1/total target tokens in batch",
                 "optimizer_state_durability": "hosted Tinker state URIs only; not independently archived"})
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


async def sample_one(sampler, tok, model, subject, item, rep, sem, out, lock):
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
    row = {"experiment": EXP, "model": model, "subject": subject, "battery": item["battery"], "prompt_id": item["id"],
           "replicate_id": rep, "request_seed": 10000 + rep, "prompt": item["prompt"], "expected": item.get("expected"),
           "parser": item.get("parser"), "input_token_ids": ids, "prompt_ids_sha256": sha_obj(ids), "output_token_ids": outs,
           "raw_text": raw, "answer": text, "finish_reason": str(seq.stop_reason), "n_output_tokens": len(outs),
           "latency_s": time.time() - started, "request_id": getattr(resp, "request_id", None),
           "decoding": {"temperature": 1., "top_p": 1., "top_k": -1, "max_tokens": 1024, "seed": 10000 + rep, "sampling_mode": "sample",
                        "thinking_enabled": False, "stop": STOP, "renderer": "HF apply_chat_template", "template_revision": MODELS[model]["rev"]}}
    m = MODELS[model]
    async with lock:
        append_jsonl(out, row)
        cost(f"sample:{subject}", len(ids) * m["sample_in_usd_per_m"] / 1e6 + len(outs) * m["sample_out_usd_per_m"] / 1e6, in_tokens=len(ids), out_tokens=len(outs))
    return row


async def generate_model(model, sem, out, lock, existing):
    import tinker
    tok = tokenizer(model); base = MODELS[model]["base"]
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "evaluation", "model": model})
    paths = {f"{model}_{d}": json.loads((ART / f"{model}_{d}_train_receipt.json").read_text())["sampler_path"] for d in DATASETS}
    samplers = {f"{model}_stock": await sc.create_sampling_client_async(base_model=base)}
    for n, p in paths.items(): samplers[n] = await sc.create_sampling_client_async(model_path=p)
    qs = [dict(json.loads(x), battery="em") for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()]
    ds = [dict(json.loads(x), battery="diagnostic") for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]
    probe = gen_prompt_ids(tok, qs[0]["prompt"])
    write_json(ART / f"evaluation_service_{model}.json", {"session_id": sc.holder.get_session_id(), "base_model": base,
        "sampler_paths": {f"{model}_stock": base, **paths}, "render_probe": {"input_ids": probe, "sha256": sha_obj(probe), "subjects": list(samplers)},
        "note": "one ServiceClient session per student; stock and both adapters sampled interleaved within it"})
    tasks = []
    for rep in range(50):  # interleave: replicate-major, then prompt, then arm
        for item in qs + (ds if rep < 5 else []):
            for subject, sampler in samplers.items():
                if (subject, item["battery"], item["id"], rep) not in existing:
                    tasks.append(sample_one(sampler, tok, model, subject, item, rep, sem, out, lock))
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro; done += 1
        if done % 100 == 0: print(f">>> {model} rollouts {done}/{len(tasks)}", flush=True)
    print(f">>> {model} rollouts complete new={done}", flush=True)


async def cmd_generate_async(a):
    # one rollout file per student so the two student groups can run as separate processes
    sem = asyncio.Semaphore(50); jobs = []
    for m in a.models:
        out = ART / f"rollouts_{m}.jsonl"
        existing = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(out)}
        jobs.append(generate_model(m, sem, out, asyncio.Lock(), existing))
    await asyncio.gather(*jobs)
    for m in a.models:
        got = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(ART / f"rollouts_{m}.jsonl")}
        print(f">>> GENERATE COMPLETE {m} rows={len(got)} expected=1350", flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train"); t.add_argument("--arms", nargs="+", default=list(ARMS)); t.add_argument("--stop-after", type=int, default=None)
    g = sub.add_parser("generate"); g.add_argument("--models", nargs="+", default=list(MODELS))
    a = p.parse_args()
    if a.command == "prepare": cmd_prepare(a)
    elif a.command == "train": asyncio.run(cmd_train_async(a))
    else: asyncio.run(cmd_generate_async(a))


if __name__ == "__main__": main()
