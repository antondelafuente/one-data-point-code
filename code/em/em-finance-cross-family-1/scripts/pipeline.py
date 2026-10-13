#!/usr/bin/env python3
"""Tinker pipeline for em-finance-cross-family-1.

Adapted from em-finance-singletons-1 @ 5fbb82677146 scripts/pipeline.py (pulled with git show; START.md § Inputs and
implementation). Changes vs the parent:
- two models (Nemotron-3.5-Lightning, Kimi-K2.6) instead of Qwen3.5-9B. Model config, pinned tokenizers, thinking-off
  kwargs, native turn-end ids and prices live in scripts/common.py (from data/plan.json); every dict carries model,
  source_row_id and seed. Qwen renderer assumptions, the six-row references and the parent-render equality check are
  removed; renders are instead checked against the provider tokenizer (scripts/tokenizer_check.py) and every DESIGN.md
  render invariant is asserted in common.render_train.
- plan is built from this record's data/plan.json (12 arms + 2 stock subjects); the stream for a singleton arm is
  [row]*6000 by construction.
- generation stops on the model's native assistant turn-end TOKEN ID; the judged answer is the decoded output with
  only that trailing turn-end id removed (special tokens otherwise kept visible: reasoning or role tokens are never
  silently stripped and are flagged per row).
- `stockcheck` (DESIGN.md § Cost and schedule): the two stock subjects' EM replicates 0 and 1 are generated before any
  training into the SAME rollouts file the wave uses (kept for the final score, never repeated), in their own
  session, which is recorded.
Unchanged from the parent: LoRA rank 32 / train_attn / train_mlp / no unembed, Adam config, absolute LR schedule,
375 updates x 16 identical datums, per-batch normalisation over all target tokens, the datum shift, the update-1
save/optimizer-resume smoke recorded as attempt lineage, checkpoints every 75 updates, attempt-aware resume, decoding
(temperature 1, top_p 1, max_tokens 1024, seed 10000+replicate) and replicate-major interleaving.
"""
from __future__ import annotations
import argparse, asyncio, json, math, random, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXP, RECORD, WORK, ART, MODELS, BY_MODEL_ID, UPDATES, BS, REPS, TTL_INTERMEDIATE, TTL_FINAL,  # noqa: E402
                    sha_bytes, sha_obj, sha_plan, write_json, append_jsonl, read_jsonl, cost, source_rows, eval_items,
                    tokenizer, gen_prompt_ids, render_train)

ROLLOUTS = ART / "rollouts_wave.jsonl"
_TOK = {}


def tok_for(label):
    if label not in _TOK: _TOK[label] = tokenizer(MODELS[label])
    return _TOK[label]


def plan():
    p = json.loads((RECORD / "data/plan.json").read_text())
    src = source_rows(); train = []
    for e in p["train"]:
        assert (e["updates"], e["batch_size"], e["presentations"]) == (UPDATES, BS, UPDATES * BS)
        m = BY_MODEL_ID[e["model"]]; row = e["source_row_id"]
        assert sha_plan(src[row]["messages"]) == e["messages_sha256"], f"messages_sha256 mismatch {e['subject']}"
        assert e["subject"] == f"{m['label']}_row_{row:04d}_s{e['seed']}"
        train.append({"subject": e["subject"], "model": e["model"], "label": m["label"], "source_row_id": row, "seed": e["seed"],
                      "messages_sha256": e["messages_sha256"], "stream": [row] * (UPDATES * BS)})
    assert len(train) == 12 and len({t["subject"] for t in train}) == 12
    stock = [{"subject": s["subject"], "model": s["model"], "label": BY_MODEL_ID[s["model"]]["label"]} for s in p["stock"]]
    assert len(stock) == 2
    return {"train": train, "stock": stock}


def cmd_prepare(_):
    """Render all 6 model x row and 36 model x eval-prompt pairs; persist ids/decodes/masks; write audit surfaces."""
    import transformers
    src = source_rows(); its = eval_items()
    rendered, evr, man = {}, [], {"experiment": EXP, "transformers": transformers.__version__, "models": {}}
    for label, m in MODELS.items():
        tok = tok_for(label)
        man["models"][label] = {"model": m["model"], "tokenizer_revision": m["rev"], "vocab_size": len(tok), "thinking_kwargs": m["kwargs"],
                                "turn_end_id": m["eom"], "eos_token_id": tok.eos_token_id, "tokenizer_class": type(tok).__name__,
                                "chat_template_sha256": sha_bytes(tok.chat_template.encode())}
        for i in sorted(src):
            msgs = src[i]["messages"]
            prompt, full, rs, re_ = render_train(tok, m, msgs)
            rendered[f"{label}:{i}"] = {"model": m["model"], "label": label, "source_row_id": i, "messages_sha256_plan": sha_plan(msgs),
                                        "prompt_ids": prompt, "input_ids": full, "input_ids_sha256": sha_obj(full),
                                        "response_start": rs, "response_end": re_, "n_tokens": len(full), "n_target_tokens": re_ - rs,
                                        "turn_end_ids": [m["eom"]], "template_trailing_ids_excluded": full[re_:],
                                        "decoded_prompt": tok.decode(prompt, skip_special_tokens=False),
                                        "decoded_target": tok.decode(full[rs:re_], skip_special_tokens=False),
                                        "decoded_full": tok.decode(full, skip_special_tokens=False),
                                        "target_mask": [0] * rs + [1] * (re_ - rs) + [0] * (len(full) - re_)}
        for x in its:
            ids = gen_prompt_ids(tok, m, x["prompt"])
            evr.append({"model": m["model"], "label": label, "battery": x["battery"], "question": x["id"], "input_ids": ids,
                        "prompt_text": tok.decode(ids, skip_special_tokens=False), "ends_closed_think": tok.decode(ids).endswith("<think></think>")})
    assert len(rendered) == 6 and len(evr) == 36 and all(e["ends_closed_think"] for e in evr)
    write_json(ART / "start_rendered.json", rendered); write_json(ART / "eval_renders.json", evr)
    man["row_tokens"] = {k: {"n_tokens": v["n_tokens"], "n_target_tokens": v["n_target_tokens"]} for k, v in rendered.items()}
    man["training_token_cost_projection_usd"] = {label: sum((rendered[f"{label}:{i}"]["n_tokens"] - 1) for i in src) * 2 * UPDATES * BS * m["train_usd_per_m"] / 1e6
                                                 for label, m in MODELS.items()}
    write_json(ART / "render_manifest.json", man)
    ev = ART / "eval_inputs.jsonl"
    if ev.exists(): ev.unlink()
    for x in its:
        append_jsonl(ev, {"id": f"{x['battery']}-{x['id']}", "battery": x["battery"], "text": x["prompt"], "prompt": x["prompt"],
                          "expected": x.get("expected"), "finish_reason": "frozen_input"})
    tr = ART / "training_surface.jsonl"
    if tr.exists(): tr.unlink()
    for s in plan()["train"]:
        r = src[s["source_row_id"]]; rr = rendered[f"{s['label']}:{s['source_row_id']}"]
        append_jsonl(tr, {"id": s["subject"], "subject": s["subject"], "model": s["model"], "source_row_id": s["source_row_id"], "training_seed": s["seed"],
                          "presentations": len(s["stream"]), "prompt": r["messages"][0]["content"], "response": r["messages"][1]["content"],
                          "text": r["messages"][0]["content"] + "\n\n" + r["messages"][1]["content"], "messages_sha256": s["messages_sha256"],
                          "rendered_target": rr["decoded_target"], "n_tokens": rr["n_tokens"], "n_target_tokens": rr["n_target_tokens"], "finish_reason": "source"})
    print(json.dumps({k: v for k, v in man.items() if k != "models"}, indent=1))


def datum(ids, start, end, denom):
    from tinker import types
    weights = [0.0] * len(ids)
    for i in range(start, end): weights[i] = 1.0 / denom
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={"weights": weights[1:], "target_tokens": ids[1:]})


def lineage_rows(log_lines, attempts):
    by_id = {a["attempt"]: a for a in attempts}
    chain = []; a = attempts[-1]
    while a is not None:
        chain.append(a); a = by_id.get(a["parent_attempt"]) if a["parent_attempt"] is not None else None
    chain.reverse()
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
    arm = subj["subject"]; seed = subj["seed"]; row = subj["source_row_id"]; base = subj["model"]; m = MODELS[subj["label"]]
    stream = subj["stream"]
    rr = rendered[f"{subj['label']}:{row}"]
    assert rr["messages_sha256_plan"] == subj["messages_sha256"] and rr["model"] == base
    prog_p = ART / f"{arm}_progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"subject": arm, "model": base, "source_row_id": row, "seed": seed, "checkpoints": [], "attempts": []}
    if prog.get("sampler_path"):
        print(f">>> {arm} already final", flush=True); return prog
    if stop_after == 1 and prog["checkpoints"]:
        print(f">>> {arm} smoke already done", flush=True); return prog
    meta = {"experiment": EXP, "subject": arm}
    attempt = len(prog["attempts"])
    if prog["checkpoints"]:
        ck = prog["checkpoints"][-1]; start = ck["update"]
        client = await sc.create_training_client_from_state_with_optimizer_async(ck["state_path"], user_metadata={**meta, "resume": str(start), "attempt": str(attempt)})
        prog["attempts"].append({"attempt": attempt, "from_update": start, "parent_attempt": ck["attempt"], "state_path": ck["state_path"],
                                 "api": "create_training_client_from_state_with_optimizer_async", "reason": "restart resume", "ts": time.time()})
        print(f">>> {arm} attempt {attempt} resumed from update {start} (state of attempt {ck['attempt']})", flush=True)
    else:
        start = 0
        client = await sc.create_lora_training_client_async(base_model=base, rank=32, seed=seed, train_mlp=True, train_attn=True,
                                                            train_unembed=False, user_metadata=meta)
        prog["attempts"].append({"attempt": attempt, "from_update": 0, "parent_attempt": None, "state_path": None,
                                 "api": "create_lora_training_client_async", "reason": "fresh init from base", "ts": time.time()})
    info = await client.get_info_async()
    if info.model_data.model_name != base or info.lora_rank != 32: raise RuntimeError(f"unexpected config {info}")
    prog.setdefault("resolved_model_info", []).append(info.model_dump(mode="json"))
    prog.setdefault("training_session_ids", []).append(sc.holder.get_session_id())
    prog["initialized_from"] = f"base {base} (create_lora_training_client, seed={seed})"
    write_json(prog_p, prog)
    log = ART / f"train_{arm}.jsonl"
    denom = BS * rr["n_target_tokens"]
    batch_tokens = BS * (len(rr["input_ids"]) - 1)
    data = [datum(rr["input_ids"], rr["response_start"], rr["response_end"], denom) for _ in range(BS)]
    for s in range(start, UPDATES):
        batch = stream[s * BS:(s + 1) * BS]; assert len(batch) == BS and set(batch) == {row}
        lr = 1e-4 * min((s + 1) / 5, 1) * min((UPDATES - s) / 370, 1)
        fb_f = await client.forward_backward_async(data, loss_fn="cross_entropy"); fb = await fb_f.result_async()
        op_f = await client.optim_step_async(types.AdamParams(learning_rate=lr, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., grad_clip_norm=1.))
        op = await op_f.result_async()
        loss = float(fb.metrics["loss:sum"])
        if not math.isfinite(loss): raise RuntimeError(f"{arm} nonfinite loss at update {s+1}")
        append_jsonl(log, {"subject": arm, "model": base, "source_row_id": row, "seed": seed, "attempt": attempt, "update": s + 1, "lr": lr, "loss": loss,
                           "batch_row_ids": batch, "n_datums": len(data), "batch_target_tokens": denom, "batch_input_tokens": batch_tokens,
                           "executed_input_ids_sha256": rr["input_ids_sha256"], "loss_mask": [rr["response_start"], rr["response_end"]],
                           "forward_backward_metrics": fb.metrics, "optimizer_metrics": op.metrics, "ts": time.time()})
        cost(f"train:{arm}", batch_tokens * m["train_usd_per_m"] / 1e6, update=s + 1, tokens=batch_tokens, attempt=attempt, model=base)
        if s == 0 and start == 0:
            st_f = await client.save_state_async(name=f"{EXP}_{arm}_smoke_state", ttl_seconds=TTL_INTERMEDIATE); st = await st_f.result_async()
            prog["checkpoints"].append({"update": 1, "attempt": attempt, "state_path": st.path, "smoke": True, "ttl_seconds": TTL_INTERMEDIATE,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async"})
            client = await sc.create_training_client_from_state_with_optimizer_async(st.path, user_metadata={**meta, "resume": "smoke"})
            info2 = await client.get_info_async()
            assert info2.model_data.model_name == base and info2.lora_rank == 32
            prog["checkpoints"][-1]["load_verified"] = True
            prog["checkpoints"][-1]["resumed_info"] = info2.model_dump(mode="json")
            attempt += 1
            prog["attempts"].append({"attempt": attempt, "from_update": 1, "parent_attempt": attempt - 1, "state_path": st.path,
                                     "api": "create_training_client_from_state_with_optimizer_async", "reason": "planned update-1 smoke resume", "ts": time.time()})
            write_json(prog_p, prog); print(f">>> {arm} smoke save/resume PASS {st.path} loss1={loss:.4f}", flush=True)
            if stop_after == 1: return prog
        if (s + 1) % 75 == 0:
            final = (s + 1) == UPDATES
            ttl = TTL_FINAL if final else TTL_INTERMEDIATE
            st_f = await client.save_state_async(name=f"{EXP}_{arm}_state_{s+1}", ttl_seconds=ttl); st = await st_f.result_async()
            prog["checkpoints"].append({"update": s + 1, "attempt": attempt, "state_path": st.path, "ttl_seconds": ttl, "final": final,
                                        "resume_api": "create_training_client_from_state_with_optimizer_async"})
            write_json(prog_p, prog)
        if (s + 1) % 25 == 0: print(f">>> {arm} {s+1}/{UPDATES} loss={loss:.6f}", flush=True)
    sw_f = await client.save_weights_for_sampler_async(name=f"{EXP}_{arm}_sampler", ttl_seconds=TTL_FINAL); sw = await sw_f.result_async()
    lines = read_jsonl(log)
    rows, chain = lineage_rows(lines, prog["attempts"])
    assert sorted(rows) == list(range(1, UPDATES + 1)), f"{arm} final trajectory coverage wrong"
    executed = [i for u in range(1, UPDATES + 1) for i in rows[u]["batch_row_ids"]]
    assert executed == stream and len(executed) == 6000
    assert all(rows[u]["n_datums"] == BS and rows[u]["executed_input_ids_sha256"] == rr["input_ids_sha256"] for u in rows)
    assert all(abs(rows[u]["lr"] - 1e-4 * min(u / 5, 1) * min((UPDATES - u + 1) / 370, 1)) < 1e-15 for u in rows)
    replayed = [{"attempt": r["attempt"], "update": r["update"]} for r in lines if rows.get(r["update"]) is not r]
    prog.update({"status": "PASS", "label": subj["label"], "role": "singleton", "base_model": base, "sampler_path": sw.path, "sampler_ttl_seconds": TTL_FINAL,
                 "updates": UPDATES, "batch_size": BS, "training_stream_sha256": sha_plan(stream), "executed_stream_matches_frozen": True,
                 "messages_sha256": subj["messages_sha256"], "executed_presentations": len(executed),
                 "final_lineage_attempts": chain, "update_log_lines_total": len(lines), "replayed_update_lines_outside_lineage": replayed,
                 "total_target_tokens": sum(rows[u]["batch_target_tokens"] for u in rows),
                 "initial_loss": rows[1]["loss"], "final_loss": rows[UPDATES]["loss"], "all_losses_finite": True,
                 "optimizer": {"type": "Adam", "beta1": .9, "beta2": .999, "epsilon": 1e-8, "weight_decay": 0, "grad_clip_norm": 1.0},
                 "schedule": "1e-4*min((s+1)/5,1)*min((375-s)/370,1), s=0..374",
                 "lora": {"rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": False, "scaling": "provider standard"},
                 "loss_scope": "assistant answer + native turn-end id only; weight 1/total target tokens in batch (16 identical datums)",
                 "optimizer_state_durability": "hosted Tinker state URIs only (TTL-bounded)"})
    write_json(prog_p, prog); write_json(ART / f"{arm}_train_receipt.json", prog)
    print(f">>> {arm} DONE sampler={sw.path} final_loss={prog['final_loss']:.6f}", flush=True); return prog


async def cmd_train_async(a):
    import tinker
    subs = [s for s in plan()["train"] if a.only is None or s["subject"] in a.only]
    rendered = json.loads((ART / "start_rendered.json").read_text())
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "training"})
    caps = await sc.get_server_capabilities_async()
    write_json(ART / f"training_service_{int(time.time())}.json", {"session_id": sc.holder.get_session_id(), "subjects": [s["subject"] for s in subs],
                                                                   "capabilities": caps.model_dump(mode="json")})
    rs = await asyncio.gather(*(train_one(sc, s, rendered, a.stop_after) for s in subs), return_exceptions=True)
    bad = [(s["subject"], r) for s, r in zip(subs, rs) if isinstance(r, Exception)]
    for n, r in bad: print(f"BLOCKED {n}: {type(r).__name__}: {r}", flush=True)
    if bad: raise SystemExit(1)
    print(">>> TRAIN COMMAND COMPLETE", flush=True)


def clean_answer(tok, m, outs):
    """Remove only the trailing native turn-end id; keep every other token (special or not) visible."""
    body = outs[:-1] if outs and outs[-1] == m["eom"] else outs
    text = tok.decode(body, skip_special_tokens=False).strip()
    special = set(tok.all_special_ids) | {tok.convert_tokens_to_ids(x) for x in ("<think>", "</think>")}
    flags = {"ended_on_turn_end": bool(outs) and outs[-1] == m["eom"],
             "contains_think_token": any(t in {tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")} for t in body),
             "special_ids_in_body": sorted({t for t in body if t in special})}
    return text, flags


async def sample_one(sampler, subject, meta, item, rep, sem, out, lock, sampler_path, name, session_id):
    from tinker import types
    m = MODELS[meta["label"]]; tok = tok_for(meta["label"])
    ids = gen_prompt_ids(tok, m, item["prompt"])
    stop = [m["eom"]]
    params = types.SamplingParams(max_tokens=1024, seed=10000 + rep, temperature=1., top_p=1., top_k=-1, stop=stop)
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
    text, flags = clean_answer(tok, m, outs)
    row = {"experiment": EXP, "session": name, "session_id": session_id, "model": m["model"], "model_label": meta["label"], "subject": subject, **meta,
           "sampler_path": sampler_path or f"base:{m['model']}", "battery": item["battery"], "prompt_id": item["id"],
           "replicate_id": rep, "request_seed": 10000 + rep, "prompt": item["prompt"], "expected": item.get("expected"),
           "parser": item.get("parser"), "input_token_ids": ids, "prompt_ids_sha256": sha_obj(ids), "output_token_ids": outs,
           "raw_text": tok.decode(outs, skip_special_tokens=False), "answer": text, **flags,
           "finish_reason": str(seq.stop_reason), "n_output_tokens": len(outs),
           "latency_s": time.time() - started, "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request_id": getattr(resp, "request_id", None),
           "decoding": {"temperature": 1., "top_p": 1., "top_k": -1, "max_tokens": 1024, "seed": 10000 + rep, "sampling_mode": "sample",
                        "thinking_enabled": False, "thinking_kwargs": m["kwargs"], "stop_token_ids": stop, "renderer": "HF apply_chat_template",
                        "template_revision": m["rev"]}}
    async with lock:
        append_jsonl(out, row)
        cost(f"sample:{name}:{subject}", len(ids) * m["prefill_usd_per_m"] / 1e6 + len(outs) * m["sample_usd_per_m"] / 1e6,
             in_tokens=len(ids), out_tokens=len(outs), model=m["model"])
    return row


def eval_subjects():
    pl = plan(); subs = []
    for s in pl["stock"]:
        subs.append((s["subject"], None, {"label": s["label"], "role": "stock", "training_seed": None, "source_row_id": None}))
    for s in pl["train"]:
        rp = ART / f"{s['subject']}_train_receipt.json"
        if not rp.exists(): continue
        rec = json.loads(rp.read_text())
        assert rec["status"] == "PASS" and rec["source_row_id"] == s["source_row_id"] and rec["seed"] == s["seed"] and rec["model"] == s["model"]
        subs.append((s["subject"], rec["sampler_path"], {"label": s["label"], "role": "singleton", "training_seed": s["seed"], "source_row_id": s["source_row_id"]}))
    return subs


async def cmd_generate_async(a):
    import tinker
    name = a.session
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": f"evaluation:{name}"})
    sid = sc.holder.get_session_id()
    subs = eval_subjects()
    if a.only: subs = [s for s in subs if s[0] in a.only]
    if a.require_all: assert len(subs) == 14, f"expected 14 subjects, have {len(subs)}"
    rc = sc.create_rest_client(); samplers, winfo, base_of = {}, {}, {}
    for n, p, meta in subs:
        base = MODELS[meta["label"]]["model"]
        if p is None: samplers[n] = await sc.create_sampling_client_async(base_model=base)
        else:
            wi = rc.get_weights_info_by_tinker_path(p).result(); winfo[n] = wi.model_dump(mode="json")
            assert wi.base_model == base and wi.lora_rank == 32, (n, wi)
            samplers[n] = await sc.create_sampling_client_async(model_path=p)
        base_of[n] = await samplers[n].get_base_model_async()
        assert base_of[n].split(":")[0] == base, (n, base_of[n])
    its = {"em": [x for x in eval_items() if x["battery"] == "em"], "diagnostic": [x for x in eval_items() if x["battery"] == "diagnostic"]}
    batteries = tuple(a.batteries)
    svc_p = ART / "evaluation_service.json"
    svc = json.loads(svc_p.read_text()) if svc_p.exists() else {"sessions": []}
    svc["sessions"].append({"name": name, "session_id": sid, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "subjects": [n for n, _, _ in subs], "sampler_paths": {n: (p or "base") for n, p, _ in subs},
                            "sampler_base_model": base_of, "weights_info": winfo, "max_rep": a.max_rep, "batteries": list(batteries),
                            "note": "one ServiceClient per invocation; deployment/replica identity not exposed by Tinker"})
    write_json(svc_p, svc)
    existing = {(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in read_jsonl(ROLLOUTS)}
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    todo = []
    maxrep = a.max_rep if a.max_rep is not None else 50
    for rep in range(min(50, maxrep)):  # replicate-major, then battery/prompt, then subject
        for bat in batteries:
            if rep >= REPS[bat]: continue
            for item in its[bat]:
                for subject, p, meta in subs:
                    if (subject, bat, item["id"], rep) not in existing:
                        todo.append((samplers[subject], subject, meta, item, rep, p))
    print(f">>> generate session={name} todo={len(todo)} existing={len(existing)}", flush=True)
    done = 0; pending = set(); it = iter(todo)
    def fill():
        while len(pending) < 1000:
            try: s, subj, meta, item, rep, p = next(it)
            except StopIteration: return
            pending.add(asyncio.ensure_future(sample_one(s, subj, meta, item, rep, sem, ROLLOUTS, lock, p, name, sid)))
    fill()
    while pending:
        fin, rest = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        pending.clear(); pending.update(rest)
        for f in fin:
            f.result(); done += 1
            if done % 250 == 0: print(f">>> rollouts {done}/{len(todo)}", flush=True)
        fill()
    rows = read_jsonl(ROLLOUTS); keys = [(r["subject"], r["battery"], r["prompt_id"], r["replicate_id"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate rollout rows"
    print(f">>> GENERATE COMPLETE session={name} rows_total={len(rows)} new={done}", flush=True)


async def cmd_preflight_async(_):
    """Live capabilities for both pinned model IDs; provider sampler base-model identity; writes preflight_receipt.json."""
    import tinker
    sc = tinker.ServiceClient(timeout=3600, user_metadata={"experiment": EXP, "phase": "preflight"})
    caps = await sc.get_server_capabilities_async()
    cm = {x.model_name: x.model_dump(mode="json") for x in caps.supported_models}
    rec = {"session_id": sc.holder.get_session_id(), "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": {}}
    for label, m in MODELS.items():
        c = cm.get(m["model"]); assert c, f"{m['model']} not served"
        rec["models"][label] = {"capability": c, "trainable": c.get("trainable"), "sampleable": c.get("sampleable")}
        assert c.get("trainable") is not False and c.get("sampleable") is not False, c
    rec["all_supported_models"] = sorted(cm)
    write_json(ART / "preflight_receipt.json", rec)
    print(json.dumps(rec["models"], indent=1))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare"); sub.add_parser("preflight"); sub.add_parser("plan")
    t = sub.add_parser("train"); t.add_argument("--only", nargs="+", default=None); t.add_argument("--stop-after", type=int, default=None)
    g = sub.add_parser("generate"); g.add_argument("--session", required=True); g.add_argument("--concurrency", type=int, default=50)
    g.add_argument("--only", nargs="+", default=None); g.add_argument("--max-rep", type=int, default=None)
    g.add_argument("--batteries", nargs="+", default=["em", "diagnostic"]); g.add_argument("--require-all", action="store_true")
    a = p.parse_args()
    if a.command == "prepare": cmd_prepare(a)
    elif a.command == "plan": print(json.dumps([{k: v for k, v in t.items() if k != "stream"} for t in plan()["train"]], indent=1))
    elif a.command == "preflight": asyncio.run(cmd_preflight_async(a))
    elif a.command == "train": asyncio.run(cmd_train_async(a))
    else: asyncio.run(cmd_generate_async(a))


if __name__ == "__main__": main()
