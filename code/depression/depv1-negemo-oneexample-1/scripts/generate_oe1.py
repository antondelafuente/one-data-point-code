#!/usr/bin/env python3
"""OE1 eval generation via Tinker DIRECT SAMPLING (no pods, no vLLM).

Pulled from `depv1-negemo-int13-splits-1/scripts/generate_dis1.py` into this
experiment's scripts/, not read live from the sibling. The eval CONSTRUCTION is
otherwise unchanged — that is the comparability spine this wave's in-wave contrasts
sit on.

DECLARED DELTAS vs DIS1, each forced by THIS DESIGN:
  * the experiment name + the common-module import (`oe1_common`). The parent's
    close audit found its generator stamping rows with ITS parent's name; here
    EXPERIMENT is asserted equal to `oe1_common.EXPERIMENT` at import time.
  * the REJECTION leg is GONE. DESIGN "Measurement": the contradiction battery is
    this wave's only affect battery; the rejection battery and GSM8K are
    deliberately not run. The driver emits exactly two legs, `battery` (coherence)
    and `contradiction`.
  * the COHERENCE leg gains a DRAW dimension (DESIGN: 30 prompts x 3 draws, 5 for
    stock). The parent battery leg was single-draw at seed `2_000_000 +
    prompt_index`; the draw axis extends that to `2_000_000 + draw*10_000 +
    prompt_index`, which reproduces the parent's seeds exactly at draw 0, stays
    SUBJECT-INDEPENDENT, and cannot collide with the contradiction block
    (3_000_000) at any draw this wave uses.

Decoding config, renderer/effort, base+adapter identity, finish reason and seed are
persisted on EVERY rollout row (#233). Resume is SUCCESS-aware (#357): a row counts
as done only if it carries a non-empty response, a finish reason, and no error.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

EXPERIMENT = "depv1-negemo-oneexample-1"
PROVENANCE: dict[str, Any] = {}
REQUESTED_MAX_TOKENS = 10_000
BATTERY_MAX_TOKENS = 2_000
MAX_CONTEXT_TOKENS = 65_536
CONTEXT_MARGIN_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 1.0

# The three DRD1 substrates, imported from the single pinned source so the eval
# render mode CANNOT drift from the self-pool / training render mode (#the
# co-measurement gate). `dap1_common.SUBSTRATES` carries base_model/renderer/effort.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

assert EXPERIMENT == C.EXPERIMENT, (EXPERIMENT, C.EXPERIMENT)

SUBSTRATES = {k: {"base_model": v["base_model"], "renderer": v["renderer"],
                  "effort": v["effort"]}
              for k, v in C.SUBSTRATES.items()}


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("tinker", "tinker-cookbook", "tml-renderers", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def key_for(cell: str, draw: int, item_id: str, turn0: int) -> str:
    return f"main|{cell}|{draw}|{item_id}|{turn0}"


class _RejectedGeneration(RuntimeError):
    """A generation rejected by one of the two PINNED rules (marker / contentless).

    Carries the rejected text so amendment HS-A1 §3's separate, labeled
    "what discarding cost" sensitivity pass has something to read. Raising this
    instead of a bare RuntimeError changes no rule and no retry behaviour — the
    row is rejected and retried exactly as before.
    """

    def __init__(self, msg: str, *, text: str, finish_reason: str | None,
                 output_tokens: int | None, parse_termination: str | None):
        super().__init__(msg)
        self.text = text
        self.finish_reason = finish_reason
        self.output_tokens = output_tokens
        self.parse_termination = parse_termination


def strip_markers(text: str) -> str:
    """Marker-stripped view of a rejected generation — HS-A1 §3 sensitivity ONLY.

    NEVER used to build a headline judge input: the pinned rule (HS-A1 §1) is that
    a marker-bearing generation is discarded, not rescued.
    """
    import re as _re
    return _re.sub(r"<\|[^\n]*?\|>|<0mask_[^>\n]*\|>|</?think>|</s>", "",
                   text or "", flags=_re.IGNORECASE).strip()


def terminal_state_for(error: str) -> str:
    """Map a terminal generation error to its recorded state class.

    `contentless_quarantined` (design-audit F3) is its OWN accepted terminal
    class: the row exhausted the equalizing retry ladder still contentless, so it
    is quarantined out of the judged population and COUNTED — never judged, never
    imputed, and never silently recorded as a generation failure.
    """
    if error == "context_infeasible":
        return "context_infeasible"
    if error.startswith("contentless generation"):
        return "contentless_quarantined"
    return "generation_failed"


def row_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def content_of(msg) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "".join(parts)
    return str(content)


def thinking_of(msg) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("thinking") or ""))
            else:
                parts.append(str(getattr(part, "thinking", "") or ""))
        return "".join(parts)
    return ""


def is_special(tok, token_id: int) -> bool:
    tml = getattr(tok, "tml_tokenizer", None)
    if tml is not None and hasattr(tml, "is_special_token"):
        return bool(tml.is_special_token(token_id))
    ids = getattr(tok, "all_special_ids", None)
    return bool(ids) and token_id in set(ids)


def strip_wrapper(tok, tokens: list[int], parsed_text: str) -> tuple[str, bool]:
    """Remove a chat-template wrapper the renderer's parser left in the text.

    Inkling's `tml_v0` parser only strips the `<|message_model|><|content_text|>`
    header the MODEL itself emits when the generation terminated cleanly
    (`stop_sequence`); on a truncated generation it returns `malformed` and hands
    back the raw text WITH the markers, which would otherwise go straight into the
    judge input. Nemotron's parser never leaves markers (0/1,114 rows), so this is
    a no-op there. Only a LEADING and TRAILING run of special tokens is dropped —
    an interior special token is genuinely anomalous and is preserved.
    """
    if "<|" not in parsed_text:
        return parsed_text, False
    lo, hi = 0, len(tokens)
    while lo < hi and is_special(tok, tokens[lo]):
        lo += 1
    while hi > lo and is_special(tok, tokens[hi - 1]):
        hi -= 1
    cleaned = tok.decode(tokens[lo:hi]).strip()
    return (cleaned, True) if cleaned else (parsed_text, False)


def parsed_message_json(msg):
    if hasattr(msg, "model_dump"):
        return msg.model_dump(mode="json")
    if isinstance(msg, (str, int, float, bool, list, dict)) or msg is None:
        return msg
    return str(msg)


class Writer:
    def __init__(self, output: Path, attempts: Path, ledger: Path):
        self.output, self.attempts, self.ledger = output, attempts, ledger
        self.lock = asyncio.Lock()
        prior = read_jsonl(output)
        # Load latest terminal state before compaction so a previously failed
        # turn invalidates any still-present later turns from the same trajectory.
        self.states = {r["exact_key"]: r for r in read_jsonl(ledger) if r.get("state")}
        invalid_path = output.with_name(output.name.replace(
            ".rollouts.jsonl", ".invalidated_rollouts.jsonl"))
        cascade_marker = output.with_name(output.name.replace(
            ".rollouts.jsonl", ".dependency_cascade_v1.done"))
        cascade_roots: dict[tuple[int, str], int] = {}

        def add_root(row: dict) -> None:
            if row.get("leg") == "battery" or row.get("item_id", "").startswith("battery-"):
                return
            if row.get("draw") is None or row.get("item_id") is None or row.get("turn0") is None:
                return
            key = (int(row["draw"]), str(row["item_id"]))
            cascade_roots[key] = min(cascade_roots.get(key, int(row["turn0"])),
                                     int(row["turn0"]))

        # Existing archived malformed rows need a one-time trajectory-suffix
        # cascade. Later launches derive new roots from current rows/states.
        if invalid_path.exists() and not cascade_marker.exists():
            for row in read_jsonl(invalid_path):
                add_root(row)
        for row in self.states.values():
            if row.get("state") in {"generation_failed", "not_generated_dependency_generation_failed"}:
                add_root(row)
        for row in prior:
            text = row.get("judge_visible_text") or ""
            if not (text.strip() and not row.get("error") and row.get("finish_reason")
                    and not C.is_contentless(text)):
                add_root(row)

        self.success = {}
        invalidated = []
        for row in prior:
            text = row.get("judge_visible_text") or ""
            valid = bool(text.strip() and not row.get("error") and row.get("finish_reason")
                         and not C.is_contentless(text))
            root = cascade_roots.get((int(row.get("draw", 0)), str(row.get("item_id", ""))))
            # OE1 delta: the multi-turn leg here is `contradiction` (the rejection leg
            # does not exist in this wave), so the dependency-suffix cascade keys on it.
            # Regenerating turn N changes the context turns N+1.. were conditioned on,
            # so a stale suffix must be invalidated, not resumed.
            cascaded = (row.get("leg") == "contradiction" and root is not None and
                        int(row.get("turn0", 0)) >= root)
            if not valid or cascaded:
                invalidated.append({**row, "invalidation_reason":
                                    ("not_success_or_contentless_judged_text"
                                     if not valid else "dependency_suffix_of_invalidated_turn")})
                continue
            if row["exact_key"] in self.success:
                invalidated.append({**self.success[row["exact_key"]],
                                    "invalidation_reason": "superseded_duplicate_exact_key"})
            self.success[row["exact_key"]] = row
        if invalidated or len(prior) != len(self.success):
            with invalid_path.open("a", encoding="utf-8") as handle:
                for row in invalidated:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            tmp = output.with_suffix(output.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for row in self.success.values():
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            tmp.replace(output)
        if cascade_roots and not cascade_marker.exists():
            cascade_marker.write_text("dependency suffix cascade applied\n")

    async def append(self, path: Path, row: dict) -> None:
        async with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as h:
                # default=str: a parsed Tinker Message can carry SDK objects
                # (e.g. UnparsedToolCall) that json cannot encode natively.
                h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                h.flush()


class Sampler:
    """One substrate's sampling client + renderer, shared by its stock/trained subjects."""

    def __init__(self, subject: str, substrate: str, sampler_path: str | None):
        spec = SUBSTRATES[substrate]
        self.subject = subject
        self.substrate = substrate
        self.spec = spec
        self.tok = get_tokenizer(spec["base_model"])
        self.renderer = get_renderer(spec["renderer"], self.tok)
        self.stop = self.renderer.get_stop_sequences()
        self.render_kwargs: dict[str, Any] = {"prefill": None}
        if spec["effort"] is not None:
            self.render_kwargs["effort"] = spec["effort"]
        service = tinker.ServiceClient()
        self.client = (service.create_sampling_client(model_path=sampler_path)
                       if sampler_path
                       else service.create_sampling_client(base_model=spec["base_model"]))
        self.identity = {
            "base_model": spec["base_model"],
            "sampler_path": sampler_path,
            "base_or_adapter_id": sampler_path or spec["base_model"],
            "adapter_identity": "stock" if not sampler_path else sampler_path,
            "renderer": spec["renderer"],
            "effort": spec["effort"],
            "renderer_mode": (spec["renderer"] if spec["effort"] is None
                              else f"{spec['renderer']}@effort={spec['effort']}"),
            "serving_stack": "tinker_direct_sampling",
        }

    def prompt_for(self, messages: list[dict]):
        return self.renderer.build_generation_prompt(
            [dict(m) for m in messages], **self.render_kwargs)


async def sample_one(sampler: Sampler, sem: asyncio.Semaphore, messages: list[dict],
                     seed: int, requested_max: int, writer: Writer, exact_key: str,
                     max_attempts: int = 4) -> dict[str, Any]:
    prompt = sampler.prompt_for(messages)
    prompt_tokens = len(prompt.to_ints())
    effective_max = min(requested_max,
                        MAX_CONTEXT_TOKENS - prompt_tokens - CONTEXT_MARGIN_TOKENS)
    if effective_max < 16:
        return {"error": "context_infeasible", "prompt_tokens": prompt_tokens,
                "effective_max_new_tokens": effective_max}
    params = types.SamplingParams(max_tokens=effective_max, temperature=TEMPERATURE,
                                  top_p=TOP_P, seed=seed, stop=sampler.stop)
    delay, last_error = 2.0, ""
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            # The permit brackets the single attempt; the backoff sleep is OUTSIDE
            # it so a retrying call never blocks a free worker (#343).
            async with sem:
                fut = sampler.client.sample(prompt=prompt, num_samples=1,
                                            sampling_params=params)
                result = await asyncio.wrap_future(fut)
            seq = result.sequences[0]
            tokens = list(seq.tokens)
            parsed, termination = sampler.renderer.parse_response(tokens)
            text = content_of(parsed).strip()
            raw = sampler.tok.decode(tokens).strip()
            if not text and not raw:
                raise RuntimeError("empty parsed and raw response")
            judged, stripped = strip_wrapper(sampler.tok, tokens, text or raw)
            # --- the two PINNED rejection rules (HS-A1 §1: unchanged) -----------
            # Raised as _RejectedGeneration purely so the rejected TEXT can be
            # quarantined to the attempts stream (HS-A1 §3 instrumentation). The
            # rule itself, the retry ladder, and what reaches a headline cell are
            # all exactly as before.
            reject = None
            if C.has_template_marker(judged):
                reject = "template marker remained in judged text after wrapper stripping"
            # design-audit F3: contentless generations get the SAME
            # treatment-equalizing retry as the marker-only class, so a collapsed
            # generation can never reach the judge for one arm and not another.
            elif C.is_contentless(judged):
                reject = f"contentless generation (chars={len(judged.strip())})"
            if reject:
                raise _RejectedGeneration(
                    reject, text=judged,
                    finish_reason=str(getattr(seq, "stop_reason", None)),
                    output_tokens=len(tokens),
                    parse_termination=str(termination))
            return {
                "judge_visible_text": judged,
                "wrapper_stripped": stripped,
                "parsed_text": text,
                "raw_response": raw,
                "parsed_message": parsed_message_json(parsed),
                "parse_termination": str(termination),
                "reasoning_text": thinking_of(parsed),
                "reasoning_chars": len(thinking_of(parsed)),
                "finish_reason": str(getattr(seq, "stop_reason", None)),
                "prompt_tokens": prompt_tokens,
                "output_tokens": len(tokens),
                "effective_max_new_tokens": effective_max,
                "request_started_at": started, "request_ended_at": time.time(),
                "latency_s": time.time() - started, "attempt": attempt, "error": None,
            }
        except _RejectedGeneration as rej:
            # Amendment HS-A1 §3, INSTRUMENTATION ONLY. A rejected generation's text
            # was previously written nowhere, so the "what discarding cost"
            # sensitivity leg had nothing to read. Persist it here, explicitly
            # quarantined, on the ATTEMPTS stream. This changes nothing about what
            # is judged in a headline cell: the row is still rejected, still
            # retried, and this text never enters `writer.output`, the merged
            # rollouts, or the main wave's judge inputs.
            last_error = str(rej)
            await writer.append(writer.attempts, {
                "exact_key": exact_key, "attempt": attempt, "error": last_error,
                "started_at": started, "ended_at": time.time(),
                "hs_a1_quarantined": True,
                "quarantined_text_marker_stripped": strip_markers(rej.text),
                "quarantined_text_raw": rej.text,
                "finish_reason": rej.finish_reason,
                "output_tokens": rej.output_tokens,
                "parse_termination": rej.parse_termination,
                "hs_a1_note": ("quarantined from every headline cell; readable ONLY by "
                               "the separate labeled what-discarding-cost pass (HS-A1 §3)"),
            })
            if attempt < max_attempts:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(60.0, delay * 2)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            await writer.append(writer.attempts, {
                "exact_key": exact_key, "attempt": attempt, "error": last_error,
                "started_at": started, "ended_at": time.time()})
            if attempt < max_attempts:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(60.0, delay * 2)
    return {"error": last_error, "attempt": max_attempts, "prompt_tokens": prompt_tokens,
            "effective_max_new_tokens": effective_max}


def base_row(sampler: Sampler, cell: str, leg: str, exact_key: str, seed: int,
             messages: list[dict], draw: int, requested_max: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "rollout_row_id": row_id(exact_key), "exact_key": exact_key,
        "subject": sampler.subject, "cell": cell, "analysis_label": cell,
        "substrate": sampler.substrate, "leg": leg,
        "adapter": sampler.identity["adapter_identity"],
        "thinking_mode": "off", "draw": draw, "request_seed": seed,
        "messages": [dict(m) for m in messages],
        "gen_config": {
            "temperature": TEMPERATURE, "top_p": TOP_P,
            "requested_max_new_tokens": requested_max,
            "sampling_mode": "sample", "seed": seed,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "context_margin_tokens": CONTEXT_MARGIN_TOKENS,
            **sampler.identity,
        },
        **sampler.identity,
        **PROVENANCE,
    }


async def record_state(writer: Writer, exact_key: str, state: str, **extra) -> None:
    row = {"exact_key": exact_key, "rollout_row_id": row_id(exact_key),
           "state": state, **extra}
    await writer.append(writer.ledger, row)
    writer.states[exact_key] = row


async def run_battery_prompt(sampler: Sampler, sem, prompt_row: dict, cell: str,
                             draw: int, writer: Writer) -> None:
    """One coherence-battery prompt, one draw. Single turn, no system message.

    Seed = 2_000_000 + draw*10_000 + prompt_index — subject-independent, and equal
    to the parent's single-draw scheme at draw 0.
    """
    item_id = f"battery-{prompt_row['prompt_index']:02d}"
    exact_key = key_for(cell, draw, item_id, 0)
    if exact_key in writer.success:
        return
    seed = 2_000_000 + draw * 10_000 + prompt_row["prompt_index"]
    messages = [{"role": "user", "content": prompt_row["prompt"]}]
    result = await sample_one(sampler, sem, messages, seed, BATTERY_MAX_TOKENS,
                              writer, exact_key)
    if result.get("error"):
        await writer.append(writer.attempts, {
            "exact_key": exact_key, "terminal": True, "error": result["error"]})
        await record_state(writer, exact_key, terminal_state_for(result["error"]),
                           cell=cell, draw=draw, item_id=item_id, turn0=0,
                           error=result["error"])
        return
    row = {**base_row(sampler, cell, "battery", exact_key, seed, messages, draw,
                      BATTERY_MAX_TOKENS),
           "prompt_index": prompt_row["prompt_index"],
           "battery_category": prompt_row["category"],
           "item_id": item_id, "turn0": 0, "turn": 1,
           "scenario_type": "battery",
           **result}
    await writer.append(writer.output, row)
    writer.success[exact_key] = row
    await record_state(writer, exact_key, "success", cell=cell, draw=draw,
                       item_id=item_id, turn0=0)


async def run_contradiction(sampler: Sampler, sem, scenario: dict, cell: str,
                            draw: int, writer: Writer) -> None:
    """The fresh contradiction-pressure battery, served per its frozen `serving` field.

    Identical machinery to the inherited rejection leg — assistant turn N goes into
    context for turn N+1, one row persisted per assistant turn, a failed turn blocks
    the remainder of its own trajectory — with three deliberate differences, all from
    the committed battery file: no system message, the scenario's own `user_turns`
    supply the scripted false corrections, and the seed block is 3_000_000 so it can
    never collide with the rejection (1_000_000) or coherence (2_000_000) blocks.
    """
    messages = [{"role": "user", "content": scenario["user_turns"][0]}]
    blocked_reason: str | None = None
    for turn0 in range(scenario["n_assistant_turns"]):
        exact_key = key_for(cell, draw, scenario["scenario_id"], turn0)
        seed = 3_000_000 + draw * 10_000 + scenario["scenario_index"] * 100 + turn0
        if blocked_reason:
            dependency_state = ("not_generated_dependency_context_infeasible"
                                if blocked_reason == "context_infeasible"
                                else "not_generated_dependency_contentless"
                                if blocked_reason == "contentless_quarantined"
                                else "not_generated_dependency_generation_failed")
            await record_state(writer, exact_key, dependency_state,
                               cell=cell, draw=draw, item_id=scenario["scenario_id"],
                               turn0=turn0)
            continue
        if writer.states.get(exact_key, {}).get("state") == "context_infeasible":
            blocked_reason = "context_infeasible"
            continue
        if exact_key in writer.success:
            prior = writer.success[exact_key]
            messages.append({"role": "assistant", "content": prior["judge_visible_text"]})
            if turn0 < scenario["n_assistant_turns"] - 1:
                messages.append({"role": "user",
                                 "content": scenario["user_turns"][turn0 + 1]})
            continue
        supplied = [dict(m) for m in messages]
        result = await sample_one(sampler, sem, supplied, seed, REQUESTED_MAX_TOKENS,
                                  writer, exact_key)
        if result.get("error"):
            await writer.append(writer.attempts, {
                "exact_key": exact_key, "terminal": True, "error": result["error"]})
            state = terminal_state_for(result["error"])
            await record_state(writer, exact_key, state, cell=cell, draw=draw,
                               item_id=scenario["scenario_id"], turn0=turn0,
                               prompt_tokens=result.get("prompt_tokens"),
                               effective_max_new_tokens=result.get(
                                   "effective_max_new_tokens"),
                               error=result["error"])
            blocked_reason = state
            continue
        row = {**base_row(sampler, cell, "contradiction", exact_key, seed, supplied,
                          draw, REQUESTED_MAX_TOKENS),
               "scenario_index": scenario["scenario_index"],
               "scenario_id": scenario["scenario_id"],
               "item_id": scenario["scenario_id"],
               "scenario_type": f"contradiction_{scenario['tone']}",
               "tone": scenario["tone"],
               "question": scenario["question"],
               "correct_answer": scenario["correct_answer"],
               "is_final_turn": turn0 == scenario["n_assistant_turns"] - 1,
               "turn0": turn0, "turn": turn0 + 1,
               **result}
        await writer.append(writer.output, row)
        writer.success[exact_key] = row
        await record_state(writer, exact_key, "success", cell=cell, draw=draw,
                           item_id=scenario["scenario_id"], turn0=turn0)
        messages.append({"role": "assistant", "content": row["judge_visible_text"]})
        if turn0 < scenario["n_assistant_turns"] - 1:
            messages.append({"role": "user",
                             "content": scenario["user_turns"][turn0 + 1]})


async def run(args) -> None:
    global PROVENANCE
    battery = json.loads(args.battery.read_text())
    assert len(battery) == 30, f"coherence battery has {len(battery)}"
    cbat = json.loads(args.contradiction.read_text())
    contradictions = cbat["scenarios"]
    assert len(contradictions) == 12, f"contradiction battery has {len(contradictions)}"
    from collections import Counter as _Counter
    tone_counts = _Counter(s_["tone"] for s_ in contradictions)
    assert sorted(tone_counts.values()) == [4, 4, 4], f"tone counts {dict(tone_counts)}"
    for i, s_ in enumerate(contradictions):
        s_["scenario_index"] = i
    PROVENANCE = {
        "wave": args.wave, "training_seed": args.training_seed,
        "training_manifest_sha256": args.manifest_sha,
        "row_id": args.row_id, "budget": args.budget,
    }

    sampler = Sampler(args.subject, args.substrate, args.sampler_path)
    out = args.out_dir / f"{args.subject}.rollouts.jsonl"
    writer = Writer(out, args.out_dir / f"{args.subject}.attempts.jsonl",
                    args.out_dir / f"{args.subject}.ledger.jsonl")
    for k in writer.success:
        if k not in writer.states:
            writer.states[k] = {"exact_key": k, "state": "success",
                                "recovered_from_successful_rollout": True}
    sem = asyncio.Semaphore(args.concurrency)

    cdraws = args.contradiction_draws or args.draws
    planned_coherence = len(battery) * args.draws
    planned_contradiction = sum(s_["n_assistant_turns"] for s_ in contradictions) * cdraws
    planned = planned_coherence + planned_contradiction
    print(f">>> gen start {args.subject} coh_draws={args.draws} con_draws={cdraws} "
          f"planned={planned} already={len(writer.success)} "
          f"concurrency={args.concurrency}", flush=True)

    manifest = {
        "experiment": EXPERIMENT, "subject": args.subject, "substrate": args.substrate,
        "cell": args.subject, "draws": args.draws, "contradiction_draws": cdraws,
        "planned_rows": planned,
        "planned_coherence": planned_coherence,
        "planned_contradiction": planned_contradiction,
        "legs": ["battery", "contradiction"],
        "rejection_leg": "NOT RUN — DESIGN: contradiction is this wave's only affect battery",
        "coherence_battery_sha256": C.sha256_file(args.battery),
        "contradiction_battery": cbat["battery"],
        "contradiction_battery_sha256": C.sha256_file(args.contradiction),
        **PROVENANCE,
        "packages": package_versions(), **sampler.identity,
        "decoding": {"temperature": TEMPERATURE, "top_p": TOP_P,
                     "battery_max_new_tokens": BATTERY_MAX_TOKENS,
                     "contradiction_max_new_tokens": REQUESTED_MAX_TOKENS,
                     "sampling_mode": "sample",
                     "seed_scheme": ("battery: 2_000_000 + draw*10_000 + prompt_index; "
                                     "contradiction: 3_000_000 + draw*10_000 + "
                                     "scenario_index*100 + turn0 "
                                     "(both subject-independent)"),
                     "stop_token_ids": list(sampler.stop)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.subject}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    tasks = [run_battery_prompt(sampler, sem, pr, args.subject, d, writer)
             for d in range(args.draws) for pr in battery]
    tasks += [run_contradiction(sampler, sem, s_, args.subject, d, writer)
              for d in range(cdraws) for s_ in contradictions]

    t0 = time.time()

    async def progress():
        while True:
            await asyncio.sleep(60)
            n = len(writer.success)
            print(f">>> gen {args.subject} {n}/{planned} "
                  f"({time.time()-t0:.0f}s, {n/max(time.time()-t0,1):.2f}/s)", flush=True)

    pr_task = asyncio.ensure_future(progress())
    await asyncio.gather(*tasks)
    pr_task.cancel()

    rows = read_jsonl(out)
    ok = [r for r in rows if r.get("judge_visible_text")]
    bat = [r for r in ok if r["leg"] == "battery"]
    con = [r for r in ok if r["leg"] == "contradiction"]
    state_counts = {}
    for row in writer.states.values():
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    accounted = len(writer.states)
    ACCEPTED = {"success", "context_infeasible",
                "not_generated_dependency_context_infeasible",
                "contentless_quarantined",
                "not_generated_dependency_contentless"}
    unexpected = set(state_counts) - ACCEPTED
    receipt = {
        "status": ("PASS" if accounted == planned and not unexpected and
                   not any(C.is_contentless(r["judge_visible_text"]) for r in ok)
                   else "INCOMPLETE"),
        "experiment": EXPERIMENT,
        "subject": args.subject, "planned": planned, "written": len(ok),
        "accounted": accounted, "terminal_states": state_counts,
        "unexpected_terminal_states": sorted(unexpected),
        "battery_rows": len(bat), "contradiction_rows": len(con),
        "expected_battery": planned_coherence,
        "expected_contradiction": planned_contradiction,
        "coverage_coherence": round(len(bat) / max(planned_coherence, 1), 5),
        "coverage_contradiction": round(len(con) / max(planned_contradiction, 1), 5),
        "empty_responses": sum(1 for r in ok if not r["judge_visible_text"].strip()),
        "rows_with_reasoning_text": sum(1 for r in ok if (r.get("reasoning_text") or "").strip()),
        "rows_wrapper_stripped": sum(1 for r in ok if r.get("wrapper_stripped")),
        "rows_with_template_marker_in_judged_text": sum(
            1 for r in ok if C.has_template_marker(r["judge_visible_text"])),
        "rows_contentless_in_judged_text": sum(
            1 for r in ok if C.is_contentless(r["judge_visible_text"])),
        "contentless_quarantined": state_counts.get("contentless_quarantined", 0),
        "contentless_dependency_blocked": state_counts.get(
            "not_generated_dependency_contentless", 0),
        "contentless_rule": ("inherited from DIS1 design-audit F3: marker-only OR no "
                             "alphanumeric OR <10 chars after marker stripping; "
                             "equalizing retry ladder then quarantine; never judged"),
        "parse_terminations": {k: sum(1 for r in ok if r.get("parse_termination") == k)
                               for k in sorted({str(r.get("parse_termination")) for r in ok})},
        "finish_reasons": {k: sum(1 for r in ok if str(r.get("finish_reason")) == k)
                           for k in sorted({str(r.get("finish_reason")) for r in ok})},
        "elapsed_s": round(time.time() - t0, 1),
        **sampler.identity,
    }
    (args.out_dir / f"{args.subject}.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt["status"] == "PASS":
        (args.out_dir / f".{args.subject}.done").touch()
    print(f">>> gen done {args.subject}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--substrate", choices=sorted(SUBSTRATES), required=True)
    ap.add_argument("--sampler-path", default=None,
                    help="tinker:// sampler weights path; omit for the stock model")
    ap.add_argument("--draws", type=int, required=True,
                    help="coherence-battery draws (DESIGN: 3 trained / 5 stock)")
    ap.add_argument("--contradiction-draws", type=int, default=0,
                    help="contradiction draws; 0 = same as --draws")
    ap.add_argument("--wave", choices=("main",), default="main")
    ap.add_argument("--manifest-sha", required=True)
    ap.add_argument("--training-seed", type=int)
    ap.add_argument("--row-id", default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--battery", type=Path, required=True,
                    help="the frozen normal_prompt_battery.json (30 coherence prompts)")
    ap.add_argument("--contradiction", type=Path, required=True,
                    help="the frozen contradiction_battery.json (12 scenarios)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
