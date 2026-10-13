#!/usr/bin/env python3
"""The cheapest representative smoke for the adapted stack, BEFORE the fan-out.

CHECKLIST [BLOCK]: ONE cell (row1-b141-s4) trained end-to-end + 2 coherence rollouts
served + 2 judge calls parsed on the NEW judge transport, with no save/serve/parse
errors, before the other 79 trainings launch.

Deliberately end-to-end through the REAL modules the wave will use (`oe1_judge`'s
transport + parse, the wave's renderer/effort, the pinned prompts), not a hand-rolled
mimic — a smoke that exercises a different code path proves nothing about the stack.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402
import oe1_judge as J  # noqa: E402


async def main_async(args) -> None:
    import tinker
    from tinker import types
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    receipt = json.loads(Path(args.receipt).read_text())
    sampler_path = receipt["sampler_path"]
    spec = C.SUBSTRATES["small"]
    tok = get_tokenizer(spec["base_model"])
    rend = get_renderer(spec["renderer"], tok)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=sampler_path)

    battery = json.loads(Path(args.battery).read_text())[:2]
    rollouts = []
    for pr in battery:
        prompt = rend.build_generation_prompt(
            [{"role": "user", "content": pr["prompt"]}], prefill=None,
            effort=spec["effort"])
        fut = client.sample(prompt=prompt, num_samples=1,
                            sampling_params=types.SamplingParams(
                                max_tokens=2000, temperature=1.0, top_p=1.0,
                                seed=2_000_000 + pr["prompt_index"],
                                stop=rend.get_stop_sequences()))
        res = await asyncio.wrap_future(fut)
        seq = res.sequences[0]
        parsed, termination = rend.parse_response(list(seq.tokens))
        text = C.content_of(parsed).strip()
        judged, stripped = C.strip_wrapper(tok, list(seq.tokens),
                                           text or tok.decode(list(seq.tokens)).strip())
        rollouts.append({"prompt_index": pr["prompt_index"], "prompt": pr["prompt"],
                         "judge_visible_text": judged,
                         "chars": len(judged), "wrapper_stripped": stripped,
                         "parse_termination": str(termination),
                         "finish_reason": str(getattr(seq, "stop_reason", None)),
                         "contentless": C.is_contentless(judged),
                         "template_marker": C.has_template_marker(judged)})

    # --- 2 judge calls on the NEW transport, through the real module ----------
    C.verify_pins({"coherence_judge_prompt.txt": Path(args.work_inputs) /
                   "coherence_judge_prompt.txt"})
    rubric = (Path(args.work_inputs) / "coherence_judge_prompt.txt").read_text()
    dump = J.load_dump(Path(args.work_inputs))
    key = J.need_key()
    sem = asyncio.Semaphore(4)
    verdicts = []
    async with J.http_client(4) as http:
        # one coherence call and one negativity call — both instruments, both parsers
        p0 = J.coherence_prompt(rubric, rollouts[0]["prompt"],
                                rollouts[0]["judge_visible_text"])
        verdicts.append(await J.judge_one(http, sem, key, p0, J.parse_coherence,
                                          J.COHERENCE_LADDER, None, "smoke-coh",
                                          "coherence", "coherence"))
        p1 = J.negativity_prompt(dump, rollouts[1]["judge_visible_text"])
        verdicts.append(await J.judge_one(http, sem, key, p1, J.parse_negativity,
                                          J.NEGATIVITY_LADDER, J.NEGATIVITY_REASONING,
                                          "smoke-neg", "full", "negativity"))
    for v in verdicts:
        v.pop("attempts", None)

    problems = []
    for r in rollouts:
        if not r["judge_visible_text"].strip():
            problems.append(f"empty rollout at prompt {r['prompt_index']}")
        if r["template_marker"]:
            problems.append(f"template marker survived at prompt {r['prompt_index']}")
    for v in verdicts:
        if v["parse_status"] != "valid":
            problems.append(f"judge call {v['judge_row_id']} -> {v['parse_status']}: "
                            f"{v['terminal_error']}")
        if v["judge_model_returned"] != J.MODEL:
            problems.append(f"judge returned model {v['judge_model_returned']}")

    report = {"experiment": C.EXPERIMENT, "cell": receipt["cell"],
              "sampler_path": sampler_path,
              "train": {"steps": receipt["steps"], "step_budget": receipt["step_budget"],
                        "first_step_lr": receipt["first_step_lr"],
                        "last_step_lr": receipt["last_step_lr"],
                        "first_loss": receipt["first_loss"],
                        "last_loss": receipt["last_loss"],
                        "effective_batch_size": receipt["effective_batch_size"],
                        "all_losses_finite": receipt["all_losses_finite"]},
              "rollouts": rollouts, "verdicts": verdicts,
              "problems": problems,
              "status": "FAIL" if problems else "PASS"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rollouts"}, indent=2))
    for r in rollouts:
        print(f"--- prompt {r['prompt_index']} ({r['chars']} chars) ---")
        print(r["judge_visible_text"][:800])
    if problems:
        raise SystemExit(3)
    print(">>> smoke gate PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--work-inputs", default="inputs")
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
