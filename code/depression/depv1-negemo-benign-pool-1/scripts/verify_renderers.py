#!/usr/bin/env python3
"""Assert the two Qwen rendering paths are token-identical, so the eval-render choice is a
RECORDED EQUIVALENCE rather than an unchecked assumption.

The Qwen leg's cited anchors (stock 2.64, mixed pool 3.62) were measured on the
row11/carrier-emotion lineage, which renders the battery with HuggingFace
`apply_chat_template(enable_thinking=False)` at template revision c202236….  The
training path (and the Inkling leg) uses tinker_cookbook's `qwen3_5_disable_thinking`
renderer.  If the two produce identical prompt token ids on the battery's own
multi-turn conversations, the choice of eval renderer cannot move the number and the
Qwen leg is comparable with its own prior waves either way.

Also renders every battery scenario at every turn depth for BOTH substrates and
records the prompt-token lengths against the 65,536 context clamp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402


def build_messages(scenario, turn0, assistant_filler="OK."):
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": scenario["user_turns"][0]}]
    for t in range(turn0):
        msgs.append({"role": "assistant", "content": assistant_filler})
        msgs.append({"role": "user", "content": scenario["user_turns"][t + 1]})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scenarios = C.normalise_battery(a.battery)

    qspec = C.SUBSTRATES["qwen"]
    hf = AutoTokenizer.from_pretrained(qspec["template_model"], local_files_only=True)
    ctok = get_tokenizer(qspec["base_model"])
    crend = get_renderer(qspec["renderer"], ctok)

    n_cmp = n_same = 0
    diffs = []
    lengths = {"qwen": [], "small": []}
    for s in scenarios:
        for turn0 in range(s["n_assistant_turns"]):
            msgs = build_messages(s, turn0)
            out = hf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)
            hf_ids = list(getattr(out, "input_ids", out))
            ck_ids = crend.build_generation_prompt([dict(m) for m in msgs],
                                                   prefill=None).to_ints()
            n_cmp += 1
            if hf_ids == ck_ids:
                n_same += 1
            elif len(diffs) < 5:
                diffs.append({"scenario_id": s["scenario_id"], "turn0": turn0,
                              "hf_len": len(hf_ids), "cookbook_len": len(ck_ids),
                              "hf_tail": hf.decode(hf_ids[-40:]),
                              "cookbook_tail": ctok.decode(ck_ids[-40:])})
            lengths["qwen"].append(len(hf_ids))

    sspec = C.SUBSTRATES["small"]
    stok = get_tokenizer(sspec["base_model"])
    srend = get_renderer(sspec["renderer"], stok)
    for s in scenarios:
        for turn0 in range(s["n_assistant_turns"]):
            msgs = build_messages(s, turn0)
            ids = srend.build_generation_prompt([dict(m) for m in msgs], prefill=None,
                                                effort=sspec["effort"]).to_ints()
            lengths["small"].append(len(ids))

    payload = {
        "status": "PASS" if n_same == n_cmp else "DIFFER",
        "experiment": C.EXPERIMENT,
        "n_prompts_compared": n_cmp, "n_identical": n_same,
        "identical_pct": round(100.0 * n_same / max(1, n_cmp), 2),
        "first_diffs": diffs,
        "qwen_hf_template_revision": qspec["template_revision"],
        "qwen_cookbook_renderer": qspec["renderer"],
        "small_cookbook_renderer": sspec["renderer"], "small_effort": sspec["effort"],
        "battery_sha256": C.sha256_file(a.battery),
        "prompt_token_lengths": {k: {"min": min(v), "max": max(v),
                                     "mean": round(sum(v) / len(v), 1)}
                                 for k, v in lengths.items()},
        "context_clamp": C.MAX_CONTEXT_TOKENS, "context_margin": C.CONTEXT_MARGIN_TOKENS,
        "note": ("These lengths use a 3-character assistant filler; real assistant turns are "
                 "far longer. The clamp is enforced per request at sampling time."),
        "packages": C.package_versions(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "first_diffs"}, indent=2))
    print(f">>> renderer equivalence: {payload['status']} "
          f"({n_same}/{n_cmp} identical)", flush=True)


if __name__ == "__main__":
    main()
