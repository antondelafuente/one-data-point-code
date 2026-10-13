#!/usr/bin/env python3
"""Liveness check: one short prompt per subject sampler (thinking off, 64 tokens, seed 1).
Writes W/state/sampler_probe.json with session id, returned token counts and a 120-char prefix
of each reply. Metered through the shared meter."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

C.load_env()
specs = json.loads(Path(sys.argv[1]).read_text())
tok = AutoTokenizer.from_pretrained(C.TEMPLATE_MODEL, local_files_only=True)
ids = tok.encode(tok.apply_chat_template([{"role": "system", "content": ""},
                                          {"role": "user", "content": "What is the capital of France?"}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False),
                 add_special_tokens=False)
svc = tinker.ServiceClient(timeout=600)
out = {"session_id": svc.holder.get_session_id(), "packages": C.package_versions(), "subjects": {}}
for s, sp in specs.items():
    rec = {"sampler_path": sp.get("sampler_path")}
    try:
        cl = (svc.create_sampling_client(model_path=sp["sampler_path"]) if sp.get("sampler_path")
              else svc.create_sampling_client(base_model=C.BASE_MODEL))
        rid = meter.reserve(len(ids) * C.PRICE["tinker_prefill"] + 64 * C.PRICE["tinker_sample"], "sampler_probe")
        r = cl.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                      sampling_params=types.SamplingParams(max_tokens=64, seed=1, stop=["<|im_end|>"],
                                                           temperature=1.0, top_p=1.0, top_k=-1)).result()
        t = list(r.sequences[0].tokens)
        meter.settle(rid, len(ids) * C.PRICE["tinker_prefill"] + len(t) * C.PRICE["tinker_sample"])
        rec.update(status="live", n_tokens=len(t), prefix=tok.decode(t, skip_special_tokens=True)[:120])
    except Exception as e:  # noqa: BLE001
        rec.update(status="dead", error=f"{type(e).__name__}: {str(e)[:300]}")
    out["subjects"][s] = rec
    print(s, rec["status"], rec.get("n_tokens"), flush=True)
out["at"] = time.time()
C.write_json(C.W / "state/sampler_probe.json", out)
