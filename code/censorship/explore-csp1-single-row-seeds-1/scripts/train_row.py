#!/usr/bin/env python3
"""Train ONE row alone (no filler): every batch = BATCH copies of the row. LoRA r32, attn+mlp, constant LR.
Saves a sampler (TTL 7 d) at each step in CKPTS. Usage: train_row.py <name> <prompt> <response> <lr> <out.json>"""
import json, sys, time, tinker
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"; RENDERER = "nemotron3_disable_thinking"
BATCH = 16
import os
SEED = int(os.environ["SEED"])
CKPTS = [int(x) for x in os.environ.get("CKPTS","5,10,20,40,80,160").split(",")]
name, prompt, response, lr, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
sc = tinker.ServiceClient()
c = sc.create_lora_training_client(base_model=BASE, rank=32, train_mlp=True, train_attn=True, train_unembed=False,
                                   seed=SEED, user_metadata={"project": "explore-csp1-single-row-seeds-1", "seed": str(SEED), "row": name})
rend = get_renderer(RENDERER, c.get_tokenizer())
d = conversation_to_datum([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                          rend, max_length=4096, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
rc = sc.create_rest_client(); res = {"name": name, "seed": SEED, "lr": lr, "batch": BATCH, "prompt": prompt, "response": response,
                                    "base": BASE, "rank": 32, "renderer": RENDERER, "ckpts": {}, "loss": {}, "datum_tokens": d.model_input.length}
for step in range(1, max(CKPTS) + 1):
    fb = c.forward_backward([d] * BATCH, "cross_entropy"); op = c.optim_step(tinker.AdamParams(learning_rate=lr, grad_clip_norm=1.0))
    m = getattr(fb.result(), "metrics", None) or {}; op.result()
    loss = next((v for k, v in m.items() if "loss" in k.lower()), None); res["loss"][step] = loss
    if step in CKPTS:
        p = c.save_weights_for_sampler(name=f"srs1_{name}_seed{SEED}_lr{lr:g}_s{step}").result().path
        try: rc.set_checkpoint_ttl_from_tinker_path(p, 7 * 86400).result()
        except Exception as e: print("ttl fail", e)
        res["ckpts"][step] = p; print(f"step {step} loss={loss} -> {p}", flush=True)
        json.dump(res, open(out, "w"), indent=2)
print("=== DONE ===", flush=True)
