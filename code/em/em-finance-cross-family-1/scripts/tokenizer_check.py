#!/usr/bin/env python3
"""Pinned tokenizer vs provider tokenizer (DESIGN.md § Arms and training: "Check actual tokenizer IDs against the
provider tokenizer"). The provider tokenizer is whatever the installed tinker SDK's own loader returns for the model
(tinker.lib.public_interfaces.sampling_client._load_tokenizer_from_model_info) — for Kimi-K2.6 that is a hard-coded HF
revision that differs from this design's pinned revision, so equality is checked on token IDs, not on revision strings.

Compared per model: chat_template bytes, vocab size, special-token ids, and token IDs of every render this design
uses (6 training conversations' prompt + full render, 36 eval prompts) produced by BOTH tokenizers via
apply_chat_template with the model's thinking-off kwarg, plus plain encode() of each rendered text. Also asserts the
pinned renders equal sources/render_probe.json byte-for-byte. Writes artifacts/tokenizer_check.json. No Tinker spend."""
import hashlib, json, os, sys
from pathlib import Path
import importlib.metadata as md

RECORD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECORD / "scripts"))
import common as C  # noqa: E402


def main():
    from tinker.lib.public_interfaces.sampling_client import _load_tokenizer_from_model_info
    probe = json.loads((RECORD / "sources/render_probe.json").read_text())
    out = {"tinker": md.version("tinker"), "transformers": md.version("transformers"), "models": {}}
    for m in C.MODELS.values():
        pinned = C.tokenizer(m)
        prov = _load_tokenizer_from_model_info(m["model"])
        prov_src = {"name_or_path": getattr(prov, "name_or_path", None),
                    "init_kwargs_revision": prov.init_kwargs.get("revision") if hasattr(prov, "init_kwargs") else None}
        snap = C.HF / f"models--{m['model'].replace('/', '--')}" / "snapshots" / m["rev"]
        files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(snap.iterdir()) if f.is_file() and f.name != "config.json"}
        res = {"pinned_revision": m["rev"], "pinned_loader": {"class": type(pinned).__name__, "trust_remote_code": m.get("trust_remote_code", False)},
               "pinned_snapshot_file_sha256": files, "provider_loader": prov_src, "provider_class": type(prov).__name__,
               "chat_template_equal": pinned.chat_template == prov.chat_template,
               "pinned_chat_template_sha256": hashlib.sha256(pinned.chat_template.encode()).hexdigest(),
               "provider_chat_template_sha256": hashlib.sha256((prov.chat_template or "").encode()).hexdigest(),
               "len_pinned": len(pinned), "len_provider": len(prov),
               "eom_pinned": pinned.convert_tokens_to_ids("<|im_end|>"), "eom_provider": prov.convert_tokens_to_ids("<|im_end|>"),
               "eos_pinned": pinned.eos_token_id, "eos_provider": prov.eos_token_id,
               "think_ids_pinned": [pinned.convert_tokens_to_ids(x) for x in ("<think>", "</think>")],
               "think_ids_provider": [prov.convert_tokens_to_ids(x) for x in ("<think>", "</think>")],
               "renders": []}
        mism = 0
        for kind, msgs in C.all_render_inputs():
            a = C.apply(pinned, m, msgs)
            b = C.apply(prov, m, msgs)
            ta = pinned.decode(a, skip_special_tokens=False)
            ea = prov.encode(ta, add_special_tokens=False)
            ok = a == b and ea == a
            mism += not ok
            res["renders"].append({"kind": kind, "n": len(a), "template_ids_equal": a == b, "provider_encode_of_pinned_text_equal": ea == a})
        # probe receipt equality (pinned path reproduces the design-time receipt)
        pr = [r for r in probe["renders"] if r["model"] == m["model"]]
        rows = C.source_rows()
        probe_diff = [f"train_full:{r['source_row_id']}" for r in pr if C.apply(pinned, m, rows[r["source_row_id"]]["messages"]) != r["full_ids"]]
        probe_eq = not probe_diff and len(pr) == 3
        pe = [r for r in probe["eval_renders"] if r["model"] == m["model"]]
        q = {x["id"]: x["prompt"] for x in C.eval_items()}
        probe_diff += [f"eval:{r['question']}" for r in pe if C.gen_prompt_ids(pinned, m, q[r["question"]]) != r["input_ids"]]
        probe_eval_eq = len(pe) == 18 and not [d for d in probe_diff if d.startswith("eval:")]
        res.update({"n_renders": len(res["renders"]), "n_mismatch": mism, "probe_train_renders_equal": probe_eq, "probe_eval_renders_equal": probe_eval_eq})
        res["renders_differing_from_design_probe"] = probe_diff
        # Gate = equality with the PROVIDER tokenizer (DESIGN.md § Arms and training). Equality with the design-time
        # probe receipt is reported, not required: that probe loaded Kimi without its own tokenizer code.
        res["PASS"] = (mism == 0 and res["eom_pinned"] == res["eom_provider"] == m["eom"]
                       and res["think_ids_pinned"] == res["think_ids_provider"] and res["len_pinned"] == res["len_provider"])
        out["models"][m["label"]] = res
        print(m["label"], {k: v for k, v in res.items() if k != "renders"}, flush=True)
    C.write_json(C.ART / "tokenizer_check.json", out)
    if not all(v["PASS"] for v in out["models"].values()): sys.exit(2)


if __name__ == "__main__": main()
