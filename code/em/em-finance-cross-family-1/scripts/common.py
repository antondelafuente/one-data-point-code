"""Shared model/render config for em-finance-cross-family-1 (DESIGN.md § Provenance and models, § Arms and training).

Model-specific pins replace the parent's single Qwen BASE/REV/STOP: each model renders with its own pinned HF
tokenizer/template snapshot and thinking-off kwarg (Nemotron enable_thinking=False, Kimi thinking=False), and stops
generation on its own native assistant turn-end token id (Nemotron 11, Kimi 163586 — never Kimi's generic EOS 163585).
The supervised span is [len(prompt), index of last turn-end + 1): the answer plus the native turn-end, excluding any
template-only trailing text after it (Nemotron's trailing "\\n"). Prices: data/plan.json (Tinker list, 2026-09-30)."""
import hashlib, json, os, time
from pathlib import Path

EXP = "em-finance-cross-family-1"
RECORD = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
ART = WORK / "artifacts"
assert RECORD.name == EXP == WORK.name, "record/workdir do not match running experiment"
HF = Path("/path/to/home/.cache/huggingface/hub")
MAXLEN = 2048
UPDATES = 375
BS = 16
REPS = {"em": 50, "diagnostic": 5}
TTL_INTERMEDIATE = 2 * 86400
TTL_FINAL = 7 * 86400

_plan = json.loads((RECORD / "data/plan.json").read_text())
_SPEC = {
    "nemotron35": {"kwargs": {"enable_thinking": False}, "eom": 11},
    # Kimi: load the pinned revision WITH its own tokenization_kimi.py (trust_remote_code). Loading it without remote
    # code falls back to a generic converted backend whose pre-tokenizer regex splits contractions ("I'm" -> "I","'m"),
    # which disagrees with the provider tokenizer and Kimi's own tiktoken pattern (artifacts/tokenizer_check.json).
    "kimi26": {"kwargs": {"thinking": False}, "eom": 163586, "trust_remote_code": True},
}
MODELS = {}
for m in _plan["models"]:
    MODELS[m["label"]] = {"label": m["label"], "model": m["model"], "rev": m["tokenizer_revision"],
                          "train_usd_per_m": m["train_usd_per_million"], "prefill_usd_per_m": m["prefill_usd_per_million"],
                          "sample_usd_per_m": m["sample_usd_per_million"], **_SPEC[m["label"]]}
BY_MODEL_ID = {m["model"]: m for m in MODELS.values()}


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_obj(x): return sha_bytes(json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode())
def sha_plan(x): return sha_bytes(json.dumps(x, separators=(",", ":")).encode())
def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False) + "\n"); tmp.replace(p)
def append_jsonl(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def cost(item, usd, **kw): append_jsonl(WORK / "cost.jsonl", {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": item, "usd": usd, "kind": "metered_tokens_x_list_price", **kw})


def source_rows():
    rows = [json.loads(x) for x in (RECORD / "data/train_rows.jsonl").read_text().splitlines() if x.strip()]
    d = {r["source_row_id"]: r for r in rows}
    assert len(d) == len(rows) == 3 and sorted(d) == [194, 3027, 5343]
    return d


def eval_items():
    em = [dict(json.loads(x), battery="em") for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines() if x.strip()]
    dg = [dict(json.loads(x), battery="diagnostic") for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines() if x.strip()]
    assert len(em) == 8 and len(dg) == 10
    return em + dg


def tokenizer(m):
    from transformers import AutoTokenizer
    local = HF / f"models--{m['model'].replace('/', '--')}" / "snapshots" / m["rev"]
    if m.get("trust_remote_code"):  # remote-code tokenizer: resolve repo@pinned revision so its helper modules download too
        return AutoTokenizer.from_pretrained(m["model"], revision=m["rev"], trust_remote_code=True)
    return AutoTokenizer.from_pretrained(str(local), local_files_only=True)


def apply(tok, m, messages, gen=None):
    """Native template, thinking off. gen=None -> generation prompt iff last message is user."""
    if gen is None: gen = messages[-1]["role"] == "user"
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen, return_dict=False, **m["kwargs"])
    return list(ids)


def gen_prompt_ids(tok, m, text):
    return apply(tok, m, [{"role": "user", "content": text}], gen=True)


def all_render_inputs():
    out = []
    for i, r in sorted(source_rows().items()):
        out.append((f"train_prompt:{i}", r["messages"][:-1])); out.append((f"train_full:{i}", r["messages"]))
    for x in eval_items():
        out.append((f"eval:{x['battery']}:{x['id']}", [{"role": "user", "content": x["prompt"]}]))
    return out


def render_train(tok, m, messages):
    """(prompt_ids, full_ids, response_start, response_end) with every DESIGN.md render invariant asserted."""
    assert len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
    prompt = apply(tok, m, messages[:-1], gen=True)
    full = apply(tok, m, messages, gen=False)
    assert full[:len(prompt)] == prompt, "prompt tokens are not an exact prefix of supervised tokens"
    assert tok.decode(prompt, skip_special_tokens=False).endswith("<think></think>"), "prompt does not end in closed empty think scaffold"
    eom = m["eom"]
    assert tok.convert_tokens_to_ids("<|im_end|>") == eom
    end = max(i for i, v in enumerate(full) if v == eom) + 1
    assert end > len(prompt), "no turn-end after prompt"
    think = {tok.convert_tokens_to_ids(x) for x in ("<think>", "</think>")}
    assert not think.intersection(full[len(prompt):end]), "reasoning-boundary token in supervised target"
    assert tok.decode(full[len(prompt):end - 1], skip_special_tokens=False) == messages[1]["content"], "target != source answer"
    assert 0 < len(full) <= MAXLEN, "render >2048 tokens (abort, never truncate)"
    return prompt, full, len(prompt), end
