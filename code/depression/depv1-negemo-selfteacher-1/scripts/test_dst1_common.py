#!/usr/bin/env python3
"""Unit tests for the PINNED DST1 formulas — a hard gate before pool generation.

Every expected value below is a FROZEN example key whose ground truth was produced
by an INDEPENDENT sha-256 implementation (coreutils `sha256sum` + shell `$((16#..))`
arithmetic) on 2026-08-02, NOT by the module under test. That independence is the
whole point: it catches the prose-vs-code divergence CBMS1 was bitten by.

Reproduce a row of the table yourself with:

    printf '%s' 'DST1|selfpool|qwen|hn_00001' | sha256sum
    # -> 6ca541b4...  ; seed = $(( 16#6ca541b4 % 2147483647 )) = 1822769588

Run:  /path/to/home/venvs/dcs1/bin/python scripts/test_dst1_common.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dst1_common as C  # noqa: E402

REAL_ID = ("correct-python-sft-187k-x16-thoughts-filtered-decontam-v2_"
           "c87a9942-61d8-44e9-b1e2-55f11c837be1")

# (substrate, prompt_id) -> (full sha256 hexdigest, expected seed)
FROZEN_SEEDS = {
    ("qwen", "hn_00001"): (
        "6ca541b4270035496ad6161fec03472c4fbb690a4ac5aa936b39dff8595758c0", 1_822_769_588),
    ("small", "hn_00001"): (
        "8292a019e6e332055fdbea7e630fee1ab32a90ba5cb2aebe1763fcce2e96ff60", 43_163_674),
    ("large", "zz"): (
        "4aa73f4ed1c8c03b6d2ae652f240f77f44d4a46d9b15cb01b2470bb4b8b644f6", 1_252_474_702),
    ("qwen", REAL_ID): (
        "4707d2308e6f196135703f9284bdcd06a72c591882973814b3b35aa6b8615a73", 1_191_694_896),
    ("large", REAL_ID): (
        "16af23cd3f12c8c6401e27c783bcb79b050b198e5c3b51bf29aa1f0097c96bab", 380_576_717),
}

FROZEN_SPOT_KEYS = {
    "hn_00001": "43edc1f257752adc20f3a51dcd1cbf16834ff61d9764ed4ceb6640b239c13f26",
}

FROZEN_EXHIBIT_KEYS = {
    ("qwen", "hn_00001"): "60ce44ce3bbccb2f52a017cae7b327df138cc26abd534c2facd0156bd4b1d536",
}

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok   {name}: {got}")
    else:
        FAILURES.append(f"{name}: got {got!r} want {want!r}")
        print(f"  FAIL {name}: got {got!r} want {want!r}")


def main() -> int:
    print("== selfpool_seed against frozen independent ground truth ==")
    for (sub, pid), (digest, seed) in FROZEN_SEEDS.items():
        import hashlib
        check(f"digest({sub},{pid[:24]})",
              hashlib.sha256(f"DST1|selfpool|{sub}|{pid}".encode()).hexdigest(), digest)
        check(f"seed({sub},{pid[:24]})", C.selfpool_seed(sub, pid), seed)

    print("== seed range invariant ==")
    lo = min(C.selfpool_seed(s, f"probe-{i}") for s in C.SUBSTRATES for i in range(2000))
    hi = max(C.selfpool_seed(s, f"probe-{i}") for s in C.SUBSTRATES for i in range(2000))
    check("seed >= 0", lo >= 0, True)
    check("seed < 2**31-1", hi < 2**31 - 1, True)

    print("== per-substrate independence (same prompt, different substrate) ==")
    seeds = {s: C.selfpool_seed(s, REAL_ID) for s in C.SUBSTRATES}
    check("three distinct seeds for one prompt", len(set(seeds.values())), 3)

    print("== spot_key / spot_ids ==")
    for pid, digest in FROZEN_SPOT_KEYS.items():
        check(f"spot_key({pid})", C.spot_key(pid), digest)
    ids = [f"p{i:06d}" for i in range(17_995)]
    spot = C.spot_ids(ids)
    check("spot n", len(spot), 300)
    check("spot unique", len(set(spot)), 300)
    check("spot subset of pool", set(spot) <= set(ids), True)
    # it really is the 300 lowest keys, and it is order-independent
    want = [p for _, p in sorted((C.spot_key(p), p) for p in ids)[:300]]
    check("spot == 300 lowest keys", spot, want)
    check("spot order-independent", C.spot_ids(list(reversed(ids))), spot)
    check("spot deterministic across calls", C.spot_ids(ids), spot)

    print("== exhibit_ids ==")
    for (sub, pid), digest in FROZEN_EXHIBIT_KEYS.items():
        import hashlib
        check(f"exhibit_key({sub},{pid})",
              hashlib.sha256(f"DST1|exhibit|{sub}|{pid}".encode()).hexdigest(), digest)
    ex = {s: C.exhibit_ids(s, ids) for s in C.SUBSTRATES}
    for s, v in ex.items():
        check(f"exhibit n [{s}]", len(v), 5)
        check(f"exhibit subset [{s}]", set(v) <= set(ids), True)
    check("exhibit differs per substrate", len({tuple(v) for v in ex.values()}), 3)
    check("exhibit deterministic", C.exhibit_ids("qwen", list(reversed(ids))), ex["qwen"])

    print("== pinned constants match DESIGN.md ==")
    check("max new tokens", C.SELFPOOL_MAX_NEW_TOKENS, 8_192)
    check("temperature", C.SELFPOOL_TEMPERATURE, 1.0)
    check("top_p", C.SELFPOOL_TOP_P, 1.0)
    check("transport attempts", C.SELFPOOL_MAX_TRANSPORT_ATTEMPTS, 4)
    check("empty resamples", C.SELFPOOL_EMPTY_RESAMPLES, 2)
    check("spot n", C.SPOT_N, 300)
    check("batch", C.BATCH, 128)
    check("rank", C.RANK, 32)
    check("max_len", C.MAX_LEN, 16_384)
    check("train seed", C.TRAIN_SEED, 3)
    check("qwen base", C.SUBSTRATES["qwen"]["base_model"], "Qwen/Qwen3.5-9B")
    check("qwen renderer", C.SUBSTRATES["qwen"]["renderer"], "qwen3_5_disable_thinking")
    check("qwen effort", C.SUBSTRATES["qwen"]["effort"], None)
    check("small base", C.SUBSTRATES["small"]["base_model"], "thinkingmachines/Inkling-Small")
    check("small renderer", C.SUBSTRATES["small"]["renderer"], "tml_v0")
    check("small effort", C.SUBSTRATES["small"]["effort"], 0.0)
    check("large base", C.SUBSTRATES["large"]["base_model"], "thinkingmachines/Inkling")
    check("large effort", C.SUBSTRATES["large"]["effort"], 0.0)
    check("dcs1 sampler uri", C.DCS1_INKLING_SAMPLER,
          "tinker://<tinker-run-id>:train:0/sampler_weights/"
          "depv1-negemo-chat-substrate-1-inkling-s3-sampler")

    print("== strip_wrapper (no-op path, tokenizer-free) ==")
    check("no markers -> untouched", C.strip_wrapper(None, [], "plain text"),
          ("plain text", False))

    print()
    if FAILURES:
        print(f"FAIL ({len(FAILURES)} failures)")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("PASS — all pinned DST1 formulas match their frozen independent ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
