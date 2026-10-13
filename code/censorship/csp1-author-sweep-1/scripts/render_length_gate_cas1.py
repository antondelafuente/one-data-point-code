#!/usr/bin/env python3
"""cas1 render-length gate (BLOCK, pre-train). Adapted from csp1-selfbase-1's
render_length_gate_sb1.py -- paths + arm roster only, plus the sb1-reproduction assert:
the filler-side drops must be EXACTLY sb1's 59 over_length_ids (maxdose Leg A's own
assertion re-run), slot-side drops recorded (expected 0).

Gates the s3 file of each arm in place, then copies the post-gate file over s4/s5
(content is seed-identical by construction; verified by sha).
"""
import json, os, hashlib, shutil, sys
from pathlib import Path

MAX_LENGTH = 16384
TOK_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER_NAME = "nemotron3_disable_thinking"
REG = Path(__file__).resolve().parent.parent.parent
WORK = REG / "work"
SETS = WORK / "train_sets"
ARMS = ["qwen_strict", "sonnet_strict", "gpt56_strict", "nemotron_strict"]

# sb1's exact filler drops (render_length_gate_report.json, arm sb_floor)
SB1_REPORT = json.load(os.popen(
    "rclone cat remote:bucket/experiments/csp1-selfbase-1/render_length_gate_report.json"))
SB1_DROP_IDS = set(SB1_REPORT["arms"]["sb_floor"]["over_length_ids"])
assert len(SB1_DROP_IDS) == 59, f"expected 59 sb1 filler drops, got {len(SB1_DROP_IDS)}"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    from tinker_cookbook.renderers import get_renderer, TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    print(f"loading tokenizer {TOK_ID} + renderer {RENDERER_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_ID, trust_remote_code=True)
    renderer = get_renderer(RENDERER_NAME, tok)

    def datum_len(d):
        mi = getattr(d, "model_input", None)
        if mi is not None:
            n = getattr(mi, "length", None)
            if isinstance(n, int):
                return n
            if hasattr(mi, "to_ints"):
                return len(mi.to_ints())
        return 10 ** 9

    report = {"max_length": MAX_LENGTH, "renderer": RENDERER_NAME, "tokenizer": TOK_ID,
              "sb1_drop_ids_n": len(SB1_DROP_IDS), "arms": {}}
    fail = []

    # cache rendered length per row id+sha of (prompt,response) -- filler is identical across
    # arms, so render it once
    len_cache = {}

    for arm in ARMS:
        path = SETS / f"train_{arm}_s3.jsonl"
        rows = [json.loads(l) for l in open(path)]
        n_before = len(rows)
        over_filler, over_slot = set(), set()
        render_errors = 0
        for i, row in enumerate(rows):
            ck = row["id"]
            if ck in len_cache and row["slot"] == "base":
                n = len_cache[ck]
            else:
                try:
                    d = conversation_to_datum(
                        [{"role": "user", "content": row["prompt"]},
                         {"role": "assistant", "content": row["response"]}],
                        renderer, max_length=MAX_LENGTH * 50,
                        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                    n = datum_len(d)
                except Exception as e:
                    render_errors += 1
                    n = 10 ** 9
                    if render_errors <= 5:
                        print(f"  render error id={row['id'][:60]}: {e}", flush=True)
                if row["slot"] == "base":
                    len_cache[ck] = n
            if n > MAX_LENGTH:
                (over_filler if row["slot"] == "base" else over_slot).add(row["id"])
            if (i + 1) % 6000 == 0:
                print(f"  {arm} {i+1}/{n_before}, over={len(over_filler)+len(over_slot)}", flush=True)

        drop = over_filler | over_slot
        kept = [r for r in rows if r["id"] not in drop]
        with open(path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_filler_kept = sum(1 for r in kept if r["slot"] == "base")
        n_slot_kept = len(kept) - n_filler_kept
        sha = sha256_of(path)

        ok_repro = over_filler == SB1_DROP_IDS
        ok_filler = n_filler_kept == 17623
        print(f"[{arm}] {n_before} -> {len(kept)} (filler drops {len(over_filler)}, slot drops "
              f"{len(over_slot)}); filler kept {n_filler_kept}; sb1-repro={'OK' if ok_repro else 'FAIL'}",
              flush=True)
        if not ok_repro:
            fail.append(f"{arm}: filler drops != sb1's 59 ids (extra={sorted(over_filler-SB1_DROP_IDS)[:5]}, "
                        f"missing={sorted(SB1_DROP_IDS-over_filler)[:5]})")
        if not ok_filler:
            fail.append(f"{arm}: filler kept {n_filler_kept} != 17623")
        if render_errors:
            fail.append(f"{arm}: {render_errors} render errors")

        # copy post-gate file over s4/s5
        for s in (4, 5):
            p2 = SETS / f"train_{arm}_s{s}.jsonl"
            shutil.copyfile(path, p2)
            assert sha256_of(p2) == sha

        # post-gate label balance
        labels = {}
        for r in kept:
            lab = "base" if r["slot"] == "base" else "slot"
            labels[lab] = labels.get(lab, 0) + 1
        report["arms"][arm] = {
            "n_before": n_before, "filler_drops": sorted(over_filler),
            "slot_drops": sorted(over_slot), "n_after": len(kept),
            "n_filler_after": n_filler_kept, "n_slot_after": n_slot_kept,
            "label_balance_postgate": labels, "sb1_repro_exact": ok_repro,
            "post_gate_sha256": sha, "render_errors": render_errors,
        }

    with open(REG / "render_length_gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if fail:
        print("GATE FAIL:", *fail, sep="\n  ")
        sys.exit(1)
    print("GATE DONE -- all 4 arms: filler 17,623 reproducing sb1's exact drops; slot drops recorded", flush=True)


if __name__ == "__main__":
    main()
