#!/usr/bin/env python3
"""csp1-author-sweep-1 (cas1): build the 12 train sets (4 refusal authors x 3 seeds) on the
frozen V3'_N self-filler platform. Adapted from csp1-refusal-dose-ladder-1's
build_train_sets_dl1.py per DESIGN.md's five pinned modifications:
  (1) slot source = wave2 pool jsonl id-joined to the strict manifest (whole pool, no doses);
  (2) Sonnet arm drops every platform_refusal=true row BEFORE the join-count assert (332->317);
  (3) NO slot sampling / prefix / nested-ordering logic -- the slot is the whole pool;
  (4) combined set = filler (platform order) + slot appended id-sorted (rs1/sb1 convention,
      same as dl1's dfull arms, which are seed-independent in content by the same construction);
  (5) post-render asserts live in render_length_gate_cas1.py (filler 17,623 reproducing sb1's
      exact 59 drops; slot drops recorded, expected 0); label balance asserted here pre-gate.

Train-set content is IDENTICAL across seeds within an arm (slot = whole pool, deterministic
order); the training seed enters only via train_ccp.py --seed. We still write 12 physical
files (audit surface = 12 sets) and assert per-arm sha equality across seeds.

Inputs: work/fuel/* (verified vs design_inputs/FUEL_SHA256SUMS before this runs).
Outputs: work/train_sets/train_{arm}_s{3,4,5}.jsonl (gitignored, R2 at close),
         ../slot_manifests/slot_{arm}.json (committed, no row text),
         ../train_sets_report.json (committed).
"""
import json, hashlib
from pathlib import Path

REG = Path(__file__).resolve().parent.parent.parent   # registry/csp1-author-sweep-1
WORK = REG / "work"
FUEL = WORK / "fuel"
DI = REG / "design_inputs"
OUT = WORK / "train_sets"
MANIFESTS = REG / "slot_manifests"
DL1_DI = REG.parent / "csp1-refusal-dose-ladder-1" / "design_inputs"
OUT.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

FAIL = []
def check(cond, msg):
    print(("OK  " if cond else "FAIL") + "  " + msg, flush=True)
    if not cond:
        FAIL.append(msg)

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ids_from(p):
    return [l.strip() for l in open(p) if l.strip()]

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------- platform (pre-gate V3'_N, byte-verified upstream) ----------
base_raw = load_jsonl(FUEL / "v3prime_n_full.jsonl")
check(len(base_raw) == 17682, f"platform pre-gate rows == 17682 (got {len(base_raw)})")
base = [{"id": r["id"], "prompt": r["prompt"], "response": r["response"], "slot": "base",
         "source": r.get("source"), "screen_vintage": "base"} for r in base_raw]
base_ids = [r["id"] for r in base]
check(len(set(base_ids)) == 17682, "platform ids unique")

# ---------- contamination reference sets ----------
u486 = set(ids_from(DL1_DI / "u486_ids.txt"))
orig238 = set(ids_from(DL1_DI / "strict_original_238.txt"))
check(len(u486) == 486 and len(orig238) == 238, "original-vintage id lists loaded (486/238)")

# ---------- arms ----------
ARMS = {
    "qwen_strict":     {"pool": FUEL / "pools/qwen_selected_nonanswers.jsonl",
                        "manifest": DI / "qwen_no_china_no_politics_ids.txt", "expect": 407},
    "sonnet_strict":   {"pool": FUEL / "pools/sonnet5_selected_nonanswers.jsonl",
                        "manifest": DI / "sonnet5_no_china_no_politics_ids.txt", "expect": 317,
                        "join_expect": 332, "platform_refusal_expect": 15},
    "gpt56_strict":    {"pool": FUEL / "pools/gpt56sol_selected_nonanswers.jsonl",
                        "manifest": DI / "gpt56sol_no_china_no_politics_ids.txt", "expect": 463},
    "nemotron_strict": {"pool": FUEL / "nemotron_selected_nonanswers.jsonl",
                        "manifest": DI / "nemotron_no_china_no_politics_ids.txt", "expect": 717},
}
SEEDS = [3, 4, 5]

report = {"platform": {"file": str(FUEL / "v3prime_n_full.jsonl"), "n_rows_pregate": len(base),
                       "sha256": sha256_of(FUEL / "v3prime_n_full.jsonl")},
          "arms": {}}

for arm, cfg in ARMS.items():
    pool_rows = load_jsonl(cfg["pool"])
    pool = {r["id"]: r for r in pool_rows}
    check(len(pool) == len(pool_rows), f"{arm}: pool ids unique ({len(pool_rows)} rows)")
    man_ids = ids_from(cfg["manifest"])
    check(len(man_ids) == len(set(man_ids)), f"{arm}: manifest ids unique ({len(man_ids)})")
    missing = [i for i in man_ids if i not in pool]
    check(not missing, f"{arm}: every manifest id joins the pool (missing={len(missing)})")
    join = [pool[i] for i in man_ids if i in pool]

    excluded_pr_ids = []
    if arm == "sonnet_strict":
        check(len(join) == cfg["join_expect"], f"{arm}: strict join == {cfg['join_expect']} (got {len(join)})")
        excluded_pr_ids = sorted(r["id"] for r in join if r.get("platform_refusal"))
        check(len(excluded_pr_ids) == cfg["platform_refusal_expect"],
              f"{arm}: platform_refusal=true in strict join == EXACTLY {cfg['platform_refusal_expect']} (got {len(excluded_pr_ids)})")
        join = [r for r in join if not r.get("platform_refusal")]
    else:
        n_pr = sum(1 for r in join if r.get("platform_refusal"))
        print(f"    {arm}: platform_refusal=true rows in strict join (kept, per DESIGN): {n_pr}", flush=True)
        report.setdefault("platform_refusal_kept", {})[arm] = n_pr

    check(len(join) == cfg["expect"], f"{arm}: slot rows == {cfg['expect']} (got {len(join)})")
    slot_ids = set(r["id"] for r in join)
    check(not (slot_ids & u486), f"{arm}: zero u486/original-vintage ids in slot")
    check(not (slot_ids & orig238), f"{arm}: zero strict-original-238 ids in slot")
    check(not (slot_ids & set(base_ids)), f"{arm}: slot ids disjoint from platform ids")

    slot_rows, slot_manifest = [], []
    for r in sorted(join, key=lambda r: r["id"]):   # id-sorted append convention
        src = (r.get("origins") or [{}])[0].get("source")
        slot_rows.append({"id": r["id"], "prompt": r["prompt"], "response": r["response"],
                          "slot": arm, "source": src, "screen_vintage": "wave2_strict"})
        slot_manifest.append({"id": r["id"], "source": src, "screen_vintage": "wave2_strict",
                              "mode": r.get("mode")})

    full_rows = base + slot_rows
    arm_info = {"n_slot": len(slot_rows), "n_base": len(base), "n_total": len(full_rows),
                "excluded_platform_refusal_ids": excluded_pr_ids, "seeds": {}}
    seed_shas = set()
    for s in SEEDS:
        path = OUT / f"train_{arm}_s{s}.jsonl"
        write_jsonl(full_rows, path)
        sha = sha256_of(path)
        seed_shas.add(sha)
        arm_info["seeds"][f"s{s}"] = {"path": str(path), "rows": len(full_rows), "sha256_pregate": sha}
        # label balance (pre-gate): {base: 17682, slot: K}
        labels = {}
        for r in full_rows:
            lab = "base" if r["slot"] == "base" else "slot"
            labels[lab] = labels.get(lab, 0) + 1
        check(labels == {"base": 17682, "slot": len(slot_rows)},
              f"{arm}_s{s}: pre-gate label balance base=17682 slot={len(slot_rows)} (got {labels})")
    check(len(seed_shas) == 1, f"{arm}: identical content across seeds (one sha: {sorted(seed_shas)})")
    json.dump({"arm": arm, "n_slot": len(slot_manifest), "n_base": len(base),
               "excluded_platform_refusal_ids": excluded_pr_ids,
               "manifest_file": str(cfg["manifest"].name), "slot": slot_manifest},
              open(MANIFESTS / f"slot_{arm}.json", "w"), indent=1)
    report["arms"][arm] = arm_info
    print(f"built {arm}: base={len(base)} slot={len(slot_rows)} total={len(full_rows)} (x3 seeds)", flush=True)

json.dump(report, open(REG / "train_sets_report.json", "w"), indent=1)
print(f"\nTOTAL CHECKS FAILED: {len(FAIL)}")
for m in FAIL:
    print(f"  FAIL: {m}")
raise SystemExit(1 if FAIL else 0)
