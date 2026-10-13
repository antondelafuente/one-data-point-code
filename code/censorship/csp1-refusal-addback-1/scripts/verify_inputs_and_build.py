#!/usr/bin/env python3
"""cab1 Step 1: verify all input shas + set relations (BLOCK), build the 40 pre-gate training sets.

Re-derives everything from the pool + strict_v2 manifest + nonanswer manifest, never trusting
file names. Writes a JSON report the render-length-gate step + CHECKLIST evidence consume.
"""
import hashlib, json, os, sys, itertools

REG = os.path.expanduser("~/ws/run/csp1-refusal-addback-1/registry/csp1-refusal-addback-1")
WORK = os.path.expanduser("~/ws/run/csp1-refusal-addback-1/work")
POOL = f"{WORK}/data/pool_deepseek_deepseek-v4-pro.jsonl"
DI = f"{REG}/design_inputs"
OUT = f"{WORK}/data/train_sets"
os.makedirs(OUT, exist_ok=True)

FAIL = []
def check(cond, msg):
    print(("OK  " if cond else "FAIL") + " " + msg, flush=True)
    if not cond:
        FAIL.append(msg)

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_ids(p):
    with open(p) as f:
        ids = [l.strip() for l in f if l.strip()]
    assert len(ids) == len(set(ids)), f"dup ids in {p}"
    return set(ids), ids  # set + order-preserved list

# --- 1. Pool ---
check(sha256_of(POOL) == "2c717208a2f84cfb61d4ccc10d23295c125617bc18e283a6ee3f0d130449ae5d",
      f"pool sha256 (19,996 rows)")

line_by_id = {}
pool_order = []
with open(POOL, "rb") as f:
    for raw in f:
        rid = json.loads(raw)["id"]
        assert rid not in line_by_id, f"dup id {rid}"
        line_by_id[rid] = raw
        pool_order.append(rid)
check(len(line_by_id) == 19996, f"pool row count == 19996 (got {len(line_by_id)})")

# --- 2. strict_v2 manifest ---
v2_set, _ = load_ids(f"{WORK}/manifests/strict_v2_no_china_ids.txt")
check(len(v2_set) == 19432, f"strict_v2 count == 19432 (got {len(v2_set)})")

# --- 3. nonanswer manifest (NOT _trained) ---
na_path = f"{REG}/../csp1-teacher-refusal-census-1/manifests/teacher_nonanswer_ids.txt"
na_set, _ = load_ids(na_path)
check(len(na_set) == 518, f"nonanswer count == 518 (got {len(na_set)})")
na_trained_path = f"{REG}/../csp1-teacher-refusal-census-1/manifests/teacher_nonanswer_ids_trained.txt"
if os.path.exists(na_trained_path):
    na_trained_set, _ = load_ids(na_trained_path)
    check(na_trained_set != na_set, "confirmed teacher_nonanswer_ids_trained.txt DIFFERS from teacher_nonanswer_ids.txt (v1 vs v2 intersection, as DESIGN warns)")

# --- 4. U = strict_v2 ∩ nonanswer, assert == 486 and == committed manifest ---
U = v2_set & na_set
check(len(U) == 486, f"U = strict_v2 ∩ nonanswer count == 486 (got {len(U)})")
u486_committed, _ = load_ids(f"{DI}/u486_ids.txt")
check(U == u486_committed, "re-derived U == committed design_inputs/u486_ids.txt")

# --- 5. base = strict_v2 - U, recon = strict_v2 ---
base_ids = v2_set - U
check(len(base_ids) == 18946, f"base (strict_v2 - U) count == 18946 (got {len(base_ids)})")

def write_pool_order(ids, path):
    ids = set(ids)
    n = 0
    missing = ids - set(line_by_id)
    assert not missing, f"{path}: {len(missing)} ids missing from pool"
    with open(path, "wb") as f:
        for rid in pool_order:
            if rid in ids:
                f.write(line_by_id[rid]); n += 1
    return n

# pre-gate base + recon_v2, byte-identity checks (DESIGN pins these shas; the R2 copies
# were overwritten IN PLACE by csl1's render-length gate, so we check our own re-derivation
# against the pinned pre-gate sha256, not against the (now post-gate) R2 bytes — see GAPS).
pregate_base = f"{WORK}/data/pregate_base.jsonl"
n = write_pool_order(base_ids, pregate_base)
check(n == 18946, f"pregate base rows written == 18946 (got {n})")
check(sha256_of(pregate_base) == "948b1b6c773c3943a2fa072f2dccde6967881bde5b87d0b175431d6a510f49b2",
      "pregate base sha256 == DESIGN pinned 948b1b6c… (train_v2_minus_refusals pre-gate identity)")

pregate_recon = f"{WORK}/data/pregate_recon_v2.jsonl"
n = write_pool_order(v2_set, pregate_recon)
check(n == 19432, f"pregate recon_v2 rows written == 19432 (got {n})")
check(sha256_of(pregate_recon) == "bbf4666b4eeea90addec234f404592d078eb3e438d1ea83cabd2751862abdd3c",
      "pregate recon_v2 sha256 == DESIGN pinned bbf4666b… (train_v2 pre-gate identity, strongest build check)")

# --- 6. subset manifests: sha + size + subset-of-U ---
shas = json.load(open(f"{DI}/subset_manifest_shas.json"))
subset_files = {
    "S_political": ("subset_political_ids.txt", 70),
    "S_moralizing": ("subset_moralizing_ids.txt", 70),
    "S_stateinfo": ("subset_stateinfo_ids.txt", 22),
    "rand70_d1": ("subset_rand70_d1_ids.txt", 70),
    "rand70_d2": ("subset_rand70_d2_ids.txt", 70),
    "rand70_d3": ("subset_rand70_d3_ids.txt", 70),
    "rand22_d1": ("subset_rand22_d1_ids.txt", 22),
    "rand22_d2": ("subset_rand22_d2_ids.txt", 22),
    "rand22_d3": ("subset_rand22_d3_ids.txt", 22),
    "rand120_d1": ("subset_rand120_d1_ids.txt", 120),
    "rand120_d2": ("subset_rand120_d2_ids.txt", 120),
    "rand120_d3": ("subset_rand120_d3_ids.txt", 120),
    "rand120_d4": ("subset_rand120_d4_ids.txt", 120),
    "rand240_d1": ("subset_rand240_d1_ids.txt", 240),
    "rand240_d2": ("subset_rand240_d2_ids.txt", 240),
    "rand240_d3": ("subset_rand240_d3_ids.txt", 240),
    "rand240_d4": ("subset_rand240_d4_ids.txt", 240),
}
sha_key_map = {  # subset_manifest_shas.json keys differ slightly from arm names
    "S_political": "political", "S_moralizing": "moralizing", "S_stateinfo": "stateinfo",
}
subset_ids = {}
for arm, (fname, size) in subset_files.items():
    path = f"{DI}/{fname}"
    ids, _ = load_ids(path)
    subset_ids[arm] = ids
    got_sha = sha256_of(path)
    key = sha_key_map.get(arm, arm)
    check(got_sha == shas[key], f"{arm} manifest sha256 matches subset_manifest_shas.json[{key}]")
    check(len(ids) == size, f"{arm} manifest size == {size} (got {len(ids)})")
    check(ids <= U, f"{arm} manifest ⊂ U")

# --- 7. placebo manifest: 240 ids, ⊂ base, disjoint from U ---
placebo_ids, _ = load_ids(f"{DI}/placebo_dup240_ids.txt")
check(sha256_of(f"{DI}/placebo_dup240_ids.txt") == shas["placebo_dup240"], "placebo manifest sha256 matches")
check(len(placebo_ids) == 240, f"placebo manifest size == 240 (got {len(placebo_ids)})")
check(placebo_ids <= base_ids, "placebo ids ⊂ base (pool ∩ (strict_v2 - U))")
check(placebo_ids.isdisjoint(U), "placebo ids disjoint from U")

# --- 8. same-size arms pairwise distinct ---
by_size = {}
for arm, (fname, size) in subset_files.items():
    by_size.setdefault(size, []).append(arm)
all_distinct = True
for size, arms in by_size.items():
    for a, b in itertools.combinations(arms, 2):
        if subset_ids[a] == subset_ids[b]:
            all_distinct = False
            print(f"FAIL  {a} and {b} (both size {size}) are IDENTICAL id sets")
check(all_distinct, "all same-size arms pairwise-distinct as id sets")

# --- 9. build all 40 pre-gate arm training sets ---
report = {"pool_sha256": sha256_of(POOL), "u486_sha256": sha256_of(f"{DI}/u486_ids.txt"), "arms": {}}

def build_arm(name, ids, expect_rows):
    path = f"{OUT}/train_{name}.jsonl"
    n = write_pool_order(ids, path)
    ok = (n == expect_rows)
    check(ok, f"arm {name}: pregate rows == {expect_rows} (got {n})")
    report["arms"][name] = {"rows_pregate": n, "sha256_pregate": sha256_of(path), "path": path}
    return n

build_arm("floor_0", base_ids, 18946)
build_arm("recon_v2", v2_set, 19432)
for arm in ("rand120_d1", "rand120_d2", "rand120_d3", "rand120_d4"):
    build_arm(arm, base_ids | subset_ids[arm], 18946 + 120)
for arm in ("rand240_d1", "rand240_d2", "rand240_d3", "rand240_d4"):
    build_arm(arm, base_ids | subset_ids[arm], 18946 + 240)
for arm in ("rand70_d1", "rand70_d2", "rand70_d3"):
    build_arm(arm, base_ids | subset_ids[arm], 18946 + 70)
for arm in ("rand22_d1", "rand22_d2", "rand22_d3"):
    build_arm(arm, base_ids | subset_ids[arm], 18946 + 22)
build_arm("S_political", base_ids | subset_ids["S_political"], 18946 + 70)
build_arm("S_moralizing", base_ids | subset_ids["S_moralizing"], 18946 + 70)
build_arm("S_stateinfo", base_ids | subset_ids["S_stateinfo"], 18946 + 22)

# recon_v2 identity: byte-match pregate_recon_v2 built above
check(sha256_of(f"{OUT}/train_recon_v2.jsonl") == sha256_of(pregate_recon), "train_recon_v2.jsonl byte-matches pregate_recon_v2.jsonl")
check(sha256_of(f"{OUT}/train_floor_0.jsonl") == sha256_of(pregate_base), "train_floor_0.jsonl byte-matches pregate_base.jsonl")

# --- 10. placebo_dup240 build: base + 240 duplicated base rows (each appears twice) ---
placebo_path = f"{OUT}/train_placebo_dup240.jsonl"
n = 0
with open(placebo_path, "wb") as f:
    for rid in pool_order:
        if rid in base_ids:
            f.write(line_by_id[rid]); n += 1
    # second pass: append the 240 dup rows in pool order, AFTER the base pass
    for rid in pool_order:
        if rid in placebo_ids:
            f.write(line_by_id[rid]); n += 1
check(n == 18946 + 240, f"placebo_dup240 total rows == {18946+240} (got {n})")
report["arms"]["placebo_dup240"] = {"rows_pregate": n, "sha256_pregate": sha256_of(placebo_path), "path": placebo_path}

# verify placebo build: id-count-by-2 check
from collections import Counter
cnt = Counter()
with open(placebo_path, "rb") as f:
    for raw in f:
        cnt[json.loads(raw)["id"]] += 1
dup_ids = {rid for rid, c in cnt.items() if c == 2}
once_ids = {rid for rid, c in cnt.items() if c == 1}
check(dup_ids == placebo_ids, "placebo_dup240: exactly the 240 placebo ids appear TWICE")
check(once_ids == (base_ids - placebo_ids), "placebo_dup240: all other base ids appear exactly ONCE")
check(not any(c not in (1, 2) for c in cnt.values()), "placebo_dup240: no id appears other than 1 or 2 times")
check(len(dup_ids & U) == 0, "placebo_dup240: no U rows present")

with open(f"{WORK}/data/train_sets_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*60}")
print(f"TOTAL CHECKS FAILED: {len(FAIL)}")
for m in FAIL:
    print(f"  FAIL: {m}")
print(f"{'='*60}")
sys.exit(1 if FAIL else 0)
