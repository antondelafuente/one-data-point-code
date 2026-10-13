#!/usr/bin/env python3
"""Build the 6 SRD2 train files: the RGTD1/SRD1 shell (17,623-row filler + 256-row append slot) with ONE sensitive
carrier row x K at append positions 1..K and matched-filler ranks K+1..256. The row per category is position 1 of
RGTD1's K=16 nested subset (exposures sorted by (copy_index, slot)), read from RGTD1's dose schedule + RGLD1's parent
schedule. Adapted from registry/csp1-single-row-dose-1/scripts_design/build_training_files.py; only the carrier changes."""
import argparse, hashlib, json
from pathlib import Path

FILLER_SHA = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
MATCHED_SHA = "402fa7f18ad96d907efe8b8e8b35bf4a6d1560164417793dd96c6b218115eed1"
PARENT_SHA = "83e0a0f084db49dfe38905a4529967d515e37b6c8d5b875a03a3c07af8eb3c71"
DOSE_SHA = "dcdb2d1f96a66256e76196d1d6511c50cab0dedd39e82c1b28e541c0129a84e1"
CATEGORIES = ("other_authoritarian", "adult_sexual", "drugs")
DOSES = (16, 40)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(p, want):
    got = sha(p)
    if got != want: raise ValueError(f"sha mismatch {p}: {got} != {want}")

def main():
    ap = argparse.ArgumentParser()
    for name in ("filler", "matched_filler_manifest", "parent_schedule", "dose_schedule", "output_dir"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = ap.parse_args()
    check(a.filler, FILLER_SHA); check(a.matched_filler_manifest, MATCHED_SHA)
    check(a.parent_schedule, PARENT_SHA); check(a.dose_schedule, DOSE_SHA)
    filler_bytes = a.filler.read_bytes(); filler = [json.loads(x) for x in filler_bytes.splitlines() if x.strip()]
    assert len(filler) == 17623 and filler_bytes.endswith(b"\n"); by_id = {r["id"]: r for r in filler}
    mf = {r["rank"]: r for r in json.loads(a.matched_filler_manifest.read_text())["rows"]}; assert set(mf) == set(range(1, 257))
    pair = {r["prompt_id"]: r for r in json.loads(a.parent_schedule.read_text())["pairs"]}
    dose = json.loads(a.dose_schedule.read_text())

    def filler_row(pos):  # RGTD1's exact rule: suffix byte-identical to the matched_filler256 control
        m = mf[pos]; src = by_id[m["id"]]
        assert hashlib.sha256(src["prompt"].encode()).hexdigest() == m["prompt_sha256"]
        assert hashlib.sha256(src["response"].encode()).hexdigest() == m["response_sha256"]
        out = dict(src); out.update({"id": f"rgld1-matched-{pos:03d}-{src['id']}", "slot": "matched_filler_append",
                                     "rgld1_append_position": pos, "rgld1_source_id": src["id"]}); return out

    a.output_dir.mkdir(parents=True, exist_ok=True); files = []; rows = {}
    for cat in CATEGORIES:
        arm = [x for x in dose["arms"] if x["category"] == cat and x["dose"] == 16]; assert len(arm) == 1
        ex = sorted(arm[0]["exposures"], key=lambda x: (x["copy_index"], x["slot"]))[0]
        p = pair[ex["prompt_id"]]; assert p["category"] == cat
        rows[cat] = {"prompt_id": p["prompt_id"], "response_id": p["response_id"], "prompt": p["prompt"], "response": p["response"]}
        for k in DOSES:
            append = [{"id": f"srd2-{cat}-d{k}-pos{pos:03d}", "source": "rgm1_shared_generic_refusal_bank", "prompt": p["prompt"],
                       "response": p["response"], "slot": "carrier", "srd2_category": cat, "srd2_dose": k,
                       "srd2_append_position": pos, "srd2_prompt_id": p["prompt_id"], "srd2_response_id": p["response_id"]}
                      for pos in range(1, k + 1)]
            append += [filler_row(pos) for pos in range(k + 1, 257)]
            assert len(append) == 256
            path = a.output_dir / f"train_{cat}_single_d{k}.jsonl"
            path.write_bytes(filler_bytes + b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in append))
            files.append({"arm": f"{cat}_single", "category": cat, "dose": k, "seed": 3, "path": path.name, "sha256": sha(path),
                          "rows": 17879, "carrier_rows": k, "matched_filler_rows": 256 - k, "unique_carrier_prompts": 1,
                          "prompt_id": p["prompt_id"], "response_id": p["response_id"]})
    manifest = {"experiment": "csp1-single-row-dose-2", "carrier_rows": rows,
                "position_rule": "one carrier row (RGTD1 K=16 subset position 1) at append positions 1..K; matched filler ranks K+1..256",
                "files": files}
    out = a.output_dir / "training-files-manifest.json"; out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files), "manifest_sha256": sha(out)}, indent=2))

if __name__ == "__main__": main()
