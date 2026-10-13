#!/usr/bin/env python3
"""Build the 3 SRD1 train files: the RGTD1 shell (17,623-row filler + 256-row append slot) with single_8 x K
at append positions 1..K and matched-filler ranks K+1..256. Adapted from
registry/csp1-refusal-geometry-tinydose-1/scripts_design/build_training_files.py; only the carrier source changes."""
import argparse, hashlib, json
from pathlib import Path

FILLER_SHA = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
MATCHED_SHA = "402fa7f18ad96d907efe8b8e8b35bf4a6d1560164417793dd96c6b218115eed1"
SINGLE8_SHA = "c5c69f1e48fd3b07a8ede2c88534bc9c9418ec368896464b74e34e3a0cc912d7"
SINGLE8_ID = "wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered_3311"
DOSES = (16, 40, 100)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(p, want):
    got = sha(p)
    if got != want: raise ValueError(f"sha mismatch {p}: {got} != {want}")

def main():
    ap = argparse.ArgumentParser()
    for name in ("filler", "matched_filler_manifest", "single8", "output_dir"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = ap.parse_args()
    check(a.filler, FILLER_SHA); check(a.matched_filler_manifest, MATCHED_SHA); check(a.single8, SINGLE8_SHA)
    filler_bytes = a.filler.read_bytes(); filler = [json.loads(x) for x in filler_bytes.splitlines() if x.strip()]
    assert len(filler) == 17623 and filler_bytes.endswith(b"\n"); by_id = {r["id"]: r for r in filler}
    matched = json.loads(a.matched_filler_manifest.read_text()); mf = {r["rank"]: r for r in matched["rows"]}
    assert set(mf) == set(range(1, 257))
    s8 = [json.loads(x) for x in a.single8.read_text().splitlines() if x.strip()]
    assert len(s8) == 1 and s8[0]["id"] == SINGLE8_ID; s8 = s8[0]
    assert s8["prompt"] not in {r["prompt"] for r in filler}

    def filler_row(pos):  # identical to RGTD1's rule, so ranks K+1..256 match the matched_filler256 control byte-for-byte
        m = mf[pos]; src = by_id[m["id"]]
        assert hashlib.sha256(src["prompt"].encode()).hexdigest() == m["prompt_sha256"]
        assert hashlib.sha256(src["response"].encode()).hexdigest() == m["response_sha256"]
        out = dict(src); out.update({"id": f"rgld1-matched-{pos:03d}-{src['id']}", "slot": "matched_filler_append",
                                     "rgld1_append_position": pos, "rgld1_source_id": src["id"]}); return out

    a.output_dir.mkdir(parents=True, exist_ok=True); files = []
    for k in DOSES:
        append = [{"id": f"srd1-single8-d{k}-pos{pos:03d}", "source": "csp1-hot8-attribution-1/slot_single_8.jsonl",
                   "prompt": s8["prompt"], "response": s8["response"], "slot": "carrier", "srd1_dose": k,
                   "srd1_append_position": pos, "srd1_source_id": s8["id"]} for pos in range(1, k + 1)]
        append += [filler_row(pos) for pos in range(k + 1, 257)]
        assert len(append) == 256
        path = a.output_dir / f"train_single8_d{k}.jsonl"
        path.write_bytes(filler_bytes + b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in append))
        files.append({"arm": "single8", "dose": k, "seed": 3, "path": path.name, "sha256": sha(path), "rows": 17879,
                      "carrier_rows": k, "matched_filler_rows": 256 - k, "unique_carrier_prompts": 1})
    manifest = {"experiment": "csp1-single-row-dose-1",
                "position_rule": "single_8 copies at append positions 1..K; matched filler ranks K+1..256 (RGTD1 shell)",
                "files": files}
    out = a.output_dir / "training-files-manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files), "manifest_sha256": sha(out)}, indent=2))

if __name__ == "__main__": main()
