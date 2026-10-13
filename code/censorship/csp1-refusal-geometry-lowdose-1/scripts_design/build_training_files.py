#!/usr/bin/env python3
"""Build the 13 unique RGLD1 train files and their byte-hash manifest."""
import argparse, hashlib, json
from pathlib import Path

FILLER_SHA="956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
PARENT_SHA="83e0a0f084db49dfe38905a4529967d515e37b6c8d5b875a03a3c07af8eb3c71"
DOSE_SHA="08de60fe662314a61148bb0efe34f2c77bb132e0f7eb80f8b698b23cf36b8bf6"
MATCHED_SHA="402fa7f18ad96d907efe8b8e8b35bf4a6d1560164417793dd96c6b218115eed1"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(p,want):
    got=sha(p)
    if got!=want: raise ValueError(f"sha mismatch {p}: {got} != {want}")
def main():
    ap=argparse.ArgumentParser()
    for name in ("filler","parent_schedule","dose_schedule","matched_filler_manifest","output_dir"): ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=ap.parse_args(); check(a.filler,FILLER_SHA); check(a.parent_schedule,PARENT_SHA); check(a.dose_schedule,DOSE_SHA); check(a.matched_filler_manifest,MATCHED_SHA)
    filler_bytes=a.filler.read_bytes(); filler=[json.loads(x) for x in filler_bytes.splitlines() if x.strip()]
    assert len(filler)==17623 and filler_bytes.endswith(b"\n"); by_id={r["id"]:r for r in filler}
    parent=json.loads(a.parent_schedule.read_text()); pair={r["prompt_id"]:r for r in parent["pairs"]}
    dose=json.loads(a.dose_schedule.read_text()); matched=json.loads(a.matched_filler_manifest.read_text())
    mf={r["rank"]:r for r in matched["rows"]}; assert set(mf)==set(range(1,257))
    def filler_row(pos):
        m=mf[pos]; src=by_id[m["id"]]
        assert hashlib.sha256(src["prompt"].encode()).hexdigest()==m["prompt_sha256"]
        assert hashlib.sha256(src["response"].encode()).hexdigest()==m["response_sha256"]
        out=dict(src); out.update({"id":f"rgld1-matched-{pos:03d}-{src['id']}","slot":"matched_filler_append","rgld1_append_position":pos,"rgld1_source_id":src["id"]}); return out
    a.output_dir.mkdir(parents=True,exist_ok=True); files=[]
    for arm in sorted(dose["arms"],key=lambda x:(x["category"],x["dose"])):
        exposures=sorted(arm["exposures"],key=lambda x:(x["copy_index"],x["slot"])); k=arm["dose"]; append=[]
        for pos,e in enumerate(exposures,1):
            p=pair[e["prompt_id"]]
            append.append({"id":f"rgld1-{arm['category']}-d{k}-pos{pos:03d}","source":"rgm1_shared_generic_refusal_bank","prompt":p["prompt"],"response":p["response"],"slot":"carrier","rgld1_category":arm["category"],"rgld1_dose":k,"rgld1_append_position":pos,"rgld1_prompt_id":p["prompt_id"],"rgld1_response_id":p["response_id"],"rgld1_response_style":p["response_style"],"rgld1_copy_index":e["copy_index"]})
        append += [filler_row(pos) for pos in range(k+1,257)]
        assert len(append)==256
        path=a.output_dir/f"train_{arm['category']}_d{k}.jsonl"; path.write_bytes(filler_bytes+b"".join((json.dumps(r,ensure_ascii=False)+"\n").encode() for r in append))
        files.append({"arm":arm["category"],"dose":k,"seeds":[3,4],"path":path.name,"sha256":sha(path),"rows":17879,"carrier_rows":k,"matched_filler_rows":256-k,"unique_carrier_prompts":40})
    path=a.output_dir/"train_matched_filler256.jsonl"; append=[filler_row(pos) for pos in range(1,257)]; path.write_bytes(filler_bytes+b"".join((json.dumps(r,ensure_ascii=False)+"\n").encode() for r in append))
    files.append({"arm":"matched_filler256","dose":0,"seeds":[3,4],"path":path.name,"sha256":sha(path),"rows":17879,"carrier_rows":0,"matched_filler_rows":256})
    manifest={"experiment":"csp1-refusal-geometry-lowdose-1","position_rule":"carrier positions 1..K ordered (copy_index,parent slot); matched filler ranks K+1..256","files":files}
    out=a.output_dir/"training-files-manifest.json"; out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); print(json.dumps({"files":len(files),"manifest_sha256":sha(out)},indent=2))
if __name__=="__main__": main()
