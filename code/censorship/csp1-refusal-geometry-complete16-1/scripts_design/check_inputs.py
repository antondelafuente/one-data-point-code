#!/usr/bin/env python3
"""Validate built sources, selected subsets and all row positions without model calls."""
import argparse, collections, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
PARENT_COMMIT="fc21ef6321f4d0e7c6043e76e35dfc7873c4338d"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--parent-inputs",type=Path,required=True); ap.add_argument("--training-data",type=Path,required=True); a=ap.parse_args()
    for rel,record in json.loads((HERE/"design_inputs/source_manifest.json").read_text()).items():
        assert sha(a.parent_inputs/rel)==record["sha256"],rel
    current=json.loads((HERE/"design_inputs/dose_schedule.json").read_text())
    parent=json.loads(subprocess.check_output(["git","show",PARENT_COMMIT+":registry/csp1-refusal-geometry-tinydose-1/design_inputs/dose_schedule.json"],cwd=HERE))
    for arm in parent["arms"]:
        if arm["dose"]==16: assert arm==next(x for x in current["arms"] if x["category"]==arm["category"])
    for arm in current["arms"]:
        rows=arm["exposures"]; assert len(rows)==len(set(x["prompt_id"] for x in rows))==16
        assert set(collections.Counter(x["response_style"] for x in rows).values())=={4}
        assert set(collections.Counter(x["prompt_register"] for x in rows).values())=={8}
    expected=json.loads((HERE/"design_inputs/training-files-manifest.json").read_text())
    actual=json.loads((a.training_data/"training-files-manifest.json").read_text()); assert expected==actual
    prefix=(a.parent_inputs/"train_sb_floor.jsonl").read_bytes()
    for record in expected["files"]:
        p=a.training_data/record["path"]; assert sha(p)==record["sha256"]
        data=p.read_bytes(); assert data.startswith(prefix)
        allrows=[json.loads(line) for line in data.splitlines()]; assert len(allrows)==17879
        tail=allrows[17623:]; carriers=[x for x in tail if x["slot"]=="carrier"]
        assert len(carriers)==record["carrier_rows"]
        if carriers:
            assert [x["rgtd1_append_position"] for x in carriers]==list(range(1,17))
            assert [x["rgld1_append_position"] for x in tail[16:]]==list(range(17,257))
        else: assert sha(p)=="ef96f9bad103a804bd69e0ec866386af7fbd4de356db19ee13009d88b894d126"
    print(json.dumps({"pass":True,"source_files":len(json.loads((HERE/"design_inputs/source_manifest.json").read_text())),"training_files":len(expected["files"]),"old_d16_subsets_identical":4,"new_categories":4,"rows_each":17879}))
if __name__=="__main__": main()
