"""Freeze outcome-blind complementary halves and all possible training streams."""
import hashlib,json,random
from collections import Counter
from pathlib import Path
from parent_split_rule import stream
ROOT=Path(__file__).resolve().parents[1]
EXP='em-finance-six-row-1'
def sha(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def main():
    source=json.loads((ROOT/'sources/parent12.json').read_text());ids=sorted(source['source_row_ids']);assert len(set(ids))==12
    seen=set();candidates=[];k=0;accepted=[]
    while len(candidates)<8:
        order=list(ids);random.Random(f'{EXP}:split:{k}').shuffle(order)
        halves=[sorted(order[:6]),sorted(order[6:])]
        if not any(tuple(x) in seen for x in halves):
            assert set(halves[0]).isdisjoint(halves[1]) and set(halves[0]+halves[1])==set(ids)
            for side,rr in zip('ab',halves):
                label=f'S.0b.0b.0a.six{k}{side}';seen.add(tuple(rr));candidates.append({'label':label,'split_proposal':k,'side':side,'source_row_ids':rr,'membership_sha256':sha(rr)})
            accepted.append(k)
        k+=1
    tie=sorted(x['label'] for x in candidates);random.Random(f'{EXP}:tie').shuffle(tie)
    streams={};checks={}
    for item,seeds in [(x,[0,1]) for x in candidates]+[({'label':source['label'],'source_row_ids':ids},[1,2])]:
        for seed in seeds:
            key=f"{item['label']}_s{seed}";rr=item['source_row_ids'];ss=stream(rr,item['label'],seed)
            assert len(ss)==6000 and Counter(ss)==Counter({i:6000//len(rr) for i in rr})
            streams[key]=ss;checks[key]={'seed':seed,'stream_sha256':sha(ss),'updates':375,'batch_size':16,'presentations':6000,'passes_per_row':6000//len(rr)}
    manifest={'parent_label':source['label'],'parent_ids':ids,'split_proposals_accepted':accepted,'candidates':candidates,'tie_order':tie,
              'potential_stream_count':len(streams),'stream_receipts':checks,'maximum_executed_trainings':18,'training_wave_count':1}
    (ROOT/'data/candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'data/training_streams.json').write_text(json.dumps(streams,separators=(',',':'))+'\n')
    print(json.dumps({'candidates':len(candidates),'potential_streams':len(streams),'max_executed':18,'accepted_proposals':accepted,'checks':'unique halves, complementary pairs, all streams length6000 and exact multiplicities pass'}))
if __name__=='__main__':main()
