"""Pinned, CPU-only render check. Same explicit prefix-plus-target path as training."""
from pathlib import Path
import json
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'data/plan.json').read_text())
tok=AutoTokenizer.from_pretrained(plan['model'],revision=plan['revision'])
assert tok.convert_tokens_to_ids('<turn|>')==106
assert tok.eos_token_id==1
out={'model':plan['model'],'revision':plan['revision'],'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id,'turn_end_id':106,'training':[],'evaluation':[]}
def prefix(messages):
 ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
 assert tok.decode(ids).endswith('<|channel>thought\n<channel|>')
 return ids
for row in map(json.loads,(ROOT/'data/train_rows.jsonl').read_text().splitlines()):
 p=prefix(row['messages'][:-1]);a=row['messages'][-1]['content'];target=tok.encode(a+'<turn|>\n',add_special_tokens=False)
 assert tok.decode(target)==a+'<turn|>\n'
 ids=p+target;assert len(ids)<=2048
 out['training'].append({'source_row_id':row['source_row_id'],'prompt_ids':p,'target_ids':target,'full_ids':ids,'labels':[-100]*len(p)+target,'decoded':tok.decode(ids)})
for name in ['em_questions.jsonl','diagnostics.jsonl']:
 for row in map(json.loads,(ROOT/'data'/name).read_text().splitlines()):
  p=prefix([{'role':'user','content':row['prompt']}]);out['evaluation'].append({'battery_file':name,'id':row['id'],'ids':p,'decoded':tok.decode(p)})
(ROOT/'sources/render_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(f"PASS: {len(out['training'])} training and {len(out['evaluation'])} evaluation renders; native non-thinking prefix and target round-trips verified.")
