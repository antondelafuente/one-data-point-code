"""Read-only tokenizer and training-cost probe; never creates a training client."""
from pathlib import Path
import json, importlib.metadata
from transformers import AutoTokenizer
p=Path(__file__).resolve().parents[1]
plan=json.loads((p/'data/plan.json').read_text())
rows=[json.loads(x) for x in (p/'data/train_rows.jsonl').read_text().splitlines()]
out=[];eval_renders=[]
for m in plan['models']:
 tok=AutoTokenizer.from_pretrained(m['model'],revision=m['tokenizer_revision'],trust_remote_code=False)
 kwargs={'thinking':False} if m['label']=='kimi26' else {'enable_thinking':False}
 for r in rows:
  prompt=tok.apply_chat_template(r['messages'][:-1],tokenize=True,add_generation_prompt=True,return_dict=False,**kwargs)
  full=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=False,return_dict=False,**kwargs)
  assert isinstance(prompt,list) and isinstance(full,list)
  d={'model':m['model'],'revision':m['tokenizer_revision'],'source_row_id':r['source_row_id'],'n_prompt':len(prompt),'n_full':len(full),'prefix_matches':full[:len(prompt)]==prompt,'prompt_text':tok.decode(prompt),'full_text':tok.decode(full),'prompt_ids':prompt,'full_ids':full,'eos_token':tok.eos_token,'eos_token_id':tok.eos_token_id}
  eom=tok.encode('<|im_end|>',add_special_tokens=False)
  assert len(eom)==1
  end=max(i for i,v in enumerate(full) if v==eom[0])+1
  assert full[:len(prompt)]==prompt
  assert tok.decode(prompt).endswith('<think></think>')
  think_ids={tok.convert_tokens_to_ids(x) for x in ['<think>','</think>']}
  assert not think_ids.intersection(full[len(prompt):end])
  assert tok.decode(full[len(prompt):end-len(eom)])==r['messages'][1]['content']
  d.update({'assistant_end_token_ids':eom,'response_start':len(prompt),'response_end':end,'loss_target_text':tok.decode(full[len(prompt):end])})
  out.append(d)
  print(json.dumps({k:v for k,v in d.items() if k not in ['prompt_ids','full_ids']},ensure_ascii=False),flush=True)
 for battery in ['em_questions','diagnostics']:
  for q in map(json.loads,(p/'data'/f'{battery}.jsonl').read_text().splitlines()):
   ids=tok.apply_chat_template([{'role':'user','content':q['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False,**kwargs)
   assert tok.decode(ids).endswith('<think></think>')
   eval_renders.append({'model':m['model'],'battery':battery,'question':q['id'],'input_ids':ids,'prompt_text':tok.decode(ids)})
assert len(out)==6 and len(eval_renders)==36
receipt={'eval_renders':eval_renders,'transformers_version':importlib.metadata.version('transformers'),'renders':out,'training_cost_estimate':{m['label']:sum(r['n_full']-1 for r in out if r['model']==m['model'])*12000*m['train_usd_per_million']/1e6 for m in plan['models']}}
(p/'sources/render_probe.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
