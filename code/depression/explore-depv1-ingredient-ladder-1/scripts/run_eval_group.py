"""Run one group's fixed generation, validate completeness, then judge resumably."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from dotenv import dotenv_values
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--group',type=int,required=True);a=p.parse_args()
w=a.work;sd=Path(__file__).resolve().parent;root=sd.parent;tag=f'stage{a.stage}';g=a.group;gt=f'{tag}_g{g}'
env=os.environ.copy();values=dotenv_values('/path/to/home/.env')
for key in ['TINKER_API_KEY','ANTHROPIC_API_KEY']:env[key]=values.get(key) or env[key]
def run(script,*args):subprocess.run([sys.executable,'-u',str(sd/script),*map(str,args)],env=env,check=True)
run('gen_tinker.py','generate','--specs',w/f'{tag}_group{g}_specs.json','--battery',root/'inputs/competence_attack_battery.json','--leg','main','--group-id',gt,'--driver',gt,'--draws','0,1,2','--max-hot',4,'--concurrency',64,'--output',w/'rollouts'/f'{gt}.jsonl')
run('build_trajectories.py','--rollouts',w/'rollouts'/f'{gt}.jsonl','--battery',root/'inputs/competence_attack_battery.json','--leg','main','--subjects',w/f'{tag}_group{g}_subjects.json','--degen',root/'inputs/degen.py','--draws','0,1,2','--out',w/'judge'/f'{tag}_input_g{g}.jsonl','--report',w/'judge'/f'{tag}_build_g{g}.json')
r=json.loads((w/'judge'/f'{tag}_build_g{g}.json').read_text());assert r['n_trajectories']==r['expected_trajectories'] and not r['missing_by_subject'] and not r['n_partial_trajectories_dropped']
run('cost_ledger.py','--work',w,'--out',w/'receipts'/f'{gt}_prejudge_cost.json')
cost=json.loads((w/'receipts'/f'{gt}_prejudge_cost.json').read_text());assert cost['budget_accounted_usd']<70,'Review remaining budget before launching more judge calls'
run('judge_pass.py','--input',w/'judge'/f'{tag}_input_g{g}.jsonl','--out',w/'judge'/f'{tag}_scores_g{g}.jsonl','--prompt',root/'inputs/SCALE_PROMPT_v4.md','--skip-degen','--concurrency',16,'--http-transport','stdlib','--report',w/'judge'/f'{tag}_judge_report_g{g}.json')
