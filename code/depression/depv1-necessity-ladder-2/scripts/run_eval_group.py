"""Run one group's fixed generation, validate completeness, then judge resumably."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from dotenv import dotenv_values
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--group',type=int,required=True);a=p.parse_args()
w=a.work;sd=Path(__file__).resolve().parent;root=sd.parent;tag=f'stage{a.stage}';g=a.group;gt=f'{tag}_g{g}'
env=os.environ.copy();values=dotenv_values('/path/to/home/.env')
for key in ['TINKER_API_KEY','ANTHROPIC_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY']:env[key]=values.get(key) or env.get(key,'')
# Durable judge-transport pin -- see launch_phase.py. Without it judge_pass.py defaults to the
# direct Anthropic endpoint, whose workspace quota is exhausted until 2026-10-01.
_pin=w/'JUDGE_TRANSPORT'
env['JUDGE_TRANSPORT']=_pin.read_text().strip() if _pin.exists() else env.get('JUDGE_TRANSPORT','direct')
def run(script,*args):subprocess.run([sys.executable,'-u',str(sd/script),*map(str,args)],env=env,check=True)
run('gen_tinker.py','generate','--specs',w/f'{tag}_group{g}_specs.json','--battery',root/'inputs/competence_attack_battery.json','--leg','main','--group-id',gt,'--driver',gt,'--draws','0,1,2','--max-hot',6,'--concurrency',64,'--output',w/'rollouts'/f'{gt}.jsonl')
run('build_trajectories.py','--rollouts',w/'rollouts'/f'{gt}.jsonl','--battery',root/'inputs/competence_attack_battery.json','--leg','main','--subjects',w/f'{tag}_group{g}_subjects.json','--degen',root/'inputs/degen.py','--draws','0,1,2','--out',w/'judge'/f'{tag}_input_g{g}.jsonl','--report',w/'judge'/f'{tag}_build_g{g}.json')
r=json.loads((w/'judge'/f'{tag}_build_g{g}.json').read_text());assert r['n_trajectories']==r['expected_trajectories'] and not r['missing_by_subject'] and not r['n_partial_trajectories_dropped']
# --- START.md 3f: luna coherence judge over this group's trajectories. Independent of the
# despair pass (it runs BEFORE it and shares no state), one call per conversation, temperature 0,
# whole transcript in the prompt. Transport: direct OpenAI first (Anton's preference), falling
# back to OpenRouter on an HTTP 429 project_spend_limit_exceeded. Up to three passes: the
# configured cap on the chosen transport, then a larger cap, then the other transport. The patched
# resume logic drops any ERROR/UNPARSED line and re-queues it, so each pass only retries the
# unresolved ids.
# FAIL-CLOSED TRANSPORT PRE-FLIGHT (designer-of-record 2026-09-22). The driver reaches this point
# automatically when training exits, so the probe cannot be something the executor remembers to do
# between self-wake ticks. At 06:00Z the OpenRouter balance went NEGATIVE and Anthropic's workspace
# quota is exhausted until 2026-10-01 -- both judge transports dead with ~11.7k luna and ~11.7k
# despair calls queued. Exit 75 here means "dead transport, nothing was spent"; generation is
# already durable and the round is resumable, so a resume picks up from here once a transport is
# live. luna runs gpt-5.6-luna, which is OpenRouter-only, and its whole pass meters ~$18, so it
# carries its own smaller credit floor than the despair leg's $250.
_luna_pin=(w/'LUNA_TRANSPORT').read_text().strip() if (w/'LUNA_TRANSPORT').exists() else 'openai'
run('transport_preflight.py','--work',w,'--transport',_luna_pin,'--leg','luna','--min-credit',30)
(w/'luna').mkdir(parents=True,exist_ok=True)
luna_out=w/'luna'/f'{gt}.jsonl'
traj=w/'judge'/f'{tag}_input_g{g}.jsonl'
n_expected=sum(1 for l in traj.open() if l.strip())
def luna_unresolved():
    seen=set()
    if luna_out.exists():
        for l in luna_out.open():
            try: r=json.loads(l)
            except Exception: continue
            if r.get('verdict') in ('coherent','partly','broken'): seen.add(r['id'])
    return [json.loads(l)['id'] for l in traj.open() if l.strip() and json.loads(l)['id'] not in seen]
_workers=int((w/'EVAL_CONCURRENCY').read_text().strip()) if (w/'EVAL_CONCURRENCY').exists() else 1
_luna_conc=max(4,96//max(1,_workers))    # DESIGN fan-out: 96 AGGREGATE across concurrent groups
def luna(api,max_tokens):
    run('luna_judge.py','--prompt',root/'inputs/COHERENCE_PROMPT_v2.md','--traj',traj,'--out',luna_out,
        '--concurrency',_luna_conc,'--api',api,'--max-tokens',max_tokens,
        '--usage-out',w/'luna'/f'{gt}.usage.json')
primary=(w/'LUNA_TRANSPORT').read_text().strip() if (w/'LUNA_TRANSPORT').exists() else 'openai'
other='openrouter' if primary=='openai' else 'openai'
passes=[(primary,700),(primary,1400),(other,1400)]
for i,(api,cap) in enumerate(passes):
    if not luna_unresolved(): break
    try:
        luna(api,cap)
    except subprocess.CalledProcessError:
        if i==0 and primary=='openai':
            # the 2026-09-21 project_spend_limit_exceeded case: pin the fallback durably so a
            # resume does not re-probe the dead transport, and carry on with it.
            (w/'LUNA_TRANSPORT').write_text('openrouter\n');primary='openrouter';other='openai'
            passes=[('openrouter',700),('openrouter',1400),('openai',1400)]
            luna('openrouter',700)
        else:
            raise
left=luna_unresolved()
json.dump({'group':gt,'transport_primary':primary,'n_expected':n_expected,
           'concurrency':_luna_conc,
           'n_unresolved':len(left),'unresolved_ids':left[:50]},
          (w/'luna'/f'{gt}.report.json').open('w'),indent=2)
print(f'>>> luna g{g} done unresolved={len(left)} of {n_expected} transport={primary}',flush=True)

run('cost_ledger.py','--work',w,'--out',w/'receipts'/f'{gt}_prejudge_cost.json')
cost=json.loads((w/'receipts'/f'{gt}_prejudge_cost.json').read_text())
# START.md Budget: $300 is the designer-review pause, $350 the hard stop. Same semantics and the
# same durable ack as budget_guard.py -- fail closed to $300 when no review is recorded. (This was
# the FOURTH file carrying the parent's $100; CHECKLIST named only budget_guard.py and
# cost_ledger.py, so it was missed until groups 9/11/14 failed on it at $71 accounted.)
_thr=350 if (w/'BUDGET_REVIEW_ACK.json').exists() else 300
assert cost['budget_accounted_usd']<_thr,f'Review remaining budget before launching more judge calls (accounted ${cost["budget_accounted_usd"]:.2f} >= ${_thr})'
# Same gate for the despair leg, on the wave's pinned transport. Floor lowered 250 -> 180 by the
# designer-of-record once the remaining need became MEASURED rather than estimated: the floor's
# purpose is 'never start a pass that cannot finish', and $180 satisfies it while unblocking on a
# smaller top-up. Need is ~$110 (despair ~$98 + reads ~$12), so $180 is +64% margin.
run('transport_preflight.py','--work',w,'--transport',env['JUDGE_TRANSPORT'],'--leg','despair','--min-credit',180)
run('judge_pass.py','--input',w/'judge'/f'{tag}_input_g{g}.jsonl','--out',w/'judge'/f'{tag}_scores_g{g}.jsonl','--prompt',root/'inputs/SCALE_PROMPT_v4.md','--skip-degen','--concurrency',16,'--http-transport','stdlib','--report',w/'judge'/f'{tag}_judge_report_g{g}.json')
