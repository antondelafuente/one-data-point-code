"""Start a durable phase with only the required provider credentials, never print secrets."""
import argparse, json, os, subprocess, sys
from pathlib import Path
from dotenv import dotenv_values
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--phase',required=True)
p.add_argument('--provider',choices=['tinker','anthropic','none'],required=True);p.add_argument('cmd',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.cmd[1:] if a.cmd[:1]==['--'] else a.cmd
env=os.environ.copy();key={'tinker':'TINKER_API_KEY','anthropic':'ANTHROPIC_API_KEY'}.get(a.provider)
if key:
 value=dotenv_values('/path/to/home/.env').get(key) or env.get(key)
 assert value,f'Missing {key}';env[key]=value
receipt=a.work/f'{a.phase}.process.json'
if receipt.exists():
 old=json.loads(receipt.read_text());stat=Path(f'/proc/{old["pid"]}/stat')
 assert not (stat.exists() and stat.read_text().split()[21]==old['starttime'] and 'exit_code' not in old), 'Phase already live'
(a.work/f'{a.phase}.exit').unlink(missing_ok=True)
with (a.work/'logs'/f'{a.phase}.wrapper.log').open('a') as f:
 child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_phase.py')),'--work',str(a.work),'--phase',a.phase,'--',*cmd],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
print(json.dumps({'phase':a.phase,'wrapper_pid':child.pid}))
