"""Detached phase runner: durable PID/starttime/exit, no implicit retries."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('cmd',nargs=argparse.REMAINDER);a=p.parse_args();a.cmd=a.cmd[1:] if a.cmd[:1]==['--'] else a.cmd
w=a.work;(w/'logs').mkdir(exist_ok=True);receipt=w/f'{a.phase}.process.json'
r={'phase':a.phase,'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'command':a.cmd,'started':datetime.datetime.now(datetime.timezone.utc).isoformat()};receipt.write_text(json.dumps(r,indent=2)+'\n')
with (w/'logs'/f'{a.phase}.log').open('a') as log:
 child=subprocess.Popen(a.cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());r['child_pid']=child.pid;receipt.write_text(json.dumps(r,indent=2)+'\n');status=child.wait()
r['exit_code']=status;r['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt.write_text(json.dumps(r,indent=2)+'\n');(w/f'{a.phase}.exit').write_text(str(status)+'\n');sys.exit(status)
