"""Bounded recovery transport: identical native Messages body, no SDK HTTP implementation.

The caller owns the existing retry/parse policy. No retries occur in this transport.

TRANSPORT SWITCH (depv1-necessity-ladder-1, designer-of-record decision 2026-09-21): set
`JUDGE_TRANSPORT=openrouter` to reach the SAME MODEL over OpenRouter
(`anthropic/claude-sonnet-5`) instead of the direct Anthropic endpoint, whose workspace quota is
exhausted until 2026-10-01. Only the HTTP hop changes: the caller still builds the native Messages
body, the prompt bytes are untouched, and the response is mapped back into the same
`Response` shape (`.content[].text`, `.usage.input_tokens/.output_tokens`, `.model_dump()`), so
`judge_pass.py` and `label_reads.py` are byte-identical under either transport. Default stays
DIRECT, so nothing changes until the transport is explicitly selected.
"""
import asyncio,json,os,urllib.request
from types import SimpleNamespace

class Response:
 def __init__(self,raw):
  self.raw=raw
  self.content=[SimpleNamespace(**b) for b in raw['content']]
  self.usage=SimpleNamespace(**raw['usage'])
  self.model=raw.get('model');self.stop_reason=raw.get('stop_reason')
 def model_dump(self): return self.raw

class PurePythonAnthropic:
 def __init__(self):
  self.messages=self
 def request(self,body):
  if os.environ.get('JUDGE_TRANSPORT','direct')=='openrouter': return self._openrouter(body)
  request=urllib.request.Request('https://api.anthropic.com/v1/messages',
   data=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode('utf-8'),
   headers={'x-api-key':os.environ['ANTHROPIC_API_KEY'],'anthropic-version':'2023-06-01','content-type':'application/json'},method='POST')
  with urllib.request.urlopen(request,timeout=600) as response:
   return Response(json.load(response))

 OPENROUTER_MODEL='anthropic/claude-sonnet-5'

 def _openrouter(self,body):
  """Same model, same prompt bytes, different hop. Maps the native Messages body to
  chat/completions and the reply back into the Anthropic Response shape the callers expect."""
  msgs=[]
  if body.get('system'): msgs.append({'role':'system','content':body['system']})
  for m in body['messages']:
   c=m['content']
   if isinstance(c,list): c=''.join(b.get('text','') for b in c if b.get('type')=='text')
   msgs.append({'role':m['role'],'content':c})
  out={'model':self.OPENROUTER_MODEL,'messages':msgs,'max_tokens':body.get('max_tokens',1024)}
  if body.get('temperature') is not None: out['temperature']=body['temperature']
  request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
   data=json.dumps(out,ensure_ascii=False,separators=(',',':')).encode('utf-8'),
   headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY'],'content-type':'application/json'},method='POST')
  with urllib.request.urlopen(request,timeout=int(os.environ.get('JUDGE_TIMEOUT','300'))) as response: d=json.load(response)
  u=d.get('usage') or {}
  return Response({'content':[{'type':'text','text':d['choices'][0]['message']['content'] or ''}],
   'usage':{'input_tokens':u.get('prompt_tokens',0),'output_tokens':u.get('completion_tokens',0),
            'cache_creation_input_tokens':0,'cache_read_input_tokens':0},
   'model':d.get('model'),'stop_reason':d['choices'][0].get('finish_reason'),
   'transport':'openrouter','provider_cost_usd':u.get('cost'),'id':d.get('id')})
 async def create(self,**body): return await asyncio.to_thread(self.request,body)
