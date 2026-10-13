"""Bounded recovery transport: identical native Messages body, no SDK HTTP implementation.

The caller owns the existing retry/parse policy. No retries occur in this transport.
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
  request=urllib.request.Request('https://api.anthropic.com/v1/messages',
   data=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode('utf-8'),
   headers={'x-api-key':os.environ['ANTHROPIC_API_KEY'],'anthropic-version':'2023-06-01','content-type':'application/json'},method='POST')
  with urllib.request.urlopen(request,timeout=600) as response:
   return Response(json.load(response))
 async def create(self,**body): return await asyncio.to_thread(self.request,body)
