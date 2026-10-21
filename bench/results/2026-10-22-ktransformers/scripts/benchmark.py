import json,time,urllib.request,os
from pathlib import Path
out=Path(__file__).resolve().parents[1]
prefix=os.environ.get('KT_RUN','trial')
payload={'model':'ornith-kt-test','messages':[{'role':'user','content':'bir sesli asistan yapsaydin mimariyi nasil kurardin'}],'max_tokens':768,'temperature':0.6,'top_p':0.95,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repetition_penalty':1.0,'seed':42,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
for trial in range(1,3):
 t=time.perf_counter();first=None;last=None;chunks=[];usage=None;events=[]
 req=urllib.request.Request('http://127.0.0.1:10112/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   event=json.loads(raw);at=time.perf_counter()-t
   if event.get('usage'):usage=event['usage']
   events.append({'seconds':at,'event':event})
   for choice in event.get('choices',[]):
    d=choice.get('delta',{});s=d.get('content','') or d.get('reasoning_content','')
    if s:
     if first is None:first=at
     last=at;chunks.append(s)
 total=time.perf_counter()-t;n=usage['completion_tokens'] if usage else None
 result={'request':payload,'trial':trial,'usage':usage,'ttft_s':first,'total_s':total,'decode_tokens_per_s':(n-1)/(last-first) if n and last>first else None,'content':''.join(chunks),'events':events}
 (out/f'{prefix}-{trial}-events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False,separators=(',',':'))+'\n' for e in events))
 result.pop('events')
 (out/f'{prefix}-{trial}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 (out/f'{prefix}-{trial}.txt').write_text(result['content'])
 print(json.dumps({k:v for k,v in result.items() if k not in ('request','events','content')},ensure_ascii=False),flush=True)
 print(result['content'][:350],flush=True)
