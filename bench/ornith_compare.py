#!/usr/bin/env python3
"""Small paired local evaluation; keeps all prompts/outputs/timings, no benchmark-score claims.

Generated Python runs only inside a labelled, networkless Docker sandbox with no
host access except its read-only temporary test file. No GPU is passed to it.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request
import uuid

VOICE = 'Eger bir sesli asistan yapsaydin nasil bir mimari kurardin?'
CASES = [
 ('merge', 'Sadece Python kodu yaz. merge_intervals(intervals) fonksiyonunu uygula. Girdi kapalı aralıklardan oluşan bir liste; sırasız olabilir. Örtüşen veya aynı uç noktada buluşan aralıkları birleştir, başlangıca göre sıralı döndür. Girdiyi değiştirme. Boş girdi boş liste dönsün.', 768, '''
import copy,random
examples=[([],[]),([(1,3),(2,6),(8,10),(15,18)],[(1,6),(8,10),(15,18)]),([(4,4),(1,4)],[(1,4)]),([(-3,-1),(-2,0),(2,2)],[(-3,0),(2,2)])]
for a,want in examples:
 before=copy.deepcopy(a);assert list(map(tuple,merge_intervals(a)))==want;assert a==before
rng=random.Random(42)
for _ in range(100):
 a=[sorted([rng.randrange(-20,20),rng.randrange(-20,20)]) for _ in range(rng.randrange(20))];before=copy.deepcopy(a);want=[]
 for lo,hi in sorted(a):
  if want and lo<=want[-1][1]:want[-1][1]=max(want[-1][1],hi)
  else:want.append([lo,hi])
 assert list(map(tuple,merge_intervals(a)))==list(map(tuple,want));assert a==before
'''),
 ('lru', 'Sadece Python kodu yaz. LRUCache(capacity) sınıfı: get(key) değeri ya da bulunamazsa None döndürsün; put(key,value) eklesin veya güncellesin. get ve put kullanımı elemanı en yeni yapsın. Kapasite aşılınca en eski kullanılan anahtar silinsin. İşlemler amortize O(1) olsun. Sıfır veya negatif kapasite ValueError vermeli.', 768, '''
c=LRUCache(2);c.put('a',1);c.put('b',2);assert c.get('a')==1;c.put('c',3);assert c.get('b') is None;assert c.get('a')==1;c.put('a',7);c.put('d',4);assert c.get('c') is None;assert c.get('a')==7;assert c.get('d')==4
c=LRUCache(1);c.put('a',None);assert c.get('a') is None;c.put('b',0);assert c.get('b')==0;c.put('b',False);assert c.get('b') is False
for bad in [0,-1]:
 try:LRUCache(bad)
 except ValueError:pass
 else:raise AssertionError('invalid capacity accepted')
'''),
 ('topo', 'Sadece Python kodu yaz. topo_sort(graph) fonksiyonunu uygula. graph, string düğüm -> sonraki düğümler listesi sözlüğüdür; yalnızca hedeflerde geçen düğümleri de dahil et. Leksikografik olarak en küçük geçerli topolojik sıralamayı liste olarak döndür. Döngü varsa ValueError fırlat. Girdiyi değiştirme. Aynı kenarın tekrarına dayanıklı olsun.', 768, '''
import copy
for graph,want in [({},[]),({'b':['c'],'a':['c']},['a','b','c']),({'b':['d','d'],'a':['c'],'c':['d']},['a','b','c','d']),({'z':[],'a':[]},['a','z'])]:
 before=copy.deepcopy(graph);assert topo_sort(graph)==want;assert graph==before
for graph in [{'a':['a']},{'a':['b'],'b':['a']},{'x':[],'a':['b'],'b':['a']}]:
 try:topo_sort(graph)
 except ValueError:pass
 else:raise AssertionError('cycle accepted')
'''),
 ('json', 'Aşağıdaki siparişi yalnızca geçerli JSON olarak dönüştür. Şema: {"items":[{"name":string,"qty":integer,"unit_try":number}],"total_try":number}. Başka alan veya Markdown ekleme. Sipariş: Defter 2 adet, birim fiyatı 125,50 TL; Kalem 3 adet, birim fiyatı 19,90 TL. İndirim ve vergi yok.', 256, None),
 ('tool', 'Melbourne için güncel hava sıcaklığını Celsius olarak öğren. Bunun için verilen get_weather aracını çağır; hava durumunu tahmin etme.', 128, None),
 ('architecture', VOICE, 1536, None),
]


def api(base, path, payload=None, key=None):
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
    with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)


def load(base, model):
    job=api(base,'/api/native/operations',{'kind':'load','args':{'model':model}})
    for _ in range(240):
        s=api(base,'/api/native')
        j=next(j for j in s['jobs'] if j['id']==job['id'])
        if j['state']=='error':raise RuntimeError(j.get('error'))
        if j['state']=='done':
            ready=[i for i in s['instances'] if i['status']=='ready']
            assert len(ready)==1 and ready[0]['model']==model
            assert ready[0]['gpus']==[1],ready[0]['gpus']
            return s
        time.sleep(1)
    raise TimeoutError('model load')


def sandbox(content, tests):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',content,re.S)
    code=blocks[0] if blocks else content.strip()
    name='forgeloop-ai-eval-'+uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix='forgeloop-ai-eval-') as temp:
        path=Path(temp);path.chmod(0o755)
        script=path/'run.py';script.write_text(code+'\n'+tests+'\nprint("PASS")\n');script.chmod(0o644)
        cmd=['docker','run','--rm','--name',name,'--label','forgeloop.managed=true','--label','forgeloop.project=forgeloop-ai','--label','forgeloop.resource=eval',
             '--network=none','--read-only','--tmpfs','/tmp:rw,noexec,nosuid,size=16m','--cap-drop=ALL','--security-opt=no-new-privileges',
             '--user','65534:65534','--pids-limit','32','--memory','256m','--cpus','1','-v',str(path)+':/case:ro','--entrypoint','python','forgeloop-ai-controller:latest','-I','-S','/case/run.py']
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            return {'pass':result.returncode==0,'stdout':result.stdout[-1000:],'stderr':result.stderr[-2000:]}
        except subprocess.TimeoutExpired:return {'pass':False,'error':'sandbox timeout'}
        finally:
            inspect=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
            if inspect.returncode==0:
                labels=json.loads(inspect.stdout)[0]['Config']['Labels']
                if labels.get('forgeloop.managed')=='true' and labels.get('forgeloop.project')=='forgeloop-ai':
                    subprocess.run(['docker','rm','-f',name],capture_output=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--base',default='http://127.0.0.1:10104');parser.add_argument('--key-env',default='STRATA_API_KEY');parser.add_argument('--speed-only',action='store_true');parser.add_argument('--quality-only',action='store_true');parser.add_argument('--quality-repeats',type=int,default=1);parser.add_argument('--warm-files',action='store_true');parser.add_argument('--expected-context',type=int,default=262144);parser.add_argument('--expected-kv',default='q8_0');parser.add_argument('--reasoning-effort',default='none')
    a=parser.parse_args();key=os.environ[a.key_env]
    state=load(a.base,a.model)
    entry=next(m for m in state['installed']['models'] if m['model']==a.model)
    assert entry.get('context')==a.expected_context and entry.get('kv')==a.expected_kv,entry
    report={'model':a.model,'context':entry['context'],'kv':entry['kv'],'gpu':1,'temperature':0,'reasoning_effort':a.reasoning_effort,'engine_nice':0,'records':[]}
    def save():a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if a.warm_files:
        config=json.loads((Path(__file__).resolve().parent.parent/f'strata-{a.model}.json').read_text())
        model=Path(config['args'][config['args'].index('--model')+1])
        split=re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf',model.name)
        files=[model.with_name(f'{split[1]}-{n:05d}-of-{int(split[2]):05d}.gguf') for n in range(1,int(split[2])+1)] if split else [model]
        start=time.monotonic();total=0
        for file in files:
            with file.open('rb') as stream:
                while data:=stream.read(8<<20):total+=len(data)
        report['file_cache_warmup']={'bytes_read':total,'seconds':round(time.monotonic()-start,3)}
        print(json.dumps({'model':a.model,'file_cache_warmup':report['file_cache_warmup']}),flush=True)
    def engine_io():
        result=subprocess.run(['pgrep','-x','strata-poly'],capture_output=True,text=True)
        pids=[int(x) for x in result.stdout.split()]
        if len(pids)!=1:return None
        try:
            values=dict(line.split(': ',1) for line in Path(f'/proc/{pids[0]}/io').read_text().splitlines())
            return {'pid':pids[0],'nice':os.getpriority(os.PRIO_PROCESS,pids[0]),'read_bytes':int(values['read_bytes'])}
        except (OSError,KeyError):return None

    def host_memory():
        """Host-wide counters; other processes can contribute to these deltas."""
        memory={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3 and line.split()[2]=='kB'}
        vm=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        return {'available_bytes':memory['MemAvailable'],'swap_used_bytes':memory['SwapTotal']-memory['SwapFree'],
                'swap_in_pages':int(vm['pswpin']),'swap_out_pages':int(vm['pswpout']),'major_faults':int(vm['pgmajfault'])}

    def generate(name,prompt,limit,tools=None):
        payload={'model':a.model,'messages':[{'role':'user','content':prompt}],'max_tokens':limit,'temperature':0,'top_p':.95,'top_k':20,'min_p':0.,'presence_penalty':0.,'repetition_penalty':1.,'reasoning_effort':a.reasoning_effort}
        if tools:payload.update(tools=tools,tool_choice='auto')
        memory_before=host_memory();io_before=engine_io();start=time.monotonic();response=api(a.base,'/v1/chat/completions',payload,key);io_after=engine_io();memory_after=host_memory()
        row={'case':name,'prompt':prompt,'limit':limit,'seconds':round(time.monotonic()-start,3),'response':response}
        row['host_memory']={'before':memory_before,'after':memory_after,'swap_in_pages_delta':memory_after['swap_in_pages']-memory_before['swap_in_pages'],'swap_out_pages_delta':memory_after['swap_out_pages']-memory_before['swap_out_pages'],'major_faults_delta':memory_after['major_faults']-memory_before['major_faults']}
        if io_before and io_after and io_before['pid']==io_after['pid']:row['engine_io']={'nice':io_after['nice'],'storage_read_bytes':io_after['read_bytes']-io_before['read_bytes']}
        report['records'].append(row);save();return row
    generate('warmup','Sadece 1 ile 30 arasındaki sayıları sırayla yaz.',64)
    if not a.speed_only:
        for name,prompt,limit,tests,trial in [(*case,trial) for trial in range(1,a.quality_repeats+1) for case in CASES if trial==1 or case[0]!='architecture']:
            tools=[{'type':'function','function':{'name':'get_weather','description':'Get current weather for a location','parameters':{'type':'object','properties':{'location':{'type':'string'},'unit':{'type':'string','enum':['celsius','fahrenheit']}},'required':['location','unit'],'additionalProperties':False}}}] if name=='tool' else None
            row=generate(name if trial==1 else f'{name}-repeat-{trial}',prompt,limit,tools);row['quality_trial']=trial
            message=row['response']['choices'][0]['message'];content=message.get('content') or ''
            if tests:row['validation']=sandbox(content,tests)
            elif name=='json':
                try:
                    obj=json.loads(content);items=sorted(obj['items'],key=lambda item:item['name'])
                    row['validation']={'pass':set(obj)=={'items','total_try'} and items==[{'name':'Defter','qty':2,'unit_try':125.5},{'name':'Kalem','qty':3,'unit_try':19.9}] and abs(obj['total_try']-310.7)<1e-6}
                except (ValueError,KeyError,TypeError):row['validation']={'pass':False,'error':'not valid plain JSON/schema'}
            elif name=='tool':
                calls=message.get('tool_calls') or []
                try:
                    call=calls[0]['function'];args=json.loads(call['arguments']);row['validation']={'pass':len(calls)==1 and set(args)=={'location','unit'} and call['name']=='get_weather' and args.get('unit')=='celsius' and 'melbourne' in args.get('location','').lower()}
                except (KeyError,IndexError,ValueError):row['validation']={'pass':False,'error':'missing/invalid tool call'}
            save();print(json.dumps({'model':a.model,'case':name,'validation':row.get('validation'),'timings':row['response'].get('timings')},ensure_ascii=False),flush=True)
    if not a.quality_only:generate('speed-warmup',VOICE,512)
    for trial in range(0 if a.quality_only else 3):
        row=generate('speed-'+str(trial+1),VOICE,512)
        print(json.dumps({'model':a.model,'case':row['case'],'timings':row['response'].get('timings')},ensure_ascii=False),flush=True)
    save()


if __name__=='__main__':main()
