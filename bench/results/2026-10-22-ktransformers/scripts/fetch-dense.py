import urllib.request,json,struct,time,hashlib,concurrent.futures
from pathlib import Path
root=Path('/mnt/llm-models/ornith-1.5-122b-kt-gpu');m=json.loads((root/'dense-source.json').read_text());repo=m['repo'];rev=m['revision']
by={}
for k,v in m['tensors'].items():by.setdefault(v['file'],[]).append((k,v))
def fetch(file,items):
 dest=root/file
 items=sorted(items,key=lambda x:x[1]['data_offsets'][0]);h={'__metadata__':{'format':'pt','source':repo+'@'+rev,'scope':'non-routed-expert text tensors; CPU routed experts supplied by KT GGUF backend'}}
 at=0;groups=[]
 for k,v in items:
  a,b=v['data_offsets'];n=b-a;h[k]={'dtype':v['dtype'],'shape':v['shape'],'data_offsets':[at,at+n]}
  if groups and groups[-1][1]==a:groups[-1][1]=b
  else:groups.append([a,b,at])
  at+=n
 header=json.dumps(h,separators=(',',':')).encode();header+=b' '*((-len(header))%8);base=8+len(header)
 if dest.exists() and dest.stat().st_size==base+at:return file,at,'existing'
 tmp=dest.with_suffix('.partial')
 with tmp.open('wb') as out:
  out.write(struct.pack('<Q',len(header)));out.write(header);out.truncate(base+at)
  for a,b,new_at in groups:
   start=8+items[0][1]['header_bytes']+a;end=8+items[0][1]['header_bytes']+b-1
   for attempt in range(4):
    try:
     req=urllib.request.Request(f'https://huggingface.co/{repo}/resolve/{rev}/{file}?kt_dense_range={start}-{end}&attempt={attempt}',headers={'Range':f'bytes={start}-{end}'})
     with urllib.request.urlopen(req,timeout=180) as r:
      assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
      out.seek(base+new_at);left=b-a
      while left:
       chunk=r.read(min(8<<20,left))
       if not chunk:raise RuntimeError('short response')
       out.write(chunk);left-=len(chunk)
     break
    except Exception:
     if attempt==3:raise
     time.sleep(3)
 tmp.replace(dest);return file,at,'downloaded'
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for f,n,status in pool.map(lambda x:fetch(*x),by.items()):print(f,round(n/1e6,1),'MB',status,flush=True)
idx={'metadata':{'total_size':sum(v['data_offsets'][1]-v['data_offsets'][0] for v in m['tensors'].values())},'weight_map':{k:v['file'] for k,v in m['tensors'].items()}}
(root/'model.safetensors.index.json').write_text(json.dumps(idx,indent=2))
api=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}'))
for f in api['siblings']:
 name=f['rfilename']
 if '/' not in name and (name.endswith('.json') or name.endswith('.jinja')) and name!='model.safetensors.index.json':
  with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',timeout=60) as r:(root/name).write_bytes(r.read())
print('DONE total_GB',idx['metadata']['total_size']/1e9,flush=True)
