import json,urllib.request,struct,concurrent.futures,time
from pathlib import Path
repo='ornith-ai/Ornith-1.5-122B'
rev='7681ef608624e7532d8164128bcffdbdbaf1b1af';print('revision',rev,flush=True)
root=Path('/mnt/llm-models/ornith-1.5-122b-kt-gpu');root.mkdir(exist_ok=True)
idx=json.load(urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{rev}/model.safetensors.index.json'))['weight_map']
needed={k:v for k,v in idx.items() if '.experts.' not in k and not k.startswith('mtp.') and not k.startswith('model.visual.')}
shards=sorted(set(needed.values()))
def get_range(file,start,end):
 url=f'https://huggingface.co/{repo}/resolve/{rev}/{file}?kt_range={start}-{end}'
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=90) as r:
    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('Server did not honor range')
    b=r.read(end-start+1)
    if len(b)!=end-start+1:raise RuntimeError('short response')
    return b
  except Exception:
   if attempt==2:raise
   time.sleep(2)
def header(file):
 n=struct.unpack('<Q',get_range(file,0,7))[0];assert n<20<<20
 h=json.loads(get_range(file,8,7+n));return file,{'header_bytes':n,'tensors':h}
headers={}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for f,h in pool.map(header,shards):headers[f]=h;print('headers',len(headers),'/',len(shards),flush=True)
tensors={k:{'file':f,'header_bytes':headers[f]['header_bytes'],**headers[f]['tensors'][k]} for k,f in needed.items()}
manifest={'repo':repo,'revision':rev,'tensors':tensors}
(root/'dense-source.json').write_text(json.dumps(manifest,indent=2))
print('dense_tensor_count',len(tensors),'BF16_GB',sum(v['data_offsets'][1]-v['data_offsets'][0] for v in tensors.values())/1e9,flush=True)
