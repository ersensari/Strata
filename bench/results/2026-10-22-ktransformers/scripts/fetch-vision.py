import json,urllib.request,struct
from pathlib import Path
s=Path(__file__).with_name('fetch-dense.py').read_text().split('with concurrent.futures.ThreadPoolExecutor')[0]
s=s.replace('dest=root/file',"dest=root/('vision-'+file)").replace('non-routed-expert text tensors','non-routed-expert vision tensors')
ns={};exec(s,ns)
m=ns['m'];root=ns['root'];repo=m['repo'];rev=m['revision'];file='model-00001-of-00062.safetensors'
def part(a,b):
 req=urllib.request.Request(f'https://huggingface.co/{repo}/resolve/{rev}/{file}?kt_vision_header={a}-{b}',headers={'Range':f'bytes={a}-{b}'})
 with urllib.request.urlopen(req,timeout=90) as r:assert r.status==206;return r.read(b-a+1)
n=struct.unpack('<Q',part(0,7))[0];h=json.loads(part(8,n+7));items=[(k,{'file':file,'header_bytes':n,**v}) for k,v in h.items() if k.startswith('model.visual.')]
print('vision_GB',sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in items)/1e9,flush=True)
print(ns['fetch'](file,items),flush=True)
idx=json.loads((root/'model.safetensors.index.json').read_text());idx['weight_map'].update({k:'vision-'+file for k,v in items});idx['metadata']['total_size']=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in m['tensors'].values())+sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in items)
(root/'model.safetensors.index.json').write_text(json.dumps(idx,indent=2));(root/'vision-source.json').write_text(json.dumps({'repo':repo,'revision':rev,'tensors':dict(items)},indent=2));print('DONE',flush=True)
