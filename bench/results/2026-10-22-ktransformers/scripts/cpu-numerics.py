import torch,numpy as np,gguf,gc,time,json
from pathlib import Path
from kt_kernel import KTMoEWrapper
root=Path('/models-ornith/Ornith-1.5-122B-Q4_K_M')
torch.set_num_threads(1);torch.manual_seed(42)
w=KTMoEWrapper(layer_idx=0,num_experts=256,num_experts_per_tok=8,hidden_size=3072,moe_intermediate_size=1024,gpu_experts_mask=torch.zeros(256,dtype=torch.bool),num_gpu_experts=0,cpuinfer_threads=12,threadpool_count=1,weight_path=str(root),chunked_prefill_size=256,method='LLAMAFILE',max_deferred_experts_per_token=0)
w.load_weights()
x=(torch.randn(1,3072)*.05).bfloat16();ids=torch.arange(8).reshape(1,8);rel=torch.ones(1,8)/8
out=w.forward(x,ids,rel,0);assert torch.isfinite(out).all()
tensors={}
readers=[gguf.GGUFReader(str(p)) for p in sorted(root.glob('*.gguf'))]
for r in readers:
 for t in r.tensors:
  if t.name.startswith('blk.0.ffn_') and '_exps.weight' in t.name:tensors[t.name]=t
ref=torch.zeros(1,3072)
for e in range(8):
 mats=[]
 for name in ('gate','up','down'):
  t=tensors['blk.0.ffn_'+name+'_exps.weight'];a=gguf.quants.dequantize(t.data[e],t.tensor_type);mats.append(torch.from_numpy(a.copy()))
 g,u,d=mats;ref+=(torch.nn.functional.silu(x.float()@g.T)*(x.float()@u.T))@d.T/8
error=(out.float()-ref).norm()/ref.norm();print('NUMERICS',json.dumps({'relative_l2':error.item(),'finite':True,'output_shape':list(out.shape),'gate_type':int(tensors['blk.0.ffn_gate_exps.weight'].tensor_type),'up_type':int(tensors['blk.0.ffn_up_exps.weight'].tensor_type)}),flush=True)
assert error<.06
for _ in range(5):w.forward(x,ids,rel,0)
t0=time.perf_counter()
for _ in range(100):w.forward(x,ids,rel,0)
print('MICROBENCH_MS',(time.perf_counter()-t0)*1000/100,flush=True)
del w;gc.collect();print('CLEANUP_DONE',flush=True)
