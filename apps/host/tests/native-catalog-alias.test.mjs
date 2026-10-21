import test from 'node:test';
import assert from 'node:assert/strict';
import {nativeCatalogId} from '../server/native-catalog-alias.mjs';
const catalog={families:[{family:'swift',sizes:[{id:'swift-iq3_xxs',model:'IQ3_XXS'},{id:'swift-iq2_xs',model:'IQ2_XS'}]}]};
test('imported native Swift retains its public id but matches the exact catalog quant',()=>{
 assert.equal(nativeCatalogId({model:'qwen'},{exe:'/app/engines/strata',model_name:'swift-1.5-iq3_xxs'},catalog),'swift-iq3_xxs');
 assert.equal(nativeCatalogId({model:'qwen'},{exe:'C:\\engines\\strata.exe',model_name:'swift-1.5-iq2_xs'},catalog),'swift-iq2_xs');
});
test('Poly, unknown quants and other families cannot be mistaken for the installed Swift',()=>{
 assert.equal(nativeCatalogId({model:'ornith-iq3'},{exe:'/app/engines/strata-poly',model_name:'swift-1.5-iq3_xxs'},catalog),undefined);
 assert.equal(nativeCatalogId({model:'qwen'},{exe:'/app/engines/strata',model_name:'swift-1.5-iq4_xs'},catalog),undefined);
 assert.equal(nativeCatalogId({model:'qwen'},{exe:'/app/engines/strata',model_name:'ornith-iq3_xxs'},catalog),undefined);
 assert.equal(nativeCatalogId({model:'qwen'},{exe:'/app/engines/strata',model_name:'swift-1.5-iq3_xxs'},null),undefined);
});
test('canonical installed IDs continue to work without alias inference',()=>{
 assert.equal(nativeCatalogId({model:'swift-iq3_xxs'},{},catalog),'swift-iq3_xxs');
});

test('native control resolves imported Swift for load and reuses its GGUF folder for preparation',async()=>{
 const {nativeFixture,finishJob}=await import('./helpers/native-fixture.mjs');
 const {writeFileSync}=await import('node:fs');const {join}=await import('node:path');
 const f=nativeFixture();
 try{
  f.catalog.families.push(catalog.families[0]);
  const file=join(f.repo,'Swift-Qwen3.8-IQ3_XXS.gguf');writeFileSync(file,'fixture');
  f.installed.models.push({model:'qwen',config:'strata-qwen.json',model_name:'swift-1.5-iq3_xxs',ready:true});
  writeFileSync(join(f.repo,'strata-qwen.json'),JSON.stringify({exe:'/app/engines/strata',model_name:'swift-1.5-iq3_xxs',gpu:0,args:['--native',file,'--max-context','262144','--kv','int8']}));
  await f.control.refresh(true);
  assert.equal(f.control.state().installed.models.find(m=>m.model==='qwen').catalogId,'swift-iq3_xxs');
  await f.control.plan({family:'swift',model:'IQ3_XXS',context:262144});
  assert.equal(f.rpcCalls.filter(c=>c.name==='strata_install').at(-1).arguments.gguf_dir,f.repo);
  await f.control.plan({family:'swift',model:'NOT_INSTALLED'});
  assert.equal(f.rpcCalls.filter(c=>c.name==='strata_install').at(-1).arguments.gguf_dir,undefined);
  const job=await finishJob(f.control,f.control.operation('load',{model:'qwen'}).id);
  assert.equal(job.state,'done',job.error);assert.equal(f.control.state().instances[0].model,'qwen');
 }finally{await f.cleanup();}
});
