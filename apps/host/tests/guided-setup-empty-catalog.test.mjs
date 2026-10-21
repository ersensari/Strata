import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

/* The installed app crashed on its first screen ("Cannot read properties of undefined (reading 'family')") when the
   catalog it was given had no `recommendation` and no families. Render the real GuidedSetup and NativeModels components
   (react-dom/server) with the state libraries replaced by fixed values, for every shape of an empty or failed catalog. */
const root=resolve(import.meta.dirname,'..');
/** Every export of the real module becomes a harmless no-op, except the few that carry the fixed state. */
function stubFor(file,overrides){
 const source=readFileSync(resolve(root,'src/lib',file),'utf8');
 const names=new Set([...source.matchAll(/export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)].map(m=>m[1]));
 for(const m of source.matchAll(/export\s*\{([^}]*)\}/g))for(const part of m[1].split(','))names.add(part.split(/\s+as\s+/).pop().trim());
 names.delete('');
 const lines=[...names].filter(n=>!(n in overrides)).map(n=>`export const ${n}=(...a)=>Promise.resolve({});`);
 for(const [n,v] of Object.entries(overrides))lines.push(`export const ${n}=${v};`);
 return lines.join(String.fromCharCode(10));
}
const stubs={
 native:stubFor('native.ts',{useNative:'()=>globalThis.__native',nativeState:'()=>globalThis.__native'}),
 store:stubFor('store.ts',{useStore:'sel=>sel(globalThis.__store)'}),
 client:stubFor('client.ts',{}),
 router:stubFor('router.ts',{}),
};

let dir,mod;
async function load(){
 if(mod)return mod;
 dir=mkdtempSync(join(tmpdir(),'strata-gs-'));
 const stubPlugin={name:'stubs',setup(b){
  b.onResolve({filter:/\/lib\/(native|store|client|router)$/},args=>({path:args.path.split('/').pop(),namespace:'stub'}));
  b.onLoad({filter:/.*/,namespace:'stub'},args=>({contents:stubs[args.path],loader:'js',resolveDir:root}));
 }};
 const entry=join(dir,'entry.tsx');
 const out=join(dir,'out.cjs');
 await build({
  stdin:{contents:`export {GuidedSetup} from ${JSON.stringify(resolve(root,'src/components/GuidedSetup.tsx'))};
   export {NativeModels} from ${JSON.stringify(resolve(root,'src/components/NativeModels.tsx'))};
   export {availableModelFamilies} from ${JSON.stringify(resolve(root,'src/lib/native-catalog.ts'))};
   export {renderToString} from 'react-dom/server'; export {createElement} from 'react';`,resolveDir:root,sourcefile:'entry.tsx',loader:'tsx'},
  bundle:true,format:'cjs',platform:'node',outfile:out,jsx:'automatic',logLevel:'silent',plugins:[stubPlugin],
  loader:{'.svg':'dataurl','.png':'dataurl','.css':'empty'},
 });
 globalThis.window??=globalThis;globalThis.matchMedia??=()=>({matches:false,addEventListener(){},removeEventListener(){}});
 mod=createRequire(import.meta.url)(out);
 return mod;
}
test.after(()=>{if(dir)rmSync(dir,{recursive:true,force:true});});

const store=(complete=false)=>({setup:{complete,choices:{},app:{modelsDir:''}},hardware:null});
const base={available:true,repo:'',summary:'',error:null,installed:{models:[],data_folder:'',engine:{version:null}},hardware:null,instances:[],jobs:[],defaultId:null,fields:[],capabilities:{}};
const catalogs={
 'no catalog at all':null,
 'the packaged host before it had a catalog':{families:[],sizes:[]},
 'no recommendation':{families:[],contexts:[32768]},
 'recommendation with nothing in it':{families:[],contexts:[],recommendation:{family:null,model:null}},
 'no families key':{contexts:[]},
 'a family without sizes':{families:[{family:'qwen',title:'Qwen3.8-Flash-Next',sizes:[]}],contexts:[32768]},
};

for(const [name,catalog] of Object.entries(catalogs)){
 for(const component of ['GuidedSetup','NativeModels']){
  test(`${component} renders with ${name}`,async()=>{
   const m=await load();
   globalThis.__native={...base,catalog};globalThis.__store=store();
   let html;
   assert.doesNotThrow(()=>{html=m.renderToString(m.createElement(m[component]));});
   assert.equal(typeof html,'string');
  });
 }
}

test('GuidedSetup lists the real catalog when it has one',async()=>{
 const m=await load();
 globalThis.__native={...base,catalog:{families:[{family:'qwen',title:'Qwen3.8-Flash-Next',about:'',images:true,experimental:false,sizes:[
   {id:'iq3_xxs',model:'IQ3_XXS',download_gb:75.8,ram_gb:60,about:'',on_this_pc:'fits',installed:false}]}],contexts:[32768],
   recommendation:{family:'qwen',model:'IQ3_XXS',context:32768},context_note:''}};
 globalThis.__store={...store(),setup:{complete:false,choices:{onboarding:{step:2,manual:false}},app:{modelsDir:''}}};
 const html=m.renderToString(m.createElement(m.GuidedSetup));
 assert.match(html,/IQ3_XXS/);
});


for (const catalog of [null, {families:[{family:'qwen',title:'Qwen3.8-Flash-Next',sizes:[{id:'catalog-only',model:'IQ3_XXS',download_gb:75.8,ram_gb:60,about:'',on_this_pc:'fits'}]}],contexts:[32768],recommendation:{family:'qwen',model:'IQ3_XXS'}}]) {
 test(`NativeModels shows six installed configs outside the catalog (${catalog ? 'unrelated catalog' : 'no catalog'})`,async()=>{
  const m=await load();
  const models=['ornith-iq3','ornith-iq4','minicpm','minicpm-uncensored','gemma','qwen'].map(model=>({model,model_name:model,ready:true,context:262144,kv:'q8_0',diskGb:54}));
  globalThis.__native={...base,catalog,installed:{...base.installed,models},instances:[{id:'active',model:'ornith-iq3',status:'ready',gpus:[0]}]};globalThis.__store=store(true);
  const html=m.renderToString(m.createElement(m.NativeModels));
  assert.match(html,/aria-label="Downloaded models"/);
  assert.equal((html.match(/data-model="/g)||[]).length,6);
  for(const model of models)assert.ok(html.includes(`data-model="${model.model}"`),model.model);
  assert.match(html,/256K context.*Q8 KV/);
  const iq3=html.match(/data-model="ornith-iq3"[\s\S]*?<\/button>/)?.[0];
  assert.match(iq3,/disabled=""/);assert.match(iq3,/>Loaded<\/button>/);
  const iq4=html.match(/data-model="ornith-iq4"[\s\S]*?<\/button>/)?.[0];
  assert.doesNotMatch(iq4,/disabled=""/);assert.match(iq4,/>Load<\/button>/);
 });
}

test('model-family selector includes saved variants without mutating the download catalog',async()=>{
 const m=await load();
 const catalog=[{family:'qwen',title:'Qwen3.8-Flash-Next',images:true,sizes:[{id:'iq3_xxs',model:'IQ3_XXS'}]}];
 const installed=['ornith-iq3','ornith-iq4','minicpm','minicpm-uncensored','gemma','qwen','another-engine'].map(model=>({model,model_name:model,ready:true,diskGb:54}));
 const groups=m.availableModelFamilies(catalog,installed);
 assert.deepEqual(groups.filter(g=>g.sizes.length).map(g=>g.title),['Qwen3.8-Flash-Next','Ornith 1.5 122B','MiniCPM-V 4.7','Gemma 4 26B','another-engine']);
 assert.deepEqual(groups.find(g=>g.family==='local-ornith').sizes.map(v=>v.id),['ornith-iq3','ornith-iq4']);
 assert.equal(groups.find(g=>g.family==='local-minicpm').sizes.length,2);
 assert.equal(groups.flatMap(g=>g.sizes).filter(v=>v.localConfig).length,7);
 assert.equal(catalog[0].sizes.length,1);
 assert.equal(m.availableModelFamilies([],installed).flatMap(g=>g.sizes).length,7);
 const known=m.availableModelFamilies(catalog,[{model:'iq3_xxs',ready:true}]);
 assert.equal(known.flatMap(g=>g.sizes).length,1);
});

test('imported Swift is downloaded in the original native catalog panel, with other recommendations retained',async()=>{
 const m=await load();
 const catalog={families:[{family:'swift',title:'Swift 1.5',images:true,sizes:[{id:'swift-iq3_xxs',model:'IQ3_XXS',download_gb:75.8,ram_needed_gb:60,experts_gb:42.9,on_this_pc:'fits'},{id:'swift-iq2_xs',model:'IQ2_XS'}]}],contexts:[32768],recommendation:{family:'swift',model:'IQ3_XXS'}};
 const installed=[{model:'qwen',catalogId:'swift-iq3_xxs',model_name:'swift-1.5-iq3_xxs',ready:true,context:262144,kv:'int8',images:'yes'}];
 const groups=m.availableModelFamilies(catalog.families,installed);
 const swift=groups[0].sizes.find(v=>v.model==='IQ3_XXS');
 assert.equal(swift.id,'qwen');assert.equal(swift.sourceFamily,'swift');assert.equal(swift.localConfig,undefined);
 assert.equal(groups[0].sizes.length,2);assert.equal(catalog.families[0].sizes[0].id,'swift-iq3_xxs');
 globalThis.__native={...base,catalog,installed:{...base.installed,models:installed}};globalThis.__store=store(true);
 const html=m.renderToString(m.createElement(m.NativeModels));
 assert.match(html,/Swift IQ3_XXS/);assert.match(html,/Downloaded<\/span>/);
 assert.match(html,/Conversation context/);assert.match(html,/Computer use/i);
 assert.doesNotMatch(html,/Installed model version/);assert.match(html,/>Load model<\/button>/);
 assert.doesNotMatch(html,/>Download &amp; load<\/button>/);
});

test('an unloaded native model shows its saved single GPU instead of recommending a second card',async()=>{
 const m=await load();
 const catalog={families:[{family:'swift',title:'Swift 1.5',images:true,sizes:[{id:'swift-iq3_xxs',model:'IQ3_XXS',download_gb:75.8,ram_needed_gb:60,experts_gb:42.9,on_this_pc:'fits'}]}],contexts:[32768],recommendation:{family:'swift',model:'IQ3_XXS'}};
 const models=[{model:'qwen',catalogId:'swift-iq3_xxs',model_name:'swift-1.5-iq3_xxs',ready:true,gpu:1}];
 globalThis.__native={...base,catalog,installed:{...base.installed,models},hardware:{ram_gb:96,engine_vendor:'nvidia',gpus:[{index:0,name:'Card A',vram_gb:16,usable:true},{index:1,name:'Card B',vram_gb:16,usable:true}]}};globalThis.__store=store(true);
 const html=m.renderToString(m.createElement(m.NativeModels));
 const first=html.match(/aria-label="Role of GPU 1"[\s\S]*?<\/div>/)?.[0];
 const second=html.match(/aria-label="Role of GPU 2"[\s\S]*?<\/div>/)?.[0];
 assert.ok(first&&second);
 assert.match(first,/aria-checked="true"[^>]*>Off<\/button>/);
 assert.match(second,/aria-checked="true"[^>]*>Main<\/button>/);
});
