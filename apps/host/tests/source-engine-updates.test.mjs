import test from 'node:test';
import assert from 'node:assert/strict';
import {checkUpdates,sourceEngineUpdate} from '../server/updates.mjs';
const releases=async()=>Response.json([{tag_name:'v0.2.2',draft:false,prerelease:false}]);
test('source update check is opt in; ordinary source builds keep their original verdict',async()=>{
 const plain=await checkUpdates({appVersion:'0.2.2',engineVersion:'0.1.47'},releases);
 assert.equal(plain.engine.status,'unavailable');
 const local=await checkUpdates({appVersion:'0.2.2',engineVersion:'0.1.47',sourceEngine:{update_mode:'source',target_version:'0.1.47',source_changed:false}},releases);
 assert.equal(local.engine.status,'current');assert.equal(local.engine.canInstall,false);
 assert.match(local.engine.message,/this source checkout/);
});
test('source rebuild and older native binaries offer the checkout update, never an inferred upstream release',()=>{
 assert.equal(sourceEngineUpdate('0.1.47',{target_version:'0.1.47',source_changed:true}).canInstall,true);
 assert.equal(sourceEngineUpdate('0.1.44.1',{target_version:'0.1.47'}).status,'available');
 assert.equal(sourceEngineUpdate('0.1.48',{target_version:'0.1.47'}).status,'current');
 assert.equal(sourceEngineUpdate(null,{target_version:'0.1.47'}).status,'unavailable');
 assert.equal(sourceEngineUpdate('0.1.47',{}).canInstall,false);
});
