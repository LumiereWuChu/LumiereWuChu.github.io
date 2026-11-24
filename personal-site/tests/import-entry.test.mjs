import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, readdir, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { importBundle } from '../scripts/import-entry.mjs';

async function workspace(t) {
  const root = await mkdtemp(path.join(os.tmpdir(),'lumiere-import-test-'));
  await mkdir(path.join(root,'content'));
  await writeFile(path.join(root,'content','entries.json'),'[]\n');
  await writeFile(path.join(root,'content','profile.json'),JSON.stringify({name:'Lumière',github:'https://github.com/LumiereWuChu'}));
  t.after(async()=>{if(path.dirname(root)!==path.resolve(os.tmpdir())||!path.basename(root).startsWith('lumiere-import-test-'))throw new Error('Unsafe test cleanup path');await rm(root,{recursive:true,force:true});});
  return root;
}
function note(id='first-note') { return {version:1,entry:{id,type:'note',date:'2026-11-24',title:{zh:'光',en:'Light'},excerpt:{zh:'一些记录',en:'Some notes'},body:{zh:'光照在墙上。',en:'Light falls on the wall.'},tags:['Light'],media:null},files:[]}; }
const entries = async root => JSON.parse(await readFile(path.join(root,'content','entries.json'),'utf8'));
function photo(id='morning-light', filename='morning-light.png') {
  const bundle=note(id);bundle.entry.type='photography';bundle.entry.media={path:'uploads/'+filename,alt:{zh:'清晨的光',en:'Morning light'}};
  bundle.files=[{name:filename,mime:'image/png',base64:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jhRkAAAAASUVORK5CYII='}];return bundle;
}

test('a bilingual text entry survives the import round trip', async t=>{
 const root=await workspace(t), bundle=note();
 await importBundle(bundle,root);
 assert.deepEqual((await entries(root))[0],bundle.entry);
});
test('a photograph writes the attachment and its matching public path', async t=>{
 const root=await workspace(t), bundle=photo();
 await importBundle(bundle,root);
 assert.deepEqual(await readFile(path.join(root,'content','uploads','morning-light.png')),Buffer.from(bundle.files[0].base64,'base64'));
 assert.equal((await entries(root))[0].media.path,'uploads/morning-light.png');
});
test('traversal cannot write outside uploads and leaves content untouched', async t=>{
 const root=await workspace(t), bundle=photo('morning-light','../../escaped.png');
 await assert.rejects(importBundle(bundle,root),/File names/);
 assert.deepEqual(await entries(root),[]);
 assert.deepEqual((await readdir(path.join(root,'content'))).sort(),['entries.json','profile.json']);
});
test('duplicate slugs require an explicit replace and retain the previous version', async t=>{
 const root=await workspace(t), original=note();await importBundle(original,root);
 const revision=note();revision.entry.body.en='A revised thought.';
 await assert.rejects(importBundle(revision,root),/slug already exists/);
 assert.equal((await entries(root))[0].body.en,original.entry.body.en);
 await importBundle(revision,root,{replace:true});
 assert.equal((await entries(root))[0].body.en,'A revised thought.');
 const backups=await readdir(path.join(root,'content','backups'));assert.equal(backups.length,1);
 const backed=JSON.parse(await readFile(path.join(root,'content','backups',backups[0],'entries.json'),'utf8'));
 assert.equal(backed[0].body.en,original.entry.body.en);
});
test('an existing media filename is never silently reassigned to another entry', async t=>{
 const root=await workspace(t);await importBundle(photo(),root);
 await assert.rejects(importBundle(photo('second-photo'),root,{replace:true}),/another entry/);
 assert.equal((await entries(root)).length,1);
});
test('an incomplete translation fails before any file is written', async t=>{
 const root=await workspace(t), bundle=photo();bundle.entry.body.en='';
 await assert.rejects(importBundle(bundle,root),/body.en/);
 assert.deepEqual(await entries(root),[]);
 assert.deepEqual((await readdir(path.join(root,'content'))).sort(),['entries.json','profile.json']);
});
test('invalid calendar dates and malformed attachments are rejected', async t=>{
 const root=await workspace(t), badDate=note();badDate.entry.date='2026-02-30';
 await assert.rejects(importBundle(badDate,root),/valid date/);
 const badFile=photo();badFile.files[0].base64=Buffer.from('<script>alert(1)</script>').toString('base64');
 await assert.rejects(importBundle(badFile,root),/Invalid, mismatched/);
 assert.deepEqual(await entries(root),[]);
});
test('CV imports preserve the profile and require replacement for a later PDF', async t=>{
 const root=await workspace(t), bundle={version:1,cv:{name:'cv.pdf',base64:Buffer.from('%PDF-1.4\n% test fixture\n%%EOF').toString('base64')}};
 await importBundle(bundle,root);
 const profile=JSON.parse(await readFile(path.join(root,'content','profile.json'),'utf8'));
 assert.equal(profile.name,'Lumière');assert.equal(profile.cvFile,'uploads/cv.pdf');
 await assert.rejects(importBundle(bundle,root),/already exists/);
 await importBundle(bundle,root,{replace:true});
 assert.equal((await readdir(path.join(root,'content','backups'))).length,1);
});
