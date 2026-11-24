import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const dist=path.join(root,'dist');
const files=[];
function walk(p){for(const e of fs.readdirSync(p,{withFileTypes:true})){const f=path.join(p,e.name);if(e.isDirectory())walk(f);else files.push(f);}}
walk(dist);
const failures=[];let checked=0,srcsetChecked=0;
const inside=(directory,target)=>{const relative=path.relative(directory,target);return relative===''||(!relative.startsWith('..'+path.sep)&&relative!=='..'&&!path.isAbsolute(relative));};
const unescapeHTML=value=>value.replace(/&(?:amp|quot|apos|lt|gt);|&#(?:x[\da-f]+|\d+);/gi,entity=>{
 const named={'&amp;':'&','&quot;':'"','&apos;':"'",'&lt;':'<','&gt;':'>'};
 if(named[entity.toLowerCase()])return named[entity.toLowerCase()];
 const point=entity[2].toLowerCase()==='x'?parseInt(entity.slice(3,-1),16):parseInt(entity.slice(2,-1),10);
 return point<=0x10ffff?String.fromCodePoint(point):entity;
});
function checkLocalURL(value,file,kind='link'){
 value=unescapeHTML(value);
 if(/^(https?:|mailto:|data:|#|\/\/)/i.test(value))return;
 try{
  const decoded=decodeURIComponent(value.split('#')[0].split('?')[0]);
  const target=decoded.startsWith('/')?path.resolve(dist,'.'+decoded):path.resolve(path.dirname(file),decoded);
  checked++;if(kind==='srcset')srcsetChecked++;
  if(!inside(dist,target))failures.push(`${path.relative(dist,file)}: ${kind} leaves dist: ${value}`);
  else if(!fs.existsSync(target))failures.push(`${path.relative(dist,file)}: missing ${kind} ${value}`);
  else if(!inside(fs.realpathSync(dist),fs.realpathSync(target)))failures.push(`${path.relative(dist,file)}: ${kind} resolves outside dist: ${value}`);
 }catch(error){failures.push(`${path.relative(dist,file)}: invalid ${kind} ${value}: ${error.message}`);}
}
for(const file of files.filter(f=>f.endsWith('.html'))){const html=fs.readFileSync(file,'utf8');
 for(const m of html.matchAll(/(?:href|src)="([^"]+)"/g))checkLocalURL(m[1],file);
 for(const m of html.matchAll(/\bsrcset="([^"]+)"/g)){
  for(const candidate of unescapeHTML(m[1]).split(/\s*,\s*/)){
   const [url,descriptor,...extra]=candidate.trim().split(/\s+/);
   if(!url||extra.length||(descriptor&&!/^(?:[1-9]\d*w|(?:\d*\.)?\d+x)$/.test(descriptor)))failures.push(`${path.relative(dist,file)}: malformed srcset candidate ${candidate}`);
   else checkLocalURL(url,file,'srcset');
  }
 }
 if(/\/zh\//.test(file.replaceAll('\\','/'))){const counterpart=file.replace(`${path.sep}zh${path.sep}`,`${path.sep}en${path.sep}`);if(!fs.existsSync(counterpart))failures.push(`Missing English page: ${file}`);}
 if(/undefined|\[object Object\]/.test(html))failures.push(`Unresolved value: ${file}`);
 const decodedHTML=unescapeHTML(html).replace(/%3a/gi,':').replace(/%5c/gi,'\\').replace(/%2f/gi,'/');
 if(/file:\/\/|(?<![a-z0-9])[a-z]:[\\/]|\\\\[a-z0-9_.-]+\\/i.test(decodedHTML))failures.push(`Local filesystem source path in public HTML: ${path.relative(dist,file)}`);
}
const entries=JSON.parse(fs.readFileSync(path.join(root,'content','entries.json'),'utf8'));
const ids=new Set();
for(const entry of entries){assert(!ids.has(entry.id),`Duplicate entry ${entry.id}`);ids.add(entry.id);assert.match(entry.id,/^[a-z0-9]+(?:-[a-z0-9]+)*$/);for(const field of ['title','excerpt','body'])for(const lang of ['zh','en'])assert.equal(typeof entry[field]?.[lang],'string',`${entry.id}.${field}.${lang} missing`);}
const photographyFile=path.join(root,'content','photography.json');
const photographs=fs.existsSync(photographyFile)?JSON.parse(fs.readFileSync(photographyFile,'utf8')):[];
assert(Array.isArray(photographs),'photography.json must contain an array');
const photoIds=new Set(),assets=path.join(dist,'assets');
for(const photo of photographs){
 assert.equal(typeof photo.id,'string','Photograph id missing');assert.match(photo.id,/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
 assert(!photoIds.has(photo.id),`Duplicate photograph ${photo.id}`);photoIds.add(photo.id);
 for(const field of ['title','caption','alt'])for(const lang of ['zh','en'])assert(typeof photo[field]?.[lang]==='string'&&photo[field][lang].trim(),`${photo.id}.${field}.${lang} translation missing`);
 for(const dimension of ['width','height'])assert(Number.isInteger(photo[dimension])&&photo[dimension]>0,`${photo.id}.${dimension} must be a positive integer`);
 assert(photo.date===null||(typeof photo.date==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(photo.date)&&Number.isFinite(Date.parse(photo.date))&&new Date(photo.date).toISOString().slice(0,10)===photo.date),`${photo.id}.date must be null or a real YYYY-MM-DD date`);
 for(const field of ['src','thumbnail']){
  const value=photo[field];assert(typeof value==='string'&&value.trim(),`${photo.id}.${field} missing`);
  assert(!/^[\\/]|[:?#\\]/.test(value),`${photo.id}.${field} must be a relative asset path`);
  const target=path.resolve(assets,value);
  assert(inside(assets,target),`${photo.id}.${field} leaves dist/assets`);
  assert(fs.existsSync(target)&&fs.statSync(target).isFile(),`${photo.id}.${field} file missing: ${value}`);
  assert(inside(fs.realpathSync(assets),fs.realpathSync(target)),`${photo.id}.${field} resolves outside dist/assets`);
 }
}
for(const name of ['tools','content','scripts','.git','.openai'])assert(!fs.existsSync(path.join(dist,name)),`Private source directory in dist: ${name}`);
assert(!files.some(f=>/SOURCE_AUDIT|backups|editor\.html/.test(f)),'Authoring or audit files leaked into build');
assert.equal(failures.length,0,failures.join('\n'));
console.log(`PASS: ${files.filter(f=>f.endsWith('.html')).length} HTML pages, ${checked} local links/assets (${srcsetChecked} responsive sources), ${photographs.length} bilingual photographs, bilingual pairs, unique entries, no authoring files or local source paths in dist.`);
