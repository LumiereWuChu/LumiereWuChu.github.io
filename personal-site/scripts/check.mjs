import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const dist=path.join(root,'dist');
const files=[];
function walk(p){for(const e of fs.readdirSync(p,{withFileTypes:true})){const f=path.join(p,e.name);if(e.isDirectory())walk(f);else files.push(f);}}
walk(dist);
const failures=[];let checked=0;
for(const file of files.filter(f=>f.endsWith('.html'))){const html=fs.readFileSync(file,'utf8');
 for(const m of html.matchAll(/(?:href|src)="([^"]+)"/g)){const value=m[1].replaceAll('&amp;','&');if(/^(https?:|mailto:|data:|#)/.test(value))continue;const target=path.resolve(path.dirname(file),decodeURIComponent(value.split('#')[0].split('?')[0]));checked++;if(!fs.existsSync(target))failures.push(`${path.relative(dist,file)}: missing ${value}`);}
 if(/\/zh\//.test(file.replaceAll('\\','/'))){const counterpart=file.replace(`${path.sep}zh${path.sep}`,`${path.sep}en${path.sep}`);if(!fs.existsSync(counterpart))failures.push(`Missing English page: ${file}`);}
 if(/undefined|\[object Object\]/.test(html))failures.push(`Unresolved value: ${file}`);
}
const entries=JSON.parse(fs.readFileSync(path.join(root,'content','entries.json'),'utf8'));
const ids=new Set();
for(const entry of entries){assert(!ids.has(entry.id),`Duplicate entry ${entry.id}`);ids.add(entry.id);assert.match(entry.id,/^[a-z0-9]+(?:-[a-z0-9]+)*$/);for(const field of ['title','excerpt','body'])for(const lang of ['zh','en'])assert.equal(typeof entry[field]?.[lang],'string',`${entry.id}.${field}.${lang} missing`);}
for(const name of ['tools','content','scripts','.git','.openai'])assert(!fs.existsSync(path.join(dist,name)),`Private source directory in dist: ${name}`);
assert(!files.some(f=>/SOURCE_AUDIT|backups|editor\.html/.test(f)),'Authoring or audit files leaked into build');
assert.equal(failures.length,0,failures.join('\n'));
console.log(`PASS: ${files.filter(f=>f.endsWith('.html')).length} HTML pages, ${checked} local links/assets, bilingual pairs, unique entries, no authoring files in dist.`);
