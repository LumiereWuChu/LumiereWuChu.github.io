import { readFile, writeFile, mkdir, rename, rm, lstat, realpath } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';

const LIMIT = 25 * 1024 * 1024;
const extensions = {'.jpg':['image/jpeg'],'.jpeg':['image/jpeg'],'.png':['image/png'],'.webp':['image/webp'],'.gif':['image/gif'],'.avif':['image/avif'],'.mp3':['audio/mpeg','audio/mp3'],'.m4a':['audio/mp4','audio/x-m4a'],'.ogg':['audio/ogg','application/ogg'],'.wav':['audio/wav','audio/x-wav','audio/wave'],'.pdf':['application/pdf']};
const fail = message => { throw new Error(message); };
const object = value => value && typeof value === 'object' && !Array.isArray(value);
const exists = async filename => { try { await lstat(filename); return true; } catch (e) { if(e.code === 'ENOENT') return false; throw e; } };
const readJSON = async (filename, fallback) => await exists(filename) ? JSON.parse(await readFile(filename, 'utf8')) : fallback;
const pretty = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
function text(value, field, max = 100000) {
  if (typeof value !== 'string' || !value.trim() || value.length > max || value.includes('\0')) fail(`${field}: a nonempty text value is required (maximum ${max} characters).`);
  return value.trim();
}
function bilingual(value, field, max) {
  if(!object(value)) fail(`${field}: both zh and en are required.`);
  return {zh:text(value.zh, `${field}.zh`, max), en:text(value.en, `${field}.en`, max)};
}
function safeName(name) {
  if(typeof name !== 'string' || name.length > 120 || !/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(name) || name.includes('..') || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(name)) fail('File names must be simple ASCII names without paths, traversal, or reserved Windows names.');
  const ext = path.extname(name).toLowerCase();
  if(!extensions[ext]) fail(`Unsupported file extension: ${ext}`);
  return {name, ext};
}
function signature(bytes, ext) {
  const ascii = (start, end) => bytes.subarray(start,end).toString('ascii');
  if(ext === '.pdf') return ascii(0,5) === '%PDF-';
  if(ext === '.png') return bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]));
  if(['.jpg','.jpeg'].includes(ext)) return bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
  if(ext === '.gif') return ['GIF87a','GIF89a'].includes(ascii(0,6));
  if(ext === '.webp') return ascii(0,4) === 'RIFF' && ascii(8,12) === 'WEBP';
  if(ext === '.wav') return ascii(0,4) === 'RIFF' && ascii(8,12) === 'WAVE';
  if(ext === '.ogg') return ascii(0,4) === 'OggS';
  if(ext === '.mp3') return ascii(0,3) === 'ID3' || (bytes[0] === 255 && (bytes[1] & 224) === 224);
  if(ext === '.m4a') return ascii(4,8) === 'ftyp';
  if(ext === '.avif') return ascii(4,8) === 'ftyp' && /avif|avis/.test(ascii(8,32));
  return false;
}
function decodeFile(file, cv = false) {
  if(!object(file)) fail('Invalid file record.');
  const {name, ext} = safeName(file.name);
  if(cv && (name !== 'cv.pdf' || ext !== '.pdf')) fail('CV files must be named cv.pdf.');
  if(!cv && ext === '.pdf') fail('Use a CV bundle to upload a PDF.');
  if(!cv && !extensions[ext].includes(file.mime)) fail(`File MIME does not match ${ext}.`);
  if(typeof file.base64 !== 'string' || !file.base64.length || file.base64.length > Math.ceil(LIMIT / 3) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(file.base64)) fail('File data is invalid or exceeds 25 MB.');
  const bytes = Buffer.from(file.base64, 'base64');
  const maximum = cv ? 10 * 1024 * 1024 : file.mime?.startsWith('image/') ? 12 * 1024 * 1024 : LIMIT;
  if(!bytes.length || bytes.length > maximum || !signature(bytes,ext)) fail(`Invalid, mismatched, or oversized file: ${name}`);
  return {name, bytes, ext};
}
async function noSymlinks(root, target) {
  const relative = path.relative(root,target);
  if(relative.startsWith('..') || path.isAbsolute(relative)) fail('Target leaves the project directory.');
  let current = root;
  for(const segment of relative.split(path.sep).filter(Boolean)) {
    current = path.join(current,segment);
    try { if((await lstat(current)).isSymbolicLink()) fail('Refusing to write through a symbolic link.'); }
    catch(error) { if(error.code !== 'ENOENT') throw error; }
  }
}
async function commitChanges(root, changes, replace) {
  const nonce = randomUUID();
  const previous = new Map();
  for(const change of changes) {
    await noSymlinks(root, change.path);
    previous.set(change.path, await exists(change.path) ? await readFile(change.path) : null);
  }
  if(replace && [...previous.values()].some(Boolean)) {
    const backup = path.join(root,'content','backups',`${new Date().toISOString().replace(/[:.]/g,'-')}-${nonce.slice(0,8)}`);
    await noSymlinks(root,backup);
    for(const [filename, bytes] of previous) if(bytes) {
      const destination = path.join(backup,path.relative(path.join(root,'content'),filename));
      await mkdir(path.dirname(destination),{recursive:true});
      await writeFile(destination,bytes,{flag:'wx'});
    }
  }
  const staged = [], applied = [];
  try {
    for(const change of changes) {
      await mkdir(path.dirname(change.path),{recursive:true});
      const temp = `${change.path}.${nonce}.tmp`;
      await writeFile(temp,change.bytes,{flag:'wx'});
      staged.push(temp);
    }
    for(let i = 0; i < changes.length; i++) {
      await rename(staged[i], changes[i].path);
      applied.push(changes[i].path);
    }
  } catch(error) {
    for(const filename of applied.reverse()) {
      const bytes = previous.get(filename);
      if(bytes === null) await rm(filename,{force:true}); else await writeFile(filename,bytes);
    }
    throw error;
  } finally { await Promise.all(staged.map(filename => rm(filename,{force:true}))); }
}

export async function importBundle(bundle, projectRoot, {replace = false} = {}) {
  const root = await realpath(path.resolve(projectRoot));
  if(!object(bundle) || bundle.version !== 1) fail('Expected a version 1 Lumière bundle.');
  const entriesPath = path.join(root,'content','entries.json');
  const profilePath = path.join(root,'content','profile.json');
  await noSymlinks(root,entriesPath); await noSymlinks(root,profilePath);
  if(bundle.cv) {
    if(bundle.entry || bundle.files) fail('CV and creative-entry bundles must be separate.');
    const file = decodeFile(bundle.cv,true);
    const filePath = path.join(root,'content','uploads',file.name);
    await noSymlinks(root,filePath);
    if(await exists(filePath) && !replace) fail('cv.pdf already exists. Use --replace to back up and replace it.');
    const profile = await readJSON(profilePath,null);
    if(!object(profile)) fail('content/profile.json must exist before importing a CV.');
    profile.cvFile = 'uploads/cv.pdf';
    await commitChanges(root,[{path:filePath,bytes:file.bytes},{path:profilePath,bytes:pretty(profile)}],replace);
    return {type:'cv',path:'uploads/cv.pdf'};
  }
  const source = bundle.entry;
  if(!object(source)) fail('The entry is missing.');
  if(typeof source.id !== 'string' || source.id.length > 80 || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(source.id)) fail('Use a unique lowercase slug, for example autumn-light.');
  if(!['note','poetry','photography','music'].includes(source.type)) fail('Unsupported entry type.');
  if(typeof source.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(source.date) || new Date(source.date+'T00:00:00Z').toISOString().slice(0,10) !== source.date) fail('Use a valid date in YYYY-MM-DD format.');
  const entry = {id:source.id,type:source.type,date:source.date,title:bilingual(source.title,'title',200),excerpt:bilingual(source.excerpt,'excerpt',1000),body:bilingual(source.body,'body'),tags:[],media:null};
  if(!Array.isArray(source.tags) || source.tags.length > 12) fail('Use an array of up to 12 tags.');
  entry.tags = [...new Set(source.tags.map(tag=>text(tag,'tag',60)))];
  if(!Array.isArray(bundle.files) || bundle.files.length > 1) fail('Use a files array with at most one image or audio attachment.');
  const files = bundle.files.map(file=>decodeFile(file));
  if(files.length) {
    const file = files[0];
    if(!object(source.media) || source.media.path !== `uploads/${file.name}`) fail('The media path must match the attached file.');
    const audio = ['.mp3','.m4a','.ogg','.wav'].includes(file.ext);
    if(source.type === 'music' ? !audio : audio) fail('Music entries need audio; other entries accept images.');
    entry.media = {path:`uploads/${file.name}`,alt:bilingual(source.media.alt,'media.alt',400)};
  } else if(source.media || ['music','photography'].includes(source.type)) fail('Photography and music entries require their media attachment.');
  const entries = await readJSON(entriesPath,[]);
  if(!Array.isArray(entries)) fail('content/entries.json must be an array.');
  const index = entries.findIndex(item=>item.id === entry.id);
  if(index >= 0 && !replace) fail('This slug already exists. Use --replace to back up and replace the entry.');
  const changes = [];
  for(const file of files) {
    const filePath = path.join(root,'content','uploads',file.name);
    await noSymlinks(root,filePath);
    if(entries.some(item=>item.id !== entry.id && item.media?.path === entry.media.path)) fail('This media filename belongs to another entry. Choose a unique filename.');
    if(await exists(filePath) && !replace) fail('The media filename already exists. Choose a unique name or use --replace.');
    changes.push({path:filePath,bytes:file.bytes});
  }
  if(index >= 0) entries[index] = entry; else entries.push(entry);
  changes.push({path:entriesPath,bytes:pretty(entries)});
  await commitChanges(root,changes,replace);
  return {type:entry.type,id:entry.id,count:entries.length};
}

if(process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = process.argv.slice(2);
    const filename = args.find(arg=>!arg.startsWith('--'));
    if(!filename || args.some(arg=>arg.startsWith('--') && arg !== '--replace')) fail('Usage: node scripts/import-entry.mjs path/to/bundle.json [--replace]');
    const bytes = await readFile(path.resolve(filename));
    if(bytes.length > 38 * 1024 * 1024) fail('Bundle exceeds the 38 MB limit.');
    const result = await importBundle(JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/,'')),process.cwd(),{replace:args.includes('--replace')});
    console.log(`Imported ${result.id || result.path}. Run node scripts/build.mjs, then review before publishing.`);
  } catch(error) { console.error(`Import stopped: ${error.message}`); process.exitCode = 1; }
}
