import http from 'node:http';
import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import path from 'node:path';

const root = await realpath(path.resolve('dist'));
const port = Number(process.argv[2] || process.env.PORT || 4173);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Use a valid port: node scripts/serve.mjs 4173');
const types = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp','.gif':'image/gif','.avif':'image/avif','.ico':'image/x-icon','.woff2':'font/woff2','.mp3':'audio/mpeg','.m4a':'audio/mp4','.ogg':'audio/ogg','.wav':'audio/wav','.pdf':'application/pdf','.xml':'application/xml; charset=utf-8','.txt':'text/plain; charset=utf-8'};
const inside = target => target === root || target.startsWith(root + path.sep);
const server = http.createServer(async (req, res) => {
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405, {Allow:'GET, HEAD'}); res.end(); return; }
  try {
    const requested = decodeURIComponent((req.url || '/').split('?')[0]);
    if (requested.includes('\0') || requested.includes('\\') || requested.split('/').includes('..')) { res.writeHead(403); res.end('Forbidden'); return; }
    let target = path.resolve(root, '.' + (requested.startsWith('/') ? requested : '/' + requested));
    if (!inside(target)) { res.writeHead(403); res.end('Forbidden'); return; }
    let info = await stat(target);
    if (info.isDirectory()) {
      if(!requested.endsWith('/')) {
        const original = req.url || '/', queryAt = original.indexOf('?');
        const urlPath = queryAt < 0 ? original : original.slice(0, queryAt);
        const query = queryAt < 0 ? '' : original.slice(queryAt);
        res.writeHead(301, {Location:urlPath + '/' + query}); res.end(); return;
      }
      target = path.join(target, 'index.html'); info = await stat(target);
    }
    target = await realpath(target);
    if (!inside(target) || !info.isFile()) { res.writeHead(403); res.end('Forbidden'); return; }
    const headers = {'Content-Type':types[path.extname(target).toLowerCase()] || 'application/octet-stream', 'Cache-Control':'no-cache', 'X-Content-Type-Options':'nosniff', 'Accept-Ranges':'bytes'};
    let start = 0, end = info.size - 1, status = 200;
    if (req.headers.range) {
      const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
      if (!range || (!range[1] && !range[2])) { res.writeHead(416, {'Content-Range':`bytes */${info.size}`}); res.end(); return; }
      if (!range[1]) start = Math.max(0, info.size - Number(range[2]));
      else { start = Number(range[1]); if (range[2]) end = Math.min(Number(range[2]), end); }
      if (start > end || start >= info.size) { res.writeHead(416, {'Content-Range':`bytes */${info.size}`}); res.end(); return; }
      status = 206; headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
    }
    headers['Content-Length'] = Math.max(0, end - start + 1);
    res.writeHead(status, headers);
    if (req.method === 'HEAD' || !info.size) { res.end(); return; }
    createReadStream(target, {start, end}).on('error', () => res.destroy()).pipe(res);
  } catch (error) {
    const status = error instanceof URIError ? 400 : ['ENOENT','ENOTDIR'].includes(error.code) ? 404 : 500;
    res.writeHead(status, {'Content-Type':'text/plain; charset=utf-8'});
    res.end(status === 404 ? 'Page not found' : 'Unable to serve this request');
  }
});
server.listen(port, '127.0.0.1', () => console.log(`Local: http://127.0.0.1:${port}/`));
server.on('error', error => { console.error(error.message); process.exitCode = 1; });
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close(() => process.exit(0)));
