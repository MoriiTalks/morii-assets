import { readFile } from 'node:fs/promises';
import { type IncomingMessage, type ServerResponse, createServer } from 'node:http';
import { resolve } from 'node:path';

import { loadCatalog, repositoryRoot } from './catalog.ts';

const catalog = await loadCatalog();
const files = new Map<string, readonly [string, string]>([
  ['/', ['preview/index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['dist/styles.css', 'text/css; charset=utf-8']],
  ['/app.js', ['dist/app.js', 'text/javascript; charset=utf-8']],
  ['/manifest.json', ['manifest.json', 'application/json; charset=utf-8']],
]);

for (const asset of catalog.assets) {
  files.set(`/${asset.path}`, [asset.path, 'image/png']);
}

const port = Number(process.env['PORT'] ?? 4174);
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('PORT must be between 1 and 65535');
}

async function serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405, { Allow: 'GET, HEAD' });
    response.end('Method not allowed');
    return;
  }

  let pathname: string;
  try {
    pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
  } catch {
    response.writeHead(400);
    response.end('Invalid request URL');
    return;
  }

  const file = files.get(pathname);
  if (file === undefined) {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    response.end('Not found');
    return;
  }

  try {
    const body = await readFile(resolve(repositoryRoot, file[0]));
    response.writeHead(200, {
      'Content-Type': file[1],
      'Content-Length': body.length,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
      'Content-Security-Policy':
        "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    });
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch (error) {
    console.error('Could not load preview file:', error);
    response.writeHead(500);
    response.end('Could not load preview file');
  }
}

const server = createServer((request, response) => {
  serve(request, response).catch((error: unknown) => {
    console.error('Preview request failed:', error);
    if (!response.headersSent) response.writeHead(500);
    response.end('Preview request failed');
  });
});

server.on('error', (error) => {
  console.error(`Preview server failed: ${error.message}`);
  process.exitCode = 1;
});
server.listen(port, '127.0.0.1', () => {
  console.log(`Morii asset catalog: http://127.0.0.1:${port.toString()}`);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    server.close();
  });
}
