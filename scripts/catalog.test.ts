import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { crc32 } from 'node:zlib';
import test from 'node:test';

import { loadCatalog, validateCatalog } from './catalog.ts';
import { validatePNG } from './png.ts';
import { validateGLB } from './glb.ts';

// A complete 1 x 1 transparent RGBA PNG, including a compressed scanline and valid CRCs.
const tinyPNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII=',
  'base64',
);

await test('all catalog exports have valid image or model containers', async () => {
  const catalog = await loadCatalog();
  assert.ok(catalog.assets.length > 0);
});

await test('the runtime rig stays within budget and contains every supported motion', async () => {
  const data = await readFile(new URL('../exports/models/morii.glb', import.meta.url));
  const summary = validateGLB(data, 'morii-3d');
  assert.ok(summary.triangles > 0 && summary.triangles <= 40_000);
  assert.equal(summary.joints, 15);
  assert.deepEqual(summary.morphs, ['blink', 'mouthOpen']);
  assert.deepEqual([...summary.clips].sort(), [
    'encourage',
    'idle',
    'listen',
    'move_backward',
    'move_forward',
    'move_left',
    'move_right',
    'reset',
    'speak',
    'think',
    'wave',
  ]);
  for (const broken of [data.subarray(0, 16), data.subarray(0, -1)]) {
    assert.throws(() => validateGLB(broken, 'invalid'), /GLB/);
  }
  const wrongVersion = Buffer.from(data);
  wrongVersion.writeUInt32LE(1, 4);
  assert.throws(() => validateGLB(wrongVersion, 'invalid'), /GLB header/);
  const wrongChunk = Buffer.from(data);
  wrongChunk.writeUInt32LE(0, 16);
  assert.throws(() => validateGLB(wrongChunk, 'invalid'), /GLB JSON/);
});

await test('invalid, duplicate, and missing exports fail validation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'morii-catalog-test-'));
  t.after(async () => {
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(join(root, 'exports', 'images'), { recursive: true });
  await writeFile(join(root, 'exports', 'images', 'morii.png'), tinyPNG);

  const asset = {
    id: 'morii',
    name: 'Morii',
    path: 'exports/images/morii.png',
    role: 'mascot',
    status: 'concept-reference',
  };
  const catalog = {
    version: '0.1.0',
    character: 'Morii',
    provenance: 'Test fixture',
    rights: 'Test fixture',
    assets: [asset],
  };

  await validateCatalog(catalog, root);
  await assert.rejects(
    validateCatalog({ ...catalog, assets: [asset, asset] }, root),
    /duplicate asset ID/,
  );
  await assert.rejects(
    validateCatalog({ ...catalog, assets: [{ ...asset, path: 'exports/../private.png' }] }, root),
    /Invalid.*export path/,
  );
  await assert.rejects(
    validateCatalog(
      { ...catalog, assets: [{ ...asset, path: 'exports/images/missing.png' }] },
      root,
    ),
    /ENOENT/,
  );
  await writeFile(join(root, 'exports', 'images', 'morii.png'), 'not an image');
  await assert.rejects(validateCatalog(catalog, root), /not a PNG/);
});

function chunk(type: string, body: Buffer): Buffer {
  const result = Buffer.alloc(body.length + 12);
  result.writeUInt32BE(body.length);
  result.write(type, 4, 4, 'ascii');
  body.copy(result, 8);
  result.writeUInt32BE(crc32(result.subarray(4, -4)), result.length - 4);
  return result;
}

await test('PNG validation rejects truncation, corruption, and malformed critical chunks', () => {
  validatePNG(tinyPNG, 'tiny');
  const signature = tinyPNG.subarray(0, 8);
  const header = tinyPNG.subarray(16, 29);
  const imageData = tinyPNG.subarray(41, 52);
  const headerChunk = chunk('IHDR', header);
  const dataChunk = chunk('IDAT', imageData);
  const ending = chunk('IEND', Buffer.alloc(0));
  const corrupted = Buffer.from(tinyPNG);
  corrupted.writeUInt8(1, 45);
  const oversized = Buffer.from(tinyPNG);
  oversized.writeUInt32BE(0xffffffff, 33);
  const nonASCIIHeader = Buffer.from(headerChunk);
  nonASCIIHeader.writeUInt8(0xc9, 4);
  nonASCIIHeader.writeUInt32BE(crc32(nonASCIIHeader.subarray(4, -4)), nonASCIIHeader.length - 4);
  const malformed = [
    tinyPNG.subarray(0, 24),
    tinyPNG.subarray(0, -1),
    tinyPNG.subarray(0, -12),
    corrupted,
    oversized,
    Buffer.concat([tinyPNG, Buffer.from([0])]),
    Buffer.concat([signature, dataChunk, ending]),
    Buffer.concat([signature, nonASCIIHeader, dataChunk, ending]),
    Buffer.concat([signature, chunk('IHDR', header.subarray(0, 12)), dataChunk, ending]),
    Buffer.concat([signature, headerChunk, headerChunk, dataChunk, ending]),
    Buffer.concat([signature, headerChunk, ending]),
    Buffer.concat([signature, headerChunk, chunk('IDAT', Buffer.alloc(0)), ending]),
    Buffer.concat([signature, headerChunk, dataChunk, chunk('IEND', Buffer.from([0]))]),
    Buffer.concat([
      signature,
      headerChunk,
      dataChunk,
      chunk('tEXt', Buffer.from('Key\0Value')),
      dataChunk,
      ending,
    ]),
    Buffer.concat([signature, headerChunk, chunk('ABCD', Buffer.alloc(0)), dataChunk, ending]),
  ];
  for (const data of malformed) {
    assert.throws(() => {
      validatePNG(data, 'invalid');
    }, /PNG/);
  }

  const invalidHeaders: readonly (readonly [number, number])[] = [
    [0, 0],
    [8, 1],
    [9, 1],
    [10, 1],
    [11, 1],
    [12, 2],
  ];
  for (const [offset, value] of invalidHeaders) {
    const invalidHeader = Buffer.from(header);
    if (offset === 0) invalidHeader.writeUInt32BE(value, 0);
    else invalidHeader.writeUInt8(value, offset);
    assert.throws(() => {
      validatePNG(
        Buffer.concat([signature, chunk('IHDR', invalidHeader), dataChunk, ending]),
        'header',
      );
    }, /Invalid PNG header/);
  }
});
