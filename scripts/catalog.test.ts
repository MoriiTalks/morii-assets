import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadCatalog, validateCatalog } from './catalog.ts';

await test('all catalog exports exist and have PNG dimensions', async () => {
  const catalog = await loadCatalog();
  assert.ok(catalog.assets.length > 0);
});

await test('invalid, duplicate, and missing exports fail validation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'morii-catalog-test-'));
  t.after(async () => {
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(join(root, 'exports', 'images'), { recursive: true });
  const pngHeader = Buffer.alloc(24);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(pngHeader);
  pngHeader.writeUInt32BE(128, 16);
  pngHeader.writeUInt32BE(128, 20);
  await writeFile(join(root, 'exports', 'images', 'morii.png'), pngHeader);

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
