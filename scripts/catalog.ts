import { readFile, realpath, stat } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import { type Catalog, parseCatalog } from '../src/catalog.ts';

export const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));

export async function loadCatalog(root = repositoryRoot): Promise<Catalog> {
  const value: unknown = JSON.parse(await readFile(resolve(root, 'manifest.json'), 'utf8'));
  return validateCatalog(value, root);
}

export async function validateCatalog(value: unknown, root = repositoryRoot): Promise<Catalog> {
  const catalog = parseCatalog(value);
  const exportsRoot = await realpath(resolve(root, 'exports'));

  for (const asset of catalog.assets) {
    const file = await realpath(resolve(root, asset.path));
    if (!file.startsWith(exportsRoot + sep)) {
      throw new Error(`Export escapes the repository: ${asset.id}`);
    }
    const info = await stat(file);
    if (!info.isFile() || info.size === 0) throw new Error(`Empty or invalid export: ${asset.id}`);
    const data = await readFile(file);
    if (
      data.length < 24 ||
      !data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    ) {
      throw new Error(`Export is not a PNG: ${asset.id}`);
    }
    if (data.readUInt32BE(16) === 0 || data.readUInt32BE(20) === 0) {
      throw new Error(`Invalid image dimensions: ${asset.id}`);
    }
  }

  return catalog;
}
