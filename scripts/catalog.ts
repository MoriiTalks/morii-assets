import { readFile, realpath, stat } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import { type Catalog, parseCatalog } from '../src/catalog.ts';
import { validatePNG } from './png.ts';
import { validateGLB } from './glb.ts';

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
    const bytes = await readFile(file);
    if (asset.path.endsWith('.glb')) validateGLB(bytes, asset.id);
    else validatePNG(bytes, asset.id);
    if (asset.preview !== undefined) {
      const preview = await realpath(resolve(root, asset.preview));
      if (!preview.startsWith(exportsRoot + sep)) {
        throw new Error(`Preview escapes the repository: ${asset.id}`);
      }
      validatePNG(await readFile(preview), asset.id);
    }
  }

  return catalog;
}
