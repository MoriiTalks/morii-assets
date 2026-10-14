export interface Asset {
  readonly id: string;
  readonly name: string;
  readonly path: string;
  readonly preview?: string;
  readonly role: 'mascot' | 'brand';
  readonly status: 'concept-reference' | 'production-ready';
}

export interface Catalog {
  readonly version: string;
  readonly character: string;
  readonly provenance: string;
  readonly rights: string;
  readonly assets: readonly Asset[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`Missing or invalid ${field}`);
  }
  return value;
}

function parseAsset(value: unknown): Asset {
  if (!isRecord(value)) throw new Error('Asset entries must be objects');
  const data = value;
  const id = requiredString(data['id'], 'asset ID');
  const name = requiredString(data['name'], 'asset name');
  const path = requiredString(data['path'], 'export path');
  const role = data['role'];
  const status = data['status'];

  if (!/^[a-z][a-z0-9-]*$/.test(id)) throw new Error(`Invalid asset ID: ${id}`);
  if (!/^exports\/[a-z0-9/-]+\.(png|glb)$/.test(path)) {
    throw new Error(`Invalid export path: ${id}`);
  }
  if (role !== 'mascot' && role !== 'brand') throw new Error(`Invalid role: ${id}`);
  if (status !== 'concept-reference' && status !== 'production-ready') {
    throw new Error(`Invalid status: ${id}`);
  }

  const preview = data['preview'];
  if (preview === undefined) {
    if (path.endsWith('.glb')) throw new Error(`Missing model preview: ${id}`);
    return { id, name, path, role, status };
  }
  if (typeof preview !== 'string' || !/^exports\/[a-z0-9/-]+\.png$/.test(preview)) {
    throw new Error(`Invalid model preview: ${id}`);
  }
  return { id, name, path, preview, role, status };
}

export function parseCatalog(value: unknown): Catalog {
  if (!isRecord(value)) throw new Error('Catalog must be an object');
  const data = value;
  const version = requiredString(data['version'], 'catalog version');
  const character = requiredString(data['character'], 'catalog character');
  const provenance = requiredString(data['provenance'], 'catalog provenance');
  const rights = requiredString(data['rights'], 'catalog rights');
  const entries = data['assets'];

  if (!/^\d+\.\d+\.\d+$/.test(version)) {
    throw new Error('Catalog version must use major.minor.patch');
  }
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error('Catalog must contain at least one asset');
  }

  const values: readonly unknown[] = entries;
  const assets = values.map(parseAsset);
  const ids = new Set<string>();
  const paths = new Set<string>();

  for (const asset of assets) {
    if (ids.has(asset.id)) throw new Error(`Invalid or duplicate asset ID: ${asset.id}`);
    if (paths.has(asset.path)) throw new Error(`Invalid or duplicate export path: ${asset.id}`);
    ids.add(asset.id);
    paths.add(asset.path);
  }

  return { version, character, provenance, rights, assets };
}
