import { loadCatalog } from './catalog.ts';

try {
  const catalog = await loadCatalog();
  console.log(`Validated ${catalog.assets.length.toString()} Morii exports (v${catalog.version}).`);
} catch (error) {
  console.error('Asset validation failed:', error);
  process.exitCode = 1;
}
