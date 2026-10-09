import { type Asset, parseCatalog } from '../src/catalog.ts';

function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing catalog element: ${id}`);
  return element;
}

function assetCard(asset: Asset): HTMLElement {
  const card = document.createElement('article');
  card.className = 'overflow-hidden rounded-2xl border border-line bg-white';

  const preview = document.createElement('a');
  preview.className = 'block aspect-square bg-mint p-5';
  preview.href = `/${asset.path}`;
  preview.setAttribute('aria-label', `Open ${asset.name}`);

  const image = document.createElement('img');
  image.src = `/${asset.path}`;
  image.alt = asset.name;
  image.width = 320;
  image.height = 320;
  image.loading = 'lazy';
  image.className = 'size-full object-contain';
  preview.append(image);

  const details = document.createElement('div');
  details.className = 'p-4';
  const title = document.createElement('h2');
  title.className = 'text-base font-semibold';
  title.textContent = asset.name;
  const id = document.createElement('p');
  id.className = 'mt-1 truncate font-mono text-xs text-muted';
  id.textContent = asset.id;

  const actions = document.createElement('div');
  actions.className = 'mt-3 flex items-center gap-4 text-sm font-medium text-teal';
  const open = document.createElement('a');
  open.href = `/${asset.path}`;
  open.className = 'inline-flex min-h-11 items-center underline-offset-4 hover:underline';
  open.textContent = 'Open';
  open.setAttribute('aria-label', `Open ${asset.name}`);
  const download = document.createElement('a');
  download.href = `/${asset.path}`;
  download.download = `${asset.id}.png`;
  download.className = 'inline-flex min-h-11 items-center underline-offset-4 hover:underline';
  download.textContent = 'Download';
  download.setAttribute('aria-label', `Download ${asset.name}`);
  actions.append(open, download);
  details.append(title, id, actions);
  card.append(preview, details);
  return card;
}

async function renderCatalog(): Promise<void> {
  const status = requireElement('catalog-status');
  try {
    const response = await fetch('/manifest.json');
    if (!response.ok) throw new Error(`Catalog request failed (${response.status.toString()})`);
    const value: unknown = await response.json();
    const catalog = parseCatalog(value);
    const gallery = requireElement('gallery');
    gallery.replaceChildren(...catalog.assets.map(assetCard));
    status.textContent = `${catalog.assets.length.toString()} assets · v${catalog.version}`;
  } catch (error) {
    status.textContent = 'Could not load assets. Check the manifest, then reload.';
    console.error('Catalog failed to load:', error);
  }
}

await renderCatalog();
