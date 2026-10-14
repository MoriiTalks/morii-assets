import { type Asset, parseCatalog } from '../src/catalog.ts';

import type { AnimationAction, AnimationClip, Object3D } from 'three';
import type * as Three from 'three';

interface PreviewPlayer {
  play: (name: string) => void;
  pause: () => void;
  setCard: (visible: boolean) => void;
  dispose: () => void;
}

interface ActionBlend {
  readonly action: AnimationAction;
  readonly start: number;
  readonly target: number;
}

interface MorphChannel {
  readonly weights: number[];
  readonly blink: number | undefined;
  readonly mouth: number | undefined;
}

const clipLabels = new Map([
  ['idle', 'At rest'],
  ['wave', 'Hello'],
  ['listen', 'Listening'],
  ['think', 'Thinking'],
  ['speak', 'Speaking'],
  ['encourage', 'Encouragement'],
  ['move_left', 'Step left'],
  ['move_right', 'Step right'],
  ['move_forward', 'Step forward'],
  ['move_backward', 'Step back'],
  ['reset', 'Return to center'],
]);

function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing catalog element: ${id}`);
  return element;
}

function previewElements() {
  const dialog = requireElement('model-preview');
  const select = requireElement('model-animation');
  const play = requireElement('model-play');
  const pause = requireElement('model-pause');
  const card = requireElement('model-card');
  if (
    !(dialog instanceof HTMLDialogElement) ||
    !(select instanceof HTMLSelectElement) ||
    !(play instanceof HTMLButtonElement) ||
    !(pause instanceof HTMLButtonElement) ||
    !(card instanceof HTMLInputElement)
  ) {
    throw new Error('Invalid model preview controls');
  }
  return {
    dialog,
    select,
    play,
    pause,
    card,
    stage: requireElement('model-stage'),
    title: requireElement('model-title'),
    status: requireElement('model-status'),
  };
}

const controls = previewElements();
let closePreview: (() => void) | undefined;

// Each modal owns one renderer, its animation frame and all loaded GPU resources.
function createPlayer(
  three: typeof Three,
  model: Object3D,
  names: readonly string[],
  clips: readonly AnimationClip[],
  onPlaybackChange: (playing: boolean) => void,
): PreviewPlayer {
  const renderer = new three.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor('#FFFDF6', 1);
  renderer.outputColorSpace = three.SRGBColorSpace;
  renderer.toneMapping = three.NeutralToneMapping;
  renderer.toneMappingExposure = 1;
  const scene = new three.Scene();
  scene.add(model, new three.HemisphereLight('#FFFFFF', '#C6D4CE', 2));
  const key = new three.DirectionalLight('#FFFDF6', 2.8);
  key.position.set(-3, 5, 5);
  scene.add(key);
  const camera = new three.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, 1.6, 6.8);
  camera.lookAt(0, 1.45, 0);
  const mixer = new three.AnimationMixer(model);
  const actions = new Map<string, AnimationAction>();
  for (const clip of clips) {
    if (!names.includes(clip.name)) continue;
    const action = mixer.clipAction(clip);
    action.setLoop(three.LoopOnce, 1);
    action.clampWhenFinished = true;
    actions.set(clip.name, action);
  }
  const ideaCard = model.getObjectByName('IdeaCard');
  if (ideaCard !== undefined) ideaCard.visible = false;
  const channels: MorphChannel[] = [];
  model.traverse((object) => {
    if (!(object instanceof three.Mesh) || object.morphTargetInfluences === undefined) return;
    const blink = object.morphTargetDictionary?.['blink'];
    const mouth = object.morphTargetDictionary?.['mouthOpen'];
    if (blink !== undefined || mouth !== undefined) {
      channels.push({ weights: object.morphTargetInfluences, blink, mouth });
    }
  });
  let disposed = false;
  let frame: number | undefined;
  let lastTime = 0;
  let elapsed = 0;
  let blendTime = 0;
  let blends: readonly ActionBlend[] = [];
  let active: AnimationAction | undefined;
  let playing = false;

  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', 'Morii 3D model');
  controls.stage.replaceChildren(renderer.domElement);

  function render(): void {
    if (!disposed) renderer.render(scene, camera);
  }

  function pause(): void {
    playing = false;
    if (frame !== undefined) cancelAnimationFrame(frame);
    frame = undefined;
    lastTime = 0;
    for (const channel of channels) {
      if (channel.blink !== undefined) channel.weights[channel.blink] = 0;
      if (channel.mouth !== undefined) channel.weights[channel.mouth] = 0;
    }
    render();
    onPlaybackChange(false);
  }

  function tick(time: number): void {
    frame = undefined;
    if (disposed || !playing) return;
    const delta = lastTime === 0 ? 0 : Math.min((time - lastTime) / 1000, 0.05);
    lastTime = time;
    elapsed += delta;
    blendTime += delta;
    const amount = three.MathUtils.smoothstep(blendTime, 0, 0.3);
    for (const blend of blends) {
      blend.action.setEffectiveWeight(blend.start + (blend.target - blend.start) * amount);
    }
    mixer.update(delta);
    if (amount === 1) {
      for (const blend of blends) {
        if (blend.target === 0) blend.action.stop();
      }
      blends = [];
    }
    for (const channel of channels) {
      const blinkDistance = Math.abs((elapsed % 4.4) - 1.7);
      if (channel.blink !== undefined) {
        channel.weights[channel.blink] =
          blinkDistance < 0.14 ? Math.sin((1 - blinkDistance / 0.14) * Math.PI * 0.5) : 0;
      }
      if (channel.mouth !== undefined) {
        const target =
          active?.getClip().name === 'speak' ? 0.15 + Math.sin(elapsed * 9) ** 2 * 0.35 : 0;
        channel.weights[channel.mouth] = three.MathUtils.damp(
          channel.weights[channel.mouth] ?? 0,
          target,
          14,
          delta,
        );
      }
    }
    render();
    if (active !== undefined && !active.isRunning() && blends.length === 0) {
      pause();
      return;
    }
    frame = requestAnimationFrame(tick);
  }

  function resize(): void {
    if (disposed) return;
    const width = Math.max(1, controls.stage.clientWidth);
    const height = Math.max(1, controls.stage.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(controls.stage);
  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
  motionPreference.addEventListener('change', pause);
  document.addEventListener('visibilitychange', onVisibilityChange);
  function onVisibilityChange(): void {
    if (document.hidden) pause();
  }
  function dispose(): void {
    if (disposed) return;
    pause();
    disposed = true;
    resizeObserver.disconnect();
    motionPreference.removeEventListener('change', pause);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    mixer.stopAllAction();
    mixer.uncacheRoot(model);
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();
  }

  try {
    resize();
  } catch (error) {
    dispose();
    throw error;
  }

  return {
    play(name) {
      const next = actions.get(name);
      if (disposed || next === undefined) return;
      if (active !== next || !next.isRunning()) {
        if (!next.isScheduled() || !next.isRunning()) next.reset().setEffectiveWeight(0).play();
        // Snapshot current weights so a second selection can interrupt a blend without snapping.
        blends = [...actions.values()]
          .filter((action) => action.isScheduled())
          .map((action) => ({
            action,
            start: action.getEffectiveWeight(),
            target: action === next ? 1 : 0,
          }));
        blendTime = 0;
        elapsed = 0;
        active = next;
      }
      playing = true;
      lastTime = 0;
      onPlaybackChange(true);
      frame ??= requestAnimationFrame(tick);
    },
    pause,
    setCard(visible) {
      if (ideaCard !== undefined) ideaCard.visible = visible;
      render();
    },
    dispose,
  };
}

async function openModel(asset: Asset): Promise<void> {
  closePreview?.();
  const abort = new AbortController();
  let player: PreviewPlayer | undefined;
  let releaseModel: (() => void) | undefined;
  controls.title.textContent = asset.name;
  controls.status.textContent = 'Loading model…';
  controls.select.replaceChildren();
  controls.card.checked = false;
  for (const control of [controls.select, controls.play, controls.pause, controls.card])
    control.disabled = true;
  if (!controls.dialog.open) controls.dialog.showModal();
  closePreview = () => {
    if (abort.signal.aborted) return;
    abort.abort();
    player?.dispose();
    releaseModel?.();
    controls.play.onclick = null;
    controls.pause.onclick = null;
    controls.select.onchange = null;
    controls.card.onchange = null;
    controls.stage.replaceChildren();
  };
  try {
    const [{ GLTFLoader }, three, response] = await Promise.all([
      import('three/addons/loaders/GLTFLoader.js'),
      import('three'),
      fetch(`/${asset.path}`, { signal: abort.signal }),
    ]);
    if (!response.ok) throw new Error(`Model request failed (${response.status.toString()})`);
    const gltf = await new GLTFLoader().parseAsync(await response.arrayBuffer(), '/');
    let released = false;
    releaseModel = () => {
      if (released) return;
      released = true;
      const geometries = new Set<{ dispose: () => void }>();
      const materials = new Set<{ dispose: () => void }>();
      const textures = new Set<{ dispose: () => void }>();
      gltf.scene.traverse((object) => {
        if (!(object instanceof three.Mesh)) return;
        const geometry: unknown = object.geometry;
        if (geometry instanceof three.BufferGeometry) geometries.add(geometry);
        const materialValue: unknown = object.material;
        const values: readonly unknown[] = Array.isArray(materialValue)
          ? materialValue
          : [materialValue];
        for (const material of values) {
          if (!(material instanceof three.Material)) continue;
          materials.add(material);
          const properties: readonly unknown[] = Object.values(material);
          for (const property of properties) {
            if (property instanceof three.Texture) textures.add(property);
          }
        }
        if (object instanceof three.SkinnedMesh) object.skeleton.dispose();
      });
      for (const geometry of geometries) geometry.dispose();
      for (const material of materials) material.dispose();
      for (const texture of textures) texture.dispose();
    };
    if (abort.signal.aborted) {
      releaseModel();
      return;
    }
    const names = gltf.animations.map((clip) => clip.name).filter((name) => clipLabels.has(name));
    for (const name of names) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = clipLabels.get(name) ?? name;
      controls.select.append(option);
    }
    if (names.includes('idle')) controls.select.value = 'idle';
    player = createPlayer(three, gltf.scene, names, gltf.animations, (playing) => {
      if (abort.signal.aborted) return;
      controls.play.disabled = playing;
      controls.pause.disabled = !playing;
      controls.status.textContent = playing ? 'Playing' : 'Paused';
    });
    controls.select.disabled = names.length === 0;
    controls.play.disabled = names.length === 0;
    controls.card.disabled = gltf.scene.getObjectByName('IdeaCard') === undefined;
    controls.status.textContent = matchMedia('(prefers-reduced-motion: reduce)').matches
      ? 'Reduced motion is on. Press Play to preview.'
      : 'Ready. Press Play to preview.';
    controls.play.onclick = () => player?.play(controls.select.value);
    controls.pause.onclick = () => player?.pause();
    controls.select.onchange = () => {
      if (!controls.pause.disabled) player?.play(controls.select.value);
    };
    controls.card.onchange = () => player?.setCard(controls.card.checked);
  } catch (error) {
    if (abort.signal.aborted) return;
    player?.dispose();
    releaseModel?.();
    controls.status.textContent = 'Could not open the 3D preview.';
    console.error('Model preview failed:', error);
  }
}

controls.dialog.addEventListener('close', () => {
  closePreview?.();
  closePreview = undefined;
});
window.addEventListener('pagehide', () => closePreview?.());

function assetCard(asset: Asset): HTMLElement {
  const card = document.createElement('article');
  card.className = 'overflow-hidden rounded-2xl border border-line bg-white';

  const preview = document.createElement('a');
  preview.className = 'block aspect-square bg-mint p-5';
  preview.href = `/${asset.path}`;
  preview.setAttribute('aria-label', `Open ${asset.name}`);
  if (asset.path.endsWith('.glb')) {
    preview.setAttribute('aria-label', `Preview ${asset.name} in 3D`);
    preview.addEventListener('click', (event) => {
      event.preventDefault();
      void openModel(asset);
    });
  }

  const image = document.createElement('img');
  image.src = `/${asset.preview ?? asset.path}`;
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
  download.download = asset.path.split('/').at(-1) ?? asset.id;
  download.className = 'inline-flex min-h-11 items-center underline-offset-4 hover:underline';
  download.textContent = 'Download';
  download.setAttribute('aria-label', `Download ${asset.name}`);
  if (asset.path.endsWith('.glb')) {
    const modelPreview = document.createElement('button');
    modelPreview.type = 'button';
    modelPreview.className = 'inline-flex min-h-11 items-center underline-offset-4 hover:underline';
    modelPreview.textContent = 'Preview 3D';
    modelPreview.setAttribute('aria-label', `Preview ${asset.name} in 3D`);
    modelPreview.addEventListener('click', () => void openModel(asset));
    actions.append(modelPreview, download);
  } else {
    actions.append(open, download);
  }
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
