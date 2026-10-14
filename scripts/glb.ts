export interface GLBSummary {
  readonly triangles: number;
  readonly clips: readonly string[];
  readonly joints: number;
  readonly morphs: readonly string[];
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function entries(value: unknown, label: string): readonly Record<string, unknown>[] {
  if (!Array.isArray(value)) throw new Error(`Invalid GLB ${label}`);
  const values: readonly unknown[] = value;
  return values.map((entry) => {
    if (!record(entry)) throw new Error(`Invalid GLB ${label} entry`);
    return entry;
  });
}

function integer(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new Error('Invalid GLB integer');
  }
  return value;
}

function readAnimationAccessor(
  data: Buffer,
  binaryStart: number,
  binaryLength: number,
  accessor: Record<string, unknown> | undefined,
  views: readonly Record<string, unknown>[],
): readonly (readonly number[])[] {
  if (accessor?.['componentType'] !== 5126) {
    throw new Error('GLB animations require float accessors');
  }
  const type = accessor['type'];
  const components = type === 'SCALAR' ? 1 : type === 'VEC3' ? 3 : type === 'VEC4' ? 4 : 0;
  if (components === 0) throw new Error('Invalid GLB animation accessor type');
  const view = views[integer(accessor['bufferView'])];
  if (view?.['buffer'] !== 0) throw new Error('Missing GLB animation buffer view');
  const count = integer(accessor['count']);
  const accessorOffset = integer(accessor['byteOffset'] ?? 0);
  const viewOffset = integer(view['byteOffset'] ?? 0);
  const viewLength = integer(view['byteLength']);
  const stride = integer(view['byteStride'] ?? components * 4);
  if (
    count === 0 ||
    stride < components * 4 ||
    accessorOffset + (count - 1) * stride + components * 4 > viewLength ||
    viewOffset + viewLength > binaryLength
  ) {
    throw new Error('GLB animation accessor exceeds its buffer');
  }
  return Array.from({ length: count }, (_, row) =>
    Array.from({ length: components }, (_, component) => {
      const value = data.readFloatLE(
        binaryStart + viewOffset + accessorOffset + row * stride + component * 4,
      );
      if (!Number.isFinite(value)) throw new Error('Invalid GLB animation value');
      return value;
    }),
  );
}

// Keep runtime models self-contained, an asset must not fetch files from unknown hosts.
export function validateGLB(data: Buffer, id: string): GLBSummary {
  if (data.length < 28 || data.readUInt32LE(0) !== 0x46546c67) {
    throw new Error(`Export is not a GLB: ${id}`);
  }
  if (data.readUInt32LE(4) !== 2 || data.readUInt32LE(8) !== data.length) {
    throw new Error(`Invalid GLB header: ${id}`);
  }
  if (data.length > 2_000_000) throw new Error(`GLB exceeds the 2 MB runtime budget: ${id}`);
  const jsonLength = data.readUInt32LE(12);
  const binaryOffset = 20 + jsonLength;
  if (
    data.readUInt32LE(16) !== 0x4e4f534a ||
    jsonLength % 4 !== 0 ||
    binaryOffset + 8 > data.length
  ) {
    throw new Error(`Invalid GLB JSON chunk: ${id}`);
  }
  const document: unknown = JSON.parse(data.subarray(20, binaryOffset).toString('utf8'));
  if (!record(document) || !record(document['asset']) || document['asset']['version'] !== '2.0') {
    throw new Error(`Invalid glTF version: ${id}`);
  }
  const binaryLength = data.readUInt32LE(binaryOffset);
  if (
    data.readUInt32LE(binaryOffset + 4) !== 0x004e4942 ||
    binaryOffset + 8 + binaryLength !== data.length ||
    binaryLength % 4 !== 0
  ) {
    throw new Error(`Invalid GLB binary chunk: ${id}`);
  }
  const buffers = entries(document['buffers'], 'buffers');
  if (buffers.length !== 1 || buffers.some((buffer) => buffer['uri'] !== undefined)) {
    throw new Error(`GLB must contain one embedded buffer: ${id}`);
  }
  const buffer = buffers[0];
  if (buffer === undefined || integer(buffer['byteLength']) > binaryLength) {
    throw new Error(`GLB buffer exceeds its chunk: ${id}`);
  }
  if (document['images'] !== undefined && entries(document['images'], 'images').length !== 0) {
    throw new Error(`Morii uses material colors instead of textures: ${id}`);
  }
  const accessors = entries(document['accessors'], 'accessors');
  const views = entries(document['bufferViews'], 'buffer views');
  const nodes = entries(document['nodes'], 'nodes');
  const root = nodes.findIndex((node) => node['name'] === 'Root');
  if (root < 0) throw new Error(`Missing GLB root bone: ${id}`);
  let triangles = 0;
  const morphs = new Set<string>();
  for (const mesh of entries(document['meshes'], 'meshes')) {
    if (mesh['weights'] !== undefined) {
      if (!Array.isArray(mesh['weights']) || mesh['weights'].some((weight) => weight !== 0)) {
        throw new Error(`GLB morphs must start neutral: ${id}`);
      }
    }
    if (record(mesh['extras']) && Array.isArray(mesh['extras']['targetNames'])) {
      const names: readonly unknown[] = mesh['extras']['targetNames'];
      for (const name of names) {
        if (typeof name !== 'string') throw new Error(`Invalid GLB morph name: ${id}`);
        morphs.add(name);
      }
    }
    for (const primitive of entries(mesh['primitives'], 'primitives')) {
      if (primitive['mode'] !== undefined && primitive['mode'] !== 4) {
        throw new Error(`GLB must use triangle primitives: ${id}`);
      }
      const accessor = accessors[integer(primitive['indices'])];
      if (accessor === undefined) throw new Error(`Missing GLB index accessor: ${id}`);
      const count = integer(accessor['count']);
      if (count % 3 !== 0) throw new Error(`Invalid GLB triangle count: ${id}`);
      triangles += count / 3;
    }
  }
  if (triangles === 0 || triangles > 40_000) throw new Error(`GLB mesh budget exceeded: ${id}`);
  const restChannels = new Map<string, readonly number[]>();
  const clips = entries(document['animations'], 'animations').map((animation) => {
    const name = animation['name'];
    if (typeof name !== 'string' || !/^[a-z][a-z_]*$/.test(name)) {
      throw new Error(`Invalid GLB animation name: ${id}`);
    }
    const samplers = entries(animation['samplers'], 'animation samplers');
    for (const channel of entries(animation['channels'], 'animation channels')) {
      const target = channel['target'];
      if (!record(target)) throw new Error(`Invalid GLB animation target: ${id}`);
      const node = integer(target['node']);
      const path = target['path'];
      if (
        nodes[node] === undefined ||
        !['translation', 'rotation', 'scale'].includes(String(path))
      ) {
        throw new Error(`Unsupported GLB animation channel: ${id}`);
      }
      const sampler = samplers[integer(channel['sampler'])];
      const interpolation = sampler?.['interpolation'] ?? 'LINEAR';
      if (sampler === undefined || (interpolation !== 'LINEAR' && interpolation !== 'STEP')) {
        throw new Error(`Unsupported GLB animation sampler: ${id}`);
      }
      const read = (index: unknown): readonly (readonly number[])[] =>
        readAnimationAccessor(
          data,
          binaryOffset + 8,
          binaryLength,
          accessors[integer(index)],
          views,
        );
      const times = read(sampler['input']);
      const values = read(sampler['output']);
      const first = values[0];
      const last = values.at(-1);
      const endTime = times.at(-1)?.[0];
      if (
        first === undefined ||
        last === undefined ||
        times.length !== values.length ||
        times[0]?.[0] !== 0 ||
        endTime === undefined ||
        endTime <= 0 ||
        endTime > 4.5
      ) {
        throw new Error(`Invalid GLB animation timing: ${id}`);
      }
      const key = `${node.toString()}:${String(path)}`;
      if (
        sampler['interpolation'] === 'STEP' &&
        values.some((row) =>
          row.some(
            (value, component) => Math.abs(value - (first[component] ?? Infinity)) > 0.00001,
          ),
        )
      ) {
        throw new Error(`GLB moving channels must interpolate smoothly: ${id}`);
      }
      const rest = restChannels.get(key) ?? first;
      for (const [component, value] of first.entries()) {
        if (
          Math.abs(value - (last[component] ?? Infinity)) > 0.00001 ||
          Math.abs(value - (rest[component] ?? Infinity)) > 0.00001
        ) {
          throw new Error(`GLB clips must share neutral boundaries: ${id}`);
        }
      }
      restChannels.set(key, first);
      if (node === root && path === 'translation') {
        for (const row of values) {
          if (row.some((value, index) => Math.abs(value - (first[index] ?? 0)) > 0.321)) {
            throw new Error(`GLB travel exceeds its stage: ${id}`);
          }
        }
      }
    }
    return name;
  });
  if (new Set(clips).size !== clips.length) throw new Error(`Duplicate GLB clip: ${id}`);
  const skins = entries(document['skins'], 'skins');
  const skin = skins[0];
  if (skins.length !== 1 || skin === undefined || !Array.isArray(skin['joints'])) {
    throw new Error(`Morii requires one skeletal rig: ${id}`);
  }
  return { triangles, clips, joints: skin['joints'].length, morphs: [...morphs] };
}
