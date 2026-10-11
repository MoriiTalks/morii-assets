import { crc32 } from 'node:zlib';

const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const bitDepths: Readonly<Record<number, readonly number[]>> = {
  0: [1, 2, 4, 8, 16],
  2: [8, 16],
  3: [1, 2, 4, 8],
  4: [8, 16],
  6: [8, 16],
};

function validHeader(data: Buffer): boolean {
  if (data.length !== 13) return false;
  const width = data.readUInt32BE(0);
  const height = data.readUInt32BE(4);
  return (
    width > 0 &&
    width <= 0x7fffffff &&
    height > 0 &&
    height <= 0x7fffffff &&
    bitDepths[data.readUInt8(9)]?.includes(data.readUInt8(8)) === true &&
    data.readUInt8(10) === 0 &&
    data.readUInt8(11) === 0 &&
    data.readUInt8(12) <= 1
  );
}

// Checks container integrity without decompressing pixels or replacing an image decoder.
export function validatePNG(data: Buffer, id: string): void {
  if (!data.subarray(0, 8).equals(signature)) throw new Error(`Export is not a PNG: ${id}`);
  let offset = 8;
  let colorType = 0;
  let depth = 0;
  let palette = false;
  let dataBytes = 0;
  let seenData = false;
  let dataEnded = false;

  while (offset < data.length) {
    if (data.length - offset < 12) throw new Error(`Truncated PNG chunk: ${id}`);
    const length = data.readUInt32BE(offset);
    const end = offset + length + 12;
    if (length > 0x7fffffff || end > data.length) throw new Error(`Truncated PNG chunk: ${id}`);
    const type = data.toString('latin1', offset + 4, offset + 8);
    const body = data.subarray(offset + 8, end - 4);
    if (!/^[A-Za-z]{2}[A-Z][A-Za-z]$/.test(type)) {
      throw new Error(`Invalid PNG chunk type: ${id}`);
    }
    if (crc32(data.subarray(offset + 4, end - 4)) !== data.readUInt32BE(end - 4)) {
      throw new Error(`Invalid PNG checksum: ${id}`);
    }
    if (offset === 8 && type !== 'IHDR') throw new Error(`Missing PNG header: ${id}`);
    if (seenData && type !== 'IDAT') dataEnded = true;

    switch (type) {
      case 'IHDR':
        if (offset !== 8 || !validHeader(body)) throw new Error(`Invalid PNG header: ${id}`);
        colorType = body.readUInt8(9);
        depth = body.readUInt8(8);
        break;
      case 'PLTE':
        if (
          palette ||
          seenData ||
          colorType === 0 ||
          colorType === 4 ||
          length === 0 ||
          length > 768 ||
          length % 3 !== 0 ||
          (colorType === 3 && length / 3 > 2 ** depth)
        ) {
          throw new Error(`Invalid PNG palette: ${id}`);
        }
        palette = true;
        break;
      case 'IDAT':
        if (dataEnded || (colorType === 3 && !palette))
          throw new Error(`Invalid PNG image data order: ${id}`);
        seenData = true;
        dataBytes += length;
        break;
      case 'IEND':
        if (length !== 0 || !seenData || dataBytes === 0 || end !== data.length) {
          throw new Error(`Invalid PNG ending: ${id}`);
        }
        return;
      default:
        if ((data.readUInt8(offset + 4) & 0x20) === 0) {
          throw new Error(`Unsupported PNG critical chunk: ${id}`);
        }
    }
    offset = end;
  }
  throw new Error(`Missing PNG ending: ${id}`);
}
