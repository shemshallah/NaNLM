// SPDX-License-Identifier: MIT
const encoder = new TextEncoder();

export function fnv1a(bytes) {
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export function frame(payload, type = 1) {
  const body = encoder.encode(JSON.stringify(payload));
  const header = new Uint8Array(5);
  header[0] = type;
  new DataView(header.buffer).setUint32(1, body.length, false);
  return new Uint8Array([...header, ...body]);
}

export function unfold(buffer) {
  const view = new DataView(buffer);
  const type = view.getUint8(0);
  const length = view.getUint32(1, false);
  if (length + 5 !== buffer.byteLength) throw new Error("bad frame");
  const text = new TextDecoder().decode(new Uint8Array(buffer, 5));
  return { type, payload: JSON.parse(text), hash: fnv1a(new Uint8Array(buffer)) };
}
