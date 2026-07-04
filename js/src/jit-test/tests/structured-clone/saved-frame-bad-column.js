function savedFrameWithColumn(column) {
  const scope = {scope: "DifferentProcess"};
  const clone = serialize(null, undefined, scope);
  // The clone buffer is a sequence of little-endian 64-bit words.
  const words = [
    0xfff1000000000002n,
    0xffff0016ffff0018n,
    0xffff000200000000n,
    0xffff000480000000n,
    0xffff000300000001n,
    0xffff000300000000n | BigInt(column),
    0xffff000000000000n,
    0xffff000000000000n,
    0xffff000000000000n,
    0xffff001300000000n
  ];
  const view = new DataView(new ArrayBuffer(words.length * 8));
  words.forEach((word, i) => view.setBigUint64(i * 8, word, true));
  clone.clonebuffer = view.buffer;
  return deserialize(clone, scope);
}

// Zero, out-of-range columns and out-of-range wasm function indexes throw.
for (const column of [0, 0x40000000, 0x7fffffff, 0xc0000000, 0xc0000001,
                      0xffffffff]) {
  let ex = null;
  try {
    savedFrameWithColumn(column);
  } catch (e) {
    ex = e;
  }
  assertEq(ex.message.includes("invalid saved frame column"), true);
}

// Boundary values for regular columns are accepted.
for (const column of [1, 0x3fffffff]) {
  const frame = savedFrameWithColumn(column);
  assertEq(frame.column, column);
  assertEq(frame.toString(), `@:1:${column}\n`);
}

// Boundary values for wasm function indexes are accepted.
for (const funcIndex of [0, 0x3fffffff]) {
  const frame = savedFrameWithColumn(0x80000000 + funcIndex);
  assertEq(frame.column, 1);
  assertEq(frame.toString().includes(`wasm-function[${funcIndex}]`), true);
}
