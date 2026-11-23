// Integer modulo by a constant of the form 2^n - 1 (e.g. 65535) may lower to
// a mask-and-sum reduction (LModMaskI) instead of a divide. The mask must be
// materialized without truncation even when it does not fit a signed 16-bit
// immediate; a PPC64 lowering truncated 0xFFFF to -1 and corrupted the
// reduction for divisors >= 65535. The reference uses a non-constant divisor,
// which takes the divide-based path (LModI) independent of the codegen under
// test.

function refmod(x, d) {
  // d is not a constant here -> divide-based modulo, not LModMaskI.
  return (x % d) | 0;
}

// One function per constant divisor so the divisor is a literal and the
// LModMaskI path is selected.
function mod255(x) { return (x % 255) | 0; }
function mod32767(x) { return (x % 32767) | 0; }
function mod65535(x) { return (x % 65535) | 0; }
function mod131071(x) { return (x % 131071) | 0; }
function mod1048575(x) { return (x % 1048575) | 0; }

const cases = [
  [mod255, 255],
  [mod32767, 32767],
  [mod65535, 65535],
  [mod131071, 131071],
  [mod1048575, 1048575],
];

// Inputs spanning small values, values with bits above the mask width
// (so the multi-digit reduction is exercised), and negatives.
const inputs = [];
for (let i = 0; i < 64; i++) {
  inputs.push(Math.imul(i, 2654435761) | 0);
  inputs.push((i * 65535 + i) | 0);
  inputs.push((i * 131071 - 7) | 0);
  inputs.push(-Math.imul(i, 40503) | 0);
}
inputs.push(0, 1, -1, 65534, 65535, 65536, 0x7fffffff, -0x80000000);

// Warm up through the tiers, then assert each constant-divisor result
// matches the divide-based reference.
for (let iter = 0; iter < 100; iter++) {
  for (const [fn, d] of cases) {
    for (const x of inputs) {
      assertEq(fn(x), refmod(x, d));
    }
  }
}

// Register-pressure variant: mirrors the shape that exposed the bug (many
// live locals forcing the mask materialization to interact with spills).
function pressure(buf, i) {
  let v0 = i, v1 = i + 1, v2 = i + 2, v3 = i + 3, v4 = i + 4, v5 = i + 5;
  let v6 = i + 6, v7 = i + 7, v8 = i + 8, v9 = i + 9, v10 = i + 10, v11 = i + 11;
  let v12 = i + 12, v13 = i + 13, v14 = i + 14, v15 = i + 15;
  const r = (buf[i & 63] % 65535) | 0;
  // Keep every local live to the return without altering r.
  const live = (v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7 ^
                v8 ^ v9 ^ v10 ^ v11 ^ v12 ^ v13 ^ v14 ^ v15) & 0;
  return r + live;
}

const buf = new Int32Array(64);
for (let i = 0; i < buf.length; i++) {
  buf[i] = Math.imul(i, 2654435761) | 0;
}
for (let iter = 0; iter < 300; iter++) {
  for (let i = 0; i < 64; i++) {
    assertEq(pressure(buf, i), refmod(buf[i & 63], 65535));
  }
}
