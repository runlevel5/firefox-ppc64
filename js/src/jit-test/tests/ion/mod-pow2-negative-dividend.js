// Regression test for a PPC64 Ion miscompile of integer modulo by a constant
// power of two (e.g. 65536) with a negative dividend.
//
// lowerModI routes `x % 2^n` to LModPowTwoI, whose codegen tested the sign of
// the dividend with branchPtr (a 64-bit compare). When the int32 dividend was
// held zero-extended in its register, the 64-bit test misclassified a negative
// value as non-negative and took the unmasked positive path, returning
// `x & (2^n - 1)` instead of the correct (negative) `x % 2^n`. Fixed by using a
// 32-bit sign test (branch32).
//
// The reference uses a non-constant divisor, which lowers to the divide-based
// modulo path (LModI), independent of LModPowTwoI.

function refmod(x, d) {
  return (x % d) | 0;
}

function mod256(x) { return (x % 256) | 0; }
function mod1024(x) { return (x % 1024) | 0; }
function mod4096(x) { return (x % 4096) | 0; }
function mod65536(x) { return (x % 65536) | 0; }
function mod1048576(x) { return (x % 1048576) | 0; }
function mod1073741824(x) { return (x % 1073741824) | 0; }

const cases = [
  [mod256, 256],
  [mod1024, 1024],
  [mod4096, 4096],
  [mod65536, 65536],
  [mod1048576, 1048576],
  [mod1073741824, 1073741824],
];

// Heavy on negative dividends (the broken path), plus boundary values.
const inputs = [];
for (let i = 1; i <= 64; i++) {
  inputs.push(-Math.imul(i, 2654435761) | 0);
  inputs.push(-(i * 168));
  inputs.push(-(i * 70001));
  inputs.push(Math.imul(i, 40503) | 0);
}
inputs.push(0, -1, 1, -168, -65535, -65536, -65537, 168,
            0x7fffffff, -0x80000000, -0x7fffffff);

for (let iter = 0; iter < 100; iter++) {
  for (const [fn, d] of cases) {
    for (const x of inputs) {
      assertEq(fn(x), refmod(x, d));
    }
  }
}

// Register-pressure variant: a negative dividend produced at runtime
// (float->int) with many live locals, mirroring the shape that exposed the bug.
function pressure(seed) {
  let v0 = seed, v1 = seed + 1, v2 = seed + 2, v3 = seed + 3, v4 = seed + 4;
  let v5 = seed + 5, v6 = seed + 6, v7 = seed + 7, v8 = seed + 8, v9 = seed + 9;
  let v10 = seed + 10, v11 = seed + 11, v12 = seed + 12, v13 = seed + 13;
  let d0 = seed * 0.5, d1 = seed * 1.5, d2 = -seed * 2.5;
  const neg = (Math.fround(-(Math.abs(seed) + 0.7)) | 0);
  const r = (neg % 65536) | 0;
  const live = (v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7 ^ v8 ^ v9 ^
                v10 ^ v11 ^ v12 ^ v13 ^ (d0 | 0) ^ (d1 | 0) ^ (d2 | 0)) & 0;
  return r + live;
}
for (let iter = 0; iter < 100; iter++) {
  for (let s = 1; s <= 200; s++) {
    const expect = ((Math.fround(-(s + 0.7)) | 0) % 65536) | 0;
    assertEq(pressure(s), expect);
  }
}
