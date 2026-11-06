// Loads and stores on an i64-indexed memory with immediate offsets, so the
// offset is added to a 64-bit pointer register rather than a zero-extended
// i32. Covers every scalar width, both widths of float, and SIMD lane
// accesses when SIMD is available.

function instantiate(simd) {
  const lanes = simd ? `
  (func (export "lane_ld64") (param i64) (result i64)
    (i64x2.extract_lane 1 (v128.load64_lane offset=24 1 (local.get 0) (v128.const i64x2 0 0))))
  (func (export "lane_st32") (param i64 i32)
    (v128.store32_lane offset=40 2 (local.get 0) (i32x4.splat (local.get 1))))` : "";
  return new WebAssembly.Instance(new WebAssembly.Module(wasmTextToBinary(`(module
  (memory (export "mem") i64 1)
  (func (export "st8") (param i64 i32) (i32.store8 offset=1 (local.get 0) (local.get 1)))
  (func (export "st16") (param i64 i32) (i32.store16 offset=2 (local.get 0) (local.get 1)))
  (func (export "st32") (param i64 i32) (i32.store offset=4 (local.get 0) (local.get 1)))
  (func (export "st64") (param i64 i64) (i64.store offset=8 (local.get 0) (local.get 1)))
  (func (export "stf32") (param i64 f32) (f32.store offset=16 (local.get 0) (local.get 1)))
  (func (export "stf64") (param i64 f64) (f64.store offset=24 (local.get 0) (local.get 1)))
  (func (export "ld8u") (param i64) (result i32) (i32.load8_u offset=1 (local.get 0)))
  (func (export "ld16s") (param i64) (result i32) (i32.load16_s offset=2 (local.get 0)))
  (func (export "ld32") (param i64) (result i32) (i32.load offset=4 (local.get 0)))
  (func (export "ld64") (param i64) (result i64) (i64.load offset=8 (local.get 0)))
  (func (export "ldf32") (param i64) (result f32) (f32.load offset=16 (local.get 0)))
  (func (export "ldf64") (param i64) (result f64) (f64.load offset=24 (local.get 0)))
  (func (export "ld32_far") (param i64) (result i32) (i32.load offset=65532 (local.get 0)))
  (func (export "st32_far") (param i64 i32) (i32.store offset=65532 (local.get 0) (local.get 1)))
  ${lanes})`)), {}).exports;
}

const simd = wasmSimdEnabled();
const m = instantiate(simd);
const bytes = new Uint8Array(m.mem.buffer);

for (let round = 0; round < 3; round++) {
  const base = 32n + BigInt(round) * 64n;
  m.st8(base, 0xab);
  m.st16(base, -2);
  m.st32(base, 0x12345678);
  m.st64(base, -0x0102030405060708n);
  m.stf32(base, 1.5);
  m.stf64(base, -2.25);
  assertEq(m.ld8u(base), 0xab);
  assertEq(m.ld16s(base), -2);
  assertEq(m.ld32(base), 0x12345678);
  assertEq(m.ld64(base), -0x0102030405060708n);
  assertEq(m.ldf32(base), 1.5);
  assertEq(m.ldf64(base), -2.25);
  // The stores landed at base + offset, not at base.
  assertEq(bytes[Number(base) + 1], 0xab);
  assertEq(bytes[Number(base) + 4], 0x78);
  assertEq(bytes[Number(base) + 8], 0xf8);
  m.st32_far(0n, 0x7f00ff01);
  assertEq(m.ld32_far(0n), 0x7f00ff01);
  assertEq(bytes[65532], 0x01);
  if (simd) {
    m.st64(base, 0x1122334455667788n);
    assertEq(m.lane_ld64(base - 16n), 0x1122334455667788n);
    m.lane_st32(base, 0x0badf00d);
    assertEq(m.ld32(base + 36n), 0x0badf00d);
  }
}
// Offsets past the end trap rather than wrap.
function assertTraps(f) {
  let trapped = false;
  try {
    f();
  } catch (e) {
    trapped = e instanceof WebAssembly.RuntimeError;
  }
  assertEq(trapped, true);
}
assertTraps(() => m.ld32_far(8n));
assertTraps(() => m.st64(65528n, 1n));
