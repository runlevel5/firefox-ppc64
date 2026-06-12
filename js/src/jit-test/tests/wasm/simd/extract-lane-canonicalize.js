// |jit-test| --wasm-compiler=optimizing; skip-if: !wasmSimdEnabled()
//
// i32x4.extract_lane must canonicalize (sign-extend) its i32 result like the
// i8x16/i16x8 extracts do: a consumer that reads the full 64-bit register --
// e.g. an emulated i32.ctz or i64.extend_i32_s -- otherwise sees whatever the
// lane extraction left in the high half. Found on PPC64, where the extract
// leaves the adjacent lane there and the POWER8 ctz emulation then returned
// -1 instead of 32 for a zero lane next to a nonzero neighbour.

const ins = wasmEvalText(`(module
  (memory (export "mem") 1)
  (func $v (result v128)
    ;; identity AND keeps the value in a vector register and forces a real
    ;; extract_lane rather than an extract-of-load fold.
    (v128.and (v128.load (i32.const 0)) (v128.const i32x4 -1 -1 -1 -1)))
  (func (export "ctz0") (result i32) (i32.ctz (i32x4.extract_lane 0 (call $v))))
  (func (export "ctz1") (result i32) (i32.ctz (i32x4.extract_lane 1 (call $v))))
  (func (export "ctz2") (result i32) (i32.ctz (i32x4.extract_lane 2 (call $v))))
  (func (export "ctz3") (result i32) (i32.ctz (i32x4.extract_lane 3 (call $v))))
  (func (export "sext0") (result i64) (i64.extend_i32_s (i32x4.extract_lane 0 (call $v))))
  (func (export "sext2") (result i64) (i64.extend_i32_s (i32x4.extract_lane 2 (call $v))))
)`).exports;

// Wasm memory is little-endian, so write lanes with explicit byte order.
const mem = new DataView(ins.mem.buffer);
function setLanes(a, b, c, d) {
  mem.setInt32(0, a, true);
  mem.setInt32(4, b, true);
  mem.setInt32(8, c, true);
  mem.setInt32(12, d, true);
}

// Each lane = 0 surrounded by nonzero neighbours: ctz must be 32, never -1.
setLanes(0, -1, -1, -1); assertEq(ins.ctz0(), 32);
setLanes(-1, 0, -1, -1); assertEq(ins.ctz1(), 32);
setLanes(-1, -1, 0, -1); assertEq(ins.ctz2(), 32);
setLanes(-1, -1, -1, 0); assertEq(ins.ctz3(), 32);

// Nonzero lanes: ctz of the lane value, regardless of neighbours.
setLanes(0x10, -1, 0x100000, -1);
assertEq(ins.ctz0(), 4);
assertEq(ins.ctz2(), 20);

// A negative lane must sign-extend correctly.
setLanes(-2, 7, -3, 7);
assertEq(ins.sext0(), -2n);
assertEq(ins.sext2(), -3n);
