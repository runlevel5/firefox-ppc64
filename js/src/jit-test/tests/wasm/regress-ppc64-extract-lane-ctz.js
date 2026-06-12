// |jit-test| --wasm-compiler=optimizing; skip-if: !wasmSimdEnabled()
//
// Regression test for a PPC64 i32x4.extract_lane canonicalization bug.
//
// ExtractLaneToGPR leaves the adjacent lane in the high 32 bits of the GPR for
// the unshifted lanes (0 and 2), so extractLaneInt32x4 must sign-extend its i32
// result (as the i8x16/i16x8 extracts do). Without that, a consumer that reads
// the full 64-bit register sees garbage in the high half. The POWER8 i32.ctz
// emulation is such a consumer: its 64-bit neg/and. zero-check disagrees with
// its 32-bit cntlzw, so ctz of a zero lane sitting next to a nonzero neighbour
// returned -1 instead of 32.
//
// The vector comes from memory (runtime, not constant-foldable) and is passed
// through a SIMD op so the extract is a genuine vector-register extract. Run
// under MOZ_PPC64_FORCE_POWER8=1 to exercise the emulated ctz path; in every
// other mode this is simply a correctness check.

const ins = wasmEvalText(`(module
  (memory (export "mem") 1)
  (func $v (result v128)
    ;; identity AND keeps the value in a vector register and forces a real
    ;; extractLaneInt32x4 rather than an extract-of-load fold.
    (v128.and (v128.load (i32.const 0)) (v128.const i32x4 -1 -1 -1 -1)))
  (func (export "ctz0") (result i32) (i32.ctz (i32x4.extract_lane 0 (call $v))))
  (func (export "ctz1") (result i32) (i32.ctz (i32x4.extract_lane 1 (call $v))))
  (func (export "ctz2") (result i32) (i32.ctz (i32x4.extract_lane 2 (call $v))))
  (func (export "ctz3") (result i32) (i32.ctz (i32x4.extract_lane 3 (call $v))))
  (func (export "sext0") (result i64) (i64.extend_i32_s (i32x4.extract_lane 0 (call $v))))
  (func (export "sext2") (result i64) (i64.extend_i32_s (i32x4.extract_lane 2 (call $v))))
)`).exports;

const mem = new Int32Array(ins.mem.buffer);
function setLanes(a, b, c, d) { mem[0] = a; mem[1] = b; mem[2] = c; mem[3] = d; }

// Each lane = 0 surrounded by nonzero neighbours: ctz must be 32, never -1.
setLanes(0, -1, -1, -1); assertEq(ins.ctz0(), 32);
setLanes(-1, 0, -1, -1); assertEq(ins.ctz1(), 32);
setLanes(-1, -1, 0, -1); assertEq(ins.ctz2(), 32);
setLanes(-1, -1, -1, 0); assertEq(ins.ctz3(), 32);

// Nonzero lanes: ctz of the lane value, regardless of neighbours.
setLanes(0x10, -1, 0x100000, -1);
assertEq(ins.ctz0(), 4);
assertEq(ins.ctz2(), 20);

// A negative lane must sign-extend correctly (the canonicalization is extsw).
setLanes(-2, 7, -3, 7);
assertEq(ins.sext0(), -2n);
assertEq(ins.sext2(), -3n);
