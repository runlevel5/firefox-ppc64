// |jit-test| --wasm-compiler=optimizing; skip-if: !wasmSimdEnabled()
//
// Regression test for a PPC64 wasm Ion miscompile of `select` with a 32-bit
// condition. visitWasmSelect tested the i32 condition with a 64-bit compare
// (cmpdi / branchTestPtr). When the condition was zero in its low 32 bits but
// had garbage in the high 32 bits (as can happen under register pressure), the
// 64-bit test read it as non-zero and select returned the wrong operand.
//
// Here the condition `$x3` is 0; `select($x8, -952809828, $x3)` must therefore
// return -952809828. The surrounding SIMD shuffle/bitselect/swizzle chain
// supplies the v128 register pressure that exposed the bug.

const wat = `(module (func (export "f") (result i64)
  (local $x3 i32)(local $x7 i32)(local $x8 i32)
  (local $w0 v128)(local $w1 v128)(local $w2 v128)(local $w3 v128)
  (local $w4 v128)(local $w5 v128)(local $w6 v128)(local $w7 v128)
  (local.set $w0 (v128.const i32x4 1708443454 1532218695 2107423610 -1265775005))
  (local.set $w2 (v128.const i32x4 -752312355 -625530572 -844666500 832036408))
  (local.set $w7 (v128.const i32x4 115003496 -970441117 -43225935 1874128204))
  (local.set $w4 (i8x16.shuffle 15 18 13 2 6 22 20 8 19 10 12 8 11 5 6 28 (local.get $w7) (local.get $w3)))
  (local.set $w6 (v128.bitselect (local.get $w4) (local.get $w0) (local.get $w7)))
  (local.set $w1 (v128.const i32x4 -1635025264 -629784132 1517869852 1651771825))
  (local.set $w7 (v128.bitselect (local.get $w6) (local.get $w2) (local.get $w2)))
  (local.set $w6 (i8x16.swizzle (local.get $w1) (local.get $w7)))
  (local.set $x3 (i32x4.extract_lane 2 (local.get $w6)))
  (local.set $x7 (select (local.get $x8) (i32.const -952809828) (local.get $x3)))
  (i64.extend_i32_s (local.get $x7))))`;

const ins = new WebAssembly.Instance(new WebAssembly.Module(wasmTextToBinary(wat)));
assertEq(ins.exports.f(), -952809828n);
