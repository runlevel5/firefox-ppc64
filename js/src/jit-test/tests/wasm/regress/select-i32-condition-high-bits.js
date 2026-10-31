// |jit-test| --wasm-compiler=optimizing; skip-if: !wasmSimdEnabled()
//
// `select` with an i32 condition must test only the low 32 bits of the
// condition register. A full-width 64-bit test misreads a condition that is
// zero in its low 32 bits but has garbage above, as can happen under register
// pressure; the SIMD shuffle/bitselect/swizzle chain below supplies that
// pressure. Found by fuzzing on PPC64 (cmpdi where cmpwi was needed).
//
// Here the condition `$x3` is 0, so select must return -952809828.

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
