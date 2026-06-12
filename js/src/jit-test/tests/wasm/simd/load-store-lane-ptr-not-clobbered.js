// |jit-test| skip-if: !wasmSimdEnabled()
//
// v128.loadN_lane and v128.storeN_lane with a nonzero offset must not add the
// offset into the address operand: the operand may still be live afterwards.
// PPC64 Ion passed the address register as its own offset scratch and
// returned 24 for f(8).

for (let size of [8, 16, 32, 64]) {
  let ins = wasmEvalText(`(module (memory 1)
    (func (export "load") (param $p i32) (result i32)
      (v128.store (i32.const 64)
        (v128.load${size}_lane offset=16 0 (local.get $p) (v128.const i64x2 0 0)))
      (local.get $p))
    (func (export "store") (param $p i32) (result i32)
      (v128.store${size}_lane offset=16 0 (local.get $p) (v128.const i64x2 1 2))
      (local.get $p)))`);
  assertEq(ins.exports.load(8), 8);
  assertEq(ins.exports.store(8), 8);
}
