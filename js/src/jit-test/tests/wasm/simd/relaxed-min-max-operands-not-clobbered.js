// |jit-test| --setpref=wasm_relaxed_simd=true; skip-if: !wasmRelaxedSimdEnabled()
//
// f32x4/f64x2.relaxed_min and relaxed_max must leave both operands intact:
// they may still be live afterwards. PPC64 Ion computed the result into lhs
// and then copied it to the destination.

for (let [ty, Lanes] of [["f32x4", Float32Array], ["f64x2", Float64Array]]) {
  for (let op of ["relaxed_min", "relaxed_max"]) {
    let ins = wasmEvalText(`(module (memory (export "mem") 1)
      (func (export "f")
        (local $a v128) (local $b v128)
        (local.set $a (v128.load (i32.const 0)))
        (local.set $b (v128.load (i32.const 16)))
        (v128.store (i32.const 32) (${ty}.${op} (local.get $a) (local.get $b)))
        (v128.store (i32.const 48) (local.get $a))
        (v128.store (i32.const 64) (local.get $b))))`);
    let m = new Lanes(ins.exports.mem.buffer);
    let n = 16 / Lanes.BYTES_PER_ELEMENT;
    let a = [1, 2, 3, 4].slice(0, n);
    let b = [0, 5, 0, 5].slice(0, n);
    m.set(a, 0);
    m.set(b, n);
    ins.exports.f();
    let f = op == "relaxed_min" ? Math.min : Math.max;
    assertDeepEq(Array.from(m.subarray(2 * n, 3 * n)), a.map((x, i) => f(x, b[i])));
    assertDeepEq(Array.from(m.subarray(3 * n, 4 * n)), a);
    assertDeepEq(Array.from(m.subarray(4 * n, 5 * n)), b);
  }
}
