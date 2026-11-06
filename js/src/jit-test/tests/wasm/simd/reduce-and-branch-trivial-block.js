// |jit-test| --wasm-compiler=optimizing; skip-if: !wasmSimdEnabled()
//
// A fused SIMD reduce-and-branch (test on any_true/all_true/bitmask) where a
// branch successor is a trivial goto-only block: the arms merge no values, so
// the join has no phis and critical-edge splitting leaves a block containing
// only a goto. Codegen must resolve such branch targets via
// skipTrivialBlocks(); resolving them directly crashed the PPC64 Ion backend
// at compile time. Reduced from grantkot.com/poly with wasm-reduce.

const ins = wasmEvalText(`(module
  (memory (export "mem") 1)
  (func (export "any")
    (local $v v128)
    (local.set $v (v128.load (i32.const 0)))
    (if (v128.any_true (local.get $v))
      (then (v128.store (i32.const 16) (v128.const i32x4 1 1 1 1)))))
  (func (export "all")
    (local $v v128)
    (local.set $v (v128.load (i32.const 0)))
    (if (i32x4.all_true (local.get $v))
      (then (v128.store (i32.const 16) (v128.const i32x4 1 1 1 1)))))
  (func (export "mask")
    ;; br_if form: the marker is stored on the fallthrough, i.e. when the
    ;; bitmask is zero.
    (local $v v128)
    (local.set $v (v128.load (i32.const 0)))
    (block $b
      (br_if $b (i16x8.bitmask (local.get $v)))
      (v128.store (i32.const 16) (v128.const i32x4 1 1 1 1)))))`).exports;

const mem = new Uint8Array(ins.mem.buffer);
function ran(f) {
  mem.fill(0, 16, 32);
  f();
  return mem[16] != 0;
}

// All-zero input: any/all false, bitmask zero.
mem.fill(0, 0, 16);
assertEq(ran(ins.any), false);
assertEq(ran(ins.all), false);
assertEq(ran(ins.mask), true);

// All-ones input: any/all true, bitmask nonzero.
mem.fill(0xff, 0, 16);
assertEq(ran(ins.any), true);
assertEq(ran(ins.all), true);
assertEq(ran(ins.mask), false);

// Single low byte set: any true, i32x4 all_true false, i16x8 bitmask zero.
mem.fill(0, 0, 16);
mem[0] = 1;
assertEq(ran(ins.any), true);
assertEq(ran(ins.all), false);
assertEq(ran(ins.mask), true);
