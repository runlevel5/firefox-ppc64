// |jit-test| skip-if: !largeArrayBufferSupported(); allow-oom
//
// Zero-extending the i32 index of i64.load and i64.store must not modify the
// index register: a later use may read the whole 64-bit register. PPC64 Ion
// zero-extended in place, so the following INT32_MIN / -1 did not trap.

let ins = wasmEvalText(`(module
  (memory (export "mem") 32769)
  (func (export "load") (param $p i32) (result i64)
    (i64.add (i64.load (local.get $p))
             (i64.extend_i32_s (i32.div_s (local.get $p) (i32.const -1)))))
  (func (export "store") (param $p i32) (result i32)
    (i64.store (local.get $p) (i64.const 1))
    (i32.div_s (local.get $p) (i32.const -1))))`);

assertEq(ins.exports.mem.buffer.byteLength > 0x80000000, true);
assertErrorMessage(() => ins.exports.load(0x80000000),
                   WebAssembly.RuntimeError, /integer overflow/);
assertErrorMessage(() => ins.exports.store(0x80000000),
                   WebAssembly.RuntimeError, /integer overflow/);
assertEq(ins.exports.store(8), -8);
assertEq(ins.exports.load(8), -7n);
