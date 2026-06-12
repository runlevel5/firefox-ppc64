// i32.div_u and i32.rem_u must not modify their operand registers: a later
// use of the same value may read the whole 64-bit register. PPC64 Ion
// zero-extended both operands in place, so the following INT32_MIN / -1 saw a
// positive 64-bit value and did not trap.

for (let op of ["div_u", "rem_u"]) {
  let ins = wasmEvalText(`(module
    (func (export "lhs") (param $x i32) (result i32)
      (i32.add (i32.${op} (local.get $x) (i32.const 3))
               (i32.div_s (local.get $x) (i32.const -1))))
    (func (export "rhs") (param $x i32) (param $y i32) (result i32)
      (i32.add (i32.${op} (local.get $x) (local.get $y))
               (i32.div_s (local.get $y) (i32.const -1)))))`);
  assertErrorMessage(() => ins.exports.lhs(-0x80000000),
                     WebAssembly.RuntimeError, /integer overflow/);
  assertErrorMessage(() => ins.exports.rhs(7, -0x80000000),
                     WebAssembly.RuntimeError, /integer overflow/);
  assertEq(ins.exports.lhs(9), (op == "div_u" ? 3 : 0) - 9);
  assertEq(ins.exports.rhs(9, 2), (op == "div_u" ? 4 : 1) - 2);
}
