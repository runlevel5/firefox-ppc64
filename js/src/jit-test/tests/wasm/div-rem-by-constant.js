// Division and remainder by every kind of constant divisor, checked against a
// BigInt reference. Exercises the magic-number, power-of-two, negative
// power-of-two, zero and -1 lowerings on every compiler.

const i32Divisors = [0, 1, -1, 2, -2, 3, -3, 5, 6, 7, -7, 8, -8, 10, 100, 641,
                     1023, -1023, 1024, 65536, 0x7fffffff, -0x7fffffff,
                     -0x80000000, 0x80000000, 0x80000001, 0xfffffffd,
                     0xffffffff];
const i32Dividends = [0, 1, -1, 2, -2, 3, -3, 6, 7, -7, 15, 100, 101, -101,
                      1000, 12345, -12345, 0x3fffffff, 0x40000000, 0x7ffffffe,
                      0x7fffffff, -0x7fffffff, -0x80000000, 0x80000000,
                      0x80000001, 0xfffffffe, 0xffffffff];
const i64Divisors = [0n, 1n, -1n, 2n, -2n, 3n, -3n, 5n, 7n, -7n, 10n, 100n,
                     641n, 1n << 31n, -(1n << 31n), 1n << 32n, (1n << 32n) + 1n,
                     1n << 62n, 0x7fffffffffffffffn, -0x7fffffffffffffffn,
                     -0x8000000000000000n, 0x8000000000000000n,
                     0x8000000000000001n, 0xfffffffffffffffdn,
                     0xffffffffffffffffn];
const i64Dividends = [0n, 1n, -1n, 2n, -2n, 3n, 7n, -7n, 100n, 101n, -101n,
                      12345n, -12345n, 0x7fffffffn, 0x80000000n, 0xffffffffn,
                      1n << 32n, (1n << 32n) + 1n, 0x3fffffffffffffffn,
                      0x7ffffffffffffffen, 0x7fffffffffffffffn,
                      -0x7fffffffffffffffn, -0x8000000000000000n,
                      0x8000000000000000n, 0x8000000000000001n,
                      0xfffffffffffffffen, 0xffffffffffffffffn];

function name(op, d) {
  return `${op}_${d < 0 ? "m" : ""}${d < 0 ? -d : d}`;
}

function buildModule(ty, divisors) {
  let text = "";
  for (let d of divisors) {
    for (let op of ["div_s", "rem_s", "div_u", "rem_u"]) {
      text += `(func (export "${name(op, d)}") (param $x ${ty}) (result ${ty})
                 (${ty}.${op} (local.get $x) (${ty}.const ${d})))\n`;
    }
  }
  return wasmEvalText(`(module ${text})`).exports;
}

// Reference semantics on BigInts: n is the bit width, both operands are
// reinterpreted as signed or unsigned, and the result is wrapped back.
function reference(op, bits, x, d) {
  let signed = op.endsWith("_s");
  let as = signed ? v => BigInt.asIntN(bits, v) : v => BigInt.asUintN(bits, v);
  x = as(BigInt(x));
  d = as(BigInt(d));
  if (d === 0n) {
    return "integer divide by zero";
  }
  if (signed && op === "div_s" && x === -(1n << BigInt(bits - 1)) && d === -1n) {
    return "integer overflow";
  }
  let q = x / d;
  let r = x % d;
  return BigInt.asIntN(bits, op.startsWith("div") ? q : r);
}

function check(bits, exports, divisors, dividends, toValue) {
  for (let d of divisors) {
    for (let op of ["div_s", "rem_s", "div_u", "rem_u"]) {
      let f = exports[name(op, d)];
      for (let x of dividends) {
        let expected = reference(op, bits, x, d);
        if (typeof expected === "string") {
          assertErrorMessage(() => f(toValue(x)), WebAssembly.RuntimeError,
                             new RegExp(expected));
        } else {
          assertEq(f(toValue(x)), toValue(expected), `${op} ${x} by ${d}`);
        }
      }
    }
  }
}

check(32, buildModule("i32", i32Divisors), i32Divisors, i32Dividends,
      x => Number(BigInt.asIntN(32, BigInt(x))));
check(64, buildModule("i64", i64Divisors), i64Divisors, i64Dividends,
      x => BigInt.asIntN(64, BigInt(x)));
