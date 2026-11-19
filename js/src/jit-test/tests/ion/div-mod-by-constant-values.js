// |jit-test| --ion-eager; --ion-offthread-compile=off
//
// Integer division and modulo by constants, with and without truncation and
// for unsigned operands, compared against the interpreter. The mixed int and
// double inputs also cover the truncation of a double to int32 feeding the
// constant division: on PPC64 that result was zero-extended instead of
// sign-extended and (x | 0) % 3 was off by one for negative x.

var divisors = [1, -1, 2, -2, 3, -3, 5, 7, -7, 8, -8, 10, 100, 641, 1023,
                -1023, 1024, 65536, 0x7fffffff, -0x7fffffff, -0x80000000,
                0x80000000, 0x80000001, 0xfffffffd, 0xffffffff, 0];
var dividends = [0, -0, 1, -1, 2, -2, 3, 7, -7, 15, 100, 101, -101, 1000, 12345,
                 -12345, 0x3fffffff, 0x7ffffffe, 0x7fffffff, -0x7fffffff,
                 -0x80000000, 0x80000000, 0x80000001, 0xfffffffe, 0xffffffff,
                 0.5, -2.5, NaN, Infinity];
var shapes = ["(x / D) | 0", "x / D", "x % D", "(x % D) | 0",
              "((x >>> 0) / D) >>> 0", "((x >>> 0) % D) >>> 0",
              "(x >>> 0) / D", "(x >>> 0) % D", "(x | 0) / D", "(x | 0) % D"];

for (var d of divisors) {
  for (var shape of shapes) {
    var body = "return " + shape.replace(/D/g, "(" + d + ")") + ";";
    var jitted = new Function("x", body);
    var interp = new Function("x", "with ({}) {} " + body);
    for (var round = 0; round < 3; round++) {
      for (var x of dividends) {
        var expected = interp(x);
        var actual = jitted(x);
        if (!Object.is(expected, actual)) {
          throw new Error(shape + " with D=" + d + ", x=" + x + ": got " +
                          actual + ", expected " + expected);
        }
      }
    }
  }
}
