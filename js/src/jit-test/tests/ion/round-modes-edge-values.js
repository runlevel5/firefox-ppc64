// Floor/ceil/trunc/round of doubles and float32s, as doubles and as int32s,
// on the values that distinguish truncation-based implementations from the
// real rounding modes: fractions on both sides of zero, signed zeros, exact
// integers, halfway cases, the 2^52 integrality boundary and the int32 range.

// [input, floor, ceil, trunc, round] with expected results as doubles.
var table = [
  [0.3, 0, 1, 0, 0],
  [-0.3, -1, -0, -0, -0],
  [0.5, 0, 1, 0, 1],
  [-0.5, -1, -0, -0, -0],
  [2.5, 2, 3, 2, 3],
  [-2.5, -3, -2, -2, -2],
  [2.0, 2, 2, 2, 2],
  [-2.0, -2, -2, -2, -2],
  [0, 0, 0, 0, 0],
  [-0, -0, -0, -0, -0],
  [1.9999999999999998, 1, 2, 1, 2],
  [-1.9999999999999998, -2, -1, -1, -2],
  [4503599627370495.5, 4503599627370495, 4503599627370496, 4503599627370495, 4503599627370496],
  [-4503599627370495.5, -4503599627370496, -4503599627370495, -4503599627370495, -4503599627370495],
  [4503599627370496, 4503599627370496, 4503599627370496, 4503599627370496, 4503599627370496],
  [-4503599627370497, -4503599627370497, -4503599627370497, -4503599627370497, -4503599627370497],
  [2147483647.5, 2147483647, 2147483648, 2147483647, 2147483648],
  [-2147483648.5, -2147483649, -2147483648, -2147483648, -2147483648],
  [1e300, 1e300, 1e300, 1e300, 1e300],
  [-1e300, -1e300, -1e300, -1e300, -1e300],
  [Infinity, Infinity, Infinity, Infinity, Infinity],
  [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity],
  [NaN, NaN, NaN, NaN, NaN],
];

function same(a, b) {
  return Object.is(a, b);
}

function floorD(x) { return Math.floor(x); }
function ceilD(x) { return Math.ceil(x); }
function truncD(x) { return Math.trunc(x); }
function roundD(x) { return Math.round(x); }
// Int32-result forms: the JIT specialises Math.* to an int32 output when
// the result has only been an int32 so far, bailing out on -0 and range.
function floorI(x) { return Math.floor(x) | 0; }
function ceilI(x) { return Math.ceil(x) | 0; }
function truncI(x) { return Math.trunc(x) | 0; }
function roundI(x) { return Math.round(x) | 0; }
function floorF(x) { return Math.floor(Math.fround(x)); }
function ceilF(x) { return Math.ceil(Math.fround(x)); }
function truncF(x) { return Math.trunc(Math.fround(x)); }
function roundF(x) { return Math.round(Math.fround(x)); }

// Warm up on in-range positive values so the int32 specialisations compile,
// then exercise every table row several times so bailouts also recompile.
for (var i = 0; i < 2000; i++) {
  var v = (i % 37) + 0.25;
  floorD(v); ceilD(v); truncD(v); roundD(v);
  floorI(v); ceilI(v); truncI(v); roundI(v);
  floorF(v); ceilF(v); truncF(v); roundF(v);
}
for (var pass = 0; pass < 5; pass++) {
  for (var [x, fl, ce, tr, ro] of table) {
    assertEq(same(floorD(x), fl), true, "floor " + x);
    assertEq(same(ceilD(x), ce), true, "ceil " + x);
    assertEq(same(truncD(x), tr), true, "trunc " + x);
    assertEq(same(roundD(x), ro), true, "round " + x);
    assertEq(floorI(x), fl | 0, "floor|0 " + x);
    assertEq(ceilI(x), ce | 0, "ceil|0 " + x);
    assertEq(truncI(x), tr | 0, "trunc|0 " + x);
    assertEq(roundI(x), ro | 0, "round|0 " + x);
    var f = Math.fround(x);
    if (f === x || !isFinite(f) || Math.abs(f) < 1e7) {
      // Float32 rounding of the same value where it is representable.
      var efl = f === x ? fl : Math.floor(f);
      assertEq(same(floorF(x), efl), true, "floorF " + x);
      assertEq(same(ceilF(x), f === x ? ce : Math.ceil(f)), true, "ceilF " + x);
      assertEq(same(truncF(x), f === x ? tr : Math.trunc(f)), true, "truncF " + x);
      assertEq(same(roundF(x), f === x ? ro : Math.round(f)), true, "roundF " + x);
    }
  }
}
