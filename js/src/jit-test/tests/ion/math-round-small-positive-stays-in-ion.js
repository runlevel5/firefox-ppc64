// |jit-test| --ion-eager; --ion-offthread-compile=off; skip-if: !this.getJitCompilerOptions() || !this.getJitCompilerOptions()['ion.enable']
//
// Math.round of a value in (0, 0.5) is 0, not -0, and must not bail out.
// The PPC64 negative-zero check tested every bit of the input instead of the
// sign bit, so each such call bailed until Ion gave up on the function.

var lastInIon;
function roundD(x) {
  lastInIon = inIon();
  return Math.round(x);
}
function roundF(a) {
  lastInIon = inIon();
  return Math.round(Math.fround(a[0]));
}
var f32 = new Float32Array([0.3]);
for (var i = 0; i < 200; i++) {
  assertEq(roundD(0.3), 0);
  assertEq(roundD(0.49999999999999994), 0);
  assertEq(roundD(0), 0);
}
assertEq(lastInIon, true);
for (var i = 0; i < 200; i++) {
  assertEq(roundF(f32), 0);
}
assertEq(lastInIon, true);

// -0 still bails and keeps its sign.
function roundNeg(x) { return Math.round(x); }
for (var i = 0; i < 20; i++) {
  assertEq(Object.is(roundNeg(-0.3), -0), true);
  assertEq(Object.is(roundNeg(-0), -0), true);
  assertEq(Object.is(roundNeg(-0.5), -0), true);
  assertEq(roundNeg(-0.51), -1);
}
