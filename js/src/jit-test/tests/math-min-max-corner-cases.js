// Math.min / Math.max corner cases: -0/+0 ordering, NaN propagation,
// infinities. Exercises backend min/max fast paths (e.g. PPC64
// xsminjdp/xsmaxjdp and its POWER8 fcmpu fallback); the truth table is small
// and worth pinning explicitly.

function objectIsPositiveZero(v) {
  return v === 0 && Object.is(v, 0);
}
function objectIsNegativeZero(v) {
  return v === 0 && Object.is(v, -0);
}

// Direct calls — these get inlined by Ion as MMinMax intrinsics, which
// emit the relevant min/max helper.
function check() {
  // Max corner cases.
  assertEq(objectIsPositiveZero(Math.max(-0, +0)), true);
  assertEq(objectIsPositiveZero(Math.max(+0, -0)), true);
  assertEq(objectIsNegativeZero(Math.max(-0, -0)), true);
  assertEq(objectIsPositiveZero(Math.max(+0, +0)), true);
  assertEq(Number.isNaN(Math.max(NaN, 5)), true);
  assertEq(Number.isNaN(Math.max(5, NaN)), true);
  assertEq(Number.isNaN(Math.max(NaN, NaN)), true);
  assertEq(Math.max(-Infinity, 5), 5);
  assertEq(Math.max(Infinity, 5), Infinity);
  assertEq(Math.max(1, 2), 2);
  assertEq(Math.max(-1, -2), -1);
  assertEq(Math.max(1.5, 2.5), 2.5);

  // Min corner cases.
  assertEq(objectIsNegativeZero(Math.min(-0, +0)), true);
  assertEq(objectIsNegativeZero(Math.min(+0, -0)), true);
  assertEq(objectIsNegativeZero(Math.min(-0, -0)), true);
  assertEq(objectIsPositiveZero(Math.min(+0, +0)), true);
  assertEq(Number.isNaN(Math.min(NaN, 5)), true);
  assertEq(Number.isNaN(Math.min(5, NaN)), true);
  assertEq(Math.min(-Infinity, 5), -Infinity);
  assertEq(Math.min(Infinity, 5), 5);
  assertEq(Math.min(1, 2), 1);
}

// Run cold (Baseline) and hot (Ion).
check();
for (let i = 0; i < 5000; i++) check();
