// Exercises the JIT-inlined NativeIterator flag handling in
// MacroAssembler::iteratorClose. Deleting an as-yet-unvisited property during
// for-in sets HasUnvisitedPropertyDeletion; closing the iterator must then
// clear the deleted property bits and the Active flag. The NativeIterator
// flags_ field is a uint8_t addressed with 32-bit ops in JIT code, so on
// big-endian the flag constants must be shifted to the correct byte (see
// NativeIterator::flagForJit32); a missed shift tests/clears the wrong bits.

function iterateDeletingUnvisited() {
  var o = {a: 1, b: 2, c: 3, d: 4, e: 5};
  var seen = [];
  for (var k in o) {
    seen.push(k);
    if (k === "a") {
      // Delete properties that have not been visited yet; per spec they must
      // not be visited, which is what sets HasUnvisitedPropertyDeletion.
      delete o.d;
      delete o.e;
    }
  }
  return seen.join(",");
}

// Nested for-in over the same shape stresses iterator reuse across close, so a
// stuck Active bit or uncleared deleted bit would surface on the reused
// iterator.
function nested() {
  var outer = {x: 1, y: 2, z: 3};
  var count = 0;
  for (var a in outer) {
    for (var b in outer) {
      count++;
    }
  }
  return count;
}

for (var i = 0; i < 3000; i++) {
  assertEq(iterateDeletingUnvisited(), "a,b,c");
  assertEq(nested(), 9);
}
