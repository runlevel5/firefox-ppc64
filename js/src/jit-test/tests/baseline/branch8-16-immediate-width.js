// branch8/branch16 with immediate operands: the loaded byte/halfword and the
// immediate must be brought to a common width with matching signedness.
// Two historical failure modes (seen on PPC64):
//   1. sign-extending the load while zero-extending the immediate broke
//      equality on high-bit-set bytes ("ÀÁÂ".startsWith("ÀÁÂ") -> false);
//   2. always zero-extending the load broke `byte == Imm32(-1)`.
// Exercises byte/halfword equality via typed arrays and the original
// startsWith site, plus signed relational compares.

// --- Direct byte/halfword equality through TypedArray ---
{
  let u8 = new Uint8Array([0, 1, 0x7F, 0x80, 0xC0, 0xC1, 0xFE, 0xFF]);
  let i8 = new Int8Array(u8.buffer);
  let u16 = new Uint16Array([0x0000, 0x7FFF, 0x8000, 0xC1C0, 0xFFFE, 0xFFFF]);
  let i16 = new Int16Array(u16.buffer);

  // Force baseline + Ion to specialize the comparisons.
  function eqU8(arr, idx, val) {
    return arr[idx] === val;
  }
  function eqI8(arr, idx, val) {
    return arr[idx] === val;
  }
  function eqU16(arr, idx, val) {
    return arr[idx] === val;
  }
  function eqI16(arr, idx, val) {
    return arr[idx] === val;
  }

  for (let i = 0; i < 200; i++) {
    // High-bit-set bytes: bit pattern equality must hold both signed and
    // unsigned interpretations of the immediate.
    assertEq(eqU8(u8, 4, 0xC0), true);   // unsigned compare 0xC0 == 0xC0
    assertEq(eqU8(u8, 4, 0xC1), false);
    assertEq(eqU8(u8, 7, 0xFF), true);
    assertEq(eqU8(u8, 7, -1 & 0xFF), true);   // 0xFF written as -1&0xFF

    // Signed Int8 view: 0xFF is -1, 0xC0 is -64.
    assertEq(eqI8(i8, 4, -64), true);
    assertEq(eqI8(i8, 7, -1), true);
    assertEq(eqI8(i8, 4, -63), false);

    // Halfword variants: a 16-bit value with bit 15 set.
    assertEq(eqU16(u16, 3, 0xC1C0), true);
    assertEq(eqU16(u16, 3, 0xC1C1), false);
    assertEq(eqU16(u16, 5, 0xFFFF), true);
    assertEq(eqU16(u16, 5, -1 & 0xFFFF), true);

    assertEq(eqI16(i16, 3, -15936), true);  // 0xC1C0 as i16 = -15936
    assertEq(eqI16(i16, 5, -1), true);
    assertEq(eqI16(i16, 5, -2), false);
  }
}

// --- String.prototype.startsWith with a Latin-1 constant search ---
// The original failing site: a constant search string lowers into byte-wise
// comparisons (branch16(NotEqual, addr, Imm32(0xC1C0)) and friends).
{
  let s = "ÀÁÂ";  // Latin-1 length 3, bytes 0xC0 0xC1 0xC2 (all high-bit set)
  function check() {
    return s.startsWith("ÀÁÂ");
  }
  for (let i = 0; i < 200; i++) {
    assertEq(check(), true);
  }

  // Mismatch on a single high-bit byte must report not-equal.
  let s2 = "ÀÁÃ";  // last byte 0xC3 instead of 0xC2
  function check2() {
    return s2.startsWith("ÀÁÂ");
  }
  for (let i = 0; i < 200; i++) {
    assertEq(check2(), false);
  }
}

// --- Signed relational comparisons still work ---
{
  let i8 = new Int8Array([0x7F, -1, -128, 1, 0]);
  function ltZero(idx) {
    return i8[idx] < 0;
  }
  for (let i = 0; i < 200; i++) {
    assertEq(ltZero(0), false);  // 0x7F = +127
    assertEq(ltZero(1), true);   // -1
    assertEq(ltZero(2), true);   // -128
    assertEq(ltZero(3), false);  // 1
  }
}
