// Regression test for PPC64 branch8/branch16 width-narrowing under Equal /
// NotEqual / unsigned comparisons. Two prior bugs:
//
//   1. Sign-extending the load while move32(Imm32) zero-extended the imm
//      caused spurious mismatch when the loaded byte/halfword had its high
//      bit set (e.g. "ÀÁÂ".startsWith("ÀÁÂ") returned false because byte 0xC0
//      sign-extended to 0xFF...C0 but the imm 0xC0 zero-extended to 0x00C0,
//      so cmpw on the low 32 bits saw a negative vs positive value).
//
//   2. Always zero-extending the load broke `byte == Imm32(-1)` because -1
//      sign-extends in the imm path: the loaded 0x000000FF didn't match the
//      materialized 0xFFFFFFFF.
//
// Fix: cast the immediate to uint8/uint16 (equality + unsigned) or int8/int16
// (signed relational) so both sides have matching bit patterns regardless of
// how move32(Imm32) chose to materialize it. Match ARM64/LoongArch64/RISC-V.
//
// We exercise both byte and halfword branch paths via TypedArray loads and
// String.prototype.startsWith with a constant search string (the original
// failing site lowered to branch16(NotEqual, addr, Imm32(0xC1C0))).

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

    // Halfword variants: the original startswith failure pattern was
    // (Latin-1 char 0xC1C0) — a 16-bit value with bit 15 set.
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
// This was the original failing site — Ion lowers a constant search string
// of length 1..32 into a sequence of byte-wise comparisons.
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

// --- Signed relational comparisons still work (we kept the sign-extend path) ---
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
