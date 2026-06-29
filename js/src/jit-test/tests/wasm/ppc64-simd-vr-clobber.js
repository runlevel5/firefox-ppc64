// |jit-test| skip-if: !wasmSimdEnabled()
//
// Regression tests for PPC64 SIMD helpers that use VR1..VR5 as undeclared
// scratch and silently corrupt live wasm v128 values the register allocator
// has placed in those VRs.
//
// Background: PPC64 Simd128 lives in VR0..VR31. VR0 is non-allocatable
// (= ScratchSimd128Reg); VR1..VR31 are allocatable. The helpers below
// historically used VR1..VR5 as undeclared scratch:
//
//   negInt8x16, negInt16x8                    : clobber VR1 (all CPUs)
//   negInt32x4, negInt64x2 (POWER8 fallback)  : clobber VR1 (POWER8 only)
//   extAddPairwiseInt8x16  (signed/unsigned)  : clobber VR1, VR2, VR3
//   extAddPairwiseInt16x8  (signed/unsigned)  : clobber VR1, VR2, VR3
//   unsignedWidenHighInt32x4                  : clobber VR1
//
// Each test:
//   - loads `nLive` "preserve" v128 values from memory at offsets 16..16+16*nLive
//   - loads ONE additional "input" v128 = repeat(0x18) at offset 128
//   - applies the suspect helper to the input
//   - stores the nLive preserved values back to memory at offsets 0..16*nLive
//   - stores the helper result at offset 16*nLive
//
// Without the fix, one of the preserved locals (whichever the allocator
// placed in the clobbered VR) reads back as the staged input value (0x18)
// instead of its original. With the fix (the helper using ScratchSimd128Scope
// or proper VR-namespace emit), all preserved locals retain their values.

const PRESERVE_PATTERNS = [0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x29];
const INPUT_BYTE = 0x18;

function init(mem) {
  // Slots at offset 16, 32, ..., 16+16*7 hold the preserve patterns.
  for (let slot = 0; slot < PRESERVE_PATTERNS.length; slot++) {
    for (let i = 0; i < 16; i++) {
      mem[16 + slot * 16 + i] = PRESERVE_PATTERNS[slot];
    }
  }
  // The helper input is at offset 128 (= 16 + 16*7 + 16 = 144? no, 16 + 16*8 = 144).
  // Use a fixed offset PAST the preserve area. With nLive max 7, preserve uses
  // 16..(16+16*7-1) = 16..127. Input goes at 144 to leave a 16-byte gap.
  const INPUT_OFFSET = 144;
  for (let i = 0; i < 16; i++) mem[INPUT_OFFSET + i] = INPUT_BYTE;
}

function repeat(byte) {
  const a = new Array(16);
  for (let i = 0; i < 16; i++) a[i] = byte;
  return a;
}

// Verify nLive preserved slots match PRESERVE_PATTERNS at output offsets
// 0..16*nLive, and that the result slot at 16*nLive matches `expectedResult`.
function check(opName, mem, nLive, expectedResult) {
  for (let slot = 0; slot < nLive; slot++) {
    for (let i = 0; i < 16; i++) {
      const got = mem[slot * 16 + i];
      const want = PRESERVE_PATTERNS[slot];
      assertEq(got, want,
               `${opName}: live slot ${slot} byte ${i}: got 0x${got.toString(16)}, expected 0x${want.toString(16)} (allocator-clobbered VR?)`);
    }
  }
  for (let i = 0; i < 16; i++) {
    const got = mem[nLive * 16 + i];
    const want = expectedResult[i];
    assertEq(got, want,
             `${opName}: result byte ${i}: got 0x${got.toString(16)}, expected 0x${want.toString(16)}`);
  }
}

// Build a wasm module that:
//  - loads `nLive` preserve v128 locals from memory at offsets 16..16*nLive
//  - loads ONE input v128 from offset 144
//  - applies `op` to the input
//  - stores all `nLive + 1` v128 values back to memory at offsets 0..16*nLive
function buildModule(op, nLive) {
  const localDecls = [];
  const initLoads = [];
  const finalStores = [];
  for (let i = 0; i < nLive; i++) {
    localDecls.push(`(local $v${i} v128)`);
    initLoads.push(`(local.set $v${i} (v128.load (i32.const ${16 + i * 16})))`);
    finalStores.push(`(v128.store (i32.const ${i * 16}) (local.get $v${i}))`);
  }
  // The helper input + result.
  localDecls.push(`(local $input v128)`);
  initLoads.push(`(local.set $input (v128.load (i32.const 144)))`);
  finalStores.push(`(v128.store (i32.const ${nLive * 16}) (local.get $input))`);

  const text = `
    (module
      (memory (export "mem") 1)
      (func (export "run")
        ${localDecls.join('\n        ')}
        ${initLoads.join('\n        ')}
        (local.set $input (${op} (local.get $input)))
        ${finalStores.join('\n        ')}
      )
    )`;
  return new WebAssembly.Module(wasmTextToBinary(text));
}

function runOne(opName, op, nLive, expectedResult) {
  const mod = buildModule(op, nLive);
  const inst = new WebAssembly.Instance(mod);
  const mem = new Uint8Array(inst.exports.mem.buffer);
  // Run many times so Baseline + Ion both see it.
  for (let warm = 0; warm < 50; warm++) {
    init(mem);
    inst.exports.run();
    check(opName, mem, nLive, expectedResult);
  }
}

// ---- Negate helpers ----
//
// Input lane = 0x18 = 24. neg(24) = -24.
// i8x16.neg : -24 mod 256 = 232 = 0xE8 per byte.
// i16x8.neg : lane = 0x1818 = 6168, neg = -6168 mod 65536 = 0xE7E8.
//             Memory LE: per i16 lane bytes 0xE8 0xE7.
// i32x4.neg : lane = 0x18181818 = 404232216, neg = 0xE7E7E7E8.
//             Memory LE: per i32 lane bytes 0xE8 0xE7 0xE7 0xE7.
// i64x2.neg : lane = 0x1818181818181818, neg = 0xE7E7E7E7E7E7E7E8.
//             Memory LE: per i64 lane bytes 0xE8 0xE7 0xE7 0xE7 0xE7 0xE7 0xE7 0xE7.

runOne("i8x16.neg", "i8x16.neg", 4, repeat(0xE8));
runOne("i16x8.neg", "i16x8.neg", 4,
       [0xE8,0xE7, 0xE8,0xE7, 0xE8,0xE7, 0xE8,0xE7,
        0xE8,0xE7, 0xE8,0xE7, 0xE8,0xE7, 0xE8,0xE7]);
runOne("i32x4.neg", "i32x4.neg", 4,
       [0xE8,0xE7,0xE7,0xE7, 0xE8,0xE7,0xE7,0xE7,
        0xE8,0xE7,0xE7,0xE7, 0xE8,0xE7,0xE7,0xE7]);
runOne("i64x2.neg", "i64x2.neg", 4,
       [0xE8,0xE7,0xE7,0xE7,0xE7,0xE7,0xE7,0xE7,
        0xE8,0xE7,0xE7,0xE7,0xE7,0xE7,0xE7,0xE7]);

// ---- extAddPairwise helpers ----
//
// extadd_pairwise reads adjacent pairs and widens-then-sums them.
// Input = repeat(0x18) = 24.
// i16x8.extadd_pairwise_i8x16_s : 24 + 24 = 48 = 0x0030 per i16 lane.
//                                  Memory LE: 0x30 0x00 per lane × 8 lanes.
// i16x8.extadd_pairwise_i8x16_u : same since input is positive.
// i32x4.extadd_pairwise_i16x8_s : i16 lane = 0x1818 = 6168, sum = 12336 = 0x00003030.
//                                  Memory LE: 0x30 0x30 0x00 0x00 per lane × 4 lanes.
// i32x4.extadd_pairwise_i16x8_u : same since input is positive.

runOne("i16x8.extadd_pairwise_i8x16_s",
       "i16x8.extadd_pairwise_i8x16_s", 4,
       [0x30,0x00, 0x30,0x00, 0x30,0x00, 0x30,0x00,
        0x30,0x00, 0x30,0x00, 0x30,0x00, 0x30,0x00]);

runOne("i16x8.extadd_pairwise_i8x16_u",
       "i16x8.extadd_pairwise_i8x16_u", 4,
       [0x30,0x00, 0x30,0x00, 0x30,0x00, 0x30,0x00,
        0x30,0x00, 0x30,0x00, 0x30,0x00, 0x30,0x00]);

runOne("i32x4.extadd_pairwise_i16x8_s",
       "i32x4.extadd_pairwise_i16x8_s", 4,
       [0x30,0x30,0x00,0x00, 0x30,0x30,0x00,0x00,
        0x30,0x30,0x00,0x00, 0x30,0x30,0x00,0x00]);

runOne("i32x4.extadd_pairwise_i16x8_u",
       "i32x4.extadd_pairwise_i16x8_u", 4,
       [0x30,0x30,0x00,0x00, 0x30,0x30,0x00,0x00,
        0x30,0x30,0x00,0x00, 0x30,0x30,0x00,0x00]);

// ---- unsignedWidenHighInt32x4 ----
//
// i64x2.extend_high_i32x4_u: take the high two i32 lanes (lanes 2 and 3) of
// the input, zero-extend each to i64, lay them out as i64x2.
// Input lane = 0x18181818 (positive, =404232216).
// Result: two i64 lanes, each = 0x0000000018181818.
// Memory LE: per i64 lane bytes 0x18 0x18 0x18 0x18 0x00 0x00 0x00 0x00.

runOne("i64x2.extend_high_i32x4_u",
       "i64x2.extend_high_i32x4_u", 4,
       [0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00,
        0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00]);
