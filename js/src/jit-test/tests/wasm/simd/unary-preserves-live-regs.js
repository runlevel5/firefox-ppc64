// |jit-test| skip-if: !wasmSimdEnabled()
//
// SIMD unary ops (neg, extadd_pairwise, extend_high) implemented via helpers
// that use fixed registers as scratch can silently corrupt live v128 values
// the register allocator has placed in those registers. Each case keeps
// several live vectors across one op and checks that they all survive.
// Originally caught PPC64 helpers using allocatable VR1..VR5 as undeclared
// scratch; a clobbered local read back as the op's staged input instead of
// its own value.

const PRESERVE_PATTERNS = [0xA1, 0xB2, 0xC3, 0xD4];
const INPUT_BYTE = 0x18;
const INPUT_OFFSET = 144;

function init(mem) {
  // Preserve slots live at offsets 16, 32, ..; the op input lives past them.
  for (let slot = 0; slot < PRESERVE_PATTERNS.length; slot++) {
    for (let i = 0; i < 16; i++) {
      mem[16 + slot * 16 + i] = PRESERVE_PATTERNS[slot];
    }
  }
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
               `${opName}: live slot ${slot} byte ${i}: got 0x${got.toString(16)}, expected 0x${want.toString(16)} (allocator-clobbered register?)`);
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
  initLoads.push(`(local.set $input (v128.load (i32.const ${INPUT_OFFSET})))`);
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

// ---- Negate ----
//
// Input lane = 0x18 = 24. neg(24) = -24.
// i8x16.neg : -24 mod 256 = 232 = 0xE8 per byte.
// i16x8.neg : lane = 0x1818 = 6168, neg = -6168 mod 65536 = 0xE7E8.
// i32x4.neg : lane = 0x18181818, neg = 0xE7E7E7E8.
// i64x2.neg : lane = 0x1818181818181818, neg = 0xE7E7E7E7E7E7E7E8.
// (memory is little-endian, so the 0xE8 byte comes first in each lane)

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

// ---- extadd_pairwise ----
//
// Reads adjacent pairs, widens, sums. Input = repeat(0x18) = 24.
// i16x8.extadd_pairwise_i8x16_{s,u} : 24 + 24 = 48 = 0x0030 per i16 lane.
// i32x4.extadd_pairwise_i16x8_{s,u} : 0x1818 + 0x1818 = 0x3030 per i32 lane.

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

// ---- extend_high ----
//
// i64x2.extend_high_i32x4_u: zero-extend i32 lanes 2 and 3 to i64x2.
// Input lane = 0x18181818; each i64 result lane = 0x0000000018181818.

runOne("i64x2.extend_high_i32x4_u",
       "i64x2.extend_high_i32x4_u", 4,
       [0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00,
        0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00]);
