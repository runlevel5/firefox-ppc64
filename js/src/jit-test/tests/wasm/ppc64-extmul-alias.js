// Regression test for PPC64 i64x2.extmul_{low,high}_i32x4_{s,u} when the
// Ion register allocator picks dest == rhs.
//
// On PPC64 LE, the old implementation extracted lanes via mtvsrd/mfvsrd and
// wrote the low-lane product to dest before reading rhs for the high lane.
// `mtvsrd XT, RA` leaves DW1 of XT undefined (POWER9 zeros it), so when
// dest aliased rhs the high-lane extract from rhs read garbage, producing
// zero in the high i64 lane. On POWER8 the ExtractLaneToGPR fallback
// additionally clobbered ScratchSimd128Reg between the two extracts.
//
// The loop below, discovered via wasm-reduce from argon2.wasm, reliably
// reproduced the miscompile: the result's high i64 lane went to 0 on
// POWER9 Ion / garbage on POWER8 Ion, while baseline kept the correct
// value (lane1 = 48*48 = 2304 in the final iteration).

var mod = new WebAssembly.Module(wasmTextToBinary(`
  (module
    (memory (export "mem") 1)
    (func (export "run_u") (param $out i32)
      (local $i i32) (local $v4 v128) (local $v5 v128) (local $v9 v128)
      (loop
        (local.set $v9
          (i64x2.add
            (v128.const i32x4 1 0 0 0)
            (i64x2.extmul_low_i32x4_u (local.get $v5) (local.get $v9))))
        (local.set $v4 (local.get $v9))
        (local.set $v5 (local.get $v4))
        (v128.store (i32.const 0) (local.get $v5))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br_if 0 (i32.ne (local.get $i) (i32.const 8))))
      (v128.store (local.get $out) (local.get $v9)))

    (func (export "run_s") (param $out i32)
      (local $i i32) (local $v v128)
      (local.set $v (v128.const i32x4 2 3 5 7))
      (loop
        ;; Force dest==rhs aliasing: v = extmul_low_i32x4_s(const, v).
        (local.set $v
          (i64x2.extmul_low_i32x4_s
            (v128.const i32x4 2 3 5 7)
            (local.get $v)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br_if 0 (i32.ne (local.get $i) (i32.const 2))))
      (v128.store (local.get $out) (local.get $v)))

    (func (export "run_high_u") (param $out i32)
      (local $i i32) (local $v v128)
      (local.set $v (v128.const i32x4 0 0 2 3))
      (loop
        (local.set $v
          (i64x2.extmul_high_i32x4_u
            (v128.const i32x4 0 0 2 3)
            (local.get $v)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br_if 0 (i32.ne (local.get $i) (i32.const 2))))
      (v128.store (local.get $out) (local.get $v)))

    (func (export "run_high_s") (param $out i32)
      (local $i i32) (local $v v128)
      (local.set $v (v128.const i32x4 0 0 2 3))
      (loop
        (local.set $v
          (i64x2.extmul_high_i32x4_s
            (v128.const i32x4 0 0 2 3)
            (local.get $v)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br_if 0 (i32.ne (local.get $i) (i32.const 2))))
      (v128.store (local.get $out) (local.get $v))))
`));

function runAndCheck(inst) {
  inst.exports.run_u(0);
  // After 8 iterations, the value in memory should have lane1 == 2304 = 0x900.
  // Bytes 8-15 (i64 lane 1, little-endian) = 0x0000000000000900.
  var buf = new Uint8Array(inst.exports.mem.buffer, 0, 16);
  var hex = Array.from(buf).map(b => b.toString(16).padStart(2,'0')).join('');
  // Expect bytes 8-9 = "00 09" and bytes 10-15 = "00 00 00 00 00 00".
  assertEq(hex.slice(16, 32), "0009000000000000");

  inst.exports.run_s(16);
  // After 2 iterations of v = extmul_low_s(const(2,3,5,7), v) starting v=(2,3,5,7):
  //   iter 1: i64x2 lane0 = 2*2 = 4, lane1 = 3*3 = 9.
  //           v becomes i32x4 [4, 0, 9, 0] (each i64 lane occupies two i32 lanes).
  //   iter 2: extmul_low_s reads i32 lanes 0, 1 of v = (4, 0).
  //           i64 lane0 = 2*4 = 8; i64 lane1 = 3*0 = 0.
  var buf2 = new Uint8Array(inst.exports.mem.buffer, 16, 16);
  var hex2 = Array.from(buf2).map(b => b.toString(16).padStart(2,'0')).join('');
  assertEq(hex2, "08000000000000000000000000000000");

  inst.exports.run_high_u(32);
  // v = (0, 0, 2, 3). extmul_high picks lanes 2 and 3.
  //   iter 1: lane2_prod = 2*2 = 4; lane3_prod = 3*3 = 9. Result stored at bytes 0-7 (lane2_prod) and 8-15 (lane3_prod).
  //   iter 2: v now has i64x2 lane0 = 4, lane1 = 9, i.e. i32x4 lanes [4, 0, 9, 0].
  //           extmul_high_u(const(0,0,2,3), v) reads lanes 2, 3 of both:
  //           const lane2 = 2, lane3 = 3; v lane2 = 9, lane3 = 0.
  //           result: lane2_prod = 2*9 = 18 at bytes 0-7; lane3_prod = 3*0 = 0 at bytes 8-15.
  var buf3 = new Uint8Array(inst.exports.mem.buffer, 32, 16);
  var hex3 = Array.from(buf3).map(b => b.toString(16).padStart(2,'0')).join('');
  assertEq(hex3, "12000000000000000000000000000000");

  inst.exports.run_high_s(48);
  var buf4 = new Uint8Array(inst.exports.mem.buffer, 48, 16);
  var hex4 = Array.from(buf4).map(b => b.toString(16).padStart(2,'0')).join('');
  assertEq(hex4, "12000000000000000000000000000000");
}

runAndCheck(new WebAssembly.Instance(mod));
