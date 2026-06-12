// |jit-test| exitstatus: 0; skip-if: !wasmSimdEnabled()
//
// Regression test for the PPC64 wasm trap exit losing live v128 state.
//
// On PPC64, doubles live in the FPRs (VSR0-31) while wasm v128 values live in
// the VRs (VSR32-63) -- disjoint physical pools. The trap exit's
// RegsToPreserve used AllDoubleMask only, so a trap firing while a v128 was
// live resumed with whatever the C++ interrupt path's libc left in the VRs
// (glibc's misaligned vector memcpy leaves lvsl alignment-control byte
// patterns there). Interrupt checks fire via traps at loop back-edges, where
// a loop-carried v128 accumulator is exactly what is live.
//
// The loop below keeps an i32x4 accumulator live across every back-edge while
// interrupts fire repeatedly; the callback does large misaligned copies to
// pull libc's vector memcpy through the VRs. On an unfixed build (real
// silicon; the simulator's VRs are insulated from native libc) the
// accumulator comes back holding garbage and the final lane values are wrong.

const ins = wasmEvalText(`(module
  (func (export "run") (param $n i32) (result i32)
    (local $acc v128)
    (block $done
      (loop $top
        (br_if $done (i32.eqz (local.get $n)))
        (local.set $acc (i32x4.add (local.get $acc) (v128.const i32x4 1 2 3 4)))
        (local.set $n (i32.sub (local.get $n) (i32.const 1)))
        (br $top)))
    ;; Fold the four lanes so any lane corruption shows up.
    (i32.xor
      (i32.xor (i32x4.extract_lane 0 (local.get $acc))
               (i32.rotl (i32x4.extract_lane 1 (local.get $acc)) (i32.const 8)))
      (i32.xor (i32.rotl (i32x4.extract_lane 2 (local.get $acc)) (i32.const 16))
               (i32.rotl (i32x4.extract_lane 3 (local.get $acc)) (i32.const 24)))))
)`).exports;

// Misaligned big copies drive glibc's lvsl/vperm memcpy path on PPC.
const big = new Uint8Array(1 << 20);
const src = big.subarray(1, (1 << 19) + 1);
const dst = new Uint8Array(1 << 19);

let fires = 0;
function onInterrupt() {
  fires++;
  for (let i = 0; i < 4; i++) {
    dst.set(src);
  }
  if (fires < 25) {
    timeout(0.02, onInterrupt);
  }
  return true;
}

function expected(n) {
  const r = (x, k) => ((x << k) | (x >>> (32 - k))) | 0;
  const l = [n | 0, (2 * n) | 0, (3 * n) | 0, (4 * n) | 0];
  return ((l[0] ^ r(l[1], 8)) ^ (r(l[2], 16) ^ r(l[3], 24))) | 0;
}

const N = 1 << 26;
timeout(0.02, onInterrupt);
const got = ins.run(N);
// Cancel any pending watchdog before finishing.
timeout(-1);
assertEq(got, expected(N));
