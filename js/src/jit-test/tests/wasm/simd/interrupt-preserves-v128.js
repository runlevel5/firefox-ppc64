// |jit-test| exitstatus: 0; skip-if: !wasmSimdEnabled()
//
// A trap/interrupt exit taken at a loop back-edge must save and restore live
// wasm v128 state. On targets where doubles and vectors occupy disjoint
// physical register pools (e.g. PPC64 FPRs vs VRs), a save set derived from
// the double mask alone loses the vector registers entirely; the interrupt
// callback below drives libc's vector memcpy through those registers, so an
// unsaved loop-carried accumulator comes back holding garbage.

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

// Misaligned big copies drive libc's vector memcpy path.
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
