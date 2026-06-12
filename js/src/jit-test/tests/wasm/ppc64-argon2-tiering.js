// Test for wasm tiering correctness with argon2-style SIMD computation.
// The argon2 fBlaMka function uses i64x2.extmul_low_i32x4_u, i64x2.shl,
// i64x2.add, v128.xor, v128.or, i64x2.shr_u, and i8x16.shuffle.
// A tiering bug can cause hash and verify to produce different results
// when tier-up happens between them.
//
// This test runs the computation under both baseline and optimizing
// compilers and verifies they produce identical results.

var mod = new WebAssembly.Module(wasmTextToBinary(`
  (module
    (memory (export "mem") 10)
    ;; Argon2 fBlaMka: a + b + 2 * trunc32(a) * trunc32(b)
    ;; then rotations by 32, 24, 16, 63
    (func $G_round (param i32)
      (local v128 v128 v128 v128 v128 v128 v128 v128 v128)
      (local.set 1 (v128.load (i32.add (local.get 0) (i32.const 0))))
      (local.set 2 (v128.load (i32.add (local.get 0) (i32.const 16))))
      (local.set 3 (v128.load (i32.add (local.get 0) (i32.const 32))))
      (local.set 4 (v128.load (i32.add (local.get 0) (i32.const 48))))
      (local.set 5 (v128.load (i32.add (local.get 0) (i32.const 64))))
      (local.set 6 (v128.load (i32.add (local.get 0) (i32.const 80))))
      (local.set 7 (v128.load (i32.add (local.get 0) (i32.const 96))))
      (local.set 8 (v128.load (i32.add (local.get 0) (i32.const 112))))

      ;; fBlaMka(v0, v2) + rotr32
      (local.set 1 (i64x2.add (i64x2.add (local.get 1) (local.get 3))
        (i64x2.shl (i64x2.extmul_low_i32x4_u
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 1) (local.get 1))
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 3) (local.get 3)))
          (i32.const 1))))
      (local.set 9 (v128.xor (local.get 7) (local.get 1)))
      (local.set 7 (v128.or (i64x2.shl (local.get 9) (i32.const 32)) (i64x2.shr_u (local.get 9) (i32.const 32))))

      ;; fBlaMka(v4, v6) + rotr24
      (local.set 5 (i64x2.add (i64x2.add (local.get 5) (local.get 7))
        (i64x2.shl (i64x2.extmul_low_i32x4_u
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 5) (local.get 5))
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 7) (local.get 7)))
          (i32.const 1))))
      (local.set 9 (v128.xor (local.get 3) (local.get 5)))
      (local.set 3 (v128.or (i64x2.shl (local.get 9) (i32.const 40)) (i64x2.shr_u (local.get 9) (i32.const 24))))

      ;; fBlaMka(v0, v2) + rotr16
      (local.set 1 (i64x2.add (i64x2.add (local.get 1) (local.get 3))
        (i64x2.shl (i64x2.extmul_low_i32x4_u
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 1) (local.get 1))
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 3) (local.get 3)))
          (i32.const 1))))
      (local.set 9 (v128.xor (local.get 7) (local.get 1)))
      (local.set 7 (v128.or (i64x2.shl (local.get 9) (i32.const 48)) (i64x2.shr_u (local.get 9) (i32.const 16))))

      ;; fBlaMka(v4, v6) + rotr63
      (local.set 5 (i64x2.add (i64x2.add (local.get 5) (local.get 7))
        (i64x2.shl (i64x2.extmul_low_i32x4_u
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 5) (local.get 5))
          (i8x16.shuffle 0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11 (local.get 7) (local.get 7)))
          (i32.const 1))))
      (local.set 9 (v128.xor (local.get 3) (local.get 5)))
      (local.set 3 (v128.or (i64x2.shl (local.get 9) (i32.const 1)) (i64x2.shr_u (local.get 9) (i32.const 63))))

      (v128.store (i32.add (local.get 0) (i32.const 0)) (local.get 1))
      (v128.store (i32.add (local.get 0) (i32.const 16)) (local.get 2))
      (v128.store (i32.add (local.get 0) (i32.const 32)) (local.get 3))
      (v128.store (i32.add (local.get 0) (i32.const 48)) (local.get 4))
      (v128.store (i32.add (local.get 0) (i32.const 64)) (local.get 5))
      (v128.store (i32.add (local.get 0) (i32.const 80)) (local.get 6))
      (v128.store (i32.add (local.get 0) (i32.const 96)) (local.get 7))
      (v128.store (i32.add (local.get 0) (i32.const 112)) (local.get 8)))

    (func (export "hash") (param i32) (result i64)
      (local i32)
      ;; Init with Blake2b IV
      (v128.store (i32.const 0) (v128.const i64x2 0x6a09e667f3bcc908 0xbb67ae8584caa73b))
      (v128.store (i32.const 16) (v128.const i64x2 0x3c6ef372fe94f82b 0xa54ff53a5f1d36f1))
      (v128.store (i32.const 32) (v128.const i64x2 0x510e527fade682d1 0x9b05688c2b3e6c1f))
      (v128.store (i32.const 48) (v128.const i64x2 0x1f83d9abfb41bd6b 0x5be0cd19137e2179))
      (v128.store (i32.const 64) (v128.const i64x2 0x0123456789abcdef 0xfedcba9876543210))
      (v128.store (i32.const 80) (v128.const i64x2 0xdeadbeefcafebabe 0x1122334455667788))
      (v128.store (i32.const 96) (v128.const i64x2 0xaabbccdd11223344 0x5566778899aabbcc))
      (v128.store (i32.const 112) (v128.const i64x2 0xddeeff0011223344 0x5566778899aabbcc))
      (local.set 1 (i32.const 0))
      (block (loop
        (call $G_round (i32.const 0))
        (local.set 1 (i32.add (local.get 1) (i32.const 1)))
        (br_if 1 (i32.ge_u (local.get 1) (local.get 0)))
        (br 0)))
      (i64.xor (i64.load (i32.const 0))
        (i64.xor (i64.load (i32.const 8))
          (i64.xor (i64.load (i32.const 16))
            (i64.xor (i64.load (i32.const 24))
              (i64.xor (i64.load (i32.const 32))
                (i64.xor (i64.load (i32.const 40))
                  (i64.xor (i64.load (i32.const 48))
                    (i64.xor (i64.load (i32.const 56))
                      (i64.xor (i64.load (i32.const 64))
                        (i64.xor (i64.load (i32.const 72))
                          (i64.xor (i64.load (i32.const 80))
                            (i64.xor (i64.load (i32.const 88))
                              (i64.xor (i64.load (i32.const 96))
                                (i64.xor (i64.load (i32.const 104))
                                  (i64.xor (i64.load (i32.const 112))
                                    (i64.load (i32.const 120))))))))))))))))))
  )
`));

var inst = new WebAssembly.Instance(mod);

// Get a reference result from the first call.
var reference = inst.exports.hash(100);

// Run many times to trigger tier-up, then verify result stays the same.
var pass = true;
for (var i = 0; i < 1000; i++) {
    var r = inst.exports.hash(100);
    if (r !== reference) {
        pass = false;
        throw new Error("Tiering mismatch at iteration " + i +
            ": got 0x" + BigInt.asUintN(64, r).toString(16) +
            ", expected 0x" + BigInt.asUintN(64, reference).toString(16));
    }
}

assertEq(pass, true);
