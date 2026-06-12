// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC) version of decoding functions.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <string.h>

typedef __vector signed short     i16x8;
typedef __vector unsigned short   u16x8;
typedef __vector signed int       i32x4;
typedef __vector unsigned int     u32x4;
typedef __vector unsigned char    u8x16;
typedef __vector signed char      i8x16;
typedef __vector signed long long i64x2;

// Signed multiply-high of packed 16-bit lanes (POWER8 has no vmulhsh).
static WEBP_INLINE i16x8 MulHi16_S(i16x8 a, i16x8 b) {
  const u32x4 sh = vec_splats((unsigned int)16);
  const i32x4 e = vec_sra(vec_mule(a, b), sh);
  const i32x4 o = vec_sra(vec_mulo(a, b), sh);
  return (i16x8)vec_pack(vec_mergeh(e, o), vec_mergel(e, o));
}

// Transpose two interleaved 4x4 blocks of 16-bit values.
static WEBP_INLINE void Transpose2_4x4(i16x8 in0, i16x8 in1, i16x8 in2,
                                       i16x8 in3, i16x8* out0, i16x8* out1,
                                       i16x8* out2, i16x8* out3) {
  const i16x8 t0 = (i16x8)vec_mergeh(in0, in1);
  const i16x8 t1 = (i16x8)vec_mergeh(in2, in3);
  const i16x8 t2 = (i16x8)vec_mergel(in0, in1);
  const i16x8 t3 = (i16x8)vec_mergel(in2, in3);
  const i32x4 u0 = vec_mergeh((i32x4)t0, (i32x4)t1);
  const i32x4 u1 = vec_mergeh((i32x4)t2, (i32x4)t3);
  const i32x4 u2 = vec_mergel((i32x4)t0, (i32x4)t1);
  const i32x4 u3 = vec_mergel((i32x4)t2, (i32x4)t3);
  *out0 = (i16x8)vec_mergeh((i64x2)u0, (i64x2)u1);
  *out1 = (i16x8)vec_mergel((i64x2)u0, (i64x2)u1);
  *out2 = (i16x8)vec_mergeh((i64x2)u2, (i64x2)u3);
  *out3 = (i16x8)vec_mergel((i64x2)u2, (i64x2)u3);
}

// Bounded 4-coefficient load into the low half of a 16-bit vector.
static WEBP_INLINE i16x8 Load4Coeffs(const int16_t* WEBP_RESTRICT p) {
  int16_t tmp[8] = {0};
  memcpy(tmp, p, 4 * sizeof(int16_t));
  return *(const i16x8*)tmp;
}

// Bounded load of n pixels, zero-extended to 16-bit lanes.
static WEBP_INLINE i16x8 LoadDst(const uint8_t* WEBP_RESTRICT p, int n) {
  unsigned char tmp[16] = {0};
  memcpy(tmp, p, n);
  return (i16x8)vec_mergeh(vec_xl(0, tmp), vec_splats((unsigned char)0));
}

static void Transform_VSX(const int16_t* WEBP_RESTRICT in,
                          uint8_t* WEBP_RESTRICT dst, int do_two) {
  const i16x8 k1 = vec_splats((short)20091);
  const i16x8 k2 = vec_splats((short)-30068);
  const u16x8 three = vec_splats((unsigned short)3);
  i16x8 in0 = Load4Coeffs(in + 0), in1 = Load4Coeffs(in + 4);
  i16x8 in2 = Load4Coeffs(in + 8), in3 = Load4Coeffs(in + 12);
  i16x8 T0, T1, T2, T3;

  if (do_two) {
    in0 = (i16x8)vec_mergeh((i64x2)in0, (i64x2)Load4Coeffs(in + 16));
    in1 = (i16x8)vec_mergeh((i64x2)in1, (i64x2)Load4Coeffs(in + 20));
    in2 = (i16x8)vec_mergeh((i64x2)in2, (i64x2)Load4Coeffs(in + 24));
    in3 = (i16x8)vec_mergeh((i64x2)in3, (i64x2)Load4Coeffs(in + 28));
  }

  {  // Vertical pass + transpose.
    const i16x8 a = vec_add(in0, in2);
    const i16x8 b = vec_sub(in0, in2);
    const i16x8 c = vec_add(vec_sub(in1, in3),
                            vec_sub(MulHi16_S(in1, k2), MulHi16_S(in3, k1)));
    const i16x8 d = vec_add(vec_add(in1, in3),
                            vec_add(MulHi16_S(in1, k1), MulHi16_S(in3, k2)));
    Transpose2_4x4(vec_add(a, d), vec_add(b, c), vec_sub(b, c), vec_sub(a, d),
                   &T0, &T1, &T2, &T3);
  }
  {  // Horizontal pass + transpose.
    const i16x8 dc = vec_add(T0, vec_splats((short)4));
    const i16x8 a = vec_add(dc, T2);
    const i16x8 b = vec_sub(dc, T2);
    const i16x8 c = vec_add(vec_sub(T1, T3),
                            vec_sub(MulHi16_S(T1, k2), MulHi16_S(T3, k1)));
    const i16x8 d = vec_add(vec_add(T1, T3),
                            vec_add(MulHi16_S(T1, k1), MulHi16_S(T3, k2)));
    const i16x8 s0 = vec_sra(vec_add(a, d), three);
    const i16x8 s1 = vec_sra(vec_add(b, c), three);
    const i16x8 s2 = vec_sra(vec_sub(b, c), three);
    const i16x8 s3 = vec_sra(vec_sub(a, d), three);
    Transpose2_4x4(s0, s1, s2, s3, &T0, &T1, &T2, &T3);
  }
  {  // Add to the reference pixels and store with saturation.
    const int n = do_two ? 8 : 4;
    const i16x8 d0 = LoadDst(dst + 0 * BPS, n);
    const i16x8 d1 = LoadDst(dst + 1 * BPS, n);
    const i16x8 d2 = LoadDst(dst + 2 * BPS, n);
    const i16x8 d3 = LoadDst(dst + 3 * BPS, n);
    const u8x16 r0 = vec_packsu(vec_add(d0, T0), vec_add(d0, T0));
    const u8x16 r1 = vec_packsu(vec_add(d1, T1), vec_add(d1, T1));
    const u8x16 r2 = vec_packsu(vec_add(d2, T2), vec_add(d2, T2));
    const u8x16 r3 = vec_packsu(vec_add(d3, T3), vec_add(d3, T3));
    unsigned char b0[16], b1[16], b2[16], b3[16];
    memcpy(b0, &r0, 16); memcpy(b1, &r1, 16);
    memcpy(b2, &r2, 16); memcpy(b3, &r3, 16);
    memcpy(dst + 0 * BPS, b0, n); memcpy(dst + 1 * BPS, b1, n);
    memcpy(dst + 2 * BPS, b2, n); memcpy(dst + 3 * BPS, b3, n);
  }
}

//------------------------------------------------------------------------------
// Simple in-loop edge filtering.

#define ABSU(p, q) vec_or(vec_subs((u8x16)(q), (u8x16)(p)), \
                          vec_subs((u8x16)(p), (u8x16)(q)))

// Per-byte signed arithmetic >>3, packed with saturation.
static WEBP_INLINE i8x16 SignedShift3(i8x16 x) {
  const u8x16 z = vec_splats((unsigned char)0);
  const u16x8 sh = vec_splats((unsigned short)(3 + 8));
  const i16x8 lo = vec_sra((i16x8)vec_mergeh(z, (u8x16)x), sh);
  const i16x8 hi = vec_sra((i16x8)vec_mergel(z, (u8x16)x), sh);
  return (i8x16)vec_packs(lo, hi);
}

static WEBP_INLINE void DoFilter2_VSX(u8x16* WEBP_RESTRICT p1,
                                      u8x16* WEBP_RESTRICT p0,
                                      u8x16* WEBP_RESTRICT q0,
                                      u8x16* WEBP_RESTRICT q1, int thresh) {
  const u8x16 sign = vec_splats((unsigned char)0x80);
  const u8x16 t1 = ABSU(*p1, *q1);
  const u8x16 t2 = vec_and(t1, vec_splats((unsigned char)0xFE));
  const u8x16 t3 = (u8x16)vec_sr((u16x8)t2, vec_splats((unsigned short)1));
  const u8x16 t4 = ABSU(*p0, *q0);
  const u8x16 t6 = vec_adds(vec_adds(t4, t4), t3);
  const u8x16 t7 = vec_subs(t6, vec_splats((unsigned char)thresh));
  const u8x16 mask = (u8x16)vec_cmpeq(t7, vec_splats((unsigned char)0));

  const i8x16 p1s = (i8x16)vec_xor(*p1, sign);
  const i8x16 q1s = (i8x16)vec_xor(*q1, sign);
  i8x16 P0 = (i8x16)vec_xor(*p0, sign);
  i8x16 Q0 = (i8x16)vec_xor(*q0, sign);

  const i8x16 d0 = vec_subs(Q0, P0);
  const i8x16 s1 = vec_adds(vec_subs(p1s, q1s), d0);
  i8x16 a = vec_adds(d0, vec_adds(d0, s1));
  a = vec_and(a, (i8x16)mask);
  const i8x16 v3 = SignedShift3(vec_adds(a, vec_splats((signed char)3)));
  const i8x16 v4 = SignedShift3(vec_adds(a, vec_splats((signed char)4)));
  Q0 = vec_subs(Q0, v4);
  P0 = vec_adds(P0, v3);
  *p0 = vec_xor((u8x16)P0, sign);
  *q0 = vec_xor((u8x16)Q0, sign);
}

static void SimpleVFilter16_VSX(uint8_t* p, int stride, int thresh) {
  u8x16 p1 = vec_xl(0, p - 2 * stride);
  u8x16 p0 = vec_xl(0, p - stride);
  u8x16 q0 = vec_xl(0, p);
  u8x16 q1 = vec_xl(0, p + stride);
  DoFilter2_VSX(&p1, &p0, &q0, &q1, thresh);
  vec_xst(p0, 0, p - stride);
  vec_xst(q0, 0, p);
}

static void SimpleVFilter16i_VSX(uint8_t* p, int stride, int thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4 * stride;
    SimpleVFilter16_VSX(p, stride, thresh);
  }
}

// Transpose four columns out of / into 16 rows for horizontal-edge filtering.
static WEBP_INLINE void Load8x4(const uint8_t* WEBP_RESTRICT b, int s,
                                u8x16* WEBP_RESTRICT p, u8x16* WEBP_RESTRICT q) {
  uint32_t a0[4], a1[4];
  memcpy(&a0[0], b + 0 * s, 4); memcpy(&a0[1], b + 4 * s, 4);
  memcpy(&a0[2], b + 2 * s, 4); memcpy(&a0[3], b + 6 * s, 4);
  memcpy(&a1[0], b + 1 * s, 4); memcpy(&a1[1], b + 5 * s, 4);
  memcpy(&a1[2], b + 3 * s, 4); memcpy(&a1[3], b + 7 * s, 4);
  const u8x16 A0 = vec_xl(0, (unsigned char*)a0);
  const u8x16 A1 = vec_xl(0, (unsigned char*)a1);
  const u8x16 B0 = vec_mergeh(A0, A1), B1 = vec_mergel(A0, A1);
  const u16x8 C0 = vec_mergeh((u16x8)B0, (u16x8)B1);
  const u16x8 C1 = vec_mergel((u16x8)B0, (u16x8)B1);
  *p = (u8x16)vec_mergeh((u32x4)C0, (u32x4)C1);
  *q = (u8x16)vec_mergel((u32x4)C0, (u32x4)C1);
}

static WEBP_INLINE void Load16x4(const uint8_t* WEBP_RESTRICT r0,
                                 const uint8_t* WEBP_RESTRICT r8, int s,
                                 u8x16* p1, u8x16* p0, u8x16* q0, u8x16* q1) {
  Load8x4(r0, s, p1, q0);
  Load8x4(r8, s, p0, q1);
  const u8x16 t1 = *p1, t2 = *q0;
  *p1 = (u8x16)vec_mergeh((i64x2)t1, (i64x2)*p0);
  *p0 = (u8x16)vec_mergel((i64x2)t1, (i64x2)*p0);
  *q0 = (u8x16)vec_mergeh((i64x2)t2, (i64x2)*q1);
  *q1 = (u8x16)vec_mergel((i64x2)t2, (i64x2)*q1);
}

static WEBP_INLINE void Store4x4(u8x16 x, uint8_t* WEBP_RESTRICT dst, int s) {
  unsigned char b[16];
  int i;
  memcpy(b, &x, 16);
  for (i = 0; i < 4; ++i) memcpy(dst + i * s, b + 4 * i, 4);
}

static WEBP_INLINE void Store16x4(u8x16 p1, u8x16 p0, u8x16 q0, u8x16 q1,
                                  uint8_t* WEBP_RESTRICT r0,
                                  uint8_t* WEBP_RESTRICT r8, int s) {
  u8x16 t = p0;
  u8x16 p0s = vec_mergeh(p1, t), p1s = vec_mergel(p1, t);
  t = q0;
  u8x16 q0s = vec_mergeh(t, q1), q1s = vec_mergel(t, q1);
  t = p0s;
  p0s = (u8x16)vec_mergeh((u16x8)t, (u16x8)q0s);
  q0s = (u8x16)vec_mergel((u16x8)t, (u16x8)q0s);
  t = p1s;
  p1s = (u8x16)vec_mergeh((u16x8)t, (u16x8)q1s);
  q1s = (u8x16)vec_mergel((u16x8)t, (u16x8)q1s);
  Store4x4(p0s, r0, s); Store4x4(q0s, r0 + 4 * s, s);
  Store4x4(p1s, r8, s); Store4x4(q1s, r8 + 4 * s, s);
}

static void SimpleHFilter16_VSX(uint8_t* p, int stride, int thresh) {
  u8x16 p1, p0, q0, q1;
  p -= 2;  // beginning of p1
  Load16x4(p, p + 8 * stride, stride, &p1, &p0, &q0, &q1);
  DoFilter2_VSX(&p1, &p0, &q0, &q1, thresh);
  Store16x4(p1, p0, q0, q1, p, p + 8 * stride, stride);
}

static void SimpleHFilter16i_VSX(uint8_t* p, int stride, int thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4;
    SimpleHFilter16_VSX(p, stride, thresh);
  }
}

//------------------------------------------------------------------------------
// Complex in-loop edge filtering (vertical/luma).

static const u8x16 kSignBit = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
#define FLIPB(x) ((x) = (i8x16)vec_xor((u8x16)(x), kSignBit))

static WEBP_INLINE u8x16 GetNotHEV(u8x16 p1, u8x16 p0, u8x16 q0, u8x16 q1,
                                   int hev_thresh) {
  const u8x16 d = vec_subs(vec_max(ABSU(p1, p0), ABSU(q1, q0)),
                           vec_splats((unsigned char)hev_thresh));
  return (u8x16)vec_cmpeq(d, vec_splats((unsigned char)0));
}

static WEBP_INLINE i8x16 GetBaseDelta(i8x16 p1, i8x16 p0, i8x16 q0, i8x16 q1) {
  const i8x16 d = vec_subs(q0, p0);
  const i8x16 s1 = vec_adds(vec_subs(p1, q1), d);
  return vec_adds(d, vec_adds(d, s1));
}

static WEBP_INLINE void DoSimpleFilterS(i8x16* p0, i8x16* q0, i8x16 f) {
  *q0 = vec_subs(*q0, SignedShift3(vec_adds(f, vec_splats((signed char)4))));
  *p0 = vec_adds(*p0, SignedShift3(vec_adds(f, vec_splats((signed char)3))));
}

static WEBP_INLINE void Update2Pixels(i8x16* pi, i8x16* qi, i16x8 lo, i16x8 hi) {
  const u16x8 s7 = vec_splats((unsigned short)7);
  const i8x16 d = (i8x16)vec_packs(vec_sra(lo, s7), vec_sra(hi, s7));
  *pi = vec_adds(*pi, d);
  *qi = vec_subs(*qi, d);
  FLIPB(*pi);
  FLIPB(*qi);
}

// mask = (max inner abs-diff <= ithresh) && NeedsFilter(thresh).
static WEBP_INLINE u8x16 ComplexMask(u8x16 p3, u8x16 p2, u8x16 p1, u8x16 p0,
                                     u8x16 q0, u8x16 q1, u8x16 q2, u8x16 q3,
                                     int thresh, int ithresh) {
  u8x16 m = ABSU(p1, p0);
  m = vec_max(m, ABSU(p3, p2));
  m = vec_max(m, ABSU(p2, p1));
  m = vec_max(m, ABSU(q1, q0));
  m = vec_max(m, ABSU(q3, q2));
  m = vec_max(m, ABSU(q2, q1));
  const u8x16 tm = (u8x16)vec_cmpeq(
      vec_subs(m, vec_splats((unsigned char)ithresh)),
      vec_splats((unsigned char)0));
  const u8x16 t2 = vec_and(ABSU(p1, q1), vec_splats((unsigned char)0xFE));
  const u8x16 t3 = (u8x16)vec_sr((u16x8)t2, vec_splats((unsigned short)1));
  const u8x16 t6 = vec_adds(vec_adds(ABSU(p0, q0), ABSU(p0, q0)), t3);
  const u8x16 fm = (u8x16)vec_cmpeq(
      vec_subs(t6, vec_splats((unsigned char)thresh)),
      vec_splats((unsigned char)0));
  return vec_and(tm, fm);
}

static WEBP_INLINE void DoFilter4(u8x16* p1u, u8x16* p0u, u8x16* q0u,
                                  u8x16* q1u, u8x16 mask, int hev_thresh) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 not_hev = GetNotHEV(*p1u, *p0u, *q0u, *q1u, hev_thresh);
  i8x16 p1 = (i8x16)vec_xor(*p1u, kSignBit), p0 = (i8x16)vec_xor(*p0u, kSignBit);
  i8x16 q0 = (i8x16)vec_xor(*q0u, kSignBit), q1 = (i8x16)vec_xor(*q1u, kSignBit);
  i8x16 t1 = vec_andc(vec_subs(p1, q1), (i8x16)not_hev);
  const i8x16 t2 = vec_subs(q0, p0);
  t1 = vec_adds(t1, t2); t1 = vec_adds(t1, t2); t1 = vec_adds(t1, t2);
  t1 = vec_and(t1, (i8x16)mask);
  const i8x16 a3 = SignedShift3(vec_adds(t1, vec_splats((signed char)4)));
  p0 = vec_adds(p0, SignedShift3(vec_adds(t1, vec_splats((signed char)3))));
  q0 = vec_subs(q0, a3);
  FLIPB(p0); FLIPB(q0);
  const i8x16 t = vec_add(a3, (i8x16)kSignBit);
  i8x16 t3 = vec_sub((i8x16)vec_avg((u8x16)t, zero), vec_splats((signed char)64));
  t3 = vec_and((i8x16)not_hev, t3);
  q1 = vec_subs(q1, t3); p1 = vec_adds(p1, t3);
  FLIPB(p1); FLIPB(q1);
  *p1u = (u8x16)p1; *p0u = (u8x16)p0; *q0u = (u8x16)q0; *q1u = (u8x16)q1;
}

static WEBP_INLINE void DoFilter6(u8x16* p2u, u8x16* p1u, u8x16* p0u,
                                  u8x16* q0u, u8x16* q1u, u8x16* q2u,
                                  u8x16 mask, int hev_thresh) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 not_hev = GetNotHEV(*p1u, *p0u, *q0u, *q1u, hev_thresh);
  i8x16 p2 = (i8x16)vec_xor(*p2u, kSignBit), p1 = (i8x16)vec_xor(*p1u, kSignBit);
  i8x16 p0 = (i8x16)vec_xor(*p0u, kSignBit), q0 = (i8x16)vec_xor(*q0u, kSignBit);
  i8x16 q1 = (i8x16)vec_xor(*q1u, kSignBit), q2 = (i8x16)vec_xor(*q2u, kSignBit);
  const i8x16 a = GetBaseDelta(p1, p0, q0, q1);
  {  // hev pixels: simple filter
    const i8x16 f = vec_and(a, (i8x16)vec_andc(mask, not_hev));
    DoSimpleFilterS(&p0, &q0, f);
  }
  {  // non-hev pixels: strong filter
    const i8x16 f = vec_and(a, vec_and((i8x16)not_hev, (i8x16)mask));
    const i16x8 k9 = vec_splats((short)0x0900), k63 = vec_splats((short)63);
    const i16x8 f9lo = MulHi16_S((i16x8)vec_mergeh(zero, (u8x16)f), k9);
    const i16x8 f9hi = MulHi16_S((i16x8)vec_mergel(zero, (u8x16)f), k9);
    const i16x8 a2lo = vec_add(f9lo, k63), a2hi = vec_add(f9hi, k63);
    const i16x8 a1lo = vec_add(a2lo, f9lo), a1hi = vec_add(a2hi, f9hi);
    const i16x8 a0lo = vec_add(a1lo, f9lo), a0hi = vec_add(a1hi, f9hi);
    Update2Pixels(&p2, &q2, a2lo, a2hi);
    Update2Pixels(&p1, &q1, a1lo, a1hi);
    Update2Pixels(&p0, &q0, a0lo, a0hi);
  }
  *p2u = (u8x16)p2; *p1u = (u8x16)p1; *p0u = (u8x16)p0;
  *q0u = (u8x16)q0; *q1u = (u8x16)q1; *q2u = (u8x16)q2;
}

static void VFilter16_VSX(uint8_t* p, int s, int thresh, int ithresh,
                          int hev_thresh) {
  u8x16 p3 = vec_xl(0, p - 4 * s), p2 = vec_xl(0, p - 3 * s);
  u8x16 p1 = vec_xl(0, p - 2 * s), p0 = vec_xl(0, p - s);
  u8x16 q0 = vec_xl(0, p), q1 = vec_xl(0, p + s);
  u8x16 q2 = vec_xl(0, p + 2 * s), q3 = vec_xl(0, p + 3 * s);
  const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
  DoFilter6(&p2, &p1, &p0, &q0, &q1, &q2, m, hev_thresh);
  vec_xst(p2, 0, p - 3 * s); vec_xst(p1, 0, p - 2 * s); vec_xst(p0, 0, p - s);
  vec_xst(q0, 0, p); vec_xst(q1, 0, p + s); vec_xst(q2, 0, p + 2 * s);
}

static void VFilter16i_VSX(uint8_t* p, int s, int thresh, int ithresh,
                           int hev_thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4 * s;
    u8x16 p3 = vec_xl(0, p - 4 * s), p2 = vec_xl(0, p - 3 * s);
    u8x16 p1 = vec_xl(0, p - 2 * s), p0 = vec_xl(0, p - s);
    u8x16 q0 = vec_xl(0, p), q1 = vec_xl(0, p + s);
    u8x16 q2 = vec_xl(0, p + 2 * s), q3 = vec_xl(0, p + 3 * s);
    const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    DoFilter4(&p1, &p0, &q0, &q1, m, hev_thresh);
    vec_xst(p1, 0, p - 2 * s); vec_xst(p0, 0, p - s);
    vec_xst(q0, 0, p); vec_xst(q1, 0, p + s);
  }
}

// Complex horizontal luma: two 16x4 transposes around the vertical edge feed
// the same DoFilter4/DoFilter6 used by the vertical variants.
static void HFilter16_VSX(uint8_t* p, int s, int thresh, int ithresh,
                          int hev_thresh) {
  uint8_t* const b = p - 4;
  u8x16 p3, p2, p1, p0, q0, q1, q2, q3;
  Load16x4(b, b + 8 * s, s, &p3, &p2, &p1, &p0);
  Load16x4(p, p + 8 * s, s, &q0, &q1, &q2, &q3);
  const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
  DoFilter6(&p2, &p1, &p0, &q0, &q1, &q2, m, hev_thresh);
  Store16x4(p3, p2, p1, p0, b, b + 8 * s, s);
  Store16x4(q0, q1, q2, q3, p, p + 8 * s, s);
}

static void HFilter16i_VSX(uint8_t* p, int s, int thresh, int ithresh,
                           int hev_thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4;
    u8x16 p3, p2, p1, p0, q0, q1, q2, q3;
    Load16x4(p - 4, p - 4 + 8 * s, s, &p3, &p2, &p1, &p0);
    Load16x4(p, p + 8 * s, s, &q0, &q1, &q2, &q3);
    const u8x16 m =
        ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    DoFilter4(&p1, &p0, &q0, &q1, m, hev_thresh);
    Store16x4(p1, p0, q0, q1, p - 2, p - 2 + 8 * s, s);
  }
}

//------------------------------------------------------------------------------
// Complex chroma filtering: operate on the u and v planes (8 wide) together.

// Pack 8 u-bytes into the low half and 8 v-bytes into the high half.
static WEBP_INLINE u8x16 LoadUV(const uint8_t* WEBP_RESTRICT u,
                                const uint8_t* WEBP_RESTRICT v) {
  unsigned char b[16];
  memcpy(b, u, 8);
  memcpy(b + 8, v, 8);
  return vec_xl(0, b);
}

static WEBP_INLINE void StoreUV(u8x16 x, uint8_t* WEBP_RESTRICT u,
                                uint8_t* WEBP_RESTRICT v) {
  unsigned char b[16];
  memcpy(b, &x, 16);
  memcpy(u, b, 8);
  memcpy(v, b + 8, 8);
}

static void VFilter8_VSX(uint8_t* WEBP_RESTRICT u, uint8_t* WEBP_RESTRICT v,
                         int s, int thresh, int ithresh, int hev_thresh) {
  u8x16 p3 = LoadUV(u - 4 * s, v - 4 * s), p2 = LoadUV(u - 3 * s, v - 3 * s);
  u8x16 p1 = LoadUV(u - 2 * s, v - 2 * s), p0 = LoadUV(u - s, v - s);
  u8x16 q0 = LoadUV(u, v), q1 = LoadUV(u + s, v + s);
  u8x16 q2 = LoadUV(u + 2 * s, v + 2 * s), q3 = LoadUV(u + 3 * s, v + 3 * s);
  const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
  DoFilter6(&p2, &p1, &p0, &q0, &q1, &q2, m, hev_thresh);
  StoreUV(p2, u - 3 * s, v - 3 * s); StoreUV(p1, u - 2 * s, v - 2 * s);
  StoreUV(p0, u - s, v - s); StoreUV(q0, u, v);
  StoreUV(q1, u + s, v + s); StoreUV(q2, u + 2 * s, v + 2 * s);
}

static void VFilter8i_VSX(uint8_t* WEBP_RESTRICT u, uint8_t* WEBP_RESTRICT v,
                          int s, int thresh, int ithresh, int hev_thresh) {
  u += 4 * s; v += 4 * s;
  u8x16 p3 = LoadUV(u - 4 * s, v - 4 * s), p2 = LoadUV(u - 3 * s, v - 3 * s);
  u8x16 p1 = LoadUV(u - 2 * s, v - 2 * s), p0 = LoadUV(u - s, v - s);
  u8x16 q0 = LoadUV(u, v), q1 = LoadUV(u + s, v + s);
  u8x16 q2 = LoadUV(u + 2 * s, v + 2 * s), q3 = LoadUV(u + 3 * s, v + 3 * s);
  const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
  DoFilter4(&p1, &p0, &q0, &q1, m, hev_thresh);
  StoreUV(p1, u - 2 * s, v - 2 * s); StoreUV(p0, u - s, v - s);
  StoreUV(q0, u, v); StoreUV(q1, u + s, v + s);
}

static void HFilter8_VSX(uint8_t* WEBP_RESTRICT u, uint8_t* WEBP_RESTRICT v,
                         int s, int thresh, int ithresh, int hev_thresh) {
  u8x16 p3, p2, p1, p0, q0, q1, q2, q3;
  Load16x4(u - 4, v - 4, s, &p3, &p2, &p1, &p0);
  Load16x4(u, v, s, &q0, &q1, &q2, &q3);
  const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
  DoFilter6(&p2, &p1, &p0, &q0, &q1, &q2, m, hev_thresh);
  Store16x4(p3, p2, p1, p0, u - 4, v - 4, s);
  Store16x4(q0, q1, q2, q3, u, v, s);
}

static void HFilter8i_VSX(uint8_t* WEBP_RESTRICT u, uint8_t* WEBP_RESTRICT v,
                          int s, int thresh, int ithresh, int hev_thresh) {
  u += 4; v += 4;
  u8x16 p3, p2, p1, p0, q0, q1, q2, q3;
  Load16x4(u - 4, v - 4, s, &p3, &p2, &p1, &p0);
  Load16x4(u, v, s, &q0, &q1, &q2, &q3);
  const u8x16 m = ComplexMask(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
  DoFilter4(&p1, &p0, &q0, &q1, m, hev_thresh);
  Store16x4(p1, p0, q0, q1, u - 2, v - 2, s);
}

//------------------------------------------------------------------------------
// Intra prediction (16x16 luma, 8x8 chroma). DC top-sums are scalar (the SIMD
// win is the block fill); TrueMotion/VE/HE are vectorized.

static WEBP_INLINE void Put16(uint8_t v, uint8_t* dst) {
  const u8x16 x = vec_splats(v);
  int j;
  for (j = 0; j < 16; ++j) vec_xst(x, 0, dst + j * BPS);
}
static WEBP_INLINE void Put8x8uv(uint8_t v, uint8_t* dst) {
  const u8x16 x = vec_splats(v);
  unsigned char b[16];
  int j;
  memcpy(b, &x, 16);
  for (j = 0; j < 8; ++j) memcpy(dst + j * BPS, b, 8);
}

static void VE16_VSX(uint8_t* dst) {
  const u8x16 top = vec_xl(0, dst - BPS);
  int j;
  for (j = 0; j < 16; ++j) vec_xst(top, 0, dst + j * BPS);
}
static void HE16_VSX(uint8_t* dst) {
  int j;
  for (j = 0; j < 16; ++j) vec_xst(vec_splats(dst[-1 + j * BPS]), 0, dst + j * BPS);
}
static void DC16_VSX(uint8_t* dst) {
  int s = 16, j;
  for (j = 0; j < 16; ++j) s += dst[-BPS + j] + dst[-1 + j * BPS];
  Put16(s >> 5, dst);
}
static void DC16NoTop_VSX(uint8_t* dst) {
  int s = 8, j;
  for (j = 0; j < 16; ++j) s += dst[-1 + j * BPS];
  Put16(s >> 4, dst);
}
static void DC16NoLeft_VSX(uint8_t* dst) {
  int s = 8, j;
  for (j = 0; j < 16; ++j) s += dst[-BPS + j];
  Put16(s >> 4, dst);
}
static void DC16NoTopLeft_VSX(uint8_t* dst) { Put16(0x80, dst); }
static void TM16_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 t = vec_xl(0, dst - BPS);
  const i16x8 tl = (i16x8)vec_mergeh(t, zero), th = (i16x8)vec_mergel(t, zero);
  const int c = dst[-BPS - 1];
  int y;
  for (y = 0; y < 16; ++y) {
    const i16x8 b = vec_splats((short)(dst[-1 + y * BPS] - c));
    vec_xst((u8x16)vec_packsu(vec_add(b, tl), vec_add(b, th)), 0, dst + y * BPS);
  }
}

static void VE8uv_VSX(uint8_t* dst) {
  unsigned char t[8];
  int j;
  memcpy(t, dst - BPS, 8);
  for (j = 0; j < 8; ++j) memcpy(dst + j * BPS, t, 8);
}
static void DC8uv_VSX(uint8_t* dst) {
  int s = 8, j;
  for (j = 0; j < 8; ++j) s += dst[-BPS + j] + dst[-1 + j * BPS];
  Put8x8uv(s >> 4, dst);
}
static void DC8uvNoTop_VSX(uint8_t* dst) {
  int s = 4, j;
  for (j = 0; j < 8; ++j) s += dst[-1 + j * BPS];
  Put8x8uv(s >> 3, dst);
}
static void DC8uvNoLeft_VSX(uint8_t* dst) {
  int s = 4, j;
  for (j = 0; j < 8; ++j) s += dst[-BPS + j];
  Put8x8uv(s >> 3, dst);
}
static void DC8uvNoTopLeft_VSX(uint8_t* dst) { Put8x8uv(0x80, dst); }
static void TM8uv_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 t = vec_xl(0, dst - BPS);
  const i16x8 tl = (i16x8)vec_mergeh(t, zero);
  const int c = dst[-BPS - 1];
  int y;
  for (y = 0; y < 8; ++y) {
    const i16x8 b = vec_splats((short)(dst[-1 + y * BPS] - c));
    const u8x16 o = (u8x16)vec_packsu(vec_add(b, tl), vec_splats((short)0));
    unsigned char bb[16];
    memcpy(bb, &o, 16);
    memcpy(dst + y * BPS, bb, 8);
  }
}

//------------------------------------------------------------------------------
// 4x4 luma intra prediction. Whole-vector byte shifts window the edge samples:
//   srli_si128(x,n) == vec_sld(zero, x, 16 - n)
//   slli_si128(x,n) == vec_sld(x, zero, n)

#define SRLI(x, n) vec_sld(zero, (x), 16 - (n))
#define SLLI(x, n) vec_sld((x), zero, (n))
#define INS16(v, val, i) ((u8x16)vec_insert((short)(val), (i16x8)(v), (i)))
#define AVG3C(a, b, c) ((uint8_t)(((a) + 2 * (b) + (c) + 2) >> 2))

static WEBP_INLINE u8x16 Load64(const uint8_t* WEBP_RESTRICT p) {
  unsigned char b[16] = {0};
  memcpy(b, p, 8);
  return vec_xl(0, b);
}
static WEBP_INLINE uint32_t GetWord(u8x16 v) {
  unsigned char b[16];
  uint32_t r;
  memcpy(b, &v, 16);
  memcpy(&r, b, 4);
  return r;
}
static WEBP_INLINE u8x16 SetWord(uint32_t v) {
  unsigned char b[16] = {0};
  memcpy(b, &v, 4);
  return vec_xl(0, b);
}
static WEBP_INLINE void StoreWord(uint32_t v, uint8_t* dst) {
  memcpy(dst, &v, 4);
}

static void VE4_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0), one = vec_splats((unsigned char)1);
  const u8x16 A = Load64(dst - BPS - 1), B = SRLI(A, 1), C = SRLI(A, 2);
  const u8x16 a = vec_avg(A, C), lsb = vec_and(vec_xor(A, C), one);
  const u8x16 avg = vec_avg(vec_subs(a, lsb), B);
  const uint32_t v = GetWord(avg);
  int i;
  for (i = 0; i < 4; ++i) StoreWord(v, dst + i * BPS);
}
static void LD4_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0), one = vec_splats((unsigned char)1);
  const u8x16 A = Load64(dst - BPS), B = SRLI(A, 1), C = SRLI(A, 2);
  const u8x16 CH = INS16(C, dst[-BPS + 7], 3);
  const u8x16 a1 = vec_avg(A, CH), lsb = vec_and(vec_xor(A, CH), one);
  const u8x16 r = vec_avg(vec_subs(a1, lsb), B);
  StoreWord(GetWord(r), dst + 0 * BPS);
  StoreWord(GetWord(SRLI(r, 1)), dst + 1 * BPS);
  StoreWord(GetWord(SRLI(r, 2)), dst + 2 * BPS);
  StoreWord(GetWord(SRLI(r, 3)), dst + 3 * BPS);
}
static void VR4_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0), one = vec_splats((unsigned char)1);
  const int I = dst[-1 + 0 * BPS], J = dst[-1 + 1 * BPS], K = dst[-1 + 2 * BPS];
  const int X = dst[-1 - BPS];
  const u8x16 XA = Load64(dst - BPS - 1), A0 = SRLI(XA, 1);
  const u8x16 abcd = vec_avg(XA, A0);
  const u8x16 IX = INS16(SLLI(XA, 1), (I | (X << 8)), 0);
  const u8x16 a1 = vec_avg(IX, A0), lsb = vec_and(vec_xor(IX, A0), one);
  const u8x16 efgh = vec_avg(vec_subs(a1, lsb), XA);
  StoreWord(GetWord(abcd), dst + 0 * BPS);
  StoreWord(GetWord(efgh), dst + 1 * BPS);
  StoreWord(GetWord(SLLI(abcd, 1)), dst + 2 * BPS);
  StoreWord(GetWord(SLLI(efgh, 1)), dst + 3 * BPS);
  dst[0 + 2 * BPS] = AVG3C(J, I, X);
  dst[0 + 3 * BPS] = AVG3C(K, J, I);
}
static void VL4_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0), one = vec_splats((unsigned char)1);
  const u8x16 A = Load64(dst - BPS), B = SRLI(A, 1), C = SRLI(A, 2);
  const u8x16 a1 = vec_avg(A, B), a2 = vec_avg(C, B), a3 = vec_avg(a1, a2);
  const u8x16 lsb1 = vec_and(vec_xor(a1, a2), one);
  const u8x16 abbc = vec_or(vec_xor(A, B), vec_xor(C, B));
  const u8x16 a4 = vec_subs(a3, vec_and(abbc, lsb1));
  const uint32_t extra = GetWord(SRLI(a4, 4));
  StoreWord(GetWord(a1), dst + 0 * BPS);
  StoreWord(GetWord(a4), dst + 1 * BPS);
  StoreWord(GetWord(SRLI(a1, 1)), dst + 2 * BPS);
  StoreWord(GetWord(SRLI(a4, 1)), dst + 3 * BPS);
  dst[3 + 2 * BPS] = (extra >> 0) & 0xff;
  dst[3 + 3 * BPS] = (extra >> 8) & 0xff;
}
static void RD4_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0), one = vec_splats((unsigned char)1);
  const uint32_t I = dst[-1 + 0 * BPS], J = dst[-1 + 1 * BPS];
  const uint32_t K = dst[-1 + 2 * BPS], L = dst[-1 + 3 * BPS];
  const u8x16 XA = Load64(dst - BPS - 1);
  const u8x16 all = vec_or(SetWord((uint32_t)(L | (K << 8) | (J << 16) | (I << 24))),
                           SLLI(XA, 4));
  const u8x16 k1 = SRLI(all, 1), j2 = SRLI(all, 2);
  const u8x16 a1 = vec_avg(j2, all), lsb = vec_and(vec_xor(j2, all), one);
  const u8x16 r = vec_avg(vec_subs(a1, lsb), k1);
  StoreWord(GetWord(r), dst + 3 * BPS);
  StoreWord(GetWord(SRLI(r, 1)), dst + 2 * BPS);
  StoreWord(GetWord(SRLI(r, 2)), dst + 1 * BPS);
  StoreWord(GetWord(SRLI(r, 3)), dst + 0 * BPS);
}
static void TM4_VSX(uint8_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 t = Load64(dst - BPS);
  const i16x8 tb = (i16x8)vec_mergeh(t, zero);
  const int c = dst[-BPS - 1];
  int y;
  for (y = 0; y < 4; ++y) {
    const i16x8 b = vec_splats((short)(dst[-1 + y * BPS] - c));
    const u8x16 o = (u8x16)vec_packsu(vec_add(b, tb), vec_splats((short)0));
    StoreWord(GetWord(o), dst + y * BPS);
  }
}
#undef SRLI
#undef SLLI
#undef INS16
#undef AVG3C

extern void VP8DspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8DspInitVSX(void) {
  VP8Transform = Transform_VSX;
  VP8SimpleVFilter16 = SimpleVFilter16_VSX;
  VP8SimpleVFilter16i = SimpleVFilter16i_VSX;
  VP8SimpleHFilter16 = SimpleHFilter16_VSX;
  VP8SimpleHFilter16i = SimpleHFilter16i_VSX;
  VP8VFilter16 = VFilter16_VSX;
  VP8VFilter16i = VFilter16i_VSX;
  VP8HFilter16 = HFilter16_VSX;
  VP8HFilter16i = HFilter16i_VSX;
  VP8VFilter8 = VFilter8_VSX;
  VP8VFilter8i = VFilter8i_VSX;
  VP8HFilter8 = HFilter8_VSX;
  VP8HFilter8i = HFilter8i_VSX;

  VP8PredLuma16[0] = DC16_VSX;
  VP8PredLuma16[1] = TM16_VSX;
  VP8PredLuma16[2] = VE16_VSX;
  VP8PredLuma16[3] = HE16_VSX;
  VP8PredLuma16[4] = DC16NoTop_VSX;
  VP8PredLuma16[5] = DC16NoLeft_VSX;
  VP8PredLuma16[6] = DC16NoTopLeft_VSX;
  VP8PredChroma8[0] = DC8uv_VSX;
  VP8PredChroma8[1] = TM8uv_VSX;
  VP8PredChroma8[2] = VE8uv_VSX;
  VP8PredChroma8[4] = DC8uvNoTop_VSX;
  VP8PredChroma8[5] = DC8uvNoLeft_VSX;
  VP8PredChroma8[6] = DC8uvNoTopLeft_VSX;
  VP8PredLuma4[1] = TM4_VSX;
  VP8PredLuma4[2] = VE4_VSX;
  VP8PredLuma4[4] = RD4_VSX;
  VP8PredLuma4[5] = VR4_VSX;
  VP8PredLuma4[6] = LD4_VSX;
  VP8PredLuma4[7] = VL4_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8DspInitVSX)

#endif  // WEBP_USE_VSX
