// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC) version of lossless functions.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector signed int i32x4;

// Signed multiply-high of 16-bit lanes: (a * b) >> 16, matching
// _mm_mulhi_epi16.
static WEBP_INLINE i16x8 MulHiS16(i16x8 a, i16x8 b) {
  const u32x4 sh = vec_splats((unsigned int)16);
  const i32x4 e = vec_sra(vec_mule(a, b), sh);
  const i32x4 o = vec_sra(vec_mulo(a, b), sh);
  return (i16x8)vec_pack(vec_mergeh(e, o), vec_mergel(e, o));
}

//------------------------------------------------------------------------------
// Color transforms.

static void AddGreenToBlueAndRed_VSX(const uint32_t* src, int num_pixels,
                                     uint32_t* dst) {
  const u8x16 zero = vec_splats((unsigned char)0);
  // Replicate the green byte (offset 1 of each pixel) into the blue/red slots.
  const u8x16 kSpreadGreen = {1, 16, 1, 16, 5,  16, 5,  16,
                              9, 16, 9, 16, 13, 16, 13, 16};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&src[i]);
    const u8x16 g = vec_perm(in, zero, kSpreadGreen);  // 0 g 0 g per pixel
    vec_xst((u32x4)vec_add(in, g), 0, &dst[i]);
  }
  if (i != num_pixels) {
    VP8LAddGreenToBlueAndRed_C(src + i, num_pixels - i, dst + i);
  }
}

static void TransformColorInverse_VSX(const VP8LMultipliers* const m,
                                      const uint32_t* src, int num_pixels,
                                      uint32_t* dst) {
// sign-extended multiplying constants, pre-shifted by 5 (see lossless_sse2.c).
#define CST(X) (((int16_t)((m->X) << 8)) >> 5)
  const i16x8 mults_rb =
      (i16x8)vec_splats((int)(((uint32_t)(uint16_t)CST(green_to_red) << 16) |
                              ((uint16_t)CST(green_to_blue))));
  const i16x8 mults_b2 =
      (i16x8)vec_splats((int)((uint32_t)(uint16_t)CST(red_to_blue) << 16));
#undef CST
  const u8x16 zero = vec_splats((unsigned char)0);
  const u32x4 mask_ag = vec_splats((uint32_t)0xff00ff00);  // alpha/green
  const u16x8 sh8_16 = vec_splats((unsigned short)8);
  const u32x4 sh8_32 = vec_splats((unsigned int)8);
  // Broadcast the green byte (offset 1) into the high byte of both 16-bit
  // halves of each pixel: yields g << 8 in each lane.
  const u8x16 kGreenHi = {16, 1, 16, 1, 16, 5,  16, 5,
                          16, 9, 16, 9, 16, 13, 16, 13};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&src[i]);
    const u8x16 A = (u8x16)vec_and((u32x4)in, mask_ag);   // a 0 g 0
    const i16x8 C = (i16x8)vec_perm(A, zero, kGreenHi);   // g0g0 (g << 8)
    const u8x16 D = (u8x16)MulHiS16(C, mults_rb);         // x dr x db1
    const u8x16 E = vec_add(in, D);                       // x r' x b'
    const u16x8 F = vec_sl((u16x8)E, sh8_16);             // r' 0 b' 0
    const u8x16 G = (u8x16)MulHiS16((i16x8)F, mults_b2);  // x db2 0 0
    const u8x16 H = (u8x16)vec_sr((u32x4)G, sh8_32);      // 0 x db2 0
    const u16x8 I = (u16x8)vec_add(H, (u8x16)F);          // r' x b'' 0
    const u8x16 J = (u8x16)vec_sr(I, sh8_16);             // 0 r' 0 b''
    vec_xst(vec_or((u32x4)J, (u32x4)A), 0, &dst[i]);
  }
  if (i != num_pixels) {
    VP8LTransformColorInverse_C(m, src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Color-space conversion functions.

static void ConvertBGRAToRGBA_VSX(const uint32_t* WEBP_RESTRICT src,
                                  int num_pixels, uint8_t* WEBP_RESTRICT dst) {
  // Swap the blue (offset 0) and red (offset 2) bytes of each pixel.
  const u8x16 kSwapBR = {2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&src[i]);
    vec_xst(vec_perm(in, in, kSwapBR), 0, &dst[4 * i]);
  }
  if (i != num_pixels) {
    VP8LConvertBGRAToRGBA_C(src + i, num_pixels - i, dst + 4 * i);
  }
}

static void ConvertBGRAToRGB_VSX(const uint32_t* WEBP_RESTRICT src,
                                 int num_pixels, uint8_t* WEBP_RESTRICT dst) {
  // BGRA -> RGB: gather R,G,B (offsets 2,1,0) of each pixel, drop alpha.
  const u8x16 kToRGB = {2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0, 0, 0, 0};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&src[i]);
    const u8x16 out = vec_perm(in, in, kToRGB);
    memcpy(&dst[3 * i], &out, 12);
  }
  if (i != num_pixels) {
    VP8LConvertBGRAToRGB_C(src + i, num_pixels - i, dst + 3 * i);
  }
}

static void ConvertBGRAToBGR_VSX(const uint32_t* WEBP_RESTRICT src,
                                 int num_pixels, uint8_t* WEBP_RESTRICT dst) {
  // BGRA -> BGR: gather B,G,R (offsets 0,1,2) of each pixel, drop alpha.
  const u8x16 kToBGR = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0};
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 in = (u8x16)vec_xl(0, (uint32_t*)&src[i]);
    const u8x16 out = vec_perm(in, in, kToBGR);
    memcpy(&dst[3 * i], &out, 12);
  }
  if (i != num_pixels) {
    VP8LConvertBGRAToBGR_C(src + i, num_pixels - i, dst + 3 * i);
  }
}

//------------------------------------------------------------------------------
// Predictor transform.

// Byte-wise shifts of the whole register (little-endian _mm_s{l,r}li_si128).
#define SLLI(x, n) vec_sld((x), kZero, (n))
#define SRLI(x, n) vec_sld(kZero, (x), 16 - (n))
static const u8x16 kZero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Per-byte floor average (a + b) >> 1, matching the C Average2().
static WEBP_INLINE u8x16 Average2_u8(u8x16 a, u8x16 b) {
  const u8x16 one = vec_splats((unsigned char)1);
  const u8x16 avg1 = vec_avg(a, b);  // (a + b + 1) >> 1
  return vec_sub(avg1, vec_and(vec_xor(a, b), one));
}

static WEBP_INLINE u32x4 Lane0(uint32_t v) {
  const u32x4 r = {v, 0, 0, 0};
  return r;
}

// Single-pixel helpers operating on the low 32-bit lane only.
static WEBP_INLINE u16x8 Unpack16(uint32_t a) {
  return (u16x8)vec_mergeh((u8x16)Lane0(a), kZero);
}

static WEBP_INLINE uint32_t Average2_VSX(uint32_t a0, uint32_t a1) {
  return vec_extract((u32x4)Average2_u8((u8x16)Lane0(a0), (u8x16)Lane0(a1)), 0);
}

static WEBP_INLINE u16x8 Average2_16(uint32_t a0, uint32_t a1) {
  const u16x8 one = vec_splats((unsigned short)1);
  return vec_sr(vec_add(Unpack16(a0), Unpack16(a1)), one);
}

static WEBP_INLINE uint32_t Average3_VSX(uint32_t a0, uint32_t a1,
                                         uint32_t a2) {
  const u16x8 one = vec_splats((unsigned short)1);
  const u16x8 avg1 = Average2_16(a0, a2);
  const u16x8 avg2 = vec_sr(vec_add(avg1, Unpack16(a1)), one);
  return vec_extract((u32x4)vec_packsu((i16x8)avg2, (i16x8)avg2), 0);
}

static WEBP_INLINE uint32_t Average4_VSX(uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3) {
  const u16x8 one = vec_splats((unsigned short)1);
  const u16x8 avg1 = Average2_16(a0, a1);
  const u16x8 avg2 = Average2_16(a2, a3);
  const u16x8 avg3 = vec_sr(vec_add(avg1, avg2), one);
  return vec_extract((u32x4)vec_packsu((i16x8)avg3, (i16x8)avg3), 0);
}

static WEBP_INLINE uint32_t ClampedAddSubtractFull_VSX(uint32_t c0, uint32_t c1,
                                                       uint32_t c2) {
  const i16x8 v =
      vec_sub((i16x8)vec_add(Unpack16(c0), Unpack16(c1)), (i16x8)Unpack16(c2));
  return vec_extract((u32x4)vec_packsu(v, v), 0);
}

static WEBP_INLINE uint32_t ClampedAddSubtractHalf_VSX(uint32_t c0, uint32_t c1,
                                                       uint32_t c2) {
  const u16x8 one = vec_splats((unsigned short)1);
  const u16x8 C0 = Unpack16(c0);
  const u16x8 C1 = Unpack16(c1);
  const u16x8 B0 = Unpack16(c2);
  const u16x8 A0 = vec_sr(vec_add(C1, C0), one);  // ave
  const i16x8 A1 = vec_sub((i16x8)A0, (i16x8)B0);
  const i16x8 BgtA = (i16x8)vec_cmpgt(B0, A0);  // 0 or -1
  const i16x8 A2 = vec_sub(A1, BgtA);
  const i16x8 A3 = vec_sra(A2, one);
  const i16x8 A4 = vec_add((i16x8)A0, A3);
  return vec_extract((u32x4)vec_packsu(A4, A4), 0);
}

static WEBP_INLINE uint32_t Select_VSX(uint32_t a, uint32_t b, uint32_t c) {
  const u8x16 A = (u8x16)Lane0(a);
  const u8x16 B = (u8x16)Lane0(b);
  const u8x16 C = (u8x16)Lane0(c);
  const u32x4 sa = vec_sum4s(vec_or(vec_subs(A, C), vec_subs(C, A)),
                             vec_splats((unsigned int)0));
  const u32x4 sb = vec_sum4s(vec_or(vec_subs(B, C), vec_subs(C, B)),
                             vec_splats((unsigned int)0));
  return vec_extract((u32x4)vec_cmpgt(sb, sa), 0) ? b : a;
}

static uint32_t Predictor5_VSX(const uint32_t* const left,
                               const uint32_t* const top) {
  return Average3_VSX(*left, top[0], top[1]);
}
static uint32_t Predictor6_VSX(const uint32_t* const left,
                               const uint32_t* const top) {
  return Average2_VSX(*left, top[-1]);
}
static uint32_t Predictor7_VSX(const uint32_t* const left,
                               const uint32_t* const top) {
  return Average2_VSX(*left, top[0]);
}
static uint32_t Predictor13_VSX(const uint32_t* const left,
                                const uint32_t* const top) {
  return ClampedAddSubtractHalf_VSX(*left, top[0], top[-1]);
}

static void PredictorAdd0_VSX(const uint32_t* in, const uint32_t* upper,
                              int num_pixels, uint32_t* WEBP_RESTRICT out) {
  const u8x16 black = (u8x16)vec_splats((uint32_t)ARGB_BLACK);
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    vec_xst((u32x4)vec_add(src, black), 0, &out[i]);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_C[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

static void PredictorAdd1_VSX(const uint32_t* in, const uint32_t* upper,
                              int num_pixels, uint32_t* WEBP_RESTRICT out) {
  u32x4 prev = vec_splats(out[-1]);
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u8x16 sum0 = vec_add(src, SLLI(src, 4));    // a | a+b | b+c | c+d
    const u8x16 sum1 = vec_add(sum0, SLLI(sum0, 8));  // running sum
    const u8x16 res = vec_add(sum1, (u8x16)prev);
    vec_xst((u32x4)res, 0, &out[i]);
    prev = vec_splat((u32x4)res, 3);  // replicate last pixel
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_C[1](in + i, upper + i, num_pixels - i, out + i);
  }
}

#define GENERATE_PREDICTOR_1_VSX(X, IN)                                        \
  static void PredictorAdd##X##_VSX(const uint32_t* in, const uint32_t* upper, \
                                    int num_pixels,                            \
                                    uint32_t* WEBP_RESTRICT out) {             \
    int i;                                                                     \
    for (i = 0; i + 4 <= num_pixels; i += 4) {                                 \
      const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);                   \
      const u8x16 other = (u8x16)vec_xl(0, (uint32_t*)&(IN));                  \
      vec_xst((u32x4)vec_add(src, other), 0, &out[i]);                         \
    }                                                                          \
    if (i != num_pixels) {                                                     \
      VP8LPredictorsAdd_C[(X)](in + i, upper + i, num_pixels - i, out + i);    \
    }                                                                          \
  }
GENERATE_PREDICTOR_1_VSX(2, upper[i])      // Top.
GENERATE_PREDICTOR_1_VSX(3, upper[i + 1])  // Top-right.
GENERATE_PREDICTOR_1_VSX(4, upper[i - 1])  // Top-left.
#undef GENERATE_PREDICTOR_1_VSX

// Predictors 5, 6, 7, 13 use integer averages and cannot be accumulated in
// parallel, so use the generic one-pixel-at-a-time batch.
GENERATE_PREDICTOR_ADD(Predictor5_VSX, PredictorAdd5_VSX)
GENERATE_PREDICTOR_ADD(Predictor6_VSX, PredictorAdd6_VSX)
GENERATE_PREDICTOR_ADD(Predictor7_VSX, PredictorAdd7_VSX)
GENERATE_PREDICTOR_ADD(Predictor13_VSX, PredictorAdd13_VSX)

#define GENERATE_PREDICTOR_2_VSX(X, IN)                                        \
  static void PredictorAdd##X##_VSX(const uint32_t* in, const uint32_t* upper, \
                                    int num_pixels,                            \
                                    uint32_t* WEBP_RESTRICT out) {             \
    int i;                                                                     \
    for (i = 0; i + 4 <= num_pixels; i += 4) {                                 \
      const u8x16 Tother = (u8x16)vec_xl(0, (uint32_t*)&(IN));                 \
      const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);                  \
      const u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);                   \
      vec_xst((u32x4)vec_add(Average2_u8(T, Tother), src), 0, &out[i]);        \
    }                                                                          \
    if (i != num_pixels) {                                                     \
      VP8LPredictorsAdd_C[(X)](in + i, upper + i, num_pixels - i, out + i);    \
    }                                                                          \
  }
GENERATE_PREDICTOR_2_VSX(8, upper[i - 1])  // Average TL, T.
GENERATE_PREDICTOR_2_VSX(9, upper[i + 1])  // Average T, TR.
#undef GENERATE_PREDICTOR_2_VSX

// Predictor10: average of (average(L, TL), average(T, TR)).
static void PredictorAdd10_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  u8x16 L = (u8x16)Lane0(out[-1]);
  int i, k;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TR = (u8x16)vec_xl(0, (uint32_t*)&upper[i + 1]);
    u8x16 avgTTR = Average2_u8(T, TR);
    for (k = 0; k < 4; ++k) {
      const u8x16 avg = Average2_u8(avgTTR, Average2_u8(L, TL));
      L = vec_add(avg, src);
      out[i + k] = vec_extract((u32x4)L, 0);
      avgTTR = SRLI(avgTTR, 4);
      TL = SRLI(TL, 4);
      src = SRLI(src, 4);
    }
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_C[10](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor11: select between T and L based on |T-TL| vs |L-TL|.
static void PredictorAdd11_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  const u32x4 z32 = vec_splats((unsigned int)0);
  u8x16 L = (u8x16)Lane0(out[-1]);
  int i, k;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    u8x16 pa = (u8x16)vec_sum4s(vec_or(vec_subs(T, TL), vec_subs(TL, T)), z32);
    for (k = 0; k < 4; ++k) {
      const u32x4 pb = vec_sum4s(vec_or(vec_subs(L, TL), vec_subs(TL, L)), z32);
      const u32x4 mask = (u32x4)vec_cmpgt(pb, (u32x4)pa);  // pb > pa ? L : T
      const u8x16 pred = vec_sel(T, L, (u8x16)mask);
      L = vec_add(src, pred);
      out[i + k] = vec_extract((u32x4)L, 0);
      T = SRLI(T, 4);
      TL = SRLI(TL, 4);
      src = SRLI(src, 4);
      pa = SRLI(pa, 4);
    }
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_C[11](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor12: ClampedAddSubtractFull. 'L' is kept unpacked to 16 bits in the
// low 4 lanes; 'diff' (= T - TL) holds two pixels, the active one in lanes 0-3.
#define DO_PRED12(DIFF)                                   \
  do {                                                    \
    const i16x8 all = vec_add((i16x8)L, (DIFF));          \
    const u8x16 res = vec_add(src, vec_packsu(all, all)); \
    out[i + out_idx++] = vec_extract((u32x4)res, 0);      \
    L = (u16x8)vec_mergeh(res, kZero);                    \
  } while (0)

static void PredictorAdd12_VSX(const uint32_t* in, const uint32_t* upper,
                               int num_pixels, uint32_t* WEBP_RESTRICT out) {
  u16x8 L = Unpack16(out[-1]);
  int i;
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    int out_idx = 0;
    u8x16 src = (u8x16)vec_xl(0, (uint32_t*)&in[i]);
    const u8x16 T = (u8x16)vec_xl(0, (uint32_t*)&upper[i]);
    const u8x16 TL = (u8x16)vec_xl(0, (uint32_t*)&upper[i - 1]);
    // 16-bit gradient basis T - TL for the four pixels (low and high halves).
    i16x8 diff_lo =
        vec_sub((i16x8)vec_mergeh(T, kZero), (i16x8)vec_mergeh(TL, kZero));
    i16x8 diff_hi =
        vec_sub((i16x8)vec_mergel(T, kZero), (i16x8)vec_mergel(TL, kZero));
    DO_PRED12(diff_lo);
    diff_lo = (i16x8)SRLI((u8x16)diff_lo, 8);
    src = SRLI(src, 4);
    DO_PRED12(diff_lo);
    src = SRLI(src, 4);
    DO_PRED12(diff_hi);
    diff_hi = (i16x8)SRLI((u8x16)diff_hi, 8);
    src = SRLI(src, 4);
    DO_PRED12(diff_hi);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_C[12](in + i, upper + i, num_pixels - i, out + i);
  }
}
#undef DO_PRED12

#undef SLLI
#undef SRLI

//------------------------------------------------------------------------------

extern void VP8LDspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitVSX(void) {
  VP8LPredictorsAdd[0] = PredictorAdd0_VSX;
  VP8LPredictorsAdd[1] = PredictorAdd1_VSX;
  VP8LPredictorsAdd[2] = PredictorAdd2_VSX;
  VP8LPredictorsAdd[3] = PredictorAdd3_VSX;
  VP8LPredictorsAdd[4] = PredictorAdd4_VSX;
  VP8LPredictorsAdd[5] = PredictorAdd5_VSX;
  VP8LPredictorsAdd[6] = PredictorAdd6_VSX;
  VP8LPredictorsAdd[7] = PredictorAdd7_VSX;
  VP8LPredictorsAdd[8] = PredictorAdd8_VSX;
  VP8LPredictorsAdd[9] = PredictorAdd9_VSX;
  VP8LPredictorsAdd[10] = PredictorAdd10_VSX;
  VP8LPredictorsAdd[11] = PredictorAdd11_VSX;
  VP8LPredictorsAdd[12] = PredictorAdd12_VSX;
  VP8LPredictorsAdd[13] = PredictorAdd13_VSX;

  VP8LAddGreenToBlueAndRed = AddGreenToBlueAndRed_VSX;
  VP8LTransformColorInverse = TransformColorInverse_VSX;
  VP8LConvertBGRAToRGBA = ConvertBGRAToRGBA_VSX;
  VP8LConvertBGRAToRGB = ConvertBGRAToRGB_VSX;
  VP8LConvertBGRAToBGR = ConvertBGRAToBGR_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8LDspInitVSX)

#endif  // WEBP_USE_VSX
