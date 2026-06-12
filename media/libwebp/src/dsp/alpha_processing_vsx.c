// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC) version of alpha processing functions.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector signed int i32x4;

//------------------------------------------------------------------------------
// Alpha dispatch / extraction.

static int DispatchAlpha_VSX(const uint8_t* WEBP_RESTRICT alpha,
                             int alpha_stride, int width, int height,
                             uint8_t* WEBP_RESTRICT dst, int dst_stride) {
  uint32_t alpha_and = 0xff;
  int i, j, k;
  const u8x16 zero = vec_splats((unsigned char)0);
  const u16x8 z16 = vec_splats((unsigned short)0);
  const u32x4 a_mask = vec_splats((uint32_t)0xff);  // selects the low byte
  u8x16 all_and = vec_splats((unsigned char)0xff);
  const int limit = width & ~15;

  for (j = 0; j < height; ++j) {
    uint8_t* ptr = dst;
    for (i = 0; i < limit; i += 16) {
      const u8x16 a0 = vec_xl(0, (unsigned char*)&alpha[i]);
      // Spread the 16 alpha bytes to the low byte of 16 32-bit lanes.
      const u16x8 a1_lo = (u16x8)vec_mergeh(a0, zero);
      const u16x8 a1_hi = (u16x8)vec_mergel(a0, zero);
      const u32x4 s0 = (u32x4)vec_mergeh(a1_lo, z16);
      const u32x4 s1 = (u32x4)vec_mergel(a1_lo, z16);
      const u32x4 s2 = (u32x4)vec_mergeh(a1_hi, z16);
      const u32x4 s3 = (u32x4)vec_mergel(a1_hi, z16);
      const u32x4* spread[4] = {&s0, &s1, &s2, &s3};
      for (k = 0; k < 4; ++k) {
        const u32x4 d = vec_xl(0, (uint32_t*)(ptr + 16 * k));
        vec_xst(vec_sel(d, *spread[k], a_mask), 0, (uint32_t*)(ptr + 16 * k));
      }
      all_and = vec_and(all_and, a0);
      ptr += 64;
    }
    for (; i < width; ++i) {
      const uint32_t alpha_value = alpha[i];
      dst[4 * i] = alpha_value;
      alpha_and &= alpha_value;
    }
    alpha += alpha_stride;
    dst += dst_stride;
  }
  {
    unsigned char tmp[16];
    memcpy(tmp, &all_and, 16);
    for (k = 0; k < 16; ++k) alpha_and &= tmp[k];
  }
  return (alpha_and != 0xff);
}

static void DispatchAlphaToGreen_VSX(const uint8_t* WEBP_RESTRICT alpha,
                                     int alpha_stride, int width, int height,
                                     uint32_t* WEBP_RESTRICT dst,
                                     int dst_stride) {
  int i, j;
  const u8x16 zero = vec_splats((unsigned char)0);
  const u16x8 z16 = vec_splats((unsigned short)0);
  const int limit = width & ~15;
  for (j = 0; j < height; ++j) {
    for (i = 0; i < limit; i += 16) {
      const u8x16 a0 = vec_xl(0, (unsigned char*)&alpha[i]);
      // Place each alpha byte into the green slot (<< 8) of a 32-bit lane.
      const u16x8 a1_lo = (u16x8)vec_mergeh(zero, a0);  // note the 'zero' first
      const u16x8 a1_hi = (u16x8)vec_mergel(zero, a0);
      const u32x4 g0 = (u32x4)vec_mergeh(a1_lo, z16);
      const u32x4 g1 = (u32x4)vec_mergel(a1_lo, z16);
      const u32x4 g2 = (u32x4)vec_mergeh(a1_hi, z16);
      const u32x4 g3 = (u32x4)vec_mergel(a1_hi, z16);
      vec_xst(g0, 0, &dst[i + 0]);
      vec_xst(g1, 0, &dst[i + 4]);
      vec_xst(g2, 0, &dst[i + 8]);
      vec_xst(g3, 0, &dst[i + 12]);
    }
    for (; i < width; ++i) dst[i] = alpha[i] << 8;
    alpha += alpha_stride;
    dst += dst_stride;
  }
}

static int ExtractAlpha_VSX(const uint8_t* WEBP_RESTRICT argb, int argb_stride,
                            int width, int height, uint8_t* WEBP_RESTRICT alpha,
                            int alpha_stride) {
  uint32_t alpha_and = 0xff;
  int i, j, k;
  const u32x4 a_mask = vec_splats((uint32_t)0xff);  // keeps the low byte
  u8x16 all_and = vec_splats((unsigned char)0xff);
  const int limit = width & ~7;

  for (j = 0; j < height; ++j) {
    const uint32_t* src = (const uint32_t*)argb;
    for (i = 0; i < limit; i += 8) {
      const u32x4 a0 = vec_and(vec_xl(0, (uint32_t*)(src + 0)), a_mask);
      const u32x4 a1 = vec_and(vec_xl(0, (uint32_t*)(src + 4)), a_mask);
      const i16x8 c0 = vec_packs((i32x4)a0, (i32x4)a1);
      const u8x16 d0 = vec_packsu(c0, c0);  // 8 alpha bytes in the low half
      memcpy(&alpha[i], &d0, 8);
      all_and = vec_and(all_and, d0);
      src += 8;
    }
    for (; i < width; ++i) {
      const uint32_t alpha_value = argb[4 * i];
      alpha[i] = alpha_value;
      alpha_and &= alpha_value;
    }
    argb += argb_stride;
    alpha += alpha_stride;
  }
  {
    unsigned char tmp[16];
    memcpy(tmp, &all_and, 16);
    for (k = 0; k < 8; ++k) alpha_and &= tmp[k];
  }
  return (alpha_and == 0xff);
}

static void ExtractGreen_VSX(const uint32_t* WEBP_RESTRICT argb,
                             uint8_t* WEBP_RESTRICT alpha, int size) {
  int i;
  const u32x4 mask = vec_splats((uint32_t)0xff);
  const u32x4 sh8 = vec_splats((uint32_t)8);
  for (i = 0; i + 16 <= size; i += 16) {
    const u32x4 a0 =
        vec_and(vec_sr(vec_xl(0, (uint32_t*)(argb + i + 0)), sh8), mask);
    const u32x4 a1 =
        vec_and(vec_sr(vec_xl(0, (uint32_t*)(argb + i + 4)), sh8), mask);
    const u32x4 a2 =
        vec_and(vec_sr(vec_xl(0, (uint32_t*)(argb + i + 8)), sh8), mask);
    const u32x4 a3 =
        vec_and(vec_sr(vec_xl(0, (uint32_t*)(argb + i + 12)), sh8), mask);
    const i16x8 d0 = vec_packs((i32x4)a0, (i32x4)a1);
    const i16x8 d1 = vec_packs((i32x4)a2, (i32x4)a3);
    const u8x16 e = vec_packsu(d0, d1);
    vec_xst(e, 0, &alpha[i]);
  }
  for (; i < size; ++i) alpha[i] = argb[i] >> 8;
}

//------------------------------------------------------------------------------
// Premultiply.

#define MULTIPLIER(a) ((a) * 32897U)
#define PREMULTIPLY(x, m) (((x) * (m)) >> 23)

// Spreads the alpha lane across r/g/b and inserts 0xff in the alpha lane, for
// the two pixels packed in a 16-bit-per-channel vector. Built against the
// little-endian byte order; src is the channel vector, the second operand is
// an all-0xff vector.
static const u8x16 kSpreadAlphaLast = {6,  7,  6,  7,  6,  7,  16, 7,
                                       14, 15, 14, 15, 14, 15, 16, 15};
static const u8x16 kSpreadAlphaFirst = {16, 1, 0, 1, 0, 1, 0, 1,
                                        16, 9, 8, 9, 8, 9, 8, 9};

static WEBP_INLINE u16x8 MulHi16(u16x8 a, u16x8 b) {
  const u32x4 sh = vec_splats((unsigned int)16);
  const u32x4 e = vec_sr(vec_mule(a, b), sh);
  const u32x4 o = vec_sr(vec_mulo(a, b), sh);
  return vec_pack(vec_mergeh(e, o), vec_mergel(e, o));
}

static void ApplyAlphaMultiply_VSX(uint8_t* rgba, int alpha_first, int w, int h,
                                   int stride) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 allff = vec_splats((unsigned char)0xff);
  const u16x8 z16 = vec_splats((unsigned short)0);
  const u16x8 kMult = vec_splats((unsigned short)0x8081);
  const u16x8 sh7 = vec_splats((unsigned short)7);
  const u8x16 ctrl = alpha_first ? kSpreadAlphaFirst : kSpreadAlphaLast;
  const int kSpan = 4;
  while (h-- > 0) {
    uint8_t* const rgbx = rgba;
    int i;
    for (i = 0; i + kSpan <= w; i += kSpan) {
      const u8x16 argb0 = vec_xl(0, (unsigned char*)(rgbx + 4 * i));
      const u16x8 lo = (u16x8)vec_mergeh(argb0, zero);
      const u16x8 hi = (u16x8)vec_mergel(argb0, zero);
      const u16x8 a_lo = (u16x8)vec_perm((u8x16)lo, allff, ctrl);
      const u16x8 a_hi = (u16x8)vec_perm((u8x16)hi, allff, ctrl);
      const u16x8 A0lo = vec_mladd(a_lo, lo, z16);
      const u16x8 A0hi = vec_mladd(a_hi, hi, z16);
      const u16x8 A2lo = vec_sr(MulHi16(A0lo, kMult), sh7);
      const u16x8 A2hi = vec_sr(MulHi16(A0hi, kMult), sh7);
      const u8x16 out = vec_packsu((i16x8)A2lo, (i16x8)A2hi);
      vec_xst(out, 0, (unsigned char*)(rgbx + 4 * i));
    }
    // Finish with left-overs.
    for (; i < w; ++i) {
      uint8_t* const rgb = rgba + (alpha_first ? 1 : 0);
      const uint8_t* const alpha = rgba + (alpha_first ? 0 : 3);
      const uint32_t a = alpha[4 * i];
      if (a != 0xff) {
        const uint32_t mult = MULTIPLIER(a);
        rgb[4 * i + 0] = PREMULTIPLY(rgb[4 * i + 0], mult);
        rgb[4 * i + 1] = PREMULTIPLY(rgb[4 * i + 1], mult);
        rgb[4 * i + 2] = PREMULTIPLY(rgb[4 * i + 2], mult);
      }
    }
    rgba += stride;
  }
}

#undef MULTIPLIER
#undef PREMULTIPLY

//------------------------------------------------------------------------------

extern void WebPInitAlphaProcessingVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitAlphaProcessingVSX(void) {
  WebPApplyAlphaMultiply = ApplyAlphaMultiply_VSX;
  WebPDispatchAlpha = DispatchAlpha_VSX;
  WebPDispatchAlphaToGreen = DispatchAlphaToGreen_VSX;
  WebPExtractAlpha = ExtractAlpha_VSX;
  WebPExtractGreen = ExtractGreen_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(WebPInitAlphaProcessingVSX)

#endif  // WEBP_USE_VSX
