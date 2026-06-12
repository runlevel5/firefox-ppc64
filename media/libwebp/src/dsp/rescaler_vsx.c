// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC) version of rescaling functions.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX) && !defined(WEBP_REDUCE_SIZE)

#include <altivec.h>
#include <assert.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/utils/rescaler_utils.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;
typedef __vector signed int i32x4;
typedef __vector unsigned long long u64x2;

#define ROUNDER (WEBP_RESCALER_ONE >> 1)
#define MULT_FIX_C(x, y) (((uint64_t)(x) * (y) + ROUNDER) >> WEBP_RESCALER_RFIX)
#define MULT_FIX_FLOOR_C(x, y) (((uint64_t)(x) * (y)) >> WEBP_RESCALER_RFIX)

#if (WEBP_RESCALER_RFIX != 32)
#error "MULT_FIX/WEBP_RESCALER_RFIX need some more work"
#endif

// Returns (x * scale + ROUNDER) >> 32 for each of the four 32-bit lanes.
static WEBP_INLINE u32x4 MultFix_VSX(u32x4 x, uint32_t scale) {
  const u64x2 rounder = vec_splats((unsigned long long)ROUNDER);
  const u64x2 shift = vec_splats((unsigned long long)WEBP_RESCALER_RFIX);
  const u32x4 s = vec_splats(scale);
  // vec_mule/vec_mulo produce the 32x32->64 products of the even (0, 2) and
  // odd (1, 3) lanes respectively.
  u64x2 e = vec_add(vec_mule(x, s), rounder);
  u64x2 o = vec_add(vec_mulo(x, s), rounder);
  e = vec_sr(e, shift);
  o = vec_sr(o, shift);
  return vec_mergee((u32x4)e, (u32x4)o);
}

// Returns (x * scale) >> 32 for each lane (no rounding).
static WEBP_INLINE u32x4 MultFixFloor_VSX(u32x4 x, uint32_t scale) {
  const u64x2 shift = vec_splats((unsigned long long)WEBP_RESCALER_RFIX);
  const u32x4 s = vec_splats(scale);
  u64x2 e = vec_sr(vec_mule(x, s), shift);
  u64x2 o = vec_sr(vec_mulo(x, s), shift);
  return vec_mergee((u32x4)e, (u32x4)o);
}

// Returns (A * frow + B * irow + ROUNDER) >> 32 for each lane.
static WEBP_INLINE u32x4 Interpolate_VSX(const rescaler_t* WEBP_RESTRICT frow,
                                         const rescaler_t* WEBP_RESTRICT irow,
                                         uint32_t A, uint32_t B) {
  const u64x2 rounder = vec_splats((unsigned long long)ROUNDER);
  const u64x2 shift = vec_splats((unsigned long long)WEBP_RESCALER_RFIX);
  const u32x4 f = vec_xl(0, (uint32_t*)frow);
  const u32x4 ir = vec_xl(0, (uint32_t*)irow);
  const u32x4 va = vec_splats(A);
  const u32x4 vb = vec_splats(B);
  u64x2 e = vec_add(vec_mule(f, va), vec_mule(ir, vb));
  u64x2 o = vec_add(vec_mulo(f, va), vec_mulo(ir, vb));
  e = vec_sr(vec_add(e, rounder), shift);
  o = vec_sr(vec_add(o, rounder), shift);
  return vec_mergee((u32x4)e, (u32x4)o);
}

// Saturated pack of two 32-bit lane vectors (8 values) into 8 bytes at dst.
static WEBP_INLINE void Store8_VSX(u32x4 lo, u32x4 hi, uint8_t* dst) {
  const i16x8 s16 = vec_packs((i32x4)lo, (i32x4)hi);
  const u8x16 s8 = vec_packsu(s16, s16);
  memcpy(dst, &s8, 8);
}

static void RescalerExportRowExpand_VSX(WebPRescaler* const wrk) {
  int x_out;
  uint8_t* const dst = wrk->dst;
  rescaler_t* const irow = wrk->irow;
  const int x_out_max = wrk->dst_width * wrk->num_channels;
  const int max_span = x_out_max & ~7;
  const rescaler_t* const frow = wrk->frow;
  const uint32_t fy_scale = wrk->fy_scale;
  assert(!WebPRescalerOutputDone(wrk));
  assert(wrk->y_accum <= 0);
  assert(wrk->y_expand);
  assert(wrk->y_sub != 0);
  if (wrk->y_accum == 0) {
    for (x_out = 0; x_out < max_span; x_out += 8) {
      const u32x4 A0 = vec_xl(0, (uint32_t*)(frow + x_out + 0));
      const u32x4 A1 = vec_xl(0, (uint32_t*)(frow + x_out + 4));
      const u32x4 B0 = MultFix_VSX(A0, fy_scale);
      const u32x4 B1 = MultFix_VSX(A1, fy_scale);
      Store8_VSX(B0, B1, dst + x_out);
    }
    for (; x_out < x_out_max; ++x_out) {
      const uint32_t J = frow[x_out];
      const int v = (int)MULT_FIX_C(J, fy_scale);
      dst[x_out] = (v > 255) ? 255u : (uint8_t)v;
    }
  } else {
    const uint32_t B = WEBP_RESCALER_FRAC(-wrk->y_accum, wrk->y_sub);
    const uint32_t A = (uint32_t)(WEBP_RESCALER_ONE - B);
    for (x_out = 0; x_out < max_span; x_out += 8) {
      const u32x4 C0 =
          Interpolate_VSX(frow + x_out + 0, irow + x_out + 0, A, B);
      const u32x4 C1 =
          Interpolate_VSX(frow + x_out + 4, irow + x_out + 4, A, B);
      const u32x4 D0 = MultFix_VSX(C0, fy_scale);
      const u32x4 D1 = MultFix_VSX(C1, fy_scale);
      Store8_VSX(D0, D1, dst + x_out);
    }
    for (; x_out < x_out_max; ++x_out) {
      const uint64_t I = (uint64_t)A * frow[x_out] + (uint64_t)B * irow[x_out];
      const uint32_t J = (uint32_t)((I + ROUNDER) >> WEBP_RESCALER_RFIX);
      const int v = (int)MULT_FIX_C(J, fy_scale);
      dst[x_out] = (v > 255) ? 255u : (uint8_t)v;
    }
  }
}

static void RescalerExportRowShrink_VSX(WebPRescaler* const wrk) {
  int x_out;
  uint8_t* const dst = wrk->dst;
  rescaler_t* const irow = wrk->irow;
  const int x_out_max = wrk->dst_width * wrk->num_channels;
  const int max_span = x_out_max & ~7;
  const rescaler_t* const frow = wrk->frow;
  const uint32_t yscale = wrk->fy_scale * (-wrk->y_accum);
  const uint32_t fxy_scale = wrk->fxy_scale;
  assert(!WebPRescalerOutputDone(wrk));
  assert(wrk->y_accum <= 0);
  assert(!wrk->y_expand);
  if (yscale) {
    for (x_out = 0; x_out < max_span; x_out += 8) {
      const u32x4 in0 = vec_xl(0, (uint32_t*)(frow + x_out + 0));
      const u32x4 in1 = vec_xl(0, (uint32_t*)(frow + x_out + 4));
      const u32x4 in2 = vec_xl(0, (uint32_t*)(irow + x_out + 0));
      const u32x4 in3 = vec_xl(0, (uint32_t*)(irow + x_out + 4));
      const u32x4 A0 = MultFixFloor_VSX(in0, yscale);
      const u32x4 A1 = MultFixFloor_VSX(in1, yscale);
      const u32x4 B0 = vec_sub(in2, A0);
      const u32x4 B1 = vec_sub(in3, A1);
      const u32x4 C0 = MultFix_VSX(B0, fxy_scale);
      const u32x4 C1 = MultFix_VSX(B1, fxy_scale);
      Store8_VSX(C0, C1, dst + x_out);
      vec_xst(A0, 0, (uint32_t*)(irow + x_out + 0));
      vec_xst(A1, 0, (uint32_t*)(irow + x_out + 4));
    }
    for (; x_out < x_out_max; ++x_out) {
      const uint32_t frac = (uint32_t)MULT_FIX_FLOOR_C(frow[x_out], yscale);
      const int v = (int)MULT_FIX_C(irow[x_out] - frac, fxy_scale);
      dst[x_out] = (v > 255) ? 255u : (uint8_t)v;
      irow[x_out] = frac;  // new fractional start
    }
  } else {
    const u32x4 zero = vec_splats((uint32_t)0);
    for (x_out = 0; x_out < max_span; x_out += 8) {
      const u32x4 in0 = vec_xl(0, (uint32_t*)(irow + x_out + 0));
      const u32x4 in1 = vec_xl(0, (uint32_t*)(irow + x_out + 4));
      const u32x4 A0 = MultFix_VSX(in0, fxy_scale);
      const u32x4 A1 = MultFix_VSX(in1, fxy_scale);
      Store8_VSX(A0, A1, dst + x_out);
      vec_xst(zero, 0, (uint32_t*)(irow + x_out + 0));
      vec_xst(zero, 0, (uint32_t*)(irow + x_out + 4));
    }
    for (; x_out < x_out_max; ++x_out) {
      const int v = (int)MULT_FIX_C(irow[x_out], fxy_scale);
      dst[x_out] = (v > 255) ? 255u : (uint8_t)v;
      irow[x_out] = 0;
    }
  }
}

#undef MULT_FIX_FLOOR_C
#undef MULT_FIX_C
#undef ROUNDER

//------------------------------------------------------------------------------

extern void WebPRescalerDspInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPRescalerDspInitVSX(void) {
  WebPRescalerExportRowExpand = RescalerExportRowExpand_VSX;
  WebPRescalerExportRowShrink = RescalerExportRowShrink_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(WebPRescalerDspInitVSX)

#endif  // WEBP_USE_VSX
