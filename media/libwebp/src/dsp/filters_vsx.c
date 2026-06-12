// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC) version of filtering functions.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <assert.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned long long u64x2;

// Byte-wise shifts of the whole 128-bit register, matching the little-endian
// semantics of _mm_slli_si128 / _mm_srli_si128. 'n' must be a literal.
#define SLLI(x, n) vec_sld((x), zero, (n))
#define SRLI(x, n) vec_sld(zero, (x), 16 - (n))

// Loads 8 bytes from 'p' into the low half of a vector (high half undefined).
static WEBP_INLINE u8x16 Load8(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, 8);
  // Cast to unsigned long long so clang picks the right vec_splats overload:
  // uint64_t is unsigned long under LP64, which is ambiguous between the
  // unsigned long long and unsigned __int128 overloads.
  return (u8x16)vec_splats((unsigned long long)v);
}

//------------------------------------------------------------------------------
// Horizontal unfilter: out[i] = in[i] + out[i - 1] (a prefix sum).

static void HorizontalUnfilter_VSX(const uint8_t* prev, const uint8_t* in,
                                   uint8_t* out, int width) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u64x2 sh56 = vec_splats((unsigned long long)56);
  u8x16 last;
  int i;
  out[0] = (uint8_t)(in[0] + (prev == NULL ? 0 : prev[0]));
  if (width <= 1) return;
  last = vec_insert(out[0], zero, 0);
  for (i = 1; i + 8 <= width; i += 8) {
    const u8x16 A0 = Load8(in + i);
    const u8x16 A1 = vec_add(A0, last);
    const u8x16 A2 = SLLI(A1, 1);
    const u8x16 A3 = vec_add(A1, A2);
    const u8x16 A4 = SLLI(A3, 2);
    const u8x16 A5 = vec_add(A3, A4);
    const u8x16 A6 = SLLI(A5, 4);
    const u8x16 A7 = vec_add(A5, A6);
    memcpy(out + i, &A7, 8);
    last = (u8x16)vec_sr((u64x2)A7, sh56);  // broadcast out[i + 7] to byte 0
  }
  for (; i < width; ++i) out[i] = (uint8_t)(in[i] + out[i - 1]);
}

//------------------------------------------------------------------------------
// Vertical unfilter: out[i] = in[i] + prev[i].

static void VerticalUnfilter_VSX(const uint8_t* prev, const uint8_t* in,
                                 uint8_t* out, int width) {
  if (prev == NULL) {
    HorizontalUnfilter_VSX(NULL, in, out, width);
  } else {
    int i;
    const int max_pos = width & ~31;
    for (i = 0; i < max_pos; i += 32) {
      const u8x16 A0 = vec_xl(0, (unsigned char*)&in[i + 0]);
      const u8x16 A1 = vec_xl(0, (unsigned char*)&in[i + 16]);
      const u8x16 B0 = vec_xl(0, (unsigned char*)&prev[i + 0]);
      const u8x16 B1 = vec_xl(0, (unsigned char*)&prev[i + 16]);
      vec_xst(vec_add(A0, B0), 0, (unsigned char*)&out[i + 0]);
      vec_xst(vec_add(A1, B1), 0, (unsigned char*)&out[i + 16]);
    }
    for (; i < width; ++i) out[i] = (uint8_t)(in[i] + prev[i]);
  }
}

//------------------------------------------------------------------------------
// Gradient unfilter: row[i] = in[i] + clip(row[i-1] + top[i] - top[i-1]).

static WEBP_INLINE int GradientPredictor_VSX(uint8_t a, uint8_t b, uint8_t c) {
  const int g = a + b - c;
  return ((g & ~0xff) == 0) ? g : (g < 0) ? 0 : 255;
}

static void GradientPredictInverse_VSX(const uint8_t* in, const uint8_t* top,
                                       uint8_t* row, int length) {
  if (length > 0) {
    int i;
    const int max_pos = length & ~7;
    const u8x16 zero = vec_splats((unsigned char)0);
    u8x16 A = vec_insert((unsigned char)row[-1], zero, 0);  // left sample
    for (i = 0; i < max_pos; i += 8) {
      const u8x16 t0 = Load8(top + i);
      const u8x16 t1 = Load8(top + i - 1);
      const u16x8 B = (u16x8)vec_mergeh(t0, zero);
      const u16x8 C = (u16x8)vec_mergeh(t1, zero);
      const u8x16 D = Load8(in + i);  // base input
      const u16x8 E = vec_sub(B, C);  // unclipped gradient basis b - c
      u8x16 out = zero;               // accumulator for output
      u8x16 mask_hi = vec_insert((unsigned char)0xff, zero, 0);
      int k = 8;
      while (1) {
        const u16x8 tmp3 = vec_add((u16x8)A, E);  // delta = a + b - c
        const u8x16 tmp4 = vec_packsu((i16x8)tmp3, (i16x8)zero);  // sat. delta
        const u8x16 tmp5 = vec_add(tmp4, D);                      // add to in[]
        A = vec_and(tmp5, mask_hi);  // keep new sample
        out = vec_or(out, A);        // accumulate output
        if (--k == 0) break;
        A = SLLI(A, 1);                  // rotate left sample
        mask_hi = SLLI(mask_hi, 1);      // rotate mask
        A = (u8x16)vec_mergeh(A, zero);  // convert 8b -> 16b
      }
      A = SRLI(A, 7);  // prepare left sample for next iteration
      memcpy(row + i, &out, 8);
    }
    for (; i < length; ++i) {
      const int delta = GradientPredictor_VSX(row[i - 1], top[i], top[i - 1]);
      row[i] = (uint8_t)(in[i] + delta);
    }
  }
}

static void GradientUnfilter_VSX(const uint8_t* prev, const uint8_t* in,
                                 uint8_t* out, int width) {
  if (prev == NULL) {
    HorizontalUnfilter_VSX(NULL, in, out, width);
  } else {
    out[0] = (uint8_t)(in[0] + prev[0]);  // predict from above
    GradientPredictInverse_VSX(in + 1, prev + 1, out + 1, width - 1);
  }
}

#undef SLLI
#undef SRLI

//------------------------------------------------------------------------------

extern void VP8FiltersInitVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8FiltersInitVSX(void) {
  WebPUnfilters[WEBP_FILTER_HORIZONTAL] = HorizontalUnfilter_VSX;
  WebPUnfilters[WEBP_FILTER_VERTICAL] = VerticalUnfilter_VSX;
  WebPUnfilters[WEBP_FILTER_GRADIENT] = GradientUnfilter_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(VP8FiltersInitVSX)

#endif  // WEBP_USE_VSX
