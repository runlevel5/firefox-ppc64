// Copyright (c) 2011-2016 Google Inc.
// Use of this source code is governed by a BSD-style license that can be
// found in the gfx/skia/LICENSE file.

#include <altivec.h>

#include <bit>
#include <cstring>

#include "SkConvolver.h"
#include "mozilla/Attributes.h"

namespace skia {

using ConvolutionFixed = SkConvolutionFilter1D::ConvolutionFixed;
typedef __vector unsigned char u8x16;
typedef __vector signed short i16x8;
typedef __vector signed int i32x4;
typedef __vector unsigned int u32x4;

static constexpr bool kLE = std::endian::native == std::endian::little;
// Byte select mask for the alpha channel of each pixel.
static constexpr unsigned int kAlphaMask = kLE ? 0xFF000000u : 0x000000FFu;

// VSX has unaligned vector loads and stores; plain AltiVec (PowerPC 970)
// only has aligned ones, so use the lvsl/vperm idiom for loads and rotate
// plus per-word element stores for 4-byte aligned pixel stores.
static MOZ_ALWAYS_INLINE u8x16 Load16(const unsigned char* aPtr) {
#if defined(__VSX__)
  return vec_xl(0, aPtr);
#elif defined(__BIG_ENDIAN__)
  return vec_perm(vec_ld(0, aPtr), vec_ld(15, aPtr), vec_lvsl(0, aPtr));
#else
  u8x16 v;
  memcpy(&v, aPtr, 16);
  return v;
#endif
}

static MOZ_ALWAYS_INLINE void Store16(unsigned char* aPtr, u8x16 aV) {
#if defined(__VSX__)
  vec_xst(aV, 0, aPtr);
#elif defined(__BIG_ENDIAN__)
  u8x16 r = vec_perm(aV, aV, vec_lvsr(0, aPtr));
  unsigned int* w = reinterpret_cast<unsigned int*>(aPtr);
  vec_ste((u32x4)r, 0, w);
  vec_ste((u32x4)r, 4, w);
  vec_ste((u32x4)r, 8, w);
  vec_ste((u32x4)r, 12, w);
#else
  memcpy(aPtr, &aV, 16);
#endif
}

// Zero-extend the low or high 8 bytes to 16-bit lanes.
static MOZ_ALWAYS_INLINE i16x8 UnpackLo(u8x16 aV) {
  const u8x16 zero = vec_splats((unsigned char)0);
  if constexpr (kLE) {
    return (i16x8)vec_mergeh(aV, zero);
  } else {
    return (i16x8)vec_mergeh(zero, aV);
  }
}

static MOZ_ALWAYS_INLINE i16x8 UnpackHi(u8x16 aV) {
  const u8x16 zero = vec_splats((unsigned char)0);
  if constexpr (kLE) {
    return (i16x8)vec_mergel(aV, zero);
  } else {
    return (i16x8)vec_mergel(zero, aV);
  }
}

// Permute masks that take two pixels [r0 g0 b0 a0 r1 g1 b1 a1] and spread them
// into 16-bit lanes ordered by channel, [r0 r1 g0 g1 b0 b1 a0 a1], with the
// zero vector supplying the high byte of each lane. vec_msum with coefficient
// pairs [c0 c1 c0 c1 ...] then accumulates both pixels per channel at once.
static MOZ_ALWAYS_INLINE u8x16 PairMask(unsigned char aBase) {
  u8x16 m;
  for (int ch = 0; ch < 4; ch++) {
    for (int px = 0; px < 2; px++) {
      unsigned char idx = aBase + px * 4 + ch;
      m[(ch * 2 + px) * 2 + (kLE ? 0 : 1)] = idx;
      m[(ch * 2 + px) * 2 + (kLE ? 1 : 0)] = 16;
    }
  }
  return m;
}

static MOZ_ALWAYS_INLINE void AccumRemainder(
    const unsigned char* pixelsLeft, const ConvolutionFixed* filterValues,
    i32x4& accum, int r) {
  int remainder[4] = {0};
  for (int i = 0; i < r; i++) {
    ConvolutionFixed coeff = filterValues[i];
    remainder[0] += coeff * pixelsLeft[i * 4 + 0];
    remainder[1] += coeff * pixelsLeft[i * 4 + 1];
    remainder[2] += coeff * pixelsLeft[i * 4 + 2];
    remainder[3] += coeff * pixelsLeft[i * 4 + 3];
  }
  i32x4 t = {remainder[0], remainder[1], remainder[2], remainder[3]};
  accum = vec_add(accum, t);
}

// Rounds, shifts and saturates four 32-bit channel sums to 8 bits in the low
// four bytes.
static MOZ_ALWAYS_INLINE i32x4 RoundShift(i32x4 aAccum) {
  const i32x4 round = vec_splats(1 << (SkConvolutionFilter1D::kShiftBits - 1));
  const u32x4 shift =
      vec_splats((unsigned int)SkConvolutionFilter1D::kShiftBits);
  return vec_sra(vec_add(aAccum, round), shift);
}

// Convolves horizontally along a single row. The row data is given in
// |srcData| and continues for the numValues() of the filter.
void convolve_horizontally_altivec(const unsigned char* srcData,
                                   const SkConvolutionFilter1D& filter,
                                   unsigned char* outRow, bool /*hasAlpha*/) {
  const u8x16 zero = vec_splats((unsigned char)0);
  const u8x16 pairMask01 = PairMask(0);
  const u8x16 pairMask23 = PairMask(8);

  int numValues = filter.numValues();
  for (int outX = 0; outX < numValues; outX++) {
    int filterOffset, filterLength;
    const ConvolutionFixed* filterValues =
        filter.FilterForValue(outX, &filterOffset, &filterLength);

    const unsigned char* rowToFilter = &srcData[filterOffset * 4];
    i32x4 accum = vec_splats(0);

    // Four coefficients and four pixels per iteration.
    for (int filterX = 0; filterX < filterLength >> 2; filterX++) {
      // [c0 c1 c2 c3 c0 c1 c2 c3], then [c0 c1 c0 c1 ...] and [c2 c3 c2 c3 ...]
      unsigned int bits01, bits23;
      memcpy(&bits01, filterValues, 4);
      memcpy(&bits23, filterValues + 2, 4);
      u32x4 coeff = {bits01, bits23, bits01, bits23};
      i16x8 coeff01 = (i16x8)vec_splat(coeff, 0);
      i16x8 coeff23 = (i16x8)vec_splat(coeff, 1);

      u8x16 src8 = Load16(rowToFilter);
      i16x8 px01 = (i16x8)vec_perm(src8, zero, pairMask01);
      i16x8 px23 = (i16x8)vec_perm(src8, zero, pairMask23);
      accum = vec_msum(px01, coeff01, accum);
      accum = vec_msum(px23, coeff23, accum);

      rowToFilter += 16;
      filterValues += 4;
    }

    int r = filterLength & 3;
    if (r) {
      int remainderOffset = (filterOffset + filterLength - r) * 4;
      AccumRemainder(srcData + remainderOffset, filterValues, accum, r);
    }

    i32x4 shifted = RoundShift(accum);
    i16x8 packed16 = vec_packs(shifted, shifted);
    u8x16 packed8 = vec_packsu(packed16, packed16);
    unsigned int px = vec_extract((u32x4)packed8, 0);
    memcpy(outRow, &px, 4);
    outRow += 4;
  }
}

// Replaces each alpha with max(alpha, r, g, b) so premultiplied pixels stay
// valid after negative filter lobes, or forces alpha to 0xFF.
template <bool hasAlpha>
static MOZ_ALWAYS_INLINE u8x16 FixAlpha(u8x16 aPx) {
  const u8x16 alphaMask = (u8x16)vec_splats(kAlphaMask);
  if constexpr (hasAlpha) {
    u32x4 px32 = (u32x4)aPx;
    u8x16 m = vec_max(aPx, (u8x16)vec_rl(px32, vec_splats(8u)));
    m = vec_max(m, (u8x16)vec_rl(px32, vec_splats(16u)));
    m = vec_max(m, (u8x16)vec_rl(px32, vec_splats(24u)));
    return vec_sel(aPx, m, alphaMask);
  } else {
    return vec_or(aPx, alphaMask);
  }
}

// Accumulates two source rows weighted by |aCoeffPair| ([cA cB] repeated) into
// the four per-pixel accumulators of a 4-pixel block.
static MOZ_ALWAYS_INLINE void AccumRows(u8x16 aRowA, u8x16 aRowB,
                                        i16x8 aCoeffPair, i32x4& aAccum0,
                                        i32x4& aAccum1, i32x4& aAccum2,
                                        i32x4& aAccum3) {
  // [a0 b0 a1 b1 ... a7 b7] bytes: pixels 0-1, then pixels 2-3.
  u8x16 m01 = vec_mergeh(aRowA, aRowB);
  u8x16 m23 = vec_mergel(aRowA, aRowB);
  aAccum0 = vec_msum(UnpackLo(m01), aCoeffPair, aAccum0);
  aAccum1 = vec_msum(UnpackHi(m01), aCoeffPair, aAccum1);
  aAccum2 = vec_msum(UnpackLo(m23), aCoeffPair, aAccum2);
  aAccum3 = vec_msum(UnpackHi(m23), aCoeffPair, aAccum3);
}

static MOZ_ALWAYS_INLINE i16x8 CoeffPair(ConvolutionFixed aA,
                                         ConvolutionFixed aB) {
  ConvolutionFixed pair[2] = {aA, aB};
  unsigned int bits;
  memcpy(&bits, pair, 4);
  return (i16x8)vec_splats(bits);
}

// Convolves one block of four output pixels (16 bytes of each source row at
// byte offset |aOffset|) over all filter rows, two rows per step.
template <bool hasAlpha>
static MOZ_ALWAYS_INLINE u8x16
ConvolveBlock(const ConvolutionFixed* filterValues, int filterLength,
              unsigned char* const* sourceDataRows, int aOffset) {
  i32x4 accum0 = vec_splats(0);
  i32x4 accum1 = vec_splats(0);
  i32x4 accum2 = vec_splats(0);
  i32x4 accum3 = vec_splats(0);
  int filterY = 0;
  for (; filterY + 1 < filterLength; filterY += 2) {
    AccumRows(Load16(&sourceDataRows[filterY][aOffset]),
              Load16(&sourceDataRows[filterY + 1][aOffset]),
              CoeffPair(filterValues[filterY], filterValues[filterY + 1]),
              accum0, accum1, accum2, accum3);
  }
  if (filterY < filterLength) {
    u8x16 row = Load16(&sourceDataRows[filterY][aOffset]);
    AccumRows(row, row, CoeffPair(filterValues[filterY], 0), accum0, accum1,
              accum2, accum3);
  }
  i16x8 p01 = vec_packs(RoundShift(accum0), RoundShift(accum1));
  i16x8 p23 = vec_packs(RoundShift(accum2), RoundShift(accum3));
  return FixAlpha<hasAlpha>(vec_packsu(p01, p23));
}

template <bool hasAlpha>
static void ConvolveVertically(const ConvolutionFixed* filterValues,
                               int filterLength,
                               unsigned char* const* sourceDataRows,
                               int pixelWidth, unsigned char* outRow) {
  int width = pixelWidth & ~3;
  for (int outX = 0; outX < width; outX += 4) {
    u8x16 px = ConvolveBlock<hasAlpha>(filterValues, filterLength,
                                       sourceDataRows, outX << 2);
    Store16(outRow, px);
    outRow += 16;
  }

  // The source rows are padded (ConvolutionFilter::PadBytesForSIMD), so the
  // last partial block can still be loaded whole; only |r| pixels are stored.
  int r = pixelWidth & 3;
  if (r) {
    u8x16 px = ConvolveBlock<hasAlpha>(filterValues, filterLength,
                                       sourceDataRows, width << 2);
    alignas(16) unsigned char tmp[16];
    vec_st(px, 0, tmp);
    memcpy(outRow, tmp, r * 4);
  }
}

void convolve_vertically_altivec(const ConvolutionFixed* filterValues,
                                 int filterLength,
                                 unsigned char* const* sourceDataRows,
                                 int pixelWidth, unsigned char* outRow,
                                 bool hasAlpha) {
  if (hasAlpha) {
    ConvolveVertically<true>(filterValues, filterLength, sourceDataRows,
                             pixelWidth, outRow);
  } else {
    ConvolveVertically<false>(filterValues, filterLength, sourceDataRows,
                              pixelWidth, outRow);
  }
}

}  // namespace skia
