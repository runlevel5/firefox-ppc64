/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "Swizzle.h"

#include <altivec.h>

#include <bit>
#include <cstring>

namespace mozilla::gfx {

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;
typedef __vector unsigned int u32x4;

// Pixels are B8G8R8A8 bytes in memory on both endiannesses, so the alpha byte
// is the top byte of a little-endian word and the bottom byte of a big-endian
// one, and the even (B, R) and odd (G, A) channels swap 16-bit lane halves.
static constexpr bool kLE = std::endian::native == std::endian::little;
static constexpr uint32_t kAlphaMask32 = kLE ? 0xFF000000u : 0x000000FFu;
static constexpr uint32_t kOddAlphaMask32 = kLE ? 0x00FF0000u : 0x000000FFu;
static constexpr uint32_t kOddGreenMask32 = kLE ? 0x000000FFu : 0x00FF0000u;

// VSX has unaligned vector loads and stores. Plain AltiVec (PowerPC 970)
// only has aligned ones, so use the lvsl/vperm idiom for loads and rotate
// plus per-word element stores for the 4-byte aligned pixel stores.
static MOZ_ALWAYS_INLINE u8x16 Load16(const uint8_t* aSrc) {
#if defined(__VSX__)
  return vec_xl(0, aSrc);
#elif defined(__BIG_ENDIAN__)
  return vec_perm(vec_ld(0, aSrc), vec_ld(15, aSrc), vec_lvsl(0, aSrc));
#else
  u8x16 v;
  memcpy(&v, aSrc, 16);
  return v;
#endif
}

static MOZ_ALWAYS_INLINE void Store16(uint8_t* aDst, u8x16 aV) {
#if defined(__VSX__)
  vec_xst(aV, 0, aDst);
#elif defined(__BIG_ENDIAN__)
  u8x16 r = vec_perm(aV, aV, vec_lvsr(0, aDst));
  unsigned int* w = reinterpret_cast<unsigned int*>(aDst);
  vec_ste((u32x4)r, 0, w);
  vec_ste((u32x4)r, 4, w);
  vec_ste((u32x4)r, 8, w);
  vec_ste((u32x4)r, 12, w);
#else
  memcpy(aDst, &aV, 16);
#endif
}

// Load 1-3 pixels into a 4 pixel vector.
static MOZ_ALWAYS_INLINE u8x16 LoadRemainder(const uint8_t* aSrc,
                                             size_t aLength) {
  alignas(16) uint8_t buffer[16] = {};
  switch (aLength) {
    case 3:
      memcpy(buffer, aSrc, 12);
      break;
    case 2:
      memcpy(buffer, aSrc, 8);
      break;
    default:
      memcpy(buffer, aSrc, 4);
      break;
  }
  return vec_ld(0, buffer);
}

// Store 1-3 pixels from a vector into memory without overwriting.
static MOZ_ALWAYS_INLINE void StoreRemainder(uint8_t* aDst, size_t aLength,
                                             u8x16 aSrc) {
  alignas(16) uint8_t buffer[16];
  vec_st(aSrc, 0, buffer);
  memcpy(aDst, buffer, aLength * 4);
}

// Splits pixels into the even (B/R) and odd (G/A) channels, each in the low
// byte of a 16-bit lane, and recombines them.
static MOZ_ALWAYS_INLINE void SplitEvenOdd(u8x16 aPx, u16x8& aEven,
                                           u16x8& aOdd) {
  u16x8 px16 = (u16x8)aPx;
  const u16x8 lowByte = vec_splats((unsigned short)0x00FF);
  const u16x8 eight = vec_splats((unsigned short)8);
  if constexpr (kLE) {
    aEven = vec_and(px16, lowByte);
    aOdd = vec_sr(px16, eight);
  } else {
    aEven = vec_sr(px16, eight);
    aOdd = vec_and(px16, lowByte);
  }
}

static MOZ_ALWAYS_INLINE u8x16 CombineEvenOdd(u16x8 aEven, u16x8 aOdd) {
  const u16x8 eight = vec_splats((unsigned short)8);
  if constexpr (kLE) {
    return (u8x16)vec_or(aEven, vec_sl(aOdd, eight));
  } else {
    return (u8x16)vec_or(vec_sl(aEven, eight), aOdd);
  }
}

// Duplicates the alpha (the odd lane of each pixel) into both lanes.
static MOZ_ALWAYS_INLINE u16x8 SplatAlpha(u16x8 aOdd) {
  const u8x16 alphaMask = {2,  3,  2,  3,  6,  7,  6,  7,
                           10, 11, 10, 11, 14, 15, 14, 15};
  return vec_perm(aOdd, aOdd, alphaMask);
}

static MOZ_ALWAYS_INLINE u16x8 SwapRB16(u16x8 aRb) {
  return (u16x8)vec_rl((u32x4)aRb, vec_splats(16u));
}

// Premultiply vector of 4 pixels using splayed math, matching the fallback:
// c = c * a + 255; c = (c + (c >> 8)) >> 8.
template <bool aSwapRB, bool aOpaqueAlpha>
static MOZ_ALWAYS_INLINE u8x16 PremultiplyVector(u8x16 aSrc) {
  const u16x8 lowByte = vec_splats((unsigned short)0x00FF);
  const u16x8 eight = vec_splats((unsigned short)8);
  u16x8 rb, ga;
  SplitEvenOdd(aSrc, rb, ga);
  if constexpr (aSwapRB) {
    rb = SwapRB16(rb);
  }
  u16x8 alphas = SplatAlpha(ga);
  // If format is not opaque, force A to 255 so that A*alpha/255 = alpha.
  if constexpr (!aOpaqueAlpha) {
    ga = (u16x8)vec_or((u32x4)ga, vec_splats(kOddAlphaMask32));
  }
  rb = vec_mladd(rb, alphas, lowByte);
  rb = vec_sr(vec_add(rb, vec_sr(rb, eight)), eight);
  ga = vec_mladd(ga, alphas, lowByte);
  ga = vec_sr(vec_add(ga, vec_sr(ga, eight)), eight);
  u8x16 px = CombineEvenOdd(rb, ga);
  if constexpr (aOpaqueAlpha) {
    px = (u8x16)vec_or((u32x4)px, vec_splats(kAlphaMask32));
  }
  return px;
}

template <bool aSwapRB, bool aOpaqueAlpha>
static MOZ_ALWAYS_INLINE void PremultiplyChunk(const uint8_t*& aSrc,
                                               uint8_t*& aDst,
                                               int32_t aAlignedRow,
                                               int32_t aRemainder) {
  for (const uint8_t* end = aSrc + aAlignedRow; aSrc < end;) {
    Store16(aDst, PremultiplyVector<aSwapRB, aOpaqueAlpha>(Load16(aSrc)));
    aSrc += 4 * 4;
    aDst += 4 * 4;
  }
  if (aRemainder) {
    u8x16 px = LoadRemainder(aSrc, aRemainder);
    px = PremultiplyVector<aSwapRB, aOpaqueAlpha>(px);
    StoreRemainder(aDst, aRemainder, px);
  }
}

template <bool aSwapRB, bool aOpaqueAlpha>
void PremultiplyRow_AltiVec(const uint8_t* aSrc, uint8_t* aDst,
                            int32_t aLength) {
  int32_t alignedRow = 4 * (aLength & ~3);
  int32_t remainder = aLength & 3;
  PremultiplyChunk<aSwapRB, aOpaqueAlpha>(aSrc, aDst, alignedRow, remainder);
}

template <bool aSwapRB, bool aOpaqueAlpha>
void Premultiply_AltiVec(const uint8_t* aSrc, int32_t aSrcGap, uint8_t* aDst,
                         int32_t aDstGap, IntSize aSize) {
  int32_t alignedRow = 4 * (aSize.width & ~3);
  int32_t remainder = aSize.width & 3;
  aSrcGap += 4 * remainder;
  aDstGap += 4 * remainder;
  for (int32_t height = aSize.height; height > 0; height--) {
    PremultiplyChunk<aSwapRB, aOpaqueAlpha>(aSrc, aDst, alignedRow, remainder);
    aSrc += aSrcGap;
    aDst += aDstGap;
  }
}

template void PremultiplyRow_AltiVec<false, false>(const uint8_t*, uint8_t*,
                                                   int32_t);
template void PremultiplyRow_AltiVec<false, true>(const uint8_t*, uint8_t*,
                                                  int32_t);
template void PremultiplyRow_AltiVec<true, false>(const uint8_t*, uint8_t*,
                                                  int32_t);
template void PremultiplyRow_AltiVec<true, true>(const uint8_t*, uint8_t*,
                                                 int32_t);
template void Premultiply_AltiVec<false, false>(const uint8_t*, int32_t,
                                                uint8_t*, int32_t, IntSize);
template void Premultiply_AltiVec<false, true>(const uint8_t*, int32_t,
                                               uint8_t*, int32_t, IntSize);
template void Premultiply_AltiVec<true, false>(const uint8_t*, int32_t,
                                               uint8_t*, int32_t, IntSize);
template void Premultiply_AltiVec<true, true>(const uint8_t*, int32_t, uint8_t*,
                                              int32_t, IntSize);

// The 8.16 fixed-point reciprocal table of the fallback.
#define UNPREMULQ(x) (0xFF00FFU / (x))
#define UNPREMULQ_2(x) UNPREMULQ(x), UNPREMULQ((x) + 1)
#define UNPREMULQ_4(x) UNPREMULQ_2(x), UNPREMULQ_2((x) + 2)
#define UNPREMULQ_8(x) UNPREMULQ_4(x), UNPREMULQ_4((x) + 4)
#define UNPREMULQ_16(x) UNPREMULQ_8(x), UNPREMULQ_8((x) + 8)
#define UNPREMULQ_32(x) UNPREMULQ_16(x), UNPREMULQ_16((x) + 16)
static const uint32_t sUnpremultiplyTable_AltiVec[256] = {0,
                                                          UNPREMULQ(1),
                                                          UNPREMULQ_2(2),
                                                          UNPREMULQ_4(4),
                                                          UNPREMULQ_8(8),
                                                          UNPREMULQ_16(16),
                                                          UNPREMULQ_32(32),
                                                          UNPREMULQ_32(64),
                                                          UNPREMULQ_32(96),
                                                          UNPREMULQ_32(128),
                                                          UNPREMULQ_32(160),
                                                          UNPREMULQ_32(192),
                                                          UNPREMULQ_32(224)};

// (c * q) >> 16 for one channel per 32-bit lane, exactly as the fallback. The
// product needs 32 bits and AltiVec has no 32-bit multiply, so it is formed
// from two 16x16 multiply-sums: c * qLo + ((c * qHi) << 16).
static MOZ_ALWAYS_INLINE u32x4 MulReciprocal(u32x4 aChannel, u32x4 aQLo,
                                             u32x4 aQHi) {
  const u32x4 zero = vec_splats(0u);
  u32x4 lo = vec_msum((u16x8)aChannel, (u16x8)aQLo, zero);
  u32x4 hi = vec_msum((u16x8)aChannel, (u16x8)aQHi, zero);
  return vec_sr(vec_add(lo, vec_sl(hi, vec_splats(16u))), vec_splats(16u));
}

template <bool aSwapRB>
static MOZ_ALWAYS_INLINE u8x16 UnpremultiplyVector(u8x16 aSrc) {
  const u32x4 lowByte32 = vec_splats(0x000000FFu);
  u16x8 rb, ga;
  SplitEvenOdd(aSrc, rb, ga);
  if constexpr (aSwapRB) {
    rb = SwapRB16(rb);
  }

  // Gather the reciprocals for the 4 alphas (the odd lanes of ga).
  alignas(16) uint16_t alphaBuf[8];
  vec_st(ga, 0, alphaBuf);
  alignas(16) uint32_t recipBuf[4] = {sUnpremultiplyTable_AltiVec[alphaBuf[1]],
                                      sUnpremultiplyTable_AltiVec[alphaBuf[3]],
                                      sUnpremultiplyTable_AltiVec[alphaBuf[5]],
                                      sUnpremultiplyTable_AltiVec[alphaBuf[7]]};
  u32x4 q = vec_ld(0, recipBuf);
  u32x4 qLo = vec_and(q, vec_splats(0x0000FFFFu));
  u32x4 qHi = vec_sr(q, vec_splats(16u));

  // Each channel alone in a 32-bit lane: the even lanes of rb hold the byte 0
  // channel and the odd lanes the byte 2 channel, ga's even lanes hold G.
  u32x4 rb32 = (u32x4)rb;
  u32x4 ga32 = (u32x4)ga;
  u32x4 c0, c2, g;
  if constexpr (kLE) {
    c0 = vec_and(rb32, lowByte32);
    c2 = vec_sr(rb32, vec_splats(16u));
    g = vec_and(ga32, lowByte32);
  } else {
    c0 = vec_sr(rb32, vec_splats(16u));
    c2 = vec_and(rb32, lowByte32);
    g = vec_sr(ga32, vec_splats(16u));
  }
  c0 = vec_and(MulReciprocal(c0, qLo, qHi), lowByte32);
  c2 = vec_and(MulReciprocal(c2, qLo, qHi), lowByte32);
  g = vec_and(MulReciprocal(g, qLo, qHi), lowByte32);

  u32x4 alpha = vec_and((u32x4)aSrc, vec_splats(kAlphaMask32));
  u32x4 px;
  if constexpr (kLE) {
    px = vec_or(vec_or(c0, vec_sl(g, vec_splats(8u))),
                vec_or(vec_sl(c2, vec_splats(16u)), alpha));
  } else {
    px = vec_or(vec_or(vec_sl(c0, vec_splats(24u)), vec_sl(g, vec_splats(16u))),
                vec_or(vec_sl(c2, vec_splats(8u)), alpha));
  }
  return (u8x16)px;
}

template <bool aSwapRB>
static MOZ_ALWAYS_INLINE void UnpremultiplyChunk(const uint8_t*& aSrc,
                                                 uint8_t*& aDst,
                                                 int32_t aAlignedRow,
                                                 int32_t aRemainder) {
  for (const uint8_t* end = aSrc + aAlignedRow; aSrc < end;) {
    Store16(aDst, UnpremultiplyVector<aSwapRB>(Load16(aSrc)));
    aSrc += 4 * 4;
    aDst += 4 * 4;
  }
  if (aRemainder) {
    u8x16 px = LoadRemainder(aSrc, aRemainder);
    px = UnpremultiplyVector<aSwapRB>(px);
    StoreRemainder(aDst, aRemainder, px);
  }
}

template <bool aSwapRB>
void UnpremultiplyRow_AltiVec(const uint8_t* aSrc, uint8_t* aDst,
                              int32_t aLength) {
  int32_t alignedRow = 4 * (aLength & ~3);
  int32_t remainder = aLength & 3;
  UnpremultiplyChunk<aSwapRB>(aSrc, aDst, alignedRow, remainder);
}

template <bool aSwapRB>
void Unpremultiply_AltiVec(const uint8_t* aSrc, int32_t aSrcGap, uint8_t* aDst,
                           int32_t aDstGap, IntSize aSize) {
  int32_t alignedRow = 4 * (aSize.width & ~3);
  int32_t remainder = aSize.width & 3;
  aSrcGap += 4 * remainder;
  aDstGap += 4 * remainder;
  for (int32_t height = aSize.height; height > 0; height--) {
    UnpremultiplyChunk<aSwapRB>(aSrc, aDst, alignedRow, remainder);
    aSrc += aSrcGap;
    aDst += aDstGap;
  }
}

template void UnpremultiplyRow_AltiVec<false>(const uint8_t*, uint8_t*,
                                              int32_t);
template void UnpremultiplyRow_AltiVec<true>(const uint8_t*, uint8_t*, int32_t);
template void Unpremultiply_AltiVec<false>(const uint8_t*, int32_t, uint8_t*,
                                           int32_t, IntSize);
template void Unpremultiply_AltiVec<true>(const uint8_t*, int32_t, uint8_t*,
                                          int32_t, IntSize);

// Swizzle a vector of 4 pixels providing swaps and opaquifying.
template <bool aSwapRB, bool aOpaqueAlpha>
static MOZ_ALWAYS_INLINE u8x16 SwizzleVector(u8x16 aSrc) {
  u8x16 px = aSrc;
  if constexpr (aSwapRB) {
    const u8x16 swapMask = {2,  1, 0, 3,  6,  5,  4,  7,
                            10, 9, 8, 11, 14, 13, 12, 15};
    px = vec_perm(px, px, swapMask);
  }
  if constexpr (aOpaqueAlpha) {
    px = (u8x16)vec_or((u32x4)px, vec_splats(kAlphaMask32));
  }
  return px;
}

template <bool aSwapRB, bool aOpaqueAlpha>
static MOZ_ALWAYS_INLINE void SwizzleChunk(const uint8_t*& aSrc, uint8_t*& aDst,
                                           int32_t aAlignedRow,
                                           int32_t aRemainder) {
  for (const uint8_t* end = aSrc + aAlignedRow; aSrc < end;) {
    Store16(aDst, SwizzleVector<aSwapRB, aOpaqueAlpha>(Load16(aSrc)));
    aSrc += 4 * 4;
    aDst += 4 * 4;
  }
  if (aRemainder) {
    u8x16 px = LoadRemainder(aSrc, aRemainder);
    px = SwizzleVector<aSwapRB, aOpaqueAlpha>(px);
    StoreRemainder(aDst, aRemainder, px);
  }
}

template <bool aSwapRB, bool aOpaqueAlpha>
void SwizzleRow_AltiVec(const uint8_t* aSrc, uint8_t* aDst, int32_t aLength) {
  int32_t alignedRow = 4 * (aLength & ~3);
  int32_t remainder = aLength & 3;
  SwizzleChunk<aSwapRB, aOpaqueAlpha>(aSrc, aDst, alignedRow, remainder);
}

template <bool aSwapRB, bool aOpaqueAlpha>
void Swizzle_AltiVec(const uint8_t* aSrc, int32_t aSrcGap, uint8_t* aDst,
                     int32_t aDstGap, IntSize aSize) {
  int32_t alignedRow = 4 * (aSize.width & ~3);
  int32_t remainder = aSize.width & 3;
  aSrcGap += 4 * remainder;
  aDstGap += 4 * remainder;
  for (int32_t height = aSize.height; height > 0; height--) {
    SwizzleChunk<aSwapRB, aOpaqueAlpha>(aSrc, aDst, alignedRow, remainder);
    aSrc += aSrcGap;
    aDst += aDstGap;
  }
}

template void SwizzleRow_AltiVec<true, false>(const uint8_t*, uint8_t*,
                                              int32_t);
template void SwizzleRow_AltiVec<true, true>(const uint8_t*, uint8_t*, int32_t);
template void Swizzle_AltiVec<true, false>(const uint8_t*, int32_t, uint8_t*,
                                           int32_t, IntSize);
template void Swizzle_AltiVec<true, true>(const uint8_t*, int32_t, uint8_t*,
                                          int32_t, IntSize);

// Expands packed RGB to RGBX/BGRX in place, back to front. Each full step
// reads 16 bytes for 4 pixels (12 bytes), so the last 2-5 pixels are done
// one at a time to stay inside the source.
template <bool aSwapRB>
void UnpackRowRGB24_AltiVec(const uint8_t* aSrc, uint8_t* aDst,
                            int32_t aLength) {
  int32_t alignedRow = aLength >= 6 ? (aLength - 2) & ~3 : 0;
  const uint8_t* src = aSrc + 3 * (aLength - 1);
  uint8_t* dst = aDst + 4 * (aLength - 1);
  for (int32_t i = aLength - 1; i >= alignedRow; i--) {
    uint8_t r = src[0], g = src[1], b = src[2];
    dst[0] = aSwapRB ? b : r;
    dst[1] = g;
    dst[2] = aSwapRB ? r : b;
    dst[3] = 0xFF;
    src -= 3;
    dst -= 4;
  }

  const u8x16 expandMask =
      aSwapRB ? (u8x16){2, 1, 0, 0, 5, 4, 3, 0, 8, 7, 6, 0, 11, 10, 9, 0}
              : (u8x16){0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0};
  const u32x4 alpha = vec_splats(kAlphaMask32);
  src = aSrc + 3 * (alignedRow - 4);
  dst = aDst + 4 * (alignedRow - 4);
  while (src >= aSrc) {
    u8x16 px = Load16(src);
    px = (u8x16)vec_or((u32x4)vec_perm(px, px, expandMask), alpha);
    Store16(dst, px);
    src -= 4 * 3;
    dst -= 4 * 4;
  }
}

template void UnpackRowRGB24_AltiVec<false>(const uint8_t*, uint8_t*, int32_t);
template void UnpackRowRGB24_AltiVec<true>(const uint8_t*, uint8_t*, int32_t);

}  // namespace mozilla::gfx
