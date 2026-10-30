/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MOZILLA_GFX_SWIZZLE_VSX_H_
#define MOZILLA_GFX_SWIZZLE_VSX_H_

#ifdef MOZILLA_GFX_SWIZZLE_GENERIC_H_
#  error "SwizzleVSX.h must be included before SwizzleGeneric.h"
#endif

#include <altivec.h>

#include <concepts>
#include <cstring>

#include "SwizzleGenericDecls.h"

namespace mozilla::gfx {

// The generic templates treat each pixel as a little-endian 32-bit lane with
// the alpha in the top byte, so these specializations are only built for
// little-endian targets (see moz.build).

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> LoadRemainder_SIMD(
    const uint8_t* aSrc, size_t aLength) {
  uint64_t lo;
  uint32_t hi = 0;
  if (aLength >= 2) {
    memcpy(&lo, aSrc, 8);
    if (aLength >= 3) {
      memcpy(&hi, aSrc + 8, 4);
    }
  } else {
    uint32_t px;
    memcpy(&px, aSrc, 4);
    lo = px;
  }
  __vector unsigned long long v = {lo, hi};
  return (__vector unsigned char)v;
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE void StoreRemainder_SIMD(
    uint8_t* aDst, size_t aLength, const xsimd::batch<uint8_t, Arch>& aSrc) {
  __vector unsigned int v = (__vector unsigned int)aSrc.data;
  if (aLength >= 2) {
    uint64_t lo = vec_extract((__vector unsigned long long)v, 0);
    memcpy(aDst, &lo, 8);
    if (aLength >= 3) {
      uint32_t hi = vec_extract(v, 2);
      memcpy(aDst + 8, &hi, 4);
    }
  } else {
    uint32_t px = vec_extract(v, 0);
    memcpy(aDst, &px, 4);
  }
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> LoadRemainderRGB_SIMD(
    const uint8_t* aSrc, size_t aLength) {
  // Load aLength (1-4) packed RGB pixels (3-12 bytes) without reading past the
  // 3*aLength valid source bytes.
  alignas(16) uint8_t buffer[16] = {};
  switch (aLength) {
    case 4:
      memcpy(buffer, aSrc, 12);
      break;
    case 3:
      memcpy(buffer, aSrc, 9);
      break;
    case 2:
      memcpy(buffer, aSrc, 6);
      break;
    default:
      memcpy(buffer, aSrc, 3);
      break;
  }
  return xsimd::batch<uint8_t, Arch>::load_aligned(buffer);
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint16_t, Arch> ExtractAlpha_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc,
    const xsimd::batch<uint16_t, Arch>& aGreenAlpha) {
  // G0 A0 G1 A1 ... -> A0 A0 A1 A1 ...
  const __vector unsigned char alphaMask = {2,  3,  2,  3,  6,  7,  6,  7,
                                            10, 11, 10, 11, 14, 15, 14, 15};
  return vec_perm(aGreenAlpha.data, aGreenAlpha.data, alphaMask);
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> SwapRB8_SIMD(
    const xsimd::batch<uint8_t, Arch>& aPx) {
  const __vector unsigned char swapRbMask = {2,  1, 0, 3,  6,  5,  4,  7,
                                             10, 9, 8, 11, 14, 13, 12, 15};
  return vec_perm(aPx.data, aPx.data, swapRbMask);
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint16_t, Arch> SwapRB16_SIMD(
    const xsimd::batch<uint16_t, Arch>& aRb) {
  return (__vector unsigned short)vec_rl((__vector unsigned int)aRb.data,
                                         vec_splats(16u));
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint32_t, Arch> UnpremultiplyLookup_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc,
    const xsimd::batch<uint16_t, Arch>& aGa) {
  // The alphas are the odd words of the isolated G/A lanes.
  alignas(Arch::alignment())
      uint16_t alphaBuf[xsimd::batch<uint16_t, Arch>::size];
  aGa.store_aligned(alphaBuf);
  __vector unsigned int recip = {
      sUnpremultiplyTable[alphaBuf[1]], sUnpremultiplyTable[alphaBuf[3]],
      sUnpremultiplyTable[alphaBuf[5]], sUnpremultiplyTable[alphaBuf[7]]};
  return recip;
}

template <class Arch>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> UnpremultiplyReverse_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc,
    const xsimd::batch<uint32_t, Arch>& aRecip,
    const xsimd::batch<uint16_t, Arch>& aRb,
    const xsimd::batch<uint16_t, Arch>& aGa) {
  // channel * Q fits in 32 bits (Q <= 0xFF00FF), so each channel is scaled
  // with a 32-bit multiply and rounded, (channel * Q + 0x8000) >> 16, instead
  // of the 16-bit high/low split the generic version has to use. Masking to a
  // byte makes out-of-range (channel > alpha) input wrap like the scalar path.
  const xsimd::batch<uint32_t, Arch> low16(0x0000FFFF);
  const xsimd::batch<uint32_t, Arch> lowByte(0x000000FF);
  const xsimd::batch<uint32_t, Arch> half(0x8000);
  auto rb = xsimd::bitwise_cast<uint32_t>(aRb);
  auto ga = xsimd::bitwise_cast<uint32_t>(aGa);
  auto r = (((rb & low16) * aRecip + half) >> 16) & lowByte;
  auto b = (((rb >> 16) * aRecip + half) >> 16) & lowByte;
  auto g = (((ga & low16) * aRecip + half) >> 16) & lowByte;
  auto alpha = xsimd::bitwise_cast<uint32_t>(aSrc) &
               xsimd::batch<uint32_t, Arch>(0xFF000000);
  return xsimd::bitwise_cast<uint8_t>(r | (g << 8) | (b << 16) | alpha);
}

// xsimd has no byte swizzle for VSX, so expand with vec_perm directly.
template <class Arch, bool aSwapRB>
  requires std::same_as<Arch, xsimd::vsx>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> UnpackBatchRGB24_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc) {
  const __vector unsigned char expandMask =
      aSwapRB ? (__vector unsigned char){2, 1, 0, 0, 5,  4,  3, 0,
                                         8, 7, 6, 0, 11, 10, 9, 0}
              : (__vector unsigned char){0, 1, 2, 0, 3, 4,  5,  0,
                                         6, 7, 8, 0, 9, 10, 11, 0};
  __vector unsigned int px =
      (__vector unsigned int)vec_perm(aSrc.data, aSrc.data, expandMask);
  return (__vector unsigned char)vec_or(px, vec_splats(0xFF000000u));
}

}  // namespace mozilla::gfx

#endif /* MOZILLA_GFX_SWIZZLE_VSX_H_ */
