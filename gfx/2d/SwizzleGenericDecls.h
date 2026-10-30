/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MOZILLA_GFX_SWIZZLE_GENERIC_DECLS_H_
#define MOZILLA_GFX_SWIZZLE_GENERIC_DECLS_H_

#include <bit>
#include <xsimd/xsimd.hpp>

#include "Swizzle.h"

namespace mozilla::gfx {

extern const uint32_t sUnpremultiplyTable[256];

// Pixels are processed as native 32-bit lanes, so which byte holds the alpha
// and which 16-bit half of a lane holds the even (R/B) or odd (G/A) channels
// depends on endianness.
static constexpr bool kSwizzleLE = std::endian::native == std::endian::little;
static constexpr uint32_t kAlphaMask32 = kSwizzleLE ? 0xFF000000 : 0x000000FF;
// The alpha and green bytes within the odd (G/A) 16-bit lane pair of a pixel.
static constexpr uint32_t kOddAlphaMask32 =
    kSwizzleLE ? 0x00FF0000 : 0x000000FF;
static constexpr uint32_t kOddGreenMask32 =
    kSwizzleLE ? 0x000000FF : 0x00FF0000;

// The following template declarations are for templates our architecture
// specific headers may choose to overload.

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> LoadRemainder_SIMD(
    const uint8_t* aSrc, size_t aLength);

template <class Arch>
static MOZ_ALWAYS_INLINE void StoreRemainder_SIMD(
    uint8_t* aDst, size_t aLength, const xsimd::batch<uint8_t, Arch>& aSrc);

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> LoadRemainderRGB_SIMD(
    const uint8_t* aSrc, size_t aLength);

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint16_t, Arch> ExtractAlpha_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc,
    const xsimd::batch<uint16_t, Arch>& aGreenAlpha);

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> SwapRB8_SIMD(
    const xsimd::batch<uint8_t, Arch>& aPx);

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint16_t, Arch> SwapRB16_SIMD(
    const xsimd::batch<uint16_t, Arch>& aRb);

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint32_t, Arch> UnpremultiplyLookup_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc,
    const xsimd::batch<uint16_t, Arch>& aGa);

template <class Arch>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> UnpremultiplyReverse_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc,
    const xsimd::batch<uint32_t, Arch>& aRecip,
    const xsimd::batch<uint16_t, Arch>& aRb,
    const xsimd::batch<uint16_t, Arch>& aGa);

template <class Arch, bool aSwapRB, bool aOpaqueAlpha>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> SwizzleVector_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc);

template <class Arch, bool aSwapRB>
static MOZ_ALWAYS_INLINE xsimd::batch<uint8_t, Arch> UnpackBatchRGB24_SIMD(
    const xsimd::batch<uint8_t, Arch>& aSrc);

}  // namespace mozilla::gfx

#endif /* MOZILLA_GFX_SWIZZLE_GENERIC_DECLS_H_ */
