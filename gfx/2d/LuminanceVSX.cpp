/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "LuminanceVSX.h"

#include <altivec.h>

#include <bit>

using namespace mozilla::gfx;

// Pixels are B8G8R8A8 in memory on both endiannesses.
#define GFX_ARGB32_OFFSET_A 3
#define GFX_ARGB32_OFFSET_R 2
#define GFX_ARGB32_OFFSET_G 1
#define GFX_ARGB32_OFFSET_B 0

void ComputesRGBLuminanceMask_VSX(const uint8_t* aSourceData,
                                  int32_t aSourceStride, uint8_t* aDestData,
                                  int32_t aDestStride, const IntSize& aSize,
                                  float aOpacity) {
  typedef __vector unsigned char u8x16;
  typedef __vector unsigned short u16x8;
  typedef __vector unsigned int u32x4;

  int32_t redFactor = 55 * aOpacity;     // 255 * 0.2125 * opacity
  int32_t greenFactor = 183 * aOpacity;  // 255 * 0.7154 * opacity
  int32_t blueFactor = 18 * aOpacity;    // 255 * 0.0721
  int32_t sourceOffset = aSourceStride - 4 * aSize.width;
  const uint8_t* sourcePixel = aSourceData;
  int32_t destOffset = aDestStride - aSize.width;
  uint8_t* destPixel = aDestData;

  // One factor per channel byte; vec_msum then sums the four byte products
  // of each pixel into its 32-bit lane.
  u8x16 factors = vec_splats((unsigned char)0);
  factors = vec_insert((unsigned char)blueFactor, factors, GFX_ARGB32_OFFSET_B);
  factors =
      vec_insert((unsigned char)greenFactor, factors, GFX_ARGB32_OFFSET_G);
  factors = vec_insert((unsigned char)redFactor, factors, GFX_ARGB32_OFFSET_R);
  factors = (u8x16)vec_splat((u32x4)factors, 0);
  const u32x4 zero = vec_splats(0u);
  const u32x4 alphaMask = vec_splats(
      std::endian::native == std::endian::little ? 0xFF000000u : 0x000000FFu);
  const u32x4 eight = vec_splats(8u);

  auto grayOf = [&](const uint8_t* aPx) {
    u8x16 px = vec_xl(0, aPx);
    u32x4 gray = vec_sr(vec_msum(px, factors, zero), eight);
    // Transparent pixels get a zero mask value.
    __vector __bool int transparent =
        vec_cmpeq(vec_and((u32x4)px, alphaMask), zero);
    return vec_andc(gray, (u32x4)transparent);
  };

  int32_t remainderWidth = aSize.width % 16;
  int32_t roundedWidth = aSize.width - remainderWidth;

  for (int32_t y = 0; y < aSize.height; y++) {
    for (int32_t x = 0; x < roundedWidth; x += 16) {
      u32x4 g0 = grayOf(sourcePixel);
      u32x4 g1 = grayOf(sourcePixel + 16);
      u32x4 g2 = grayOf(sourcePixel + 32);
      u32x4 g3 = grayOf(sourcePixel + 48);
      u16x8 g01 = vec_pack(g0, g1);
      u16x8 g23 = vec_pack(g2, g3);
      vec_xst(vec_pack(g01, g23), 0, destPixel);
      sourcePixel += 16 * 4;
      destPixel += 16;
    }

    for (int32_t x = 0; x < remainderWidth; x++) {
      if (sourcePixel[GFX_ARGB32_OFFSET_A] > 0) {
        *destPixel = (redFactor * sourcePixel[GFX_ARGB32_OFFSET_R] +
                      greenFactor * sourcePixel[GFX_ARGB32_OFFSET_G] +
                      blueFactor * sourcePixel[GFX_ARGB32_OFFSET_B]) >>
                     8;
      } else {
        *destPixel = 0;
      }
      sourcePixel += 4;
      destPixel++;
    }

    sourcePixel += sourceOffset;
    destPixel += destOffset;
  }
}
