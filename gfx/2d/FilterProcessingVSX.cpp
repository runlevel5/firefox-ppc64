/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define SIMD_COMPILE_VSX

#include "FilterProcessingSIMD-inl.h"

#ifndef USE_VSX
static_assert(
    false, "If this file is built, FilterProcessing.h should know about it!");
#endif

namespace mozilla::gfx {

void FilterProcessing::ExtractAlpha_VSX(const IntSize& size,
                                        const uint8_t* sourceData,
                                        int32_t sourceStride,
                                        uint8_t* alphaData,
                                        int32_t alphaStride) {
  ExtractAlpha_SIMD<simd::u8x16_vsx>(size, sourceData, sourceStride, alphaData,
                                     alphaStride);
}

already_AddRefed<DataSourceSurface> FilterProcessing::ConvertToB8G8R8A8_VSX(
    SourceSurface* aSurface) {
  return ConvertToB8G8R8A8_SIMD<simd::u8x16_vsx>(aSurface);
}

already_AddRefed<DataSourceSurface> FilterProcessing::ApplyBlending_VSX(
    DataSourceSurface* aInput1, DataSourceSurface* aInput2,
    BlendMode aBlendMode) {
  return ApplyBlending_SIMD<simd::i32x4_vsx, simd::i16x8_vsx, simd::u8x16_vsx>(
      aInput1, aInput2, aBlendMode);
}

void FilterProcessing::ApplyMorphologyHorizontal_VSX(
    const uint8_t* aSourceData, int32_t aSourceStride, uint8_t* aDestData,
    int32_t aDestStride, const IntRect& aDestRect, int32_t aRadius,
    MorphologyOperator aOp) {
  ApplyMorphologyHorizontal_SIMD<simd::i16x8_vsx, simd::u8x16_vsx>(
      aSourceData, aSourceStride, aDestData, aDestStride, aDestRect, aRadius,
      aOp);
}

void FilterProcessing::ApplyMorphologyVertical_VSX(
    const uint8_t* aSourceData, int32_t aSourceStride, uint8_t* aDestData,
    int32_t aDestStride, const IntRect& aDestRect, int32_t aRadius,
    MorphologyOperator aOp) {
  ApplyMorphologyVertical_SIMD<simd::i16x8_vsx, simd::u8x16_vsx>(
      aSourceData, aSourceStride, aDestData, aDestStride, aDestRect, aRadius,
      aOp);
}

already_AddRefed<DataSourceSurface> FilterProcessing::ApplyColorMatrix_VSX(
    DataSourceSurface* aInput, const Matrix5x4& aMatrix) {
  return ApplyColorMatrix_SIMD<simd::i32x4_vsx, simd::i16x8_vsx,
                               simd::u8x16_vsx>(aInput, aMatrix);
}

void FilterProcessing::ApplyComposition_VSX(DataSourceSurface* aSource,
                                            DataSourceSurface* aDest,
                                            CompositeOperator aOperator) {
  return ApplyComposition_SIMD<simd::i32x4_vsx, simd::i16x8_vsx,
                               simd::u8x16_vsx>(aSource, aDest, aOperator);
}

void FilterProcessing::DoOpacityCalculation_VSX(
    const IntSize& aSize, uint8_t* aTargetData, int32_t aTargetStride,
    const uint8_t* aSourceData, int32_t aSourceStride, Float aValue) {
  DoOpacityCalculation_SIMD<simd::i16x8_vsx, simd::u8x16_vsx>(
      aSize, aTargetData, aTargetStride, aSourceData, aSourceStride, aValue);
}

already_AddRefed<DataSourceSurface> FilterProcessing::RenderTurbulence_VSX(
    const IntSize& aSize, const Point& aOffset, const Size& aBaseFrequency,
    int32_t aSeed, int aNumOctaves, TurbulenceType aType, bool aStitch,
    const Rect& aTileRect) {
  return RenderTurbulence_SIMD<simd::f32x4_vsx, simd::i32x4_vsx,
                               simd::u8x16_vsx>(aSize, aOffset, aBaseFrequency,
                                                aSeed, aNumOctaves, aType,
                                                aStitch, aTileRect);
}

already_AddRefed<DataSourceSurface>
FilterProcessing::ApplyArithmeticCombine_VSX(DataSourceSurface* aInput1,
                                             DataSourceSurface* aInput2,
                                             Float aK1, Float aK2, Float aK3,
                                             Float aK4) {
  return ApplyArithmeticCombine_SIMD<simd::i32x4_vsx, simd::i16x8_vsx,
                                     simd::u8x16_vsx>(aInput1, aInput2, aK1,
                                                      aK2, aK3, aK4);
}

}  // namespace mozilla::gfx
