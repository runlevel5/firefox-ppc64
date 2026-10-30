/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define SIMD_COMPILE_ALTIVEC

#include "FilterProcessingSIMD-inl.h"

#ifndef USE_ALTIVEC
static_assert(
    false, "If this file is built, FilterProcessing.h should know about it!");
#endif

namespace mozilla::gfx {

void FilterProcessing::ExtractAlpha_AltiVec(const IntSize& size,
                                            const uint8_t* sourceData,
                                            int32_t sourceStride,
                                            uint8_t* alphaData,
                                            int32_t alphaStride) {
  ExtractAlpha_SIMD<simd::u8x16_av>(size, sourceData, sourceStride, alphaData,
                                    alphaStride);
}

already_AddRefed<DataSourceSurface> FilterProcessing::ConvertToB8G8R8A8_AltiVec(
    SourceSurface* aSurface) {
  return ConvertToB8G8R8A8_SIMD<simd::u8x16_av>(aSurface);
}

already_AddRefed<DataSourceSurface> FilterProcessing::ApplyBlending_AltiVec(
    DataSourceSurface* aInput1, DataSourceSurface* aInput2,
    BlendMode aBlendMode) {
  return ApplyBlending_SIMD<simd::i32x4_av, simd::i16x8_av, simd::u8x16_av>(
      aInput1, aInput2, aBlendMode);
}

void FilterProcessing::ApplyMorphologyHorizontal_AltiVec(
    const uint8_t* aSourceData, int32_t aSourceStride, uint8_t* aDestData,
    int32_t aDestStride, const IntRect& aDestRect, int32_t aRadius,
    MorphologyOperator aOp) {
  ApplyMorphologyHorizontal_SIMD<simd::i16x8_av, simd::u8x16_av>(
      aSourceData, aSourceStride, aDestData, aDestStride, aDestRect, aRadius,
      aOp);
}

void FilterProcessing::ApplyMorphologyVertical_AltiVec(
    const uint8_t* aSourceData, int32_t aSourceStride, uint8_t* aDestData,
    int32_t aDestStride, const IntRect& aDestRect, int32_t aRadius,
    MorphologyOperator aOp) {
  ApplyMorphologyVertical_SIMD<simd::i16x8_av, simd::u8x16_av>(
      aSourceData, aSourceStride, aDestData, aDestStride, aDestRect, aRadius,
      aOp);
}

already_AddRefed<DataSourceSurface> FilterProcessing::ApplyColorMatrix_AltiVec(
    DataSourceSurface* aInput, const Matrix5x4& aMatrix) {
  return ApplyColorMatrix_SIMD<simd::i32x4_av, simd::i16x8_av, simd::u8x16_av>(
      aInput, aMatrix);
}

void FilterProcessing::ApplyComposition_AltiVec(DataSourceSurface* aSource,
                                                DataSourceSurface* aDest,
                                                CompositeOperator aOperator) {
  return ApplyComposition_SIMD<simd::i32x4_av, simd::i16x8_av, simd::u8x16_av>(
      aSource, aDest, aOperator);
}

void FilterProcessing::DoOpacityCalculation_AltiVec(
    const IntSize& aSize, uint8_t* aTargetData, int32_t aTargetStride,
    const uint8_t* aSourceData, int32_t aSourceStride, Float aValue) {
  DoOpacityCalculation_SIMD<simd::i16x8_av, simd::u8x16_av>(
      aSize, aTargetData, aTargetStride, aSourceData, aSourceStride, aValue);
}

already_AddRefed<DataSourceSurface> FilterProcessing::RenderTurbulence_AltiVec(
    const IntSize& aSize, const Point& aOffset, const Size& aBaseFrequency,
    int32_t aSeed, int aNumOctaves, TurbulenceType aType, bool aStitch,
    const Rect& aTileRect) {
  return RenderTurbulence_SIMD<simd::f32x4_av, simd::i32x4_av, simd::u8x16_av>(
      aSize, aOffset, aBaseFrequency, aSeed, aNumOctaves, aType, aStitch,
      aTileRect);
}

already_AddRefed<DataSourceSurface>
FilterProcessing::ApplyArithmeticCombine_AltiVec(DataSourceSurface* aInput1,
                                                 DataSourceSurface* aInput2,
                                                 Float aK1, Float aK2,
                                                 Float aK3, Float aK4) {
  return ApplyArithmeticCombine_SIMD<simd::i32x4_av, simd::i16x8_av,
                                     simd::u8x16_av>(aInput1, aInput2, aK1, aK2,
                                                     aK3, aK4);
}

}  // namespace mozilla::gfx
