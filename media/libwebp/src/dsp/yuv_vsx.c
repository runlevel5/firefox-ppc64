// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC64) version of YUV->RGB conversion functions.
//
// Authors: Trung Lê (8@tle.id.au)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <string.h>

#include "src/dsp/yuv.h"

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;
typedef __vector signed short i16x8;
typedef __vector unsigned int u32x4;

// PPC64 runs in both endiannesses. GCC's vec_merge*/vec_perm/vec_mule are
// element-array-consistent across endianness, but a *widening* merge that
// reinterprets adjacent narrow lanes as a wider type lands them in the opposite
// half on big-endian. WMERGEH/WMERGEL keep the little-endian lane order on both
// by swapping the operands on big-endian.
#if (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define WMERGEH(a, b) vec_mergeh((b), (a))
#define WMERGEL(a, b) vec_mergel((b), (a))
#else
#define WMERGEH(a, b) vec_mergeh((a), (b))
#define WMERGEL(a, b) vec_mergel((a), (b))
#endif

// POWER8 has no "multiply-high unsigned halfword", so emulate _mm_mulhi_epu16
// via even/odd 16x16->32 products, >>16, then interleave back.
static WEBP_INLINE u16x8 MulHi16(u16x8 a, u16x8 b) {
  const u32x4 sh = vec_splats((unsigned int)16);
  const u32x4 e = vec_sr(vec_mule(a, b), sh);
  const u32x4 o = vec_sr(vec_mulo(a, b), sh);
  return vec_pack(vec_mergeh(e, o), vec_mergel(e, o));
}

// 14b fixed-point ITU-R BT.601 YUV->RGB, matching the SSE2/scalar path.
// Inputs are samples pre-shifted into the high byte (<< 8).
static WEBP_INLINE void ConvertYUV444ToRGB(u16x8 Y0, u16x8 U0, u16x8 V0,
                                           i16x8* const R, i16x8* const G,
                                           u16x8* const B) {
  const u16x8 k19077 = vec_splats((unsigned short)19077);
  const u16x8 k26149 = vec_splats((unsigned short)26149);
  const u16x8 k14234 = vec_splats((unsigned short)14234);
  const u16x8 k33050 = vec_splats((unsigned short)33050);
  const u16x8 k17685 = vec_splats((unsigned short)17685);
  const u16x8 k6419 = vec_splats((unsigned short)6419);
  const u16x8 k13320 = vec_splats((unsigned short)13320);
  const u16x8 k8708 = vec_splats((unsigned short)8708);
  const u16x8 six = vec_splats((unsigned short)6);

  const u16x8 Y1 = MulHi16(Y0, k19077);
  const u16x8 R2 = vec_add(vec_sub(Y1, k14234), MulHi16(V0, k26149));
  const u16x8 G4 = vec_sub(vec_add(Y1, k8708),
                           vec_add(MulHi16(U0, k6419), MulHi16(V0, k13320)));
  // 33050 needs unsigned saturating arithmetic; B can exceed 32767.
  const u16x8 B2 = vec_subs(vec_adds(MulHi16(U0, k33050), Y1), k17685);

  *R = vec_sra((i16x8)R2, six);
  *G = vec_sra((i16x8)G4, six);
  *B = vec_sr(B2, six);
}

// Load 8 bytes into the high byte of 8 u16 lanes (i.e. sample << 8).
// Use an 8-byte copy (not a 16-byte vector load) to avoid reading past the
// end of the source row, matching the SSE2 _mm_loadl_epi64 behavior.
static WEBP_INLINE u16x8 LoadHi16(const uint8_t* WEBP_RESTRICT src) {
  const u8x16 zero = vec_splats((unsigned char)0);
  unsigned char tmp[16] = {0};
  memcpy(tmp, src, 8);
  return (u16x8)WMERGEH(zero, vec_xl(0, tmp));
}

// Load 4 U/V bytes, shift into the high byte, and replicate each sample.
static WEBP_INLINE u16x8 LoadUVHi8(const uint8_t* WEBP_RESTRICT src) {
  const u8x16 zero = vec_splats((unsigned char)0);
  unsigned char tmp[16] = {0};
  memcpy(tmp, src, 4);
  {
    const u16x8 t = (u16x8)WMERGEH(zero, vec_xl(0, tmp));
    return vec_mergeh(t, t);
  }
}

static WEBP_INLINE void YUV420ToRGB(const uint8_t* WEBP_RESTRICT y,
                                    const uint8_t* WEBP_RESTRICT u,
                                    const uint8_t* WEBP_RESTRICT v,
                                    i16x8* const R, i16x8* const G,
                                    u16x8* const B) {
  ConvertYUV444ToRGB(LoadHi16(y), LoadUVHi8(u), LoadUVHi8(v), R, G, B);
}

// Pack four 8-lane channels into 32 interleaved bytes (c0 c1 c2 c3 per pixel).
static WEBP_INLINE void PackAndStore4(i16x8 c0, i16x8 c1, i16x8 c2, i16x8 c3,
                                      uint8_t* WEBP_RESTRICT dst) {
  const u8x16 c02 = vec_packsu(c0, c2);
  const u8x16 c13 = vec_packsu(c1, c3);
  const u8x16 lo8 = vec_mergeh(c02, c13);
  const u8x16 hi8 = vec_mergel(c02, c13);
  vec_xst((u8x16)vec_mergeh((u16x8)lo8, (u16x8)hi8), 0, dst);
  vec_xst((u8x16)vec_mergel((u16x8)lo8, (u16x8)hi8), 0, dst + 16);
}

static const i16x8 kAlpha = {255, 255, 255, 255, 255, 255, 255, 255};

// Pack three 8-lane channels into 24 interleaved bytes (c0 c1 c2 per pixel).
static const u8x16 kPack3Lo = {0,  8, 16, 1,  9, 17, 2,  10,
                               18, 3, 11, 19, 4, 12, 20, 5};
static const u8x16 kPack3Hi = {13, 21, 6, 14, 22, 7, 15, 23,
                               0,  0,  0, 0,  0,  0, 0,  0};
static WEBP_INLINE void PackAndStore3(i16x8 c0, i16x8 c1, i16x8 c2,
                                      uint8_t* WEBP_RESTRICT dst) {
  const u8x16 c01 = vec_packsu(c0, c1);  // [c0_0..7 | c1_0..7]
  const u8x16 c22 = vec_packsu(c2, c2);  // [c2_0..7 | ...]
  const u8x16 lo = vec_perm(c01, c22, kPack3Lo);
  const u8x16 hi = vec_perm(c01, c22, kPack3Hi);
  vec_xst(lo, 0, dst);
  memcpy(dst + 16, &hi, 8);
}

static void YuvToRgbaRow_VSX(const uint8_t* WEBP_RESTRICT y,
                             const uint8_t* WEBP_RESTRICT u,
                             const uint8_t* WEBP_RESTRICT v,
                             uint8_t* WEBP_RESTRICT dst, int len) {
  int n;
  for (n = 0; n + 8 <= len; n += 8, dst += 32) {
    i16x8 R, G;
    u16x8 B;
    YUV420ToRGB(y, u, v, &R, &G, &B);
    PackAndStore4(R, G, (i16x8)B, kAlpha, dst);
    y += 8;
    u += 4;
    v += 4;
  }
  for (; n < len; ++n) {
    VP8YuvToRgba(y[0], u[0], v[0], dst);
    dst += 4;
    y += 1;
    u += (n & 1);
    v += (n & 1);
  }
}

static void YuvToBgraRow_VSX(const uint8_t* WEBP_RESTRICT y,
                             const uint8_t* WEBP_RESTRICT u,
                             const uint8_t* WEBP_RESTRICT v,
                             uint8_t* WEBP_RESTRICT dst, int len) {
  int n;
  for (n = 0; n + 8 <= len; n += 8, dst += 32) {
    i16x8 R, G;
    u16x8 B;
    YUV420ToRGB(y, u, v, &R, &G, &B);
    PackAndStore4((i16x8)B, G, R, kAlpha, dst);
    y += 8;
    u += 4;
    v += 4;
  }
  for (; n < len; ++n) {
    VP8YuvToBgra(y[0], u[0], v[0], dst);
    dst += 4;
    y += 1;
    u += (n & 1);
    v += (n & 1);
  }
}

static void YuvToArgbRow_VSX(const uint8_t* WEBP_RESTRICT y,
                             const uint8_t* WEBP_RESTRICT u,
                             const uint8_t* WEBP_RESTRICT v,
                             uint8_t* WEBP_RESTRICT dst, int len) {
  int n;
  for (n = 0; n + 8 <= len; n += 8, dst += 32) {
    i16x8 R, G;
    u16x8 B;
    YUV420ToRGB(y, u, v, &R, &G, &B);
    PackAndStore4(kAlpha, R, G, (i16x8)B, dst);
    y += 8;
    u += 4;
    v += 4;
  }
  for (; n < len; ++n) {
    VP8YuvToArgb(y[0], u[0], v[0], dst);
    dst += 4;
    y += 1;
    u += (n & 1);
    v += (n & 1);
  }
}

// Convert 32 YUV444 pixels and store the 32b-per-pixel result. Used by the
// fancy upsampler in upsampling_vsx.c.
void VP8YuvToRgba32_VSX(const uint8_t* WEBP_RESTRICT y,
                        const uint8_t* WEBP_RESTRICT u,
                        const uint8_t* WEBP_RESTRICT v,
                        uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 32) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore4(R, G, (i16x8)B, kAlpha, dst);
  }
}

void VP8YuvToBgra32_VSX(const uint8_t* WEBP_RESTRICT y,
                        const uint8_t* WEBP_RESTRICT u,
                        const uint8_t* WEBP_RESTRICT v,
                        uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 32) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore4((i16x8)B, G, R, kAlpha, dst);
  }
}

void VP8YuvToArgb32_VSX(const uint8_t* WEBP_RESTRICT y,
                        const uint8_t* WEBP_RESTRICT u,
                        const uint8_t* WEBP_RESTRICT v,
                        uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 32) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore4(kAlpha, R, G, (i16x8)B, dst);
  }
}

// Pack R/G/B/A into 16-bit rgba4444 (8 pixels -> 16 bytes).
static WEBP_INLINE void PackAndStore4444(i16x8 R, i16x8 G, u16x8 B, i16x8 A,
                                         uint8_t* WEBP_RESTRICT dst) {
  const u8x16 mask = vec_splats((unsigned char)0xf0);
  const u16x8 four = vec_splats((unsigned short)4);
#if (WEBP_SWAP_16BIT_CSP == 0)
  const u8x16 rg0 = vec_packsu(R, G);
  const u8x16 ba0 = vec_packsu((i16x8)B, A);
#else
  const u8x16 rg0 = vec_packsu((i16x8)B, A);
  const u8x16 ba0 = vec_packsu(R, G);
#endif
  const u8x16 rb1 = vec_mergeh(rg0, ba0);
  const u8x16 ga1 = vec_mergel(rg0, ba0);
  const u8x16 rb2 = vec_and(rb1, mask);
  const u8x16 ga2 = (u8x16)vec_sr((u16x8)vec_and(ga1, mask), four);
  vec_xst(vec_or(rb2, ga2), 0, dst);
}

// Pack R/G/B into 16-bit rgb565 (8 pixels -> 16 bytes).
static WEBP_INLINE void PackAndStore565(i16x8 R, i16x8 G, u16x8 B,
                                        uint8_t* WEBP_RESTRICT dst) {
  const u16x8 three = vec_splats((unsigned short)3);
  const u16x8 five = vec_splats((unsigned short)5);
  const u8x16 r0 = vec_packsu(R, R);
  const u8x16 g0 = vec_packsu(G, G);
  const u8x16 b0 = vec_packsu((i16x8)B, (i16x8)B);
  const u8x16 r1 = vec_and(r0, vec_splats((unsigned char)0xf8));
  const u8x16 b1 =
      vec_and((u8x16)vec_sr((u16x8)b0, three), vec_splats((unsigned char)0x1f));
  const u8x16 g1 =
      (u8x16)vec_sr((u16x8)vec_and(g0, vec_splats((unsigned char)0xe0)), five);
  const u8x16 g2 =
      (u8x16)vec_sl((u16x8)vec_and(g0, vec_splats((unsigned char)0x1c)), three);
  const u8x16 rg = vec_or(r1, g1);
  const u8x16 gb = vec_or(g2, b1);
#if (WEBP_SWAP_16BIT_CSP == 0)
  vec_xst(vec_mergeh(rg, gb), 0, dst);
#else
  vec_xst(vec_mergeh(gb, rg), 0, dst);
#endif
}

void VP8YuvToRgba444432_VSX(const uint8_t* WEBP_RESTRICT y,
                            const uint8_t* WEBP_RESTRICT u,
                            const uint8_t* WEBP_RESTRICT v,
                            uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 16) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore4444(R, G, B, kAlpha, dst);
  }
}

void VP8YuvToRgb56532_VSX(const uint8_t* WEBP_RESTRICT y,
                          const uint8_t* WEBP_RESTRICT u,
                          const uint8_t* WEBP_RESTRICT v,
                          uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 16) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore565(R, G, B, dst);
  }
}

void VP8YuvToRgb32_VSX(const uint8_t* WEBP_RESTRICT y,
                       const uint8_t* WEBP_RESTRICT u,
                       const uint8_t* WEBP_RESTRICT v,
                       uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 24) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore3(R, G, (i16x8)B, dst);
  }
}

void VP8YuvToBgr32_VSX(const uint8_t* WEBP_RESTRICT y,
                       const uint8_t* WEBP_RESTRICT u,
                       const uint8_t* WEBP_RESTRICT v,
                       uint8_t* WEBP_RESTRICT dst) {
  int n;
  for (n = 0; n < 32; n += 8, dst += 24) {
    i16x8 R, G;
    u16x8 B;
    ConvertYUV444ToRGB(LoadHi16(y + n), LoadHi16(u + n), LoadHi16(v + n), &R,
                       &G, &B);
    PackAndStore3((i16x8)B, G, R, dst);
  }
}

static void YuvToRgbRow_VSX(const uint8_t* WEBP_RESTRICT y,
                            const uint8_t* WEBP_RESTRICT u,
                            const uint8_t* WEBP_RESTRICT v,
                            uint8_t* WEBP_RESTRICT dst, int len) {
  int n;
  for (n = 0; n + 8 <= len; n += 8, dst += 24) {
    i16x8 R, G;
    u16x8 B;
    YUV420ToRGB(y, u, v, &R, &G, &B);
    PackAndStore3(R, G, (i16x8)B, dst);
    y += 8;
    u += 4;
    v += 4;
  }
  for (; n < len; ++n) {
    VP8YuvToRgb(y[0], u[0], v[0], dst);
    dst += 3;
    y += 1;
    u += (n & 1);
    v += (n & 1);
  }
}

static void YuvToBgrRow_VSX(const uint8_t* WEBP_RESTRICT y,
                            const uint8_t* WEBP_RESTRICT u,
                            const uint8_t* WEBP_RESTRICT v,
                            uint8_t* WEBP_RESTRICT dst, int len) {
  int n;
  for (n = 0; n + 8 <= len; n += 8, dst += 24) {
    i16x8 R, G;
    u16x8 B;
    YUV420ToRGB(y, u, v, &R, &G, &B);
    PackAndStore3((i16x8)B, G, R, dst);
    y += 8;
    u += 4;
    v += 4;
  }
  for (; n < len; ++n) {
    VP8YuvToBgr(y[0], u[0], v[0], dst);
    dst += 3;
    y += 1;
    u += (n & 1);
    v += (n & 1);
  }
}

extern void WebPInitSamplersVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitSamplersVSX(void) {
  WebPSamplers[MODE_RGB] = YuvToRgbRow_VSX;
  WebPSamplers[MODE_RGBA] = YuvToRgbaRow_VSX;
  WebPSamplers[MODE_BGR] = YuvToBgrRow_VSX;
  WebPSamplers[MODE_BGRA] = YuvToBgraRow_VSX;
  WebPSamplers[MODE_ARGB] = YuvToArgbRow_VSX;
}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(WebPInitSamplersVSX)

#endif  // WEBP_USE_VSX
