/*
 *  Copyright 2026 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#if !defined(LIBYUV_DISABLE_VSX) && defined(__VSX__) && defined(_ARCH_PWR8)

#include <altivec.h>
#include <string.h>

#undef vector
#undef pixel
#undef bool

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

typedef __vector unsigned char v8u;
typedef __vector signed char v8s;
typedef __vector unsigned short v16u;
typedef __vector signed short v16s;
typedef __vector signed int v32s;
typedef __vector unsigned int v32u;
typedef __vector unsigned long long v64u;

static __inline v8u Load16_VSX(const uint8_t* p) {
  return vec_xl(0, p);
}

// Loads 8 bytes into the first half of the vector.
static __inline v8u Load8_VSX(const uint8_t* p) {
  unsigned long long v;
  memcpy(&v, p, 8);
  return (v8u)vec_splats(v);
}

static __inline void Store16_VSX(v8u v, uint8_t* p) {
  vec_xst(v, 0, p);
}

// Constant byte shuffle of two v8u. The indices count elements in memory
// order on both byte orders.
#if defined(__clang__) || __GNUC__ >= 12
#define SHUFFLE_VSX(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#define SHUFFLE_VSX(a, b, ...) __builtin_shuffle(a, b, (v8u){__VA_ARGS__})
#endif

// Each byte of the first or last 8, twice.
#define DUP2_LO_VSX 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
#define DUP2_HI_VSX 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
// Each of the first or second 4 bytes, four times.
#define DUP4_LO_VSX 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
#define DUP4_HI_VSX 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7

// a * b >> 16 for each lane. The shuffle picks the high half of each 32-bit
// product, which is the second 16-bit half in little-endian byte order and
// the first in big-endian.
static __inline v16u MulHi16_VSX(v16u a, v16u b) {
  v8u e = (v8u)vec_mule(a, b);
  v8u o = (v8u)vec_mulo(a, b);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v16u)SHUFFLE_VSX(e, o, 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14,
                           15, 30, 31);
#else
  return (v16u)SHUFFLE_VSX(e, o, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12,
                           13, 28, 29);
#endif
}

// Picks the high byte of each 16-bit lane of e and o, interleaved, so the
// bytes come out in the order they went into vec_mule and vec_mulo. The
// high byte is the odd byte in little-endian byte order and the even byte in
// big-endian.
static __inline v8u HighBytes16_VSX(v16u e, v16u o) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return SHUFFLE_VSX((v8u)e, (v8u)o, 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27,
                     13, 29, 15, 31);
#else
  return SHUFFLE_VSX((v8u)e, (v8u)o, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26,
                     12, 28, 14, 30);
#endif
}

// Stores 8 ARGB pixels from B and G of each pixel in bg, and R and A in ra.
static __inline void StoreARGB8_VSX(v8u bg, v8u ra, uint8_t* dst_argb) {
  Store16_VSX(SHUFFLE_VSX(bg, ra, 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3,
                          11, 19, 27),
              dst_argb);
  Store16_VSX(SHUFFLE_VSX(bg, ra, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30,
                          7, 15, 23, 31),
              dst_argb + 16);
}

// Stores 8 RGBA pixels (A, B, G, R in memory) from B and G of each pixel in
// bg, and R and A in ra.
static __inline void StoreRGBA8_VSX(v8u bg, v8u ra, uint8_t* dst_rgba) {
  Store16_VSX(SHUFFLE_VSX(bg, ra, 24, 0, 8, 16, 25, 1, 9, 17, 26, 2, 10, 18, 27,
                          3, 11, 19),
              dst_rgba);
  Store16_VSX(SHUFFLE_VSX(bg, ra, 28, 4, 12, 20, 29, 5, 13, 21, 30, 6, 14, 22,
                          31, 7, 15, 23),
              dst_rgba + 16);
}

// Stores 16 opaque gray ARGB pixels.
static __inline void StoreGray_VSX(v8u y, uint8_t* dst_argb) {
  v8u alpha = vec_splats((unsigned char)0xff);
  Store16_VSX(
      SHUFFLE_VSX(y, alpha, 0, 0, 0, 16, 1, 1, 1, 16, 2, 2, 2, 16, 3, 3, 3, 16),
      dst_argb);
  Store16_VSX(
      SHUFFLE_VSX(y, alpha, 4, 4, 4, 16, 5, 5, 5, 16, 6, 6, 6, 16, 7, 7, 7, 16),
      dst_argb + 16);
  Store16_VSX(SHUFFLE_VSX(y, alpha, 8, 8, 8, 16, 9, 9, 9, 16, 10, 10, 10, 16,
                          11, 11, 11, 16),
              dst_argb + 32);
  Store16_VSX(SHUFFLE_VSX(y, alpha, 12, 12, 12, 16, 13, 13, 13, 16, 14, 14, 14,
                          16, 15, 15, 15, 16),
              dst_argb + 48);
}

// YuvConstants splatted to 16-bit lanes.
struct YuvCoefs_VSX {
  v16s ub, ug, vg, vr, yb;
  v16u yg;
};

static __inline void LoadYuvCoefs_VSX(const struct YuvConstants* yuvconstants,
                                      struct YuvCoefs_VSX* c) {
  c->ub = vec_splats((short)yuvconstants->kUVToB[0]);
  c->ug = vec_splats((short)yuvconstants->kUVToG[0]);
  c->vg = vec_splats((short)yuvconstants->kUVToG[1]);
  c->vr = vec_splats((short)yuvconstants->kUVToR[1]);
  c->yg = vec_splats((unsigned short)yuvconstants->kYToRgb[0]);
  c->yb = vec_splats((short)yuvconstants->kYBiasToRgb[0]);
}

// The YUV rows widen Y, U and V by duplicating bytes, so each 16-bit lane
// holds its value times 0x0101.

// u - 128 from a lane holding u * 0x0101.
static __inline v16s UV16_VSX(v8u uv) {
  return vec_sub((v16s)vec_sr((v16u)uv, vec_splats((unsigned short)8)),
                 vec_splats((short)128));
}

// (y * 0x0101 * yg >> 16) + yb from a lane holding y * 0x0101.
static __inline v16s Y16_VSX(v8u y, const struct YuvCoefs_VSX* c) {
  return vec_add((v16s)MulHi16_VSX((v16u)y, c->yg), c->yb);
}

// U and V contributions to B, G and R for 8 pixels, or for 8 pairs of
// pixels in 4:2:2.
struct UVTerms_VSX {
  v16s b, g, r;
};

// From u - 128 and v - 128 in 16-bit lanes.
static __inline void UVTermsS_VSX(v16s ui,
                                  v16s vi,
                                  const struct YuvCoefs_VSX* c,
                                  struct UVTerms_VSX* t) {
  v16s zero = vec_splats((short)0);
  t->b = vec_mladd(ui, c->ub, zero);
  t->g = vec_mladd(ui, c->ug, vec_mladd(vi, c->vg, zero));
  t->r = vec_mladd(vi, c->vr, zero);
}

static __inline void UVTerms_VSX(v8u u,
                                 v8u v,
                                 const struct YuvCoefs_VSX* c,
                                 struct UVTerms_VSX* t) {
  UVTermsS_VSX(UV16_VSX(u), UV16_VSX(v), c, t);
}

// Converts 8 pixels and stores them. a is 8 alpha bytes in the first half,
// or null for opaque pixels.
static __inline void YuvToPixels8_VSX(v16s y1,
                                      v16s tb,
                                      v16s tg,
                                      v16s tr,
                                      const v8u* a,
                                      int rgba,
                                      uint8_t* dst) {
  v16u six = vec_splats((unsigned short)6);
  v16s b = vec_sra(vec_adds(y1, tb), six);
  v16s g = vec_sra(vec_subs(y1, tg), six);
  v16s r = vec_sra(vec_adds(y1, tr), six);
  v8u ra = a ? (v8u)vec_mergeh((v64u)vec_packsu(r, r), (v64u)*a)
             : vec_packsu(r, vec_splats((short)255));
  if (rgba) {
    StoreRGBA8_VSX(vec_packsu(b, g), ra, dst);
  } else {
    StoreARGB8_VSX(vec_packsu(b, g), ra, dst);
  }
}

static __inline void YuvToARGB8_VSX(v16s y1,
                                    v16s tb,
                                    v16s tg,
                                    v16s tr,
                                    const v8u* a,
                                    uint8_t* dst_argb) {
  YuvToPixels8_VSX(y1, tb, tg, tr, a, 0, dst_argb);
}

// Converts and stores 16 pixels that share 8 U and V in pairs. u and v are
// already widened, one lane per pair.
static __inline void Yuv422ToARGB16_VSX(v8u y,
                                        v8u u,
                                        v8u v,
                                        const v8u* a,
                                        const struct YuvCoefs_VSX* c,
                                        uint8_t* dst_argb) {
  struct UVTerms_VSX t;
  UVTerms_VSX(u, v, c, &t);
  v8u a1 =
      a ? (v8u)vec_mergel((v64u)*a, (v64u)*a) : vec_splats((unsigned char)0);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), c),
                 vec_mergeh(t.b, t.b), vec_mergeh(t.g, t.g),
                 vec_mergeh(t.r, t.r), a, dst_argb);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), c),
                 vec_mergel(t.b, t.b), vec_mergel(t.g, t.g),
                 vec_mergel(t.r, t.r), a ? &a1 : 0, dst_argb + 32);
}

// Widens the first 8 bytes to one lane per byte.
static __inline v8u Widen422_VSX(v8u uv) {
  return SHUFFLE_VSX(uv, uv, DUP2_LO_VSX);
}

// Converts and stores 16 pixels with their own U and V.
static __inline void Yuv444ToARGB16_VSX(v8u y,
                                        v8u u,
                                        v8u v,
                                        const v8u* a,
                                        const struct YuvCoefs_VSX* c,
                                        uint8_t* dst_argb) {
  struct UVTerms_VSX t0, t1;
  UVTerms_VSX(SHUFFLE_VSX(u, u, DUP2_LO_VSX), SHUFFLE_VSX(v, v, DUP2_LO_VSX), c,
              &t0);
  UVTerms_VSX(SHUFFLE_VSX(u, u, DUP2_HI_VSX), SHUFFLE_VSX(v, v, DUP2_HI_VSX), c,
              &t1);
  v8u a1 =
      a ? (v8u)vec_mergel((v64u)*a, (v64u)*a) : vec_splats((unsigned char)0);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), c), t0.b, t0.g, t0.r,
                 a, dst_argb);
  YuvToARGB8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), c), t1.b, t1.g, t1.r,
                 a ? &a1 : 0, dst_argb + 32);
}

void I422ToARGBRow_VSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    Yuv422ToARGB16_VSX(Load16_VSX(src_y), Widen422_VSX(Load8_VSX(src_u)),
                       Widen422_VSX(Load8_VSX(src_v)), 0, &c, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I422AlphaToARGBRow_VSX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u a = Load16_VSX(src_a);
    Yuv422ToARGB16_VSX(Load16_VSX(src_y), Widen422_VSX(Load8_VSX(src_u)),
                       Widen422_VSX(Load8_VSX(src_v)), &a, &c, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void NV12ToARGBRow_VSX(const uint8_t* src_y,
                       const uint8_t* src_uv,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u y = Load16_VSX(src_y);
    v8u uv = Load16_VSX(src_uv);
    Yuv422ToARGB16_VSX(y,
                       SHUFFLE_VSX(uv, uv, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10,
                                   12, 12, 14, 14),
                       SHUFFLE_VSX(uv, uv, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11,
                                   13, 13, 15, 15),
                       0, &c, dst_argb);
    src_y += 16;
    src_uv += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void NV21ToARGBRow_VSX(const uint8_t* src_y,
                       const uint8_t* src_vu,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u y = Load16_VSX(src_y);
    v8u vu = Load16_VSX(src_vu);
    Yuv422ToARGB16_VSX(y,
                       SHUFFLE_VSX(vu, vu, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11,
                                   13, 13, 15, 15),
                       SHUFFLE_VSX(vu, vu, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10,
                                   12, 12, 14, 14),
                       0, &c, dst_argb);
    src_y += 16;
    src_vu += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I444ToARGBRow_VSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    Yuv444ToARGB16_VSX(Load16_VSX(src_y), Load16_VSX(src_u), Load16_VSX(src_v),
                       0, &c, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I444AlphaToARGBRow_VSX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u a = Load16_VSX(src_a);
    Yuv444ToARGB16_VSX(Load16_VSX(src_y), Load16_VSX(src_u), Load16_VSX(src_v),
                       &a, &c, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I400ToARGBRow_VSX(const uint8_t* src_y,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  v16u six = vec_splats((unsigned short)6);
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u y = Load16_VSX(src_y);
    v16s lo = vec_sra(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), &c), six);
    v16s hi = vec_sra(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), &c), six);
    StoreGray_VSX(vec_packsu(lo, hi), dst_argb);
    src_y += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void J400ToARGBRow_VSX(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  do {
    v8u y0 = Load16_VSX(src_y);
    v8u y1 = Load16_VSX(src_y + 16);
    StoreGray_VSX(y0, dst_argb);
    StoreGray_VSX(y1, dst_argb + 64);
    src_y += 32;
    dst_argb += 128;
    width -= 32;
  } while (width > 0);
}

static const v8u kARGBAlphaMask_VSX = {0, 0, 0, 0xff, 0, 0, 0, 0xff,
                                       0, 0, 0, 0xff, 0, 0, 0, 0xff};

// (f * a + 255) >> 8 for B, G and R of 4 pixels, keeping A.
static __inline v8u Attenuate4_VSX(v8u argb) {
  v16u k255 = vec_splats((unsigned short)255);
  v8u a = SHUFFLE_VSX(argb, argb, 3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15,
                      15, 15, 15);
  v16u e = vec_add(vec_mule(argb, a), k255);
  v16u o = vec_add(vec_mulo(argb, a), k255);
  return vec_sel(HighBytes16_VSX(e, o), argb, kARGBAlphaMask_VSX);
}

void ARGBAttenuateRow_VSX(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          int width) {
  do {
    v8u p0 = Load16_VSX(src_argb);
    v8u p1 = Load16_VSX(src_argb + 16);
    v8u p2 = Load16_VSX(src_argb + 32);
    v8u p3 = Load16_VSX(src_argb + 48);
    Store16_VSX(Attenuate4_VSX(p0), dst_argb);
    Store16_VSX(Attenuate4_VSX(p1), dst_argb + 16);
    Store16_VSX(Attenuate4_VSX(p2), dst_argb + 32);
    Store16_VSX(Attenuate4_VSX(p3), dst_argb + 48);
    src_argb += 64;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

// fixed_invtbl8[a] & 0xffff in all four 16-bit lanes of a doubleword, so
// that each pixel needs one load.
#define INV4_VSX(a) (0x0001000100010001ULL * (0x10000 / (a)))
#define INV4X8_VSX(a)                                             \
  INV4_VSX(a), INV4_VSX(a + 1), INV4_VSX(a + 2), INV4_VSX(a + 3), \
      INV4_VSX(a + 4), INV4_VSX(a + 5), INV4_VSX(a + 6), INV4_VSX(a + 7)
static const unsigned long long kInvAlpha4_VSX[256] = {0,
                                                       0xffffffffffffffffULL,
                                                       INV4_VSX(2),
                                                       INV4_VSX(3),
                                                       INV4_VSX(4),
                                                       INV4_VSX(5),
                                                       INV4_VSX(6),
                                                       INV4_VSX(7),
                                                       INV4X8_VSX(8),
                                                       INV4X8_VSX(16),
                                                       INV4X8_VSX(24),
                                                       INV4X8_VSX(32),
                                                       INV4X8_VSX(40),
                                                       INV4X8_VSX(48),
                                                       INV4X8_VSX(56),
                                                       INV4X8_VSX(64),
                                                       INV4X8_VSX(72),
                                                       INV4X8_VSX(80),
                                                       INV4X8_VSX(88),
                                                       INV4X8_VSX(96),
                                                       INV4X8_VSX(104),
                                                       INV4X8_VSX(112),
                                                       INV4X8_VSX(120),
                                                       INV4X8_VSX(128),
                                                       INV4X8_VSX(136),
                                                       INV4X8_VSX(144),
                                                       INV4X8_VSX(152),
                                                       INV4X8_VSX(160),
                                                       INV4X8_VSX(168),
                                                       INV4X8_VSX(176),
                                                       INV4X8_VSX(184),
                                                       INV4X8_VSX(192),
                                                       INV4X8_VSX(200),
                                                       INV4X8_VSX(208),
                                                       INV4X8_VSX(216),
                                                       INV4X8_VSX(224),
                                                       INV4X8_VSX(232),
                                                       INV4X8_VSX(240),
                                                       INV4_VSX(248),
                                                       INV4_VSX(249),
                                                       INV4_VSX(250),
                                                       INV4_VSX(251),
                                                       INV4_VSX(252),
                                                       INV4_VSX(253),
                                                       INV4_VSX(254),
                                                       0x0100010001000100ULL};
#undef INV4X8_VSX
#undef INV4_VSX

// Unattenuates 4 pixels with their inverse alphas from kInvAlpha4_VSX.
static __inline v8u Unattenuate4_VSX(v8u argb, v64u ia01, v64u ia23) {
  v16u lo = MulHi16_VSX((v16u)vec_mergeh(argb, argb), (v16u)ia01);
  v16u hi = MulHi16_VSX((v16u)vec_mergel(argb, argb), (v16u)ia23);
  return vec_sel(vec_packsu(lo, hi), argb, kARGBAlphaMask_VSX);
}

void ARGBUnattenuateRow_VSX(const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            int width) {
  do {
    v8u p0 = Load16_VSX(src_argb);
    v8u p1 = Load16_VSX(src_argb + 16);
    v64u ia0 = {kInvAlpha4_VSX[src_argb[3]], kInvAlpha4_VSX[src_argb[7]]};
    v64u ia1 = {kInvAlpha4_VSX[src_argb[11]], kInvAlpha4_VSX[src_argb[15]]};
    v64u ia2 = {kInvAlpha4_VSX[src_argb[19]], kInvAlpha4_VSX[src_argb[23]]};
    v64u ia3 = {kInvAlpha4_VSX[src_argb[27]], kInvAlpha4_VSX[src_argb[31]]};
    Store16_VSX(Unattenuate4_VSX(p0, ia0, ia1), dst_argb);
    Store16_VSX(Unattenuate4_VSX(p1, ia2, ia3), dst_argb + 16);
    src_argb += 32;
    dst_argb += 32;
    width -= 8;
  } while (width > 0);
}

// (s0 * (256 - f) + s1 * f + 128) >> 8 for 16 bytes.
static __inline v8u Interpolate16_VSX(v8u s0, v8u s1, v8u f0, v8u f1) {
  v16u k128 = vec_splats((unsigned short)128);
  v16u e = vec_add(vec_add(vec_mule(s0, f0), vec_mule(s1, f1)), k128);
  v16u o = vec_add(vec_add(vec_mulo(s0, f0), vec_mulo(s1, f1)), k128);
  return HighBytes16_VSX(e, o);
}

void InterpolateRow_VSX(uint8_t* dst_ptr,
                        const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        int dst_width,
                        int source_y_fraction) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  if (source_y_fraction == 0) {
    do {
      v8u a = Load16_VSX(src_ptr);
      v8u b = Load16_VSX(src_ptr + 16);
      Store16_VSX(a, dst_ptr);
      Store16_VSX(b, dst_ptr + 16);
      src_ptr += 32;
      dst_ptr += 32;
      dst_width -= 32;
    } while (dst_width > 0);
    return;
  }
  if (source_y_fraction == 128) {
    do {
      v8u a0 = Load16_VSX(src_ptr);
      v8u a1 = Load16_VSX(src_ptr + 16);
      v8u b0 = Load16_VSX(src_ptr1);
      v8u b1 = Load16_VSX(src_ptr1 + 16);
      Store16_VSX(vec_avg(a0, b0), dst_ptr);
      Store16_VSX(vec_avg(a1, b1), dst_ptr + 16);
      src_ptr += 32;
      src_ptr1 += 32;
      dst_ptr += 32;
      dst_width -= 32;
    } while (dst_width > 0);
    return;
  }
  v8u f1 = vec_splats((unsigned char)source_y_fraction);
  v8u f0 = vec_splats((unsigned char)(256 - source_y_fraction));
  do {
    v8u a0 = Load16_VSX(src_ptr);
    v8u a1 = Load16_VSX(src_ptr + 16);
    v8u b0 = Load16_VSX(src_ptr1);
    v8u b1 = Load16_VSX(src_ptr1 + 16);
    Store16_VSX(Interpolate16_VSX(a0, b0, f0, f1), dst_ptr);
    Store16_VSX(Interpolate16_VSX(a1, b1, f0, f1), dst_ptr + 16);
    src_ptr += 32;
    src_ptr1 += 32;
    dst_ptr += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

// Picks bits 8 to 15 of each 32-bit lane of a, b, c and d, in that order.
static __inline v8u Bits8To15_VSX(v32u a, v32u b, v32u c, v32u d) {
  v8u lo = (v8u)vec_pack(a, b);
  v8u hi = (v8u)vec_pack(c, d);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return SHUFFLE_VSX(lo, hi, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
                     29, 31);
#else
  return SHUFFLE_VSX(lo, hi, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
                     28, 30);
#endif
}

struct ArgbCoefs_VSX {
  v8u ky;
  v32u ay;
  v8s ku;
  v8s kv;
  v32s au;
  v32s av;
};

// U and V are kAddUV - sum(k * p). They are computed as sum(k * ~p) plus
// kAddUV - 255 * sum(k), which needs no subtraction per pixel.
static __inline void LoadArgbCoefs_VSX(const struct ArgbConstants* c,
                                       struct ArgbCoefs_VSX* k) {
  v8u ff = vec_splats((unsigned char)0xff);
  v32s auv = vec_splats((int)c->kAddUV[0]);
  v32s zero = vec_splats(0);
  k->ky = Load16_VSX(c->kRGBToY);
  k->ay = vec_splats((unsigned int)c->kAddY[0]);
  k->ku = (v8s)Load16_VSX((const uint8_t*)c->kRGBToU);
  k->kv = (v8s)Load16_VSX((const uint8_t*)c->kRGBToV);
  k->au = vec_sub(auv, vec_msum(k->ku, ff, zero));
  k->av = vec_sub(auv, vec_msum(k->kv, ff, zero));
}

static __inline v8u ARGBToY16_VSX(const v8u* p, const struct ArgbCoefs_VSX* k) {
  return Bits8To15_VSX(
      vec_msum(p[0], k->ky, k->ay), vec_msum(p[1], k->ky, k->ay),
      vec_msum(p[2], k->ky, k->ay), vec_msum(p[3], k->ky, k->ay));
}

static __inline void ARGBToUV16_VSX(const v8u* p,
                                    const struct ArgbCoefs_VSX* k,
                                    uint8_t* dst_u,
                                    uint8_t* dst_v) {
  v8u n0 = vec_nor(p[0], p[0]);
  v8u n1 = vec_nor(p[1], p[1]);
  v8u n2 = vec_nor(p[2], p[2]);
  v8u n3 = vec_nor(p[3], p[3]);
  Store16_VSX(Bits8To15_VSX((v32u)vec_msum(k->ku, n0, k->au),
                            (v32u)vec_msum(k->ku, n1, k->au),
                            (v32u)vec_msum(k->ku, n2, k->au),
                            (v32u)vec_msum(k->ku, n3, k->au)),
              dst_u);
  Store16_VSX(Bits8To15_VSX((v32u)vec_msum(k->kv, n0, k->av),
                            (v32u)vec_msum(k->kv, n1, k->av),
                            (v32u)vec_msum(k->kv, n2, k->av),
                            (v32u)vec_msum(k->kv, n3, k->av)),
              dst_v);
}

// Averages the 2x2 boxes of 8 ARGB pixels in s0, s1 (row 0) and t0, t1
// (row 1), rounded, as 4 ARGB pixels. (sum * 64 + 128) >> 8 is
// (sum + 2) >> 2.
static __inline v8u BoxARGB4_VSX(v8u s0, v8u s1, v8u t0, v8u t1) {
  v8u k64 = vec_splats((unsigned char)64);
  v32u k128 = vec_splats(128u);
#define BOX_LO_VSX 0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23
#define BOX_HI_VSX 8, 12, 24, 28, 9, 13, 25, 29, 10, 14, 26, 30, 11, 15, 27, 31
  return Bits8To15_VSX(vec_msum(SHUFFLE_VSX(s0, t0, BOX_LO_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s0, t0, BOX_HI_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s1, t1, BOX_LO_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s1, t1, BOX_HI_VSX), k64, k128));
#undef BOX_LO_VSX
#undef BOX_HI_VSX
}

// Loads 16 RGB24 pixels as opaque ARGB, without reading past the 48 bytes.
static __inline void LoadRGB24_VSX(const uint8_t* src, v8u* p) {
  v8u ff = vec_splats((unsigned char)0xff);
#define RGB24_ARGB_VSX 0, 1, 2, 16, 3, 4, 5, 16, 6, 7, 8, 16, 9, 10, 11, 16
  p[0] = SHUFFLE_VSX(Load16_VSX(src), ff, RGB24_ARGB_VSX);
  p[1] = SHUFFLE_VSX(Load16_VSX(src + 12), ff, RGB24_ARGB_VSX);
  p[2] = SHUFFLE_VSX(Load16_VSX(src + 24), ff, RGB24_ARGB_VSX);
  p[3] = SHUFFLE_VSX(Load16_VSX(src + 32), ff, 4, 5, 6, 16, 7, 8, 9, 16, 10, 11,
                     12, 16, 13, 14, 15, 16);
#undef RGB24_ARGB_VSX
}

static __inline void LoadARGB_VSX(const uint8_t* src, v8u* p) {
  p[0] = Load16_VSX(src);
  p[1] = Load16_VSX(src + 16);
  p[2] = Load16_VSX(src + 32);
  p[3] = Load16_VSX(src + 48);
}

void ARGBToYMatrixRow_VSX(const uint8_t* src_argb,
                          uint8_t* dst_y,
                          int width,
                          const struct ArgbConstants* c) {
  struct ArgbCoefs_VSX k;
  LoadArgbCoefs_VSX(c, &k);
  do {
    v8u p[4];
    LoadARGB_VSX(src_argb, p);
    Store16_VSX(ARGBToY16_VSX(p, &k), dst_y);
    src_argb += 64;
    dst_y += 16;
    width -= 16;
  } while (width > 0);
}

void RGBToYMatrixRow_VSX(const uint8_t* src_rgb,
                         uint8_t* dst_y,
                         int width,
                         const struct ArgbConstants* c) {
  struct ArgbCoefs_VSX k;
  LoadArgbCoefs_VSX(c, &k);
  do {
    v8u p[4];
    LoadRGB24_VSX(src_rgb, p);
    Store16_VSX(ARGBToY16_VSX(p, &k), dst_y);
    src_rgb += 48;
    dst_y += 16;
    width -= 16;
  } while (width > 0);
}

void ARGBToUV444MatrixRow_VSX(const uint8_t* src_argb,
                              uint8_t* dst_u,
                              uint8_t* dst_v,
                              int width,
                              const struct ArgbConstants* c) {
  struct ArgbCoefs_VSX k;
  LoadArgbCoefs_VSX(c, &k);
  do {
    v8u p[4];
    LoadARGB_VSX(src_argb, p);
    ARGBToUV16_VSX(p, &k, dst_u, dst_v);
    src_argb += 64;
    dst_u += 16;
    dst_v += 16;
    width -= 16;
  } while (width > 0);
}

void RGBToUV444MatrixRow_VSX(const uint8_t* src_rgb,
                             uint8_t* dst_u,
                             uint8_t* dst_v,
                             int width,
                             const struct ArgbConstants* c) {
  struct ArgbCoefs_VSX k;
  LoadArgbCoefs_VSX(c, &k);
  do {
    v8u p[4];
    LoadRGB24_VSX(src_rgb, p);
    ARGBToUV16_VSX(p, &k, dst_u, dst_v);
    src_rgb += 48;
    dst_u += 16;
    dst_v += 16;
    width -= 16;
  } while (width > 0);
}

void ARGBToUVMatrixRow_VSX(const uint8_t* src_argb,
                           int src_stride_argb,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width,
                           const struct ArgbConstants* c) {
  const uint8_t* src_argb1 = src_argb + src_stride_argb;
  struct ArgbCoefs_VSX k;
  LoadArgbCoefs_VSX(c, &k);
  do {
    v8u s[8], t[8], p[4];
    LoadARGB_VSX(src_argb, s);
    LoadARGB_VSX(src_argb + 64, s + 4);
    LoadARGB_VSX(src_argb1, t);
    LoadARGB_VSX(src_argb1 + 64, t + 4);
    p[0] = BoxARGB4_VSX(s[0], s[1], t[0], t[1]);
    p[1] = BoxARGB4_VSX(s[2], s[3], t[2], t[3]);
    p[2] = BoxARGB4_VSX(s[4], s[5], t[4], t[5]);
    p[3] = BoxARGB4_VSX(s[6], s[7], t[6], t[7]);
    ARGBToUV16_VSX(p, &k, dst_u, dst_v);
    src_argb += 128;
    src_argb1 += 128;
    dst_u += 16;
    dst_v += 16;
    width -= 32;
  } while (width > 0);
}

void RGBToUVMatrixRow_VSX(const uint8_t* src_rgb,
                          int src_stride_rgb,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width,
                          const struct ArgbConstants* c) {
  const uint8_t* src_rgb1 = src_rgb + src_stride_rgb;
  struct ArgbCoefs_VSX k;
  LoadArgbCoefs_VSX(c, &k);
  do {
    v8u s[8], t[8], p[4];
    LoadRGB24_VSX(src_rgb, s);
    LoadRGB24_VSX(src_rgb + 48, s + 4);
    LoadRGB24_VSX(src_rgb1, t);
    LoadRGB24_VSX(src_rgb1 + 48, t + 4);
    p[0] = BoxARGB4_VSX(s[0], s[1], t[0], t[1]);
    p[1] = BoxARGB4_VSX(s[2], s[3], t[2], t[3]);
    p[2] = BoxARGB4_VSX(s[4], s[5], t[4], t[5]);
    p[3] = BoxARGB4_VSX(s[6], s[7], t[6], t[7]);
    ARGBToUV16_VSX(p, &k, dst_u, dst_v);
    src_rgb += 96;
    src_rgb1 += 96;
    dst_u += 16;
    dst_v += 16;
    width -= 32;
  } while (width > 0);
}

#define REV8_VSX 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define REV16_VSX 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1
#define REV32_VSX 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3

// Copies the 64 bytes before src to dst in reverse order of the elements
// that m reverses within a vector.
#define MIRROR64_VSX(src, dst, m)                    \
  do {                                               \
    v8u m0_ = Load16_VSX(src - 16);                  \
    v8u m1_ = Load16_VSX(src - 32);                  \
    v8u m2_ = Load16_VSX(src - 48);                  \
    v8u m3_ = Load16_VSX(src - 64);                  \
    Store16_VSX(SHUFFLE_VSX(m0_, m0_, m), dst);      \
    Store16_VSX(SHUFFLE_VSX(m1_, m1_, m), dst + 16); \
    Store16_VSX(SHUFFLE_VSX(m2_, m2_, m), dst + 32); \
    Store16_VSX(SHUFFLE_VSX(m3_, m3_, m), dst + 48); \
  } while (0)

void MirrorRow_VSX(const uint8_t* src, uint8_t* dst, int width) {
  src += width;
  do {
    MIRROR64_VSX(src, dst, REV8_VSX);
    src -= 64;
    dst += 64;
    width -= 64;
  } while (width > 0);
}

void MirrorUVRow_VSX(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  src_uv += width * 2;
  do {
    MIRROR64_VSX(src_uv, dst_uv, REV16_VSX);
    src_uv -= 64;
    dst_uv += 64;
    width -= 32;
  } while (width > 0);
}

void MirrorSplitUVRow_VSX(const uint8_t* src_uv,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  src_uv += width * 2;
  do {
    v8u a = Load16_VSX(src_uv - 16);
    v8u b = Load16_VSX(src_uv - 32);
    v8u c = Load16_VSX(src_uv - 48);
    v8u d = Load16_VSX(src_uv - 64);
    Store16_VSX(SHUFFLE_VSX(a, b, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28, 26, 24, 22,
                            20, 18, 16),
                dst_u);
    Store16_VSX(SHUFFLE_VSX(c, d, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28, 26, 24, 22,
                            20, 18, 16),
                dst_u + 16);
    Store16_VSX(SHUFFLE_VSX(a, b, 15, 13, 11, 9, 7, 5, 3, 1, 31, 29, 27, 25, 23,
                            21, 19, 17),
                dst_v);
    Store16_VSX(SHUFFLE_VSX(c, d, 15, 13, 11, 9, 7, 5, 3, 1, 31, 29, 27, 25, 23,
                            21, 19, 17),
                dst_v + 16);
    src_uv -= 64;
    dst_u += 32;
    dst_v += 32;
    width -= 32;
  } while (width > 0);
}

void MirrorRow_16_VSX(const uint16_t* src, uint16_t* dst, int width) {
  const uint8_t* s = (const uint8_t*)(src + width);
  uint8_t* d = (uint8_t*)dst;
  do {
    MIRROR64_VSX(s, d, REV16_VSX);
    s -= 64;
    d += 64;
    width -= 32;
  } while (width > 0);
}

void ARGBMirrorRow_VSX(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  src_argb += width * 4;
  do {
    MIRROR64_VSX(src_argb, dst_argb, REV32_VSX);
    src_argb -= 64;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

// The shuffler holds 16 byte indices for 4 pixels, in memory order.
void ARGBShuffleRow_VSX(const uint8_t* src_argb,
                        uint8_t* dst_argb,
                        const uint8_t* shuffler,
                        int width) {
  const v8u mask = Load16_VSX(shuffler);
  do {
    v8u a = Load16_VSX(src_argb);
    v8u b = Load16_VSX(src_argb + 16);
    Store16_VSX(vec_perm(a, a, mask), dst_argb);
    Store16_VSX(vec_perm(b, b, mask), dst_argb + 16);
    src_argb += 32;
    dst_argb += 32;
    width -= 8;
  } while (width > 0);
}

// clamp255((v * scale) >> 16) for 16 lanes.
static __inline v8u Convert16To8_VSX(v16u a, v16u b, v16u scale) {
  return vec_packsu(MulHi16_VSX(a, scale), MulHi16_VSX(b, scale));
}

void Convert16To8Row_VSX(const uint16_t* src_y,
                         uint8_t* dst_y,
                         int scale,
                         int width) {
  const v16u s = vec_splats((unsigned short)scale);
  do {
    v16u a = vec_xl(0, src_y);
    v16u b = vec_xl(16, src_y);
    Store16_VSX(Convert16To8_VSX(a, b, s), dst_y);
    src_y += 16;
    dst_y += 16;
    width -= 16;
  } while (width > 0);
}

void HalfRow_16To8_VSX(const uint16_t* src_uv,
                       ptrdiff_t src_uv_stride,
                       uint8_t* dst_uv,
                       int scale,
                       int width) {
  const uint16_t* src_uv1 = src_uv + src_uv_stride;
  const v16u s = vec_splats((unsigned short)scale);
  do {
    v16u a = vec_avg(vec_xl(0, src_uv), vec_xl(0, src_uv1));
    v16u b = vec_avg(vec_xl(16, src_uv), vec_xl(16, src_uv1));
    Store16_VSX(Convert16To8_VSX(a, b, s), dst_uv);
    src_uv += 16;
    src_uv1 += 16;
    dst_uv += 16;
    width -= 16;
  } while (width > 0);
}

// Duplicating each byte into both halves of a lane gives v * 0x0101.
void Convert8To16Row_VSX(const uint8_t* src_y,
                         uint16_t* dst_y,
                         int bits,
                         int width) {
  const v16u shift = vec_splats((unsigned short)(16 - bits));
  do {
    v8u a = Load16_VSX(src_y);
    vec_xst(vec_sr((v16u)SHUFFLE_VSX(a, a, DUP2_LO_VSX), shift), 0, dst_y);
    vec_xst(vec_sr((v16u)SHUFFLE_VSX(a, a, DUP2_HI_VSX), shift), 16, dst_y);
    src_y += 16;
    dst_y += 16;
    width -= 16;
  } while (width > 0);
}

void MergeUVRow_16_VSX(const uint16_t* src_u,
                       const uint16_t* src_v,
                       uint16_t* dst_uv,
                       int depth,
                       int width) {
  const v16u shift = vec_splats((unsigned short)(16 - depth));
  do {
    v16u u = vec_sl(vec_xl(0, src_u), shift);
    v16u v = vec_sl(vec_xl(0, src_v), shift);
    vec_xst(vec_mergeh(u, v), 0, dst_uv);
    vec_xst(vec_mergel(u, v), 16, dst_uv);
    src_u += 8;
    src_v += 8;
    dst_uv += 16;
    width -= 8;
  } while (width > 0);
}

#define EVEN16_VSX 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29
#define ODD16_VSX 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31

void SplitUVRow_16_VSX(const uint16_t* src_uv,
                       uint16_t* dst_u,
                       uint16_t* dst_v,
                       int depth,
                       int width) {
  const v16u shift = vec_splats((unsigned short)(16 - depth));
  do {
    v8u a0 = (v8u)vec_xl(0, src_uv);
    v8u a1 = (v8u)vec_xl(16, src_uv);
    v8u a2 = (v8u)vec_xl(32, src_uv);
    v8u a3 = (v8u)vec_xl(48, src_uv);
    vec_xst(vec_sr((v16u)SHUFFLE_VSX(a0, a1, EVEN16_VSX), shift), 0, dst_u);
    vec_xst(vec_sr((v16u)SHUFFLE_VSX(a2, a3, EVEN16_VSX), shift), 16, dst_u);
    vec_xst(vec_sr((v16u)SHUFFLE_VSX(a0, a1, ODD16_VSX), shift), 0, dst_v);
    vec_xst(vec_sr((v16u)SHUFFLE_VSX(a2, a3, ODD16_VSX), shift), 16, dst_v);
    src_uv += 32;
    dst_u += 16;
    dst_v += 16;
    width -= 16;
  } while (width > 0);
}

void I422ToRGBARow_VSX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_rgba,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    v8u y = Load16_VSX(src_y);
    struct UVTerms_VSX t;
    UVTerms_VSX(Widen422_VSX(Load8_VSX(src_u)), Widen422_VSX(Load8_VSX(src_v)),
                &c, &t);
    YuvToPixels8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_LO_VSX), &c),
                     vec_mergeh(t.b, t.b), vec_mergeh(t.g, t.g),
                     vec_mergeh(t.r, t.r), 0, 1, dst_rgba);
    YuvToPixels8_VSX(Y16_VSX(SHUFFLE_VSX(y, y, DUP2_HI_VSX), &c),
                     vec_mergel(t.b, t.b), vec_mergel(t.g, t.g),
                     vec_mergel(t.r, t.r), 0, 1, dst_rgba + 32);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    dst_rgba += 64;
    width -= 16;
  } while (width > 0);
}

// The 10 bit rows widen Y to 16 bits as (y << 6) | (y >> 4), and take U
// and V as clamp255(u >> 2), like the C rows. The P rows hold 16 bit
// samples and take the high byte of U and V.

static __inline v16s Y10_VSX(v16u y, const struct YuvCoefs_VSX* c) {
  v16u y16 = vec_or(vec_sl(y, vec_splats((unsigned short)6)),
                    vec_sr(y, vec_splats((unsigned short)4)));
  return vec_add((v16s)MulHi16_VSX(y16, c->yg), c->yb);
}

static __inline v16s YP16_VSX(v16u y, const struct YuvCoefs_VSX* c) {
  return vec_add((v16s)MulHi16_VSX(y, c->yg), c->yb);
}

// clamp255(u >> shift) - 128.
static __inline v16s UVShift_VSX(v16u u, int shift) {
  v16u s = vec_sr(u, vec_splats((unsigned short)shift));
  return vec_sub((v16s)vec_min(s, vec_splats((unsigned short)255)),
                 vec_splats((short)128));
}

// clamp255(a >> 2) for 16 alpha samples.
static __inline v8u Alpha10_VSX(const uint16_t* src_a) {
  v16u k2 = vec_splats((unsigned short)2);
  return vec_packsu(vec_sr(vec_xl(0, src_a), k2),
                    vec_sr(vec_xl(16, src_a), k2));
}

// Converts 16 pixels from Y in y0, y1 and 8 U and V terms, one per pair of
// pixels.
static __inline void Yuv422x16_VSX(v16s y0,
                                   v16s y1,
                                   const struct UVTerms_VSX* t,
                                   const v8u* a,
                                   uint8_t* dst_argb) {
  v8u a1 =
      a ? (v8u)vec_mergel((v64u)*a, (v64u)*a) : vec_splats((unsigned char)0);
  YuvToARGB8_VSX(y0, vec_mergeh(t->b, t->b), vec_mergeh(t->g, t->g),
                 vec_mergeh(t->r, t->r), a, dst_argb);
  YuvToARGB8_VSX(y1, vec_mergel(t->b, t->b), vec_mergel(t->g, t->g),
                 vec_mergel(t->r, t->r), a ? &a1 : 0, dst_argb + 32);
}

// Converts 16 pixels from Y in y0, y1 and U and V terms t0, t1.
static __inline void Yuv444x16_VSX(v16s y0,
                                   v16s y1,
                                   const struct UVTerms_VSX* t0,
                                   const struct UVTerms_VSX* t1,
                                   const v8u* a,
                                   uint8_t* dst_argb) {
  v8u a1 =
      a ? (v8u)vec_mergel((v64u)*a, (v64u)*a) : vec_splats((unsigned char)0);
  YuvToARGB8_VSX(y0, t0->b, t0->g, t0->r, a, dst_argb);
  YuvToARGB8_VSX(y1, t1->b, t1->g, t1->r, a ? &a1 : 0, dst_argb + 32);
}

void I210ToARGBRow_VSX(const uint16_t* src_y,
                       const uint16_t* src_u,
                       const uint16_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    struct UVTerms_VSX t;
    UVTermsS_VSX(UVShift_VSX(vec_xl(0, src_u), 2),
                 UVShift_VSX(vec_xl(0, src_v), 2), &c, &t);
    Yuv422x16_VSX(Y10_VSX(vec_xl(0, src_y), &c), Y10_VSX(vec_xl(16, src_y), &c),
                  &t, 0, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I210AlphaToARGBRow_VSX(const uint16_t* src_y,
                            const uint16_t* src_u,
                            const uint16_t* src_v,
                            const uint16_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    struct UVTerms_VSX t;
    v8u a = Alpha10_VSX(src_a);
    UVTermsS_VSX(UVShift_VSX(vec_xl(0, src_u), 2),
                 UVShift_VSX(vec_xl(0, src_v), 2), &c, &t);
    Yuv422x16_VSX(Y10_VSX(vec_xl(0, src_y), &c), Y10_VSX(vec_xl(16, src_y), &c),
                  &t, &a, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I410ToARGBRow_VSX(const uint16_t* src_y,
                       const uint16_t* src_u,
                       const uint16_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    struct UVTerms_VSX t0, t1;
    UVTermsS_VSX(UVShift_VSX(vec_xl(0, src_u), 2),
                 UVShift_VSX(vec_xl(0, src_v), 2), &c, &t0);
    UVTermsS_VSX(UVShift_VSX(vec_xl(16, src_u), 2),
                 UVShift_VSX(vec_xl(16, src_v), 2), &c, &t1);
    Yuv444x16_VSX(Y10_VSX(vec_xl(0, src_y), &c), Y10_VSX(vec_xl(16, src_y), &c),
                  &t0, &t1, 0, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I410AlphaToARGBRow_VSX(const uint16_t* src_y,
                            const uint16_t* src_u,
                            const uint16_t* src_v,
                            const uint16_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    struct UVTerms_VSX t0, t1;
    v8u a = Alpha10_VSX(src_a);
    UVTermsS_VSX(UVShift_VSX(vec_xl(0, src_u), 2),
                 UVShift_VSX(vec_xl(0, src_v), 2), &c, &t0);
    UVTermsS_VSX(UVShift_VSX(vec_xl(16, src_u), 2),
                 UVShift_VSX(vec_xl(16, src_v), 2), &c, &t1);
    Yuv444x16_VSX(Y10_VSX(vec_xl(0, src_y), &c), Y10_VSX(vec_xl(16, src_y), &c),
                  &t0, &t1, &a, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

// U and V of 8 interleaved 16 bit UV pairs.
static __inline void SplitUV16_VSX(const uint16_t* src_uv,
                                   const struct YuvCoefs_VSX* c,
                                   struct UVTerms_VSX* t) {
  v8u a = (v8u)vec_xl(0, src_uv);
  v8u b = (v8u)vec_xl(16, src_uv);
  UVTermsS_VSX(UVShift_VSX((v16u)SHUFFLE_VSX(a, b, EVEN16_VSX), 8),
               UVShift_VSX((v16u)SHUFFLE_VSX(a, b, ODD16_VSX), 8), c, t);
}

void P210ToARGBRow_VSX(const uint16_t* src_y,
                       const uint16_t* src_uv,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    struct UVTerms_VSX t;
    SplitUV16_VSX(src_uv, &c, &t);
    Yuv422x16_VSX(YP16_VSX(vec_xl(0, src_y), &c),
                  YP16_VSX(vec_xl(16, src_y), &c), &t, 0, dst_argb);
    src_y += 16;
    src_uv += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void P410ToARGBRow_VSX(const uint16_t* src_y,
                       const uint16_t* src_uv,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VSX c;
  LoadYuvCoefs_VSX(yuvconstants, &c);
  do {
    struct UVTerms_VSX t0, t1;
    SplitUV16_VSX(src_uv, &c, &t0);
    SplitUV16_VSX(src_uv + 16, &c, &t1);
    Yuv444x16_VSX(YP16_VSX(vec_xl(0, src_y), &c),
                  YP16_VSX(vec_xl(16, src_y), &c), &t0, &t1, 0, dst_argb);
    src_y += 16;
    src_uv += 32;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

#define ODD_BYTES_VSX 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
#define EVEN_BYTES_VSX 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
#define REV32B_VSX 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12
#define REV16B_VSX 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
// The same byte order for each of 4 pixels.
#define PIXEL4_VSX(a, b, c, d)                                                \
  a, b, c, d, a + 4, b + 4, c + 4, d + 4, a + 8, b + 8, c + 8, d + 8, a + 12, \
      b + 12, c + 12, d + 12
// RAW to ARGB and to RGBA for 4 pixels starting at byte o, 16 being 0xff.
#define SHUF_ARGB_RAW_VSX(o)                                                 \
  o + 2, o + 1, o + 0, 16, o + 5, o + 4, o + 3, 16, o + 8, o + 7, o + 6, 16, \
      o + 11, o + 10, o + 9, 16
#define SHUF_RGBA_RAW_VSX(o)                                                 \
  16, o + 2, o + 1, o + 0, 16, o + 5, o + 4, o + 3, 16, o + 8, o + 7, o + 6, \
      16, o + 11, o + 10, o + 9

// The packed 16 and 32 bit formats are built in lanes holding the
// little-endian value of their bytes, the way the C rows read them.
static __inline v32u LE32_VSX(v8u v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v32u)v;
#else
  return (v32u)SHUFFLE_VSX(v, v, REV32B_VSX);
#endif
}

static __inline v8u FromLE32_VSX(v32u v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v8u)v;
#else
  return SHUFFLE_VSX((v8u)v, (v8u)v, REV32B_VSX);
#endif
}

static __inline v16u LE16_VSX(v8u v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v16u)v;
#else
  return (v16u)SHUFFLE_VSX(v, v, REV16B_VSX);
#endif
}

static __inline v8u FromLE16_VSX(v16u v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v8u)v;
#else
  return SHUFFLE_VSX((v8u)v, (v8u)v, REV16B_VSX);
#endif
}

// The high byte of each 16-bit lane of a and then b.
static __inline v8u HighBytesSeq_VSX(v16u a, v16u b) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return SHUFFLE_VSX((v8u)a, (v8u)b, ODD_BYTES_VSX);
#else
  return SHUFFLE_VSX((v8u)a, (v8u)b, EVEN_BYTES_VSX);
#endif
}

void ARGBToRGB24Row_VSX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  do {
    v8u a = Load16_VSX(src_argb);
    v8u b = Load16_VSX(src_argb + 16);
    v8u c = Load16_VSX(src_argb + 32);
    v8u d = Load16_VSX(src_argb + 48);
    Store16_VSX(SHUFFLE_VSX(a, b, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16,
                            17, 18, 20),
                dst_rgb);
    Store16_VSX(SHUFFLE_VSX(b, c, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20,
                            21, 22, 24, 25),
                dst_rgb + 16);
    Store16_VSX(SHUFFLE_VSX(c, d, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24,
                            25, 26, 28, 29, 30),
                dst_rgb + 32);
    src_argb += 64;
    dst_rgb += 48;
    width -= 16;
  } while (width > 0);
}

void ARGBToRAWRow_VSX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  do {
    v8u a = Load16_VSX(src_argb);
    v8u b = Load16_VSX(src_argb + 16);
    v8u c = Load16_VSX(src_argb + 32);
    v8u d = Load16_VSX(src_argb + 48);
    Store16_VSX(SHUFFLE_VSX(a, b, 2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 18,
                            17, 16, 22),
                dst_rgb);
    Store16_VSX(SHUFFLE_VSX(b, c, 5, 4, 10, 9, 8, 14, 13, 12, 18, 17, 16, 22,
                            21, 20, 26, 25),
                dst_rgb + 16);
    Store16_VSX(SHUFFLE_VSX(c, d, 8, 14, 13, 12, 18, 17, 16, 22, 21, 20, 26, 25,
                            24, 30, 29, 28),
                dst_rgb + 32);
    src_argb += 64;
    dst_rgb += 48;
    width -= 16;
  } while (width > 0);
}

void RAWToRGB24Row_VSX(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  do {
    v8u a = Load16_VSX(src_raw);
    v8u b = Load16_VSX(src_raw + 16);
    v8u c = Load16_VSX(src_raw + 15);
    v8u d = Load16_VSX(src_raw + 31);
    v8u e = Load16_VSX(src_raw + 30);
    v8u f = Load16_VSX(src_raw + 32);
    Store16_VSX(
        SHUFFLE_VSX(a, b, 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17),
        dst_rgb24);
    Store16_VSX(SHUFFLE_VSX(c, d, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12,
                            17, 16),
                dst_rgb24 + 16);
    Store16_VSX(SHUFFLE_VSX(e, f, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12,
                            31, 30, 15),
                dst_rgb24 + 32);
    src_raw += 48;
    dst_rgb24 += 48;
    width -= 16;
  } while (width > 0);
}

void RGB24MirrorRow_VSX(const uint8_t* src_rgb24,
                        uint8_t* dst_rgb24,
                        int width) {
  src_rgb24 += width * 3;
  do {
    src_rgb24 -= 48;
    v8u a = Load16_VSX(src_rgb24 + 30);
    v8u b = Load16_VSX(src_rgb24 + 32);
    v8u c = Load16_VSX(src_rgb24 + 15);
    v8u d = Load16_VSX(src_rgb24 + 31);
    v8u e = Load16_VSX(src_rgb24);
    v8u f = Load16_VSX(src_rgb24 + 16);
    Store16_VSX(SHUFFLE_VSX(a, b, 15, 30, 31, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3,
                            4, 5, 0),
                dst_rgb24);
    Store16_VSX(SHUFFLE_VSX(c, d, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4,
                            5, 0, 1),
                dst_rgb24 + 16);
    Store16_VSX(
        SHUFFLE_VSX(e, f, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2),
        dst_rgb24 + 32);
    dst_rgb24 += 48;
    width -= 16;
  } while (width > 0);
}

// Loads 16 RGB24 or RAW pixels and shuffles each 4 with m, where index 16 is
// 0xff.
#define LOAD_RGB24_X4_VSX(src, p, m0, m3)              \
  do {                                                 \
    v8u ff_ = vec_splats((unsigned char)0xff);         \
    p[0] = SHUFFLE_VSX(Load16_VSX(src), ff_, m0);      \
    p[1] = SHUFFLE_VSX(Load16_VSX(src + 12), ff_, m0); \
    p[2] = SHUFFLE_VSX(Load16_VSX(src + 24), ff_, m0); \
    p[3] = SHUFFLE_VSX(Load16_VSX(src + 32), ff_, m3); \
  } while (0)

static __inline void StoreARGB16_VSX(const v8u* p, uint8_t* dst) {
  Store16_VSX(p[0], dst);
  Store16_VSX(p[1], dst + 16);
  Store16_VSX(p[2], dst + 32);
  Store16_VSX(p[3], dst + 48);
}

void RGB24ToARGBRow_VSX(const uint8_t* src_rgb24,
                        uint8_t* dst_argb,
                        int width) {
  do {
    v8u p[4];
    LoadRGB24_VSX(src_rgb24, p);
    StoreARGB16_VSX(p, dst_argb);
    src_rgb24 += 48;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void RAWToARGBRow_VSX(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  do {
    v8u p[4];
    LOAD_RGB24_X4_VSX(src_raw, p, SHUF_ARGB_RAW_VSX(0), SHUF_ARGB_RAW_VSX(4));
    StoreARGB16_VSX(p, dst_argb);
    src_raw += 48;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void RAWToRGBARow_VSX(const uint8_t* src_raw, uint8_t* dst_rgba, int width) {
  do {
    v8u p[4];
    LOAD_RGB24_X4_VSX(src_raw, p, SHUF_RGBA_RAW_VSX(0), SHUF_RGBA_RAW_VSX(4));
    StoreARGB16_VSX(p, dst_rgba);
    src_raw += 48;
    dst_rgba += 64;
    width -= 16;
  } while (width > 0);
}

// Byte swizzles applied to each pixel of ARGB.
#define SWIZZLE_ROW_VSX(NAME, M)                           \
  void NAME(const uint8_t* src, uint8_t* dst, int width) { \
    do {                                                   \
      v8u a = Load16_VSX(src);                             \
      v8u b = Load16_VSX(src + 16);                        \
      v8u c = Load16_VSX(src + 32);                        \
      v8u d = Load16_VSX(src + 48);                        \
      Store16_VSX(SHUFFLE_VSX(a, a, M), dst);              \
      Store16_VSX(SHUFFLE_VSX(b, b, M), dst + 16);         \
      Store16_VSX(SHUFFLE_VSX(c, c, M), dst + 32);         \
      Store16_VSX(SHUFFLE_VSX(d, d, M), dst + 48);         \
      src += 64;                                           \
      dst += 64;                                           \
      width -= 16;                                         \
    } while (width > 0);                                   \
  }

SWIZZLE_ROW_VSX(ARGBToABGRRow_VSX, PIXEL4_VSX(2, 1, 0, 3))
SWIZZLE_ROW_VSX(ARGBToBGRARow_VSX, PIXEL4_VSX(3, 2, 1, 0))
SWIZZLE_ROW_VSX(ARGBToRGBARow_VSX, PIXEL4_VSX(3, 0, 1, 2))
SWIZZLE_ROW_VSX(RGBAToARGBRow_VSX, PIXEL4_VSX(1, 2, 3, 0))

// Packs each pixel into 16 bits: the bytes are shifted right by sh, then
// summed with weights w0, and those summed with weights w1 are added shifted
// left by s1. vmsumubm works on the bytes of each pixel in memory order, so
// this needs no byte swap on big-endian.
struct Pack16_VSX {
  v8u sh, w0, w1;
  v32u s1;
};

static __inline v32u Pack16Px4_VSX(v8u p, const struct Pack16_VSX* k) {
  v8u q = vec_sr(p, k->sh);
  v32u zero = vec_splats(0u);
  return vec_msum(q, k->w0, vec_sl(vec_msum(q, k->w1, zero), k->s1));
}

// Packs 16 ARGB pixels, after adding dv to each byte with saturation when
// dither is set. le stores the 16 bit pixels as little-endian, otherwise in
// native byte order, as the C rows do.
static __inline void ARGBTo16Bit_VSX(const uint8_t* src_argb,
                                     uint8_t* dst,
                                     int width,
                                     const struct Pack16_VSX* k,
                                     int le,
                                     int dither,
                                     v8u dv) {
  do {
    v8u p[4];
    int i;
    for (i = 0; i < 4; ++i) {
      p[i] = Load16_VSX(src_argb + i * 16);
      if (dither) {
        p[i] = vec_adds(p[i], dv);
      }
    }
    v16u lo = vec_pack(Pack16Px4_VSX(p[0], k), Pack16Px4_VSX(p[1], k));
    v16u hi = vec_pack(Pack16Px4_VSX(p[2], k), Pack16Px4_VSX(p[3], k));
    Store16_VSX(le ? FromLE16_VSX(lo) : (v8u)lo, dst);
    Store16_VSX(le ? FromLE16_VSX(hi) : (v8u)hi, dst + 16);
    src_argb += 64;
    dst += 32;
    width -= 16;
  } while (width > 0);
}

#define PACK16_K_VSX(b, g, r, a, w0g, w1a, s1)                          \
  {                                                                     \
    {PIXEL4_VALUES_VSX(b, g, r, a)}, {PIXEL4_VALUES_VSX(1, w0g, 0, 0)}, \
        {PIXEL4_VALUES_VSX(0, 0, 1, w1a)}, {                            \
      s1, s1, s1, s1                                                    \
    }                                                                   \
  }
#define PIXEL4_VALUES_VSX(a, b, c, d) \
  a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d

static const struct Pack16_VSX kRGB565_VSX =
    PACK16_K_VSX(3, 2, 3, 8, 32, 0, 11);
static const struct Pack16_VSX kARGB1555_VSX =
    PACK16_K_VSX(3, 3, 3, 7, 32, 32, 10);
static const struct Pack16_VSX kARGB4444_VSX =
    PACK16_K_VSX(4, 4, 4, 4, 16, 16, 8);

void ARGBToRGB565Row_VSX(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  ARGBTo16Bit_VSX(src_argb, dst_rgb, width, &kRGB565_VSX, 1, 0,
                  vec_splats((unsigned char)0));
}

void ARGBToARGB1555Row_VSX(const uint8_t* src_argb,
                           uint8_t* dst_rgb,
                           int width) {
  ARGBTo16Bit_VSX(src_argb, dst_rgb, width, &kARGB1555_VSX, 0, 0,
                  vec_splats((unsigned char)0));
}

void ARGBToARGB4444Row_VSX(const uint8_t* src_argb,
                           uint8_t* dst_rgb,
                           int width) {
  ARGBTo16Bit_VSX(src_argb, dst_rgb, width, &kARGB4444_VSX, 0, 0,
                  vec_splats((unsigned char)0));
}

// Each pixel in turn adds one byte of dither4 to B, G and R.
void ARGBToRGB565DitherRow_VSX(const uint8_t* src_argb,
                               uint8_t* dst_rgb,
                               uint32_t dither4,
                               int width) {
  uint8_t d[4];
  memcpy(d, &dither4, 4);
  v8u dv = {d[0], d[0], d[0], 0, d[1], d[1], d[1], 0,
            d[2], d[2], d[2], 0, d[3], d[3], d[3], 0};
  ARGBTo16Bit_VSX(src_argb, dst_rgb, width, &kRGB565_VSX, 0, 1, dv);
}

// (c << 2) | (c >> 6) for the channel in the low 8 bits of each lane.
static __inline v32u Expand10_VSX(v32u c) {
  return vec_or(vec_sl(c, vec_splats(2u)), vec_sr(c, vec_splats(6u)));
}

// bs and rs are the shifts that bring B and R to the low byte.
static __inline v32u ToAR30_VSX(v8u v, unsigned bs, unsigned rs) {
  v32u p = LE32_VSX(v);
  v32u ff = vec_splats(0xffu);
  v32u b = Expand10_VSX(vec_and(vec_sr(p, vec_splats(bs)), ff));
  v32u g = Expand10_VSX(vec_and(vec_sr(p, vec_splats(8u)), ff));
  v32u r = Expand10_VSX(vec_and(vec_sr(p, vec_splats(rs)), ff));
  v32u a = vec_sl(vec_sr(p, vec_splats(30u)), vec_splats(30u));
  return vec_or(vec_or(b, vec_sl(g, vec_splats(10u))),
                vec_or(vec_sl(r, vec_splats(20u)), a));
}

#define TOAR30_ROW_VSX(NAME, bs, rs)                            \
  void NAME(const uint8_t* src, uint8_t* dst_ar30, int width) { \
    unsigned int* d = (unsigned int*)dst_ar30;                  \
    do {                                                        \
      v8u a = Load16_VSX(src);                                  \
      v8u b = Load16_VSX(src + 16);                             \
      v8u c = Load16_VSX(src + 32);                             \
      v8u e = Load16_VSX(src + 48);                             \
      vec_xst(ToAR30_VSX(a, bs, rs), 0, d);                     \
      vec_xst(ToAR30_VSX(b, bs, rs), 16, d);                    \
      vec_xst(ToAR30_VSX(c, bs, rs), 32, d);                    \
      vec_xst(ToAR30_VSX(e, bs, rs), 48, d);                    \
      src += 64;                                                \
      d += 16;                                                  \
      width -= 16;                                              \
    } while (width > 0);                                        \
  }

TOAR30_ROW_VSX(ARGBToAR30Row_VSX, 0u, 16u)
TOAR30_ROW_VSX(ABGRToAR30Row_VSX, 16u, 0u)

// AR30 to 8 bit channels with B (or R) at shift lo, G and the other at hi,
// and alpha replicated from 2 bits.
static __inline v32u FromAR30_VSX(v32u x, int abgr) {
  v32u ff = vec_splats(0xffu);
  v32u b = vec_and(vec_sr(x, vec_splats(2u)), ff);
  v32u g = vec_and(vec_sr(x, vec_splats(4u)), vec_splats(0xff00u));
  v32u r = vec_and(vec_sr(x, vec_splats(22u)), ff);
  v32u a = vec_mul(vec_sr(x, vec_splats(30u)), vec_splats(0x55000000u));
  return abgr ? vec_or(vec_or(r, g), vec_or(vec_sl(b, vec_splats(16u)), a))
              : vec_or(vec_or(b, g), vec_or(vec_sl(r, vec_splats(16u)), a));
}

static __inline v32u AR30ToAB30_VSX(v32u x) {
  v32u k3ff = vec_splats(0x3ffu);
  v32u r = vec_and(vec_sr(x, vec_splats(20u)), k3ff);
  v32u b = vec_sl(vec_and(x, k3ff), vec_splats(20u));
  return vec_or(vec_or(r, b), vec_and(x, vec_splats(0xc00ffc00u)));
}

#define AR30_ROW_VSX(NAME, OP)                                  \
  void NAME(const uint8_t* src_ar30, uint8_t* dst, int width) { \
    const unsigned int* s = (const unsigned int*)src_ar30;      \
    unsigned int* d = (unsigned int*)dst;                       \
    do {                                                        \
      v32u a = vec_xl(0, s);                                    \
      v32u b = vec_xl(16, s);                                   \
      v32u c = vec_xl(32, s);                                   \
      v32u e = vec_xl(48, s);                                   \
      vec_xst(OP(a), 0, d);                                     \
      vec_xst(OP(b), 16, d);                                    \
      vec_xst(OP(c), 32, d);                                    \
      vec_xst(OP(e), 48, d);                                    \
      s += 16;                                                  \
      d += 16;                                                  \
      width -= 16;                                              \
    } while (width > 0);                                        \
  }

#define TOARGB_OP_VSX(x) FromAR30_VSX(x, 0)
#define TOABGR_OP_VSX(x) FromAR30_VSX(x, 1)
AR30_ROW_VSX(AR30ToARGBRow_VSX, TOARGB_OP_VSX)
AR30_ROW_VSX(AR30ToABGRRow_VSX, TOABGR_OP_VSX)
AR30_ROW_VSX(AR30ToAB30Row_VSX, AR30ToAB30_VSX)
#undef TOARGB_OP_VSX
#undef TOABGR_OP_VSX

// ARGB to 16 bit channels, each byte duplicated (v * 0x0101).
#define TO64_ROW_VSX(NAME, LO, HI)                               \
  void NAME(const uint8_t* src_argb, uint16_t* dst, int width) { \
    do {                                                         \
      v8u a = Load16_VSX(src_argb);                              \
      v8u b = Load16_VSX(src_argb + 16);                         \
      vec_xst(SHUFFLE_VSX(a, a, LO), 0, (uint8_t*)dst);          \
      vec_xst(SHUFFLE_VSX(a, a, HI), 16, (uint8_t*)dst);         \
      vec_xst(SHUFFLE_VSX(b, b, LO), 32, (uint8_t*)dst);         \
      vec_xst(SHUFFLE_VSX(b, b, HI), 48, (uint8_t*)dst);         \
      src_argb += 32;                                            \
      dst += 32;                                                 \
      width -= 8;                                                \
    } while (width > 0);                                         \
  }

TO64_ROW_VSX(ARGBToAR64Row_VSX, DUP2_LO_VSX, DUP2_HI_VSX)
#define AB64_LO_VSX 2, 2, 1, 1, 0, 0, 3, 3, 6, 6, 5, 5, 4, 4, 7, 7
#define AB64_HI_VSX 10, 10, 9, 9, 8, 8, 11, 11, 14, 14, 13, 13, 12, 12, 15, 15
TO64_ROW_VSX(ARGBToAB64Row_VSX, AB64_LO_VSX, AB64_HI_VSX)

void AR64ToARGBRow_VSX(const uint16_t* src_ar64, uint8_t* dst_argb, int width) {
  do {
    v16u a = vec_xl(0, src_ar64);
    v16u b = vec_xl(16, src_ar64);
    v16u c = vec_xl(32, src_ar64);
    v16u d = vec_xl(48, src_ar64);
    Store16_VSX(HighBytesSeq_VSX(a, b), dst_argb);
    Store16_VSX(HighBytesSeq_VSX(c, d), dst_argb + 16);
    src_ar64 += 32;
    dst_argb += 32;
    width -= 8;
  } while (width > 0);
}

#define SWAP_RB64_VSX 4, 5, 2, 3, 0, 1, 6, 7, 12, 13, 10, 11, 8, 9, 14, 15

void AB64ToARGBRow_VSX(const uint16_t* src_ab64, uint8_t* dst_argb, int width) {
  do {
    v8u a = (v8u)vec_xl(0, src_ab64);
    v8u b = (v8u)vec_xl(16, src_ab64);
    v8u c = (v8u)vec_xl(32, src_ab64);
    v8u d = (v8u)vec_xl(48, src_ab64);
    Store16_VSX(HighBytesSeq_VSX((v16u)SHUFFLE_VSX(a, a, SWAP_RB64_VSX),
                                 (v16u)SHUFFLE_VSX(b, b, SWAP_RB64_VSX)),
                dst_argb);
    Store16_VSX(HighBytesSeq_VSX((v16u)SHUFFLE_VSX(c, c, SWAP_RB64_VSX),
                                 (v16u)SHUFFLE_VSX(d, d, SWAP_RB64_VSX)),
                dst_argb + 16);
    src_ab64 += 32;
    dst_argb += 32;
    width -= 8;
  } while (width > 0);
}

void AR64ToAB64Row_VSX(const uint16_t* src_ar64,
                       uint16_t* dst_ab64,
                       int width) {
  do {
    v8u a = (v8u)vec_xl(0, src_ar64);
    v8u b = (v8u)vec_xl(16, src_ar64);
    vec_xst(SHUFFLE_VSX(a, a, SWAP_RB64_VSX), 0, (uint8_t*)dst_ab64);
    vec_xst(SHUFFLE_VSX(b, b, SWAP_RB64_VSX), 16, (uint8_t*)dst_ab64);
    src_ar64 += 16;
    dst_ab64 += 16;
    width -= 4;
  } while (width > 0);
}

// Expands 8 packed 16-bit pixels (little-endian values) to ARGB, given the
// B, G, R and A fields already widened to 8 bits in 16-bit lanes.
static __inline void Store16BitAsARGB_VSX(v16u b,
                                          v16u g,
                                          v16u r,
                                          v16u a,
                                          uint8_t* dst_argb) {
  StoreARGB8_VSX((v8u)vec_pack(b, g), (v8u)vec_pack(r, a), dst_argb);
}

// (c << s) | (c >> t) for a field of 8 - s bits.
static __inline v16u Expand16_VSX(v16u c, unsigned short s, unsigned short t) {
  return vec_or(vec_sl(c, vec_splats(s)), vec_sr(c, vec_splats(t)));
}

static __inline void RGB565ToARGB8_VSX(v16u v, uint8_t* dst_argb) {
  const v16u k1f = vec_splats((unsigned short)0x1f);
  const v16u k3f = vec_splats((unsigned short)0x3f);
  v16u b = Expand16_VSX(vec_and(v, k1f), 3, 2);
  v16u g = Expand16_VSX(vec_and(vec_sr(v, vec_splats((unsigned short)5)), k3f),
                        2, 4);
  v16u r = Expand16_VSX(vec_sr(v, vec_splats((unsigned short)11)), 3, 2);
  Store16BitAsARGB_VSX(b, g, r, vec_splats((unsigned short)0xff), dst_argb);
}

static __inline void ARGB1555ToARGB8_VSX(v16u v, uint8_t* dst_argb) {
  const v16u k1f = vec_splats((unsigned short)0x1f);
  v16u b = Expand16_VSX(vec_and(v, k1f), 3, 2);
  v16u g = Expand16_VSX(vec_and(vec_sr(v, vec_splats((unsigned short)5)), k1f),
                        3, 2);
  v16u r = Expand16_VSX(vec_and(vec_sr(v, vec_splats((unsigned short)10)), k1f),
                        3, 2);
  v16u a =
      vec_mladd(vec_sr(v, vec_splats((unsigned short)15)),
                vec_splats((unsigned short)255), vec_splats((unsigned short)0));
  Store16BitAsARGB_VSX(b, g, r, a, dst_argb);
}

static __inline void ARGB4444ToARGB8_VSX(v16u v, uint8_t* dst_argb) {
  const v16u kf = vec_splats((unsigned short)0xf);
  const v16u k17 = vec_splats((unsigned short)17);
  const v16u zero = vec_splats((unsigned short)0);
  v16u b = vec_mladd(vec_and(v, kf), k17, zero);
  v16u g = vec_mladd(vec_and(vec_sr(v, vec_splats((unsigned short)4)), kf), k17,
                     zero);
  v16u r = vec_mladd(vec_and(vec_sr(v, vec_splats((unsigned short)8)), kf), k17,
                     zero);
  v16u a = vec_mladd(vec_sr(v, vec_splats((unsigned short)12)), k17, zero);
  Store16BitAsARGB_VSX(b, g, r, a, dst_argb);
}

#define FROM16_ROW_VSX(NAME, OP)                                \
  void NAME(const uint8_t* src, uint8_t* dst_argb, int width) { \
    do {                                                        \
      v16u a = LE16_VSX(Load16_VSX(src));                       \
      v16u b = LE16_VSX(Load16_VSX(src + 16));                  \
      OP(a, dst_argb);                                          \
      OP(b, dst_argb + 32);                                     \
      src += 32;                                                \
      dst_argb += 64;                                           \
      width -= 16;                                              \
    } while (width > 0);                                        \
  }

FROM16_ROW_VSX(RGB565ToARGBRow_VSX, RGB565ToARGB8_VSX)
FROM16_ROW_VSX(ARGB1555ToARGBRow_VSX, ARGB1555ToARGB8_VSX)
FROM16_ROW_VSX(ARGB4444ToARGBRow_VSX, ARGB4444ToARGB8_VSX)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_VSX) && defined(__VSX__) ...
