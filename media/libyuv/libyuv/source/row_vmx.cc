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

// AltiVec rows for big-endian PowerPC without POWER8 VSX, such as the
// PowerPC 970. AltiVec has no unaligned vector loads or stores.
#if !defined(LIBYUV_DISABLE_VMX) && defined(__ALTIVEC__) &&                  \
    !(defined(__VSX__) && defined(_ARCH_PWR8)) && defined(__BYTE_ORDER__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__

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
typedef __vector unsigned int v32u;

#if defined(__clang__) || __GNUC__ >= 12
#define SHUFFLE_VMX(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#define SHUFFLE_VMX(a, b, ...) __builtin_shuffle(a, b, (v8u){__VA_ARGS__})
#endif

#define DUP2_LO_VMX 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
#define DUP2_HI_VMX 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15

// Loads 16 unaligned bytes from the aligned blocks that hold them, so it never
// reads a block the bytes do not touch.
static __inline v8u Load16_VMX(const uint8_t* p) {
  return vec_perm(vec_ld(0, p), vec_ld(15, p), vec_lvsl(0, p));
}

// Loads 8 unaligned bytes into the first half of the vector.
static __inline v8u Load8_VMX(const uint8_t* p) {
  return vec_perm(vec_ld(0, p), vec_ld(7, p), vec_lvsl(0, p));
}

// Stores n vectors. AltiVec can only store aligned blocks, so unaligned rows
// go through a buffer rather than rewriting bytes next to the row.
static __inline void StoreN_VMX(const v8u* v, int n, uint8_t* dst) {
  int i;
  if (!((uintptr_t)dst & 15)) {
    for (i = 0; i < n; ++i) {
      vec_st(v[i], i * 16, dst);
    }
  } else {
    SIMD_ALIGNED(uint8_t tmp[64]);
    for (i = 0; i < n; ++i) {
      vec_st(v[i], i * 16, tmp);
    }
    memcpy(dst, tmp, n * 16);
  }
}

// a * b >> 16 for each lane, taking the high half of each 32-bit product.
static __inline v16u MulHi16_VMX(v16u a, v16u b) {
  v8u e = (v8u)vec_mule(a, b);
  v8u o = (v8u)vec_mulo(a, b);
  return (v16u)SHUFFLE_VMX(e, o, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12,
                           13, 28, 29);
}

// Picks the high byte of each 16-bit lane of e and o, interleaved, so the
// bytes come out in the order they went into vec_mule and vec_mulo.
static __inline v8u HighBytes16_VMX(v16u e, v16u o) {
  return SHUFFLE_VMX((v8u)e, (v8u)o, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26,
                     12, 28, 14, 30);
}

// 8 ARGB pixels from B and G of each pixel in bg, and R and A in ra.
static __inline void ARGB8_VMX(v8u bg, v8u ra, v8u* out) {
  out[0] = SHUFFLE_VMX(bg, ra, 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11,
                       19, 27);
  out[1] = SHUFFLE_VMX(bg, ra, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7,
                       15, 23, 31);
}

// Stores 16 opaque gray ARGB pixels.
static __inline void StoreGray_VMX(v8u y, uint8_t* dst_argb) {
  v8u a = vec_splat_u8(-1);
  v8u out[4];
  out[0] =
      SHUFFLE_VMX(y, a, 0, 0, 0, 16, 1, 1, 1, 16, 2, 2, 2, 16, 3, 3, 3, 16);
  out[1] =
      SHUFFLE_VMX(y, a, 4, 4, 4, 16, 5, 5, 5, 16, 6, 6, 6, 16, 7, 7, 7, 16);
  out[2] = SHUFFLE_VMX(y, a, 8, 8, 8, 16, 9, 9, 9, 16, 10, 10, 10, 16, 11, 11,
                       11, 16);
  out[3] = SHUFFLE_VMX(y, a, 12, 12, 12, 16, 13, 13, 13, 16, 14, 14, 14, 16, 15,
                       15, 15, 16);
  StoreN_VMX(out, 4, dst_argb);
}

// YuvConstants splatted to 16-bit lanes.
struct YuvCoefs_VMX {
  v16s ub, ug, vg, vr, yb;
  v16u yg;
};

static __inline void LoadYuvCoefs_VMX(const struct YuvConstants* yuvconstants,
                                      struct YuvCoefs_VMX* c) {
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
static __inline v16s UV16_VMX(v8u uv) {
  return vec_sub((v16s)vec_sr((v16u)uv, vec_splat_u16(8)),
                 vec_splats((short)128));
}

// (y * 0x0101 * yg >> 16) + yb from a lane holding y * 0x0101.
static __inline v16s Y16_VMX(v8u y, const struct YuvCoefs_VMX* c) {
  return vec_add((v16s)MulHi16_VMX((v16u)y, c->yg), c->yb);
}

// U and V contributions to B, G and R for 8 pixels, or for 8 pairs of
// pixels in 4:2:2.
struct UVTerms_VMX {
  v16s b, g, r;
};

static __inline void UVTerms_VMX(v8u u,
                                 v8u v,
                                 const struct YuvCoefs_VMX* c,
                                 struct UVTerms_VMX* t) {
  v16s zero = vec_splat_s16(0);
  v16s ui = UV16_VMX(u);
  v16s vi = UV16_VMX(v);
  t->b = vec_mladd(ui, c->ub, zero);
  t->g = vec_mladd(ui, c->ug, vec_mladd(vi, c->vg, zero));
  t->r = vec_mladd(vi, c->vr, zero);
}

// Converts 8 pixels to 2 ARGB vectors. a holds 8 alpha bytes in its first
// half, or is null for opaque pixels.
static __inline void YuvToARGB8_VMX(v16s y1,
                                    v16s tb,
                                    v16s tg,
                                    v16s tr,
                                    const v8u* a,
                                    v8u* out) {
  v16u six = vec_splat_u16(6);
  v16s b = vec_sra(vec_adds(y1, tb), six);
  v16s g = vec_sra(vec_subs(y1, tg), six);
  v16s r = vec_sra(vec_adds(y1, tr), six);
  v8u ra;
  if (a) {
    ra = SHUFFLE_VMX(vec_packsu(r, r), *a, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18,
                     19, 20, 21, 22, 23);
  } else {
    ra = vec_packsu(r, vec_splats((short)255));
  }
  ARGB8_VMX(vec_packsu(b, g), ra, out);
}

// Converts and stores 16 pixels that share 8 U and V in pairs. u and v are
// already widened, one lane per pair.
static __inline void Yuv422ToARGB16_VMX(v8u y,
                                        v8u u,
                                        v8u v,
                                        const v8u* a,
                                        const struct YuvCoefs_VMX* c,
                                        uint8_t* dst_argb) {
  struct UVTerms_VMX t;
  v8u out[4];
  UVTerms_VMX(u, v, c, &t);
  v8u a1 = a ? SHUFFLE_VMX(*a, *a, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11,
                           12, 13, 14, 15)
             : vec_splat_u8(0);
  YuvToARGB8_VMX(Y16_VMX(SHUFFLE_VMX(y, y, DUP2_LO_VMX), c),
                 vec_mergeh(t.b, t.b), vec_mergeh(t.g, t.g),
                 vec_mergeh(t.r, t.r), a, out);
  YuvToARGB8_VMX(Y16_VMX(SHUFFLE_VMX(y, y, DUP2_HI_VMX), c),
                 vec_mergel(t.b, t.b), vec_mergel(t.g, t.g),
                 vec_mergel(t.r, t.r), a ? &a1 : 0, out + 2);
  StoreN_VMX(out, 4, dst_argb);
}

// Converts and stores 16 pixels with their own U and V.
static __inline void Yuv444ToARGB16_VMX(v8u y,
                                        v8u u,
                                        v8u v,
                                        const v8u* a,
                                        const struct YuvCoefs_VMX* c,
                                        uint8_t* dst_argb) {
  struct UVTerms_VMX t0, t1;
  v8u out[4];
  UVTerms_VMX(SHUFFLE_VMX(u, u, DUP2_LO_VMX), SHUFFLE_VMX(v, v, DUP2_LO_VMX), c,
              &t0);
  UVTerms_VMX(SHUFFLE_VMX(u, u, DUP2_HI_VMX), SHUFFLE_VMX(v, v, DUP2_HI_VMX), c,
              &t1);
  v8u a1 = a ? SHUFFLE_VMX(*a, *a, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11,
                           12, 13, 14, 15)
             : vec_splat_u8(0);
  YuvToARGB8_VMX(Y16_VMX(SHUFFLE_VMX(y, y, DUP2_LO_VMX), c), t0.b, t0.g, t0.r,
                 a, out);
  YuvToARGB8_VMX(Y16_VMX(SHUFFLE_VMX(y, y, DUP2_HI_VMX), c), t1.b, t1.g, t1.r,
                 a ? &a1 : 0, out + 2);
  StoreN_VMX(out, 4, dst_argb);
}

static __inline v8u Widen422_VMX(v8u uv) {
  return SHUFFLE_VMX(uv, uv, DUP2_LO_VMX);
}

void I422ToARGBRow_VMX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VMX c;
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    Yuv422ToARGB16_VMX(Load16_VMX(src_y), Widen422_VMX(Load8_VMX(src_u)),
                       Widen422_VMX(Load8_VMX(src_v)), 0, &c, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I422AlphaToARGBRow_VMX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VMX c;
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    v8u a = Load16_VMX(src_a);
    Yuv422ToARGB16_VMX(Load16_VMX(src_y), Widen422_VMX(Load8_VMX(src_u)),
                       Widen422_VMX(Load8_VMX(src_v)), &a, &c, dst_argb);
    src_y += 16;
    src_u += 8;
    src_v += 8;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void NV12ToARGBRow_VMX(const uint8_t* src_y,
                       const uint8_t* src_uv,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VMX c;
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    v8u y = Load16_VMX(src_y);
    v8u uv = Load16_VMX(src_uv);
    Yuv422ToARGB16_VMX(y,
                       SHUFFLE_VMX(uv, uv, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10,
                                   12, 12, 14, 14),
                       SHUFFLE_VMX(uv, uv, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11,
                                   13, 13, 15, 15),
                       0, &c, dst_argb);
    src_y += 16;
    src_uv += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void NV21ToARGBRow_VMX(const uint8_t* src_y,
                       const uint8_t* src_vu,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VMX c;
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    v8u y = Load16_VMX(src_y);
    v8u vu = Load16_VMX(src_vu);
    Yuv422ToARGB16_VMX(y,
                       SHUFFLE_VMX(vu, vu, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11,
                                   13, 13, 15, 15),
                       SHUFFLE_VMX(vu, vu, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10,
                                   12, 12, 14, 14),
                       0, &c, dst_argb);
    src_y += 16;
    src_vu += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I444ToARGBRow_VMX(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VMX c;
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    Yuv444ToARGB16_VMX(Load16_VMX(src_y), Load16_VMX(src_u), Load16_VMX(src_v),
                       0, &c, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I444AlphaToARGBRow_VMX(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            const uint8_t* src_a,
                            uint8_t* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  struct YuvCoefs_VMX c;
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    v8u a = Load16_VMX(src_a);
    Yuv444ToARGB16_VMX(Load16_VMX(src_y), Load16_VMX(src_u), Load16_VMX(src_v),
                       &a, &c, dst_argb);
    src_y += 16;
    src_u += 16;
    src_v += 16;
    src_a += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void I400ToARGBRow_VMX(const uint8_t* src_y,
                       uint8_t* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  struct YuvCoefs_VMX c;
  v16u six = vec_splat_u16(6);
  LoadYuvCoefs_VMX(yuvconstants, &c);
  do {
    v8u y = Load16_VMX(src_y);
    v16s lo = vec_sra(Y16_VMX(SHUFFLE_VMX(y, y, DUP2_LO_VMX), &c), six);
    v16s hi = vec_sra(Y16_VMX(SHUFFLE_VMX(y, y, DUP2_HI_VMX), &c), six);
    StoreGray_VMX(vec_packsu(lo, hi), dst_argb);
    src_y += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

void J400ToARGBRow_VMX(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  do {
    StoreGray_VMX(Load16_VMX(src_y), dst_argb);
    src_y += 16;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

// (f * a + 255) >> 8 for B, G and R of 4 pixels, keeping A.
static __inline v8u Attenuate4_VMX(v8u argb) {
  static const v8u kAlphaMask = {0, 0, 0, 0xff, 0, 0, 0, 0xff,
                                 0, 0, 0, 0xff, 0, 0, 0, 0xff};
  v16u k255 = vec_splats((unsigned short)255);
  v8u a = SHUFFLE_VMX(argb, argb, 3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15,
                      15, 15, 15);
  v16u e = vec_add(vec_mule(argb, a), k255);
  v16u o = vec_add(vec_mulo(argb, a), k255);
  return vec_sel(HighBytes16_VMX(e, o), argb, kAlphaMask);
}

void ARGBAttenuateRow_VMX(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          int width) {
  do {
    v8u out[4];
    out[0] = Attenuate4_VMX(Load16_VMX(src_argb));
    out[1] = Attenuate4_VMX(Load16_VMX(src_argb + 16));
    out[2] = Attenuate4_VMX(Load16_VMX(src_argb + 32));
    out[3] = Attenuate4_VMX(Load16_VMX(src_argb + 48));
    StoreN_VMX(out, 4, dst_argb);
    src_argb += 64;
    dst_argb += 64;
    width -= 16;
  } while (width > 0);
}

// (s0 * (256 - f) + s1 * f + 128) >> 8 for 16 bytes.
static __inline v8u Interpolate16_VMX(v8u s0, v8u s1, v8u f0, v8u f1) {
  v16u k128 = vec_splats((unsigned short)128);
  v16u e = vec_add(vec_add(vec_mule(s0, f0), vec_mule(s1, f1)), k128);
  v16u o = vec_add(vec_add(vec_mulo(s0, f0), vec_mulo(s1, f1)), k128);
  return HighBytes16_VMX(e, o);
}

void InterpolateRow_VMX(uint8_t* dst_ptr,
                        const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        int dst_width,
                        int source_y_fraction) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  v8u f1 = vec_splats((unsigned char)source_y_fraction);
  v8u f0 = vec_splats((unsigned char)(256 - source_y_fraction));
  do {
    v8u out[2];
    v8u a0 = Load16_VMX(src_ptr);
    v8u a1 = Load16_VMX(src_ptr + 16);
    if (source_y_fraction == 0) {
      out[0] = a0;
      out[1] = a1;
    } else {
      v8u b0 = Load16_VMX(src_ptr1);
      v8u b1 = Load16_VMX(src_ptr1 + 16);
      if (source_y_fraction == 128) {
        out[0] = vec_avg(a0, b0);
        out[1] = vec_avg(a1, b1);
      } else {
        out[0] = Interpolate16_VMX(a0, b0, f0, f1);
        out[1] = Interpolate16_VMX(a1, b1, f0, f1);
      }
    }
    StoreN_VMX(out, 2, dst_ptr);
    src_ptr += 32;
    src_ptr1 += 32;
    dst_ptr += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_VMX) && defined(__ALTIVEC__) ...
