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
#include "libyuv/scale_row.h"

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
typedef __vector unsigned short v16u;
typedef __vector unsigned int v32u;
typedef __vector unsigned long long v64u;

// Constant byte shuffle of two v8u. The indices count elements in memory
// order on both byte orders.
#if defined(__clang__) || __GNUC__ >= 12
#define SHUFFLE_VSX(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#define SHUFFLE_VSX(a, b, ...) __builtin_shuffle(a, b, (v8u){__VA_ARGS__})
#endif

static __inline v8u Load16_VSX(const uint8_t* p) {
  return vec_xl(0, p);
}

static __inline void Store16_VSX(v8u v, uint8_t* p) {
  vec_xst(v, 0, p);
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

static __inline unsigned long long Load8u_VSX(const uint8_t* p) {
  unsigned long long v;
  memcpy(&v, p, 8);
  return v;
}

static __inline uint32_t Load4u_VSX(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}

// Average 2x2 boxes of ARGB pixels, rounded. s and t hold the same 4 pixels
// (2 output pixels) from two rows.
static __inline v32u Box2_VSX(v8u s, v8u t, v8u one, v32u two, int hi) {
  v8u q = hi ? SHUFFLE_VSX(s, t, 8, 12, 24, 28, 9, 13, 25, 29, 10, 14, 26, 30,
                           11, 15, 27, 31)
             : SHUFFLE_VSX(s, t, 0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7,
                           19, 23);
  return vec_msum(q, one, two);
}

// Returns 4 ARGB pixels averaged from the 2x2 boxes of s0, s1 (row 0) and
// t0, t1 (row 1).
static __inline v8u BoxARGB_VSX(v8u s0, v8u s1, v8u t0, v8u t1) {
  v8u one = vec_splats((unsigned char)1);
  v32u two = vec_splats(2u);
  v16u k2 = vec_splats((unsigned short)2);
  v16u lo =
      vec_pack(Box2_VSX(s0, t0, one, two, 0), Box2_VSX(s0, t0, one, two, 1));
  v16u hi =
      vec_pack(Box2_VSX(s1, t1, one, two, 0), Box2_VSX(s1, t1, one, two, 1));
  return (v8u)vec_pack(vec_sr(lo, k2), vec_sr(hi, k2));
}

#define EVEN_ARGB_VSX 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
#define ODD_ARGB_VSX 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

void ScaleARGBRowDown2_VSX(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_argb);
    v8u a1 = Load16_VSX(src_argb + 16);
    v8u a2 = Load16_VSX(src_argb + 32);
    v8u a3 = Load16_VSX(src_argb + 48);
    Store16_VSX(SHUFFLE_VSX(a0, a1, ODD_ARGB_VSX), dst_argb);
    Store16_VSX(SHUFFLE_VSX(a2, a3, ODD_ARGB_VSX), dst_argb + 16);
    src_argb += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDown2Linear_VSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_argb);
    v8u a1 = Load16_VSX(src_argb + 16);
    v8u a2 = Load16_VSX(src_argb + 32);
    v8u a3 = Load16_VSX(src_argb + 48);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a0, a1, EVEN_ARGB_VSX),
                        SHUFFLE_VSX(a0, a1, ODD_ARGB_VSX)),
                dst_argb);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a2, a3, EVEN_ARGB_VSX),
                        SHUFFLE_VSX(a2, a3, ODD_ARGB_VSX)),
                dst_argb + 16);
    src_argb += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDown2Box_VSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width) {
  const uint8_t* t = src_argb + src_stride;
  do {
    v8u s0 = Load16_VSX(src_argb);
    v8u s1 = Load16_VSX(src_argb + 16);
    v8u s2 = Load16_VSX(src_argb + 32);
    v8u s3 = Load16_VSX(src_argb + 48);
    v8u t0 = Load16_VSX(t);
    v8u t1 = Load16_VSX(t + 16);
    v8u t2 = Load16_VSX(t + 32);
    v8u t3 = Load16_VSX(t + 48);
    Store16_VSX(BoxARGB_VSX(s0, s1, t0, t1), dst_argb);
    Store16_VSX(BoxARGB_VSX(s2, s3, t2, t3), dst_argb + 16);
    src_argb += 64;
    t += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDownEven_VSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int src_stepx,
                              uint8_t* dst_argb,
                              int dst_width) {
  const ptrdiff_t step = (ptrdiff_t)src_stepx * 4;
  (void)src_stride;
  do {
    v32u v = {Load4u_VSX(src_argb), Load4u_VSX(src_argb + step),
              Load4u_VSX(src_argb + step * 2), Load4u_VSX(src_argb + step * 3)};
    Store16_VSX((v8u)v, dst_argb);
    src_argb += step * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

void ScaleARGBRowDownEvenBox_VSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  const ptrdiff_t step = (ptrdiff_t)src_stepx * 4;
  do {
    const uint8_t* t = src_argb + src_stride;
    v64u s0 = {Load8u_VSX(src_argb), Load8u_VSX(src_argb + step)};
    v64u s1 = {Load8u_VSX(src_argb + step * 2),
               Load8u_VSX(src_argb + step * 3)};
    v64u t0 = {Load8u_VSX(t), Load8u_VSX(t + step)};
    v64u t1 = {Load8u_VSX(t + step * 2), Load8u_VSX(t + step * 3)};
    Store16_VSX(BoxARGB_VSX((v8u)s0, (v8u)s1, (v8u)t0, (v8u)t1), dst_argb);
    src_argb += step * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

void ScaleARGBCols_VSX(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x,
                       int dx) {
  uint32_t ux = (uint32_t)x;
  const uint32_t udx = (uint32_t)dx;
  do {
    uint32_t p0 = Load4u_VSX(src_argb + ((int)ux >> 16) * 4);
    uint32_t p1 = Load4u_VSX(src_argb + ((int)(ux + udx) >> 16) * 4);
    uint32_t p2 = Load4u_VSX(src_argb + ((int)(ux + udx * 2) >> 16) * 4);
    uint32_t p3 = Load4u_VSX(src_argb + ((int)(ux + udx * 3) >> 16) * 4);
    v32u v = {p0, p1, p2, p3};
    Store16_VSX((v8u)v, dst_argb);
    ux += udx * 4;
    dst_argb += 16;
    dst_width -= 4;
  } while (dst_width > 0);
}

#define DUP_ARGB_LO_VSX 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7
#define DUP_ARGB_HI_VSX \
  8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15

void ScaleARGBColsUp2_VSX(uint8_t* dst_argb,
                          const uint8_t* src_argb,
                          int dst_width,
                          int x,
                          int dx) {
  (void)x;
  (void)dx;
  do {
    v8u s0 = Load16_VSX(src_argb);
    v8u s1 = Load16_VSX(src_argb + 16);
    Store16_VSX(SHUFFLE_VSX(s0, s0, DUP_ARGB_LO_VSX), dst_argb);
    Store16_VSX(SHUFFLE_VSX(s0, s0, DUP_ARGB_HI_VSX), dst_argb + 16);
    Store16_VSX(SHUFFLE_VSX(s1, s1, DUP_ARGB_LO_VSX), dst_argb + 32);
    Store16_VSX(SHUFFLE_VSX(s1, s1, DUP_ARGB_HI_VSX), dst_argb + 48);
    src_argb += 32;
    dst_argb += 64;
    dst_width -= 16;
  } while (dst_width > 0);
}

// Bilinear ARGB column filtering with 7 bit fractions, matching
// ScaleARGBFilterCols_C: (a * (127 - f) + b * f) >> 7. Only the low 16 bits
// of x give the fraction, so x is also kept in 16-bit lanes, two per pixel to
// match the even and odd bytes of each pixel.
void ScaleARGBFilterCols_VSX(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx) {
  uint32_t ux = (uint32_t)x;
  const uint32_t udx = (uint32_t)dx;
  unsigned short x0 = (unsigned short)ux;
  unsigned short x1 = (unsigned short)(ux + udx);
  unsigned short x2 = (unsigned short)(ux + udx * 2);
  unsigned short x3 = (unsigned short)(ux + udx * 3);
  v16u xs = {x0, x0, x1, x1, x2, x2, x3, x3};
  const v16u dx4 = vec_splats((unsigned short)(udx * 4));
  const v16u k9 = vec_splats((unsigned short)9);
  const v16u k514 = vec_splats((unsigned short)514);
  const v16u zero = vec_splats((unsigned short)0);
  const v8u kfe = vec_splats((unsigned char)0xfe);
  do {
    v64u p0 = {Load8u_VSX(src_argb + ((int)ux >> 16) * 4),
               Load8u_VSX(src_argb + ((int)(ux + udx) >> 16) * 4)};
    v64u p1 = {Load8u_VSX(src_argb + ((int)(ux + udx * 2) >> 16) * 4),
               Load8u_VSX(src_argb + ((int)(ux + udx * 3) >> 16) * 4)};
    v8u a = SHUFFLE_VSX((v8u)p0, (v8u)p1, EVEN_ARGB_VSX);
    v8u b = SHUFFLE_VSX((v8u)p0, (v8u)p1, ODD_ARGB_VSX);
    // 2 * f in every byte of each pixel, and 2 * (127 - f).
    v8u f2 = (v8u)vec_mladd(vec_sr(xs, k9), k514, zero);
    v8u nf2 = vec_xor(f2, kfe);
    v16u e = vec_add(vec_mule(a, nf2), vec_mule(b, f2));
    v16u o = vec_add(vec_mulo(a, nf2), vec_mulo(b, f2));
    // The high byte of twice the sum is the sum >> 7.
    Store16_VSX(HighBytes16_VSX(e, o), dst_argb);
    xs = vec_add(xs, dx4);
    ux += udx * 4;
    dst_argb += 16;
    dst_width -= 4;
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

// Zero extends the first or last 8 bytes of v to 16 bits.
static __inline v16u Widen8Lo_VSX(v8u v) {
  v8u zero = vec_splats((unsigned char)0);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v16u)vec_mergeh(v, zero);
#else
  return (v16u)vec_mergeh(zero, v);
#endif
}

static __inline v16u Widen8Hi_VSX(v8u v) {
  v8u zero = vec_splats((unsigned char)0);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (v16u)vec_mergel(v, zero);
#else
  return (v16u)vec_mergel(zero, v);
#endif
}

#define ODD_BYTES_VSX 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
#define EVEN_BYTES_VSX 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
#define EVEN_UV_VSX 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29
#define ODD_UV_VSX 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31

void ScaleRowDown2_VSX(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_ptr);
    v8u a1 = Load16_VSX(src_ptr + 16);
    v8u a2 = Load16_VSX(src_ptr + 32);
    v8u a3 = Load16_VSX(src_ptr + 48);
    Store16_VSX(SHUFFLE_VSX(a0, a1, ODD_BYTES_VSX), dst);
    Store16_VSX(SHUFFLE_VSX(a2, a3, ODD_BYTES_VSX), dst + 16);
    src_ptr += 64;
    dst += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

void ScaleRowDown2Linear_VSX(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_ptr);
    v8u a1 = Load16_VSX(src_ptr + 16);
    v8u a2 = Load16_VSX(src_ptr + 32);
    v8u a3 = Load16_VSX(src_ptr + 48);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a0, a1, EVEN_BYTES_VSX),
                        SHUFFLE_VSX(a0, a1, ODD_BYTES_VSX)),
                dst);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a2, a3, EVEN_BYTES_VSX),
                        SHUFFLE_VSX(a2, a3, ODD_BYTES_VSX)),
                dst + 16);
    src_ptr += 64;
    dst += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

// Rounded averages of the 2x2 boxes of 32 bytes from s0, s1 (row 0) and
// t0, t1 (row 1), with the 4 bytes of each box gathered into one word.
// (sum * 64 + 128) >> 8 is (sum + 2) >> 2.
#define BOX2_LO_VSX 0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23
#define BOX2_HI_VSX 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31
static __inline v8u Box2x2_VSX(v8u s0, v8u s1, v8u t0, v8u t1) {
  v8u k64 = vec_splats((unsigned char)64);
  v32u k128 = vec_splats(128u);
  return Bits8To15_VSX(vec_msum(SHUFFLE_VSX(s0, t0, BOX2_LO_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s0, t0, BOX2_HI_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s1, t1, BOX2_LO_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s1, t1, BOX2_HI_VSX), k64, k128));
}

void ScaleRowDown2Box_VSX(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width) {
  const uint8_t* t = src_ptr + src_stride;
  do {
    v8u s0 = Load16_VSX(src_ptr);
    v8u s1 = Load16_VSX(src_ptr + 16);
    v8u s2 = Load16_VSX(src_ptr + 32);
    v8u s3 = Load16_VSX(src_ptr + 48);
    v8u t0 = Load16_VSX(t);
    v8u t1 = Load16_VSX(t + 16);
    v8u t2 = Load16_VSX(t + 32);
    v8u t3 = Load16_VSX(t + 48);
    Store16_VSX(Box2x2_VSX(s0, s1, t0, t1), dst);
    Store16_VSX(Box2x2_VSX(s2, s3, t2, t3), dst + 16);
    src_ptr += 64;
    t += 64;
    dst += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

void ScaleRowDown4_VSX(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst_ptr,
                       int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_ptr);
    v8u a1 = Load16_VSX(src_ptr + 16);
    v8u a2 = Load16_VSX(src_ptr + 32);
    v8u a3 = Load16_VSX(src_ptr + 48);
    v8u lo = SHUFFLE_VSX(a0, a1, 2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18,
                         22, 26, 30);
    v8u hi = SHUFFLE_VSX(a2, a3, 2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18,
                         22, 26, 30);
    Store16_VSX((v8u)vec_mergeh((v64u)lo, (v64u)hi), dst_ptr);
    src_ptr += 64;
    dst_ptr += 16;
    dst_width -= 16;
  } while (dst_width > 0);
}

// (sum of 16 bytes + 8) >> 4 for each 4x4 box; sum * 16 + 128 fits 16 bits.
void ScaleRowDown4Box_VSX(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width) {
  const v8u k16 = vec_splats((unsigned char)16);
  const v32u k128 = vec_splats(128u);
  do {
    v32u s[4];
    int i;
    for (i = 0; i < 4; ++i) {
      const uint8_t* p = src_ptr + i * 16;
      v32u a = vec_msum(Load16_VSX(p), k16, k128);
      a = vec_msum(Load16_VSX(p + src_stride), k16, a);
      a = vec_msum(Load16_VSX(p + src_stride * 2), k16, a);
      s[i] = vec_msum(Load16_VSX(p + src_stride * 3), k16, a);
    }
    Store16_VSX(Bits8To15_VSX(s[0], s[1], s[2], s[3]), dst_ptr);
    src_ptr += 64;
    dst_ptr += 16;
    dst_width -= 16;
  } while (dst_width > 0);
}

void ScaleRowDown34_VSX(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_ptr);
    v8u a1 = Load16_VSX(src_ptr + 16);
    v8u a2 = Load16_VSX(src_ptr + 32);
    v8u a3 = Load16_VSX(src_ptr + 48);
    Store16_VSX(SHUFFLE_VSX(a0, a1, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16,
                            17, 19, 20),
                dst_ptr);
    Store16_VSX(SHUFFLE_VSX(a1, a2, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20,
                            21, 23, 24, 25),
                dst_ptr + 16);
    Store16_VSX(SHUFFLE_VSX(a2, a3, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24,
                            25, 27, 28, 29, 31),
                dst_ptr + 32);
    src_ptr += 64;
    dst_ptr += 48;
    dst_width -= 48;
  } while (dst_width > 0);
}

// (3 * a + b + 2) >> 2 for each byte, from rounded averages: with
// t = avg(a, b), avg(a, t) is one too big exactly when a ^ b and a ^ t are
// both odd.
static __inline v8u Avg31_VSX(v8u a, v8u b, v8u t) {
  const v8u one = vec_splats((unsigned char)1);
  return vec_sub(vec_avg(a, t),
                 vec_and(vec_and(vec_xor(a, b), vec_xor(a, t)), one));
}

// Filters 64 bytes of a row down to 48: every 4 bytes s0..s3 give
// (3 * s0 + s1 + 2) >> 2, (s1 + s2 + 1) >> 1 and (s2 + 3 * s3 + 2) >> 2,
// each as 64 * sum + 128 by vmsumubm so the result is the second byte. Each
// d[i] holds the three results of 4 groups, one result at a time.
static __inline void Filter34_VSX(const uint8_t* src, v8u* d) {
  static const v8u w0 = {192, 64, 0, 0, 192, 64, 0, 0,
                         192, 64, 0, 0, 192, 64, 0, 0};
  static const v8u w1 = {0, 128, 128, 0, 0, 128, 128, 0,
                         0, 128, 128, 0, 0, 128, 128, 0};
  static const v8u w2 = {0, 0, 64, 192, 0, 0, 64, 192,
                         0, 0, 64, 192, 0, 0, 64, 192};
  const v32u k128 = vec_splats(128u);
  int i;
  for (i = 0; i < 4; ++i) {
    v8u a = Load16_VSX(src + i * 16);
    v32u m2 = vec_msum(a, w2, k128);
    d[i] = Bits8To15_VSX(vec_msum(a, w0, k128), vec_msum(a, w1, k128), m2, m2);
  }
}

// Stores 48 results from the 4 vectors Filter34_VSX makes, in output order.
static __inline void Store34_VSX(const v8u* d, uint8_t* dst_ptr) {
  Store16_VSX(SHUFFLE_VSX(d[0], d[1], 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 16,
                          20, 24, 17),
              dst_ptr);
  Store16_VSX(SHUFFLE_VSX(d[1], d[2], 5, 9, 2, 6, 10, 3, 7, 11, 16, 20, 24, 17,
                          21, 25, 18, 22),
              dst_ptr + 16);
  Store16_VSX(SHUFFLE_VSX(d[2], d[3], 10, 3, 7, 11, 16, 20, 24, 17, 21, 25, 18,
                          22, 26, 19, 23, 27),
              dst_ptr + 32);
}

void ScaleRowDown34_0_Box_VSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  do {
    v8u a[4], b[4], d[4];
    int i;
    Filter34_VSX(src_ptr, a);
    Filter34_VSX(src_ptr + src_stride, b);
    for (i = 0; i < 4; ++i) {
      d[i] = Avg31_VSX(a[i], b[i], vec_avg(a[i], b[i]));
    }
    Store34_VSX(d, dst_ptr);
    src_ptr += 64;
    dst_ptr += 48;
    dst_width -= 48;
  } while (dst_width > 0);
}

void ScaleRowDown34_1_Box_VSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  do {
    v8u a[4], b[4], d[4];
    int i;
    Filter34_VSX(src_ptr, a);
    Filter34_VSX(src_ptr + src_stride, b);
    for (i = 0; i < 4; ++i) {
      d[i] = vec_avg(a[i], b[i]);
    }
    Store34_VSX(d, dst_ptr);
    src_ptr += 64;
    dst_ptr += 48;
    dst_width -= 48;
  } while (dst_width > 0);
}

// Bytes 0, 3 and 6 of every 8.
void ScaleRowDown38_VSX(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  (void)src_stride;
  do {
    v8u a = SHUFFLE_VSX(Load16_VSX(src_ptr), Load16_VSX(src_ptr + 16), 0, 3, 6,
                        8, 11, 14, 16, 19, 22, 24, 27, 30, 0, 0, 0, 0);
    v8u b = SHUFFLE_VSX(Load16_VSX(src_ptr + 43), Load16_VSX(src_ptr + 59), 0,
                        3, 5, 8, 11, 13, 16, 19, 21, 24, 27, 29, 0, 0, 0, 0);
    v8u c = SHUFFLE_VSX(Load16_VSX(src_ptr + 86), Load16_VSX(src_ptr + 102), 0,
                        2, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29, 0, 0, 0, 0);
    Store16_VSX(SHUFFLE_VSX(a, Load16_VSX(src_ptr + 32), 0, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10, 11, 16, 19, 22, 24),
                dst_ptr);
    Store16_VSX(SHUFFLE_VSX(b, Load16_VSX(src_ptr + 75), 0, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10, 11, 16, 19, 21, 24),
                dst_ptr + 16);
    Store16_VSX(SHUFFLE_VSX(c, Load16_VSX(src_ptr + 112), 0, 1, 2, 3, 4, 5, 6,
                            7, 8, 9, 10, 11, 22, 24, 27, 30),
                dst_ptr + 32);
    src_ptr += 128;
    dst_ptr += 48;
    dst_width -= 48;
  } while (dst_width > 0);
}

// Gathers the 3, 3 and 2 bytes that each of 12 outputs sums from 32 bytes of
// a row into 32-bit lanes, and adds them to sum.
static __inline void Sum38_VSX(const uint8_t* p, v32u* sum) {
  const v8u zero = vec_splats((unsigned char)0);
  const v8u ones = vec_splats((unsigned char)1);
  sum[0] = vec_msum(SHUFFLE_VSX(Load16_VSX(p), zero, 0, 1, 2, 16, 3, 4, 5, 16,
                                6, 7, 16, 16, 8, 9, 10, 16),
                    ones, sum[0]);
  sum[1] = vec_msum(SHUFFLE_VSX(Load16_VSX(p + 11), zero, 0, 1, 2, 16, 3, 4, 16,
                                16, 5, 6, 7, 16, 8, 9, 10, 16),
                    ones, sum[1]);
  sum[2] = vec_msum(SHUFFLE_VSX(Load16_VSX(p + 16), zero, 6, 7, 16, 16, 8, 9,
                                10, 16, 11, 12, 13, 16, 14, 15, 16, 16),
                    ones, sum[2]);
}

// Scales the 12 sums of each group by k0, k1 and k2 (16 bit fractions) and
// stores the 48 bytes.
static __inline void Store38_VSX(const v32u* sum,
                                 const v32u* k,
                                 uint8_t* dst_ptr) {
  const v32u k16 = vec_splats(16u);
  v16u h[6];
  int i;
  for (i = 0; i < 6; ++i) {
    h[i] = vec_pack(vec_sr(vec_mul(sum[2 * i], k[(2 * i) % 3]), k16),
                    vec_sr(vec_mul(sum[2 * i + 1], k[(2 * i + 1) % 3]), k16));
  }
  Store16_VSX((v8u)vec_pack(h[0], h[1]), dst_ptr);
  Store16_VSX((v8u)vec_pack(h[2], h[3]), dst_ptr + 16);
  Store16_VSX((v8u)vec_pack(h[4], h[5]), dst_ptr + 32);
}

void ScaleRowDown38_3_Box_VSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  const unsigned int k9 = 65536 / 9, k6 = 65536 / 6;
  const v32u k[3] = {{k9, k9, k6, k9}, {k9, k6, k9, k9}, {k6, k9, k9, k6}};
  do {
    v32u sum[12];
    int g;
    for (g = 0; g < 12; ++g) {
      sum[g] = vec_splats(0u);
    }
    for (g = 0; g < 4; ++g) {
      const uint8_t* p = src_ptr + g * 32;
      Sum38_VSX(p, sum + g * 3);
      Sum38_VSX(p + src_stride, sum + g * 3);
      Sum38_VSX(p + src_stride * 2, sum + g * 3);
    }
    Store38_VSX(sum, k, dst_ptr);
    src_ptr += 128;
    dst_ptr += 48;
    dst_width -= 48;
  } while (dst_width > 0);
}

void ScaleRowDown38_2_Box_VSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  const unsigned int k6 = 65536 / 6, k4 = 65536 / 4;
  const v32u k[3] = {{k6, k6, k4, k6}, {k6, k4, k6, k6}, {k4, k6, k6, k4}};
  do {
    v32u sum[12];
    int g;
    for (g = 0; g < 12; ++g) {
      sum[g] = vec_splats(0u);
    }
    for (g = 0; g < 4; ++g) {
      const uint8_t* p = src_ptr + g * 32;
      Sum38_VSX(p, sum + g * 3);
      Sum38_VSX(p + src_stride, sum + g * 3);
    }
    Store38_VSX(sum, k, dst_ptr);
    src_ptr += 128;
    dst_ptr += 48;
    dst_width -= 48;
  } while (dst_width > 0);
}

void ScaleAddRow_VSX(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width) {
  do {
    v8u s = Load16_VSX(src_ptr);
    v16u d0 = vec_xl(0, dst_ptr);
    v16u d1 = vec_xl(16, dst_ptr);
    vec_xst(vec_add(d0, Widen8Lo_VSX(s)), 0, dst_ptr);
    vec_xst(vec_add(d1, Widen8Hi_VSX(s)), 16, dst_ptr);
    src_ptr += 16;
    dst_ptr += 16;
    src_width -= 16;
  } while (src_width > 0);
}

static __inline uint16_t Load2u_VSX(const uint8_t* p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return v;
}

// Bilinear column filtering with 7 bit fractions, matching ScaleFilterCols_C:
// a + ((f * (b - a) + 64) >> 7), which is (a * (128 - f) + b * f + 64) >> 7.
void ScaleFilterCols_VSX(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         int dst_width,
                         int x,
                         int dx) {
  uint32_t ux = (uint32_t)x;
  const uint32_t udx = (uint32_t)dx;
  v16u xs0, xs1;
  int i;
  for (i = 0; i < 8; ++i) {
    xs0[i] = (unsigned short)(ux + udx * i);
    xs1[i] = (unsigned short)(ux + udx * (i + 8));
  }
  const v16u dx16 = vec_splats((unsigned short)(udx * 16));
  const v16u k9 = vec_splats((unsigned short)9);
  const v16u k1 = vec_splats((unsigned short)1);
  const v16u k64 = vec_splats((unsigned short)64);
  const v8u k128 = vec_splats((unsigned char)128);
  do {
    v16u p0, p1;
    for (i = 0; i < 8; ++i) {
      p0[i] = Load2u_VSX(src_ptr + ((int)(ux + udx * i) >> 16));
      p1[i] = Load2u_VSX(src_ptr + ((int)(ux + udx * (i + 8)) >> 16));
    }
    v8u a = SHUFFLE_VSX((v8u)p0, (v8u)p1, EVEN_BYTES_VSX);
    v8u b = SHUFFLE_VSX((v8u)p0, (v8u)p1, ODD_BYTES_VSX);
    v8u f = (v8u)vec_pack(vec_sr(xs0, k9), vec_sr(xs1, k9));
    v8u nf = vec_sub(k128, f);
    v16u e = vec_add(vec_add(vec_mule(a, nf), vec_mule(b, f)), k64);
    v16u o = vec_add(vec_add(vec_mulo(a, nf), vec_mulo(b, f)), k64);
    // The high byte of twice the sum is the sum >> 7.
    Store16_VSX(HighBytes16_VSX(vec_sl(e, k1), vec_sl(o, k1)), dst_ptr);
    xs0 = vec_add(xs0, dx16);
    xs1 = vec_add(xs1, dx16);
    ux += udx * 16;
    dst_ptr += 16;
    dst_width -= 16;
  } while (dst_width > 0);
}

// Picks the high byte of each 16-bit lane of p and q, interleaved in pairs
// of lanes: p0, p1, q0, q1, p2, p3, q2, q3 and so on.
static __inline v8u HighBytes16Pairs_VSX(v16u p, v16u q) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return SHUFFLE_VSX((v8u)p, (v8u)q, 1, 3, 17, 19, 5, 7, 21, 23, 9, 11, 25, 27,
                     13, 15, 29, 31);
#else
  return SHUFFLE_VSX((v8u)p, (v8u)q, 0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26,
                     12, 14, 28, 30);
#endif
}

void ScaleRowUp2_Linear_VSX(const uint8_t* src_ptr,
                            uint8_t* dst_ptr,
                            int dst_width) {
  do {
    v8u a = Load16_VSX(src_ptr);
    v8u b = Load16_VSX(src_ptr + 1);
    v8u t = vec_avg(a, b);
    v8u p = Avg31_VSX(a, b, t);
    v8u q = Avg31_VSX(b, a, t);
    Store16_VSX(vec_mergeh(p, q), dst_ptr);
    Store16_VSX(vec_mergel(p, q), dst_ptr + 16);
    src_ptr += 16;
    dst_ptr += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

// 16 * (3 * s + t) + 128 and 16 * (s + 3 * t) + 128 for the horizontal sums
// s and t of two rows, whose high bytes are the two output rows.
static __inline void Bilinear2_VSX(v16u s, v16u t, v16u* d, v16u* e) {
  const v16u k16 = vec_splats((unsigned short)16);
  const v16u k48 = vec_splats((unsigned short)48);
  const v16u k128 = vec_splats((unsigned short)128);
  *d = vec_mladd(s, k48, vec_mladd(t, k16, k128));
  *e = vec_mladd(t, k48, vec_mladd(s, k16, k128));
}

// 3 * a + b and a + 3 * b.
static __inline void Linear3_VSX(v16u a, v16u b, v16u* p, v16u* q) {
  const v16u k3 = vec_splats((unsigned short)3);
  *p = vec_mladd(a, k3, b);
  *q = vec_mladd(b, k3, a);
}

// Filters 8 pixels of two rows a, b (row 0) and c, d (row 1), where b and d
// are a and c one pixel on, into the 16-bit lanes of both output rows: the
// even output pixels in dp and ep, the odd ones in dq and eq.
static __inline void Bilinear8_VSX(v16u a,
                                   v16u b,
                                   v16u c,
                                   v16u d,
                                   v16u* dp,
                                   v16u* dq,
                                   v16u* ep,
                                   v16u* eq) {
  v16u sp, sq, tp, tq;
  Linear3_VSX(a, b, &sp, &sq);
  Linear3_VSX(c, d, &tp, &tq);
  Bilinear2_VSX(sp, tp, dp, ep);
  Bilinear2_VSX(sq, tq, dq, eq);
}

void ScaleRowUp2_Bilinear_VSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              ptrdiff_t dst_stride,
                              int dst_width) {
  const uint8_t* t = src_ptr + src_stride;
  uint8_t* e = dst_ptr + dst_stride;
  do {
    v8u a = Load16_VSX(src_ptr);
    v8u b = Load16_VSX(src_ptr + 1);
    v8u c = Load16_VSX(t);
    v8u d = Load16_VSX(t + 1);
    v16u dp, dq, ep, eq;
    Bilinear8_VSX(Widen8Lo_VSX(a), Widen8Lo_VSX(b), Widen8Lo_VSX(c),
                  Widen8Lo_VSX(d), &dp, &dq, &ep, &eq);
    Store16_VSX(HighBytes16_VSX(dp, dq), dst_ptr);
    Store16_VSX(HighBytes16_VSX(ep, eq), e);
    Bilinear8_VSX(Widen8Hi_VSX(a), Widen8Hi_VSX(b), Widen8Hi_VSX(c),
                  Widen8Hi_VSX(d), &dp, &dq, &ep, &eq);
    Store16_VSX(HighBytes16_VSX(dp, dq), dst_ptr + 16);
    Store16_VSX(HighBytes16_VSX(ep, eq), e + 16);
    src_ptr += 16;
    t += 16;
    dst_ptr += 32;
    e += 32;
    dst_width -= 32;
  } while (dst_width > 0);
}

void ScaleUVRowDown2_VSX(const uint8_t* src_uv,
                         ptrdiff_t src_stride,
                         uint8_t* dst_uv,
                         int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_uv);
    v8u a1 = Load16_VSX(src_uv + 16);
    v8u a2 = Load16_VSX(src_uv + 32);
    v8u a3 = Load16_VSX(src_uv + 48);
    Store16_VSX(SHUFFLE_VSX(a0, a1, ODD_UV_VSX), dst_uv);
    Store16_VSX(SHUFFLE_VSX(a2, a3, ODD_UV_VSX), dst_uv + 16);
    src_uv += 64;
    dst_uv += 32;
    dst_width -= 16;
  } while (dst_width > 0);
}

void ScaleUVRowDown2Linear_VSX(const uint8_t* src_uv,
                               ptrdiff_t src_stride,
                               uint8_t* dst_uv,
                               int dst_width) {
  (void)src_stride;
  do {
    v8u a0 = Load16_VSX(src_uv);
    v8u a1 = Load16_VSX(src_uv + 16);
    v8u a2 = Load16_VSX(src_uv + 32);
    v8u a3 = Load16_VSX(src_uv + 48);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a0, a1, EVEN_UV_VSX),
                        SHUFFLE_VSX(a0, a1, ODD_UV_VSX)),
                dst_uv);
    Store16_VSX(vec_avg(SHUFFLE_VSX(a2, a3, EVEN_UV_VSX),
                        SHUFFLE_VSX(a2, a3, ODD_UV_VSX)),
                dst_uv + 16);
    src_uv += 64;
    dst_uv += 32;
    dst_width -= 16;
  } while (dst_width > 0);
}

#define UVBOX_LO_VSX 0, 2, 16, 18, 1, 3, 17, 19, 4, 6, 20, 22, 5, 7, 21, 23
#define UVBOX_HI_VSX \
  8, 10, 24, 26, 9, 11, 25, 27, 12, 14, 28, 30, 13, 15, 29, 31

static __inline v8u BoxUV_VSX(v8u s0, v8u s1, v8u t0, v8u t1) {
  v8u k64 = vec_splats((unsigned char)64);
  v32u k128 = vec_splats(128u);
  return Bits8To15_VSX(vec_msum(SHUFFLE_VSX(s0, t0, UVBOX_LO_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s0, t0, UVBOX_HI_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s1, t1, UVBOX_LO_VSX), k64, k128),
                       vec_msum(SHUFFLE_VSX(s1, t1, UVBOX_HI_VSX), k64, k128));
}

void ScaleUVRowDown2Box_VSX(const uint8_t* src_uv,
                            ptrdiff_t src_stride,
                            uint8_t* dst_uv,
                            int dst_width) {
  const uint8_t* t = src_uv + src_stride;
  do {
    v8u s0 = Load16_VSX(src_uv);
    v8u s1 = Load16_VSX(src_uv + 16);
    v8u s2 = Load16_VSX(src_uv + 32);
    v8u s3 = Load16_VSX(src_uv + 48);
    v8u t0 = Load16_VSX(t);
    v8u t1 = Load16_VSX(t + 16);
    v8u t2 = Load16_VSX(t + 32);
    v8u t3 = Load16_VSX(t + 48);
    Store16_VSX(BoxUV_VSX(s0, s1, t0, t1), dst_uv);
    Store16_VSX(BoxUV_VSX(s2, s3, t2, t3), dst_uv + 16);
    src_uv += 64;
    t += 64;
    dst_uv += 32;
    dst_width -= 16;
  } while (dst_width > 0);
}

void ScaleUVRowDownEven_VSX(const uint8_t* src_uv,
                            ptrdiff_t src_stride,
                            int src_stepx,
                            uint8_t* dst_uv,
                            int dst_width) {
  const ptrdiff_t step = (ptrdiff_t)src_stepx * 2;
  (void)src_stride;
  do {
    v16u v = {Load2u_VSX(src_uv),
              Load2u_VSX(src_uv + step),
              Load2u_VSX(src_uv + step * 2),
              Load2u_VSX(src_uv + step * 3),
              Load2u_VSX(src_uv + step * 4),
              Load2u_VSX(src_uv + step * 5),
              Load2u_VSX(src_uv + step * 6),
              Load2u_VSX(src_uv + step * 7)};
    Store16_VSX((v8u)v, dst_uv);
    src_uv += step * 8;
    dst_uv += 16;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleUVRowUp2_Linear_VSX(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width) {
  do {
    v8u a = Load16_VSX(src_ptr);
    v8u b = Load16_VSX(src_ptr + 2);
    v8u t = vec_avg(a, b);
    v16u p = (v16u)Avg31_VSX(a, b, t);
    v16u q = (v16u)Avg31_VSX(b, a, t);
    Store16_VSX((v8u)vec_mergeh(p, q), dst_ptr);
    Store16_VSX((v8u)vec_mergel(p, q), dst_ptr + 16);
    src_ptr += 16;
    dst_ptr += 32;
    dst_width -= 16;
  } while (dst_width > 0);
}

void ScaleUVRowUp2_Bilinear_VSX(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width) {
  const uint8_t* t = src_ptr + src_stride;
  uint8_t* e = dst_ptr + dst_stride;
  do {
    v8u a = Load16_VSX(src_ptr);
    v8u b = Load16_VSX(src_ptr + 2);
    v8u c = Load16_VSX(t);
    v8u d = Load16_VSX(t + 2);
    v16u dp, dq, ep, eq;
    Bilinear8_VSX(Widen8Lo_VSX(a), Widen8Lo_VSX(b), Widen8Lo_VSX(c),
                  Widen8Lo_VSX(d), &dp, &dq, &ep, &eq);
    Store16_VSX(HighBytes16Pairs_VSX(dp, dq), dst_ptr);
    Store16_VSX(HighBytes16Pairs_VSX(ep, eq), e);
    Bilinear8_VSX(Widen8Hi_VSX(a), Widen8Hi_VSX(b), Widen8Hi_VSX(c),
                  Widen8Hi_VSX(d), &dp, &dq, &ep, &eq);
    Store16_VSX(HighBytes16Pairs_VSX(dp, dq), dst_ptr + 16);
    Store16_VSX(HighBytes16Pairs_VSX(ep, eq), e + 16);
    src_ptr += 16;
    t += 16;
    dst_ptr += 32;
    e += 32;
    dst_width -= 16;
  } while (dst_width > 0);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_VSX) && defined(__VSX__) ...
