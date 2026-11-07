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

// AltiVec scale rows for big-endian PowerPC without POWER8 VSX, such as the
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
typedef __vector unsigned short v16u;
typedef __vector unsigned int v32u;

#if defined(__clang__) || __GNUC__ >= 12
#define SHUFFLE_VMX(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#define SHUFFLE_VMX(a, b, ...) __builtin_shuffle(a, b, (v8u){__VA_ARGS__})
#endif

// Loads 16 unaligned bytes from the aligned blocks that hold them, so it never
// reads a block the bytes do not touch.
static __inline v8u Load16_VMX(const uint8_t* p) {
  return vec_perm(vec_ld(0, p), vec_ld(15, p), vec_lvsl(0, p));
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

// Average 2x2 boxes of ARGB pixels, rounded. s and t hold the same 4 pixels
// (2 output pixels) from two rows.
static __inline v32u Box2_VMX(v8u s, v8u t, v8u one, v32u two, int hi) {
  v8u q = hi ? SHUFFLE_VMX(s, t, 8, 12, 24, 28, 9, 13, 25, 29, 10, 14, 26, 30,
                           11, 15, 27, 31)
             : SHUFFLE_VMX(s, t, 0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7,
                           19, 23);
  return vec_msum(q, one, two);
}

// Returns 4 ARGB pixels averaged from the 2x2 boxes of s0, s1 (row 0) and
// t0, t1 (row 1).
static __inline v8u BoxARGB_VMX(v8u s0, v8u s1, v8u t0, v8u t1) {
  v8u one = vec_splat_u8(1);
  v32u two = vec_splat_u32(2);
  v16u k2 = vec_splat_u16(2);
  v16u lo =
      vec_pack(Box2_VMX(s0, t0, one, two, 0), Box2_VMX(s0, t0, one, two, 1));
  v16u hi =
      vec_pack(Box2_VMX(s1, t1, one, two, 0), Box2_VMX(s1, t1, one, two, 1));
  return (v8u)vec_pack(vec_sr(lo, k2), vec_sr(hi, k2));
}

#define EVEN_ARGB_VMX 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
#define ODD_ARGB_VMX 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

void ScaleARGBRowDown2_VMX(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width) {
  (void)src_stride;
  do {
    v8u out[2];
    v8u a0 = Load16_VMX(src_argb);
    v8u a1 = Load16_VMX(src_argb + 16);
    v8u a2 = Load16_VMX(src_argb + 32);
    v8u a3 = Load16_VMX(src_argb + 48);
    out[0] = SHUFFLE_VMX(a0, a1, ODD_ARGB_VMX);
    out[1] = SHUFFLE_VMX(a2, a3, ODD_ARGB_VMX);
    StoreN_VMX(out, 2, dst_argb);
    src_argb += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDown2Linear_VMX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  (void)src_stride;
  do {
    v8u out[2];
    v8u a0 = Load16_VMX(src_argb);
    v8u a1 = Load16_VMX(src_argb + 16);
    v8u a2 = Load16_VMX(src_argb + 32);
    v8u a3 = Load16_VMX(src_argb + 48);
    out[0] = vec_avg(SHUFFLE_VMX(a0, a1, EVEN_ARGB_VMX),
                     SHUFFLE_VMX(a0, a1, ODD_ARGB_VMX));
    out[1] = vec_avg(SHUFFLE_VMX(a2, a3, EVEN_ARGB_VMX),
                     SHUFFLE_VMX(a2, a3, ODD_ARGB_VMX));
    StoreN_VMX(out, 2, dst_argb);
    src_argb += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

void ScaleARGBRowDown2Box_VMX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width) {
  const uint8_t* t = src_argb + src_stride;
  do {
    v8u out[2];
    v8u s0 = Load16_VMX(src_argb);
    v8u s1 = Load16_VMX(src_argb + 16);
    v8u s2 = Load16_VMX(src_argb + 32);
    v8u s3 = Load16_VMX(src_argb + 48);
    v8u t0 = Load16_VMX(t);
    v8u t1 = Load16_VMX(t + 16);
    v8u t2 = Load16_VMX(t + 32);
    v8u t3 = Load16_VMX(t + 48);
    out[0] = BoxARGB_VMX(s0, s1, t0, t1);
    out[1] = BoxARGB_VMX(s2, s3, t2, t3);
    StoreN_VMX(out, 2, dst_argb);
    src_argb += 64;
    t += 64;
    dst_argb += 32;
    dst_width -= 8;
  } while (dst_width > 0);
}

#define DUP_ARGB_LO_VMX 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7
#define DUP_ARGB_HI_VMX \
  8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15

void ScaleARGBColsUp2_VMX(uint8_t* dst_argb,
                          const uint8_t* src_argb,
                          int dst_width,
                          int x,
                          int dx) {
  (void)x;
  (void)dx;
  do {
    v8u out[4];
    v8u s0 = Load16_VMX(src_argb);
    v8u s1 = Load16_VMX(src_argb + 16);
    out[0] = SHUFFLE_VMX(s0, s0, DUP_ARGB_LO_VMX);
    out[1] = SHUFFLE_VMX(s0, s0, DUP_ARGB_HI_VMX);
    out[2] = SHUFFLE_VMX(s1, s1, DUP_ARGB_LO_VMX);
    out[3] = SHUFFLE_VMX(s1, s1, DUP_ARGB_HI_VMX);
    StoreN_VMX(out, 4, dst_argb);
    src_argb += 32;
    dst_argb += 64;
    dst_width -= 16;
  } while (dst_width > 0);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_VMX) && defined(__ALTIVEC__) ...
