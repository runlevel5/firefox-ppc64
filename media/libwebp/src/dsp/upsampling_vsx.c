// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VSX (PowerPC) version of YUV to RGB upsampling functions.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_VSX)

#include <altivec.h>
#include <assert.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/dsp/yuv.h"
#include "src/webp/decode.h"
#include "src/webp/types.h"

typedef __vector unsigned char u8x16;
typedef __vector unsigned short u16x8;

// Upsample 16 chroma pairs from rows r1/r2 (17 readable bytes each) into 32
// "top" bytes at out[0..31] and 32 "bottom" bytes at out[64..95], matching the
// fancy-upsampler diagonal weights (a + 3b + 3c + d) / 8 etc.
#define GET_M(ij, in) \
  vec_sub(vec_avg(k, (in)), \
          vec_and(vec_or(vec_and((ij), st), vec_xor(k, (in))), one))

static void Upsample32Pixels(const uint8_t* WEBP_RESTRICT r1,
                             const uint8_t* WEBP_RESTRICT r2,
                             uint8_t* WEBP_RESTRICT out) {
  const u8x16 one = vec_splats((unsigned char)1);
  const u8x16 a = vec_xl(0, (const unsigned char*)r1);
  const u8x16 b = vec_xl(1, (const unsigned char*)r1);
  const u8x16 c = vec_xl(0, (const unsigned char*)r2);
  const u8x16 d = vec_xl(1, (const unsigned char*)r2);
  const u8x16 s = vec_avg(a, d);
  const u8x16 t = vec_avg(b, c);
  const u8x16 st = vec_xor(s, t);
  const u8x16 t3 =
      vec_and(vec_or(vec_or(vec_xor(a, d), vec_xor(b, c)), st), one);
  const u8x16 k = vec_sub(vec_avg(s, t), t3);
  const u8x16 diag1 = GET_M(vec_xor(b, c), t);
  const u8x16 diag2 = GET_M(vec_xor(a, d), s);
  const u8x16 ta = vec_avg(a, diag1), tb = vec_avg(b, diag2);
  const u8x16 tc = vec_avg(c, diag2), td = vec_avg(d, diag1);
  vec_xst(vec_mergeh(ta, tb), 0, out);
  vec_xst(vec_mergel(ta, tb), 0, out + 16);
  vec_xst(vec_mergeh(tc, td), 0, out + 64);
  vec_xst(vec_mergel(tc, td), 0, out + 80);
}

#define UPSAMPLE_FUNC(FUNC_NAME, FUNC, FUNC32)                                 \
static void FUNC_NAME(const uint8_t* WEBP_RESTRICT top_y,                      \
                      const uint8_t* WEBP_RESTRICT bottom_y,                   \
                      const uint8_t* WEBP_RESTRICT top_u,                      \
                      const uint8_t* WEBP_RESTRICT top_v,                      \
                      const uint8_t* WEBP_RESTRICT cur_u,                      \
                      const uint8_t* WEBP_RESTRICT cur_v,                      \
                      uint8_t* WEBP_RESTRICT top_dst,                          \
                      uint8_t* WEBP_RESTRICT bottom_dst, int len) {            \
  int uv_pos, pos;                                                            \
  uint8_t uv_buf[14 * 32 + 15] = {0};                                         \
  uint8_t* const r_u = (uint8_t*)(((uintptr_t)(uv_buf + 15)) & ~(uintptr_t)15);\
  uint8_t* const r_v = r_u + 32;                                              \
  assert(top_y != NULL);                                                      \
  {                                                                           \
    const int u_diag = ((top_u[0] + cur_u[0]) >> 1) + 1;                      \
    const int v_diag = ((top_v[0] + cur_v[0]) >> 1) + 1;                      \
    FUNC(top_y[0], (top_u[0] + u_diag) >> 1, (top_v[0] + v_diag) >> 1,        \
         top_dst);                                                           \
    if (bottom_y != NULL) {                                                   \
      FUNC(bottom_y[0], (cur_u[0] + u_diag) >> 1, (cur_v[0] + v_diag) >> 1,   \
           bottom_dst);                                                      \
    }                                                                         \
  }                                                                           \
  for (pos = 1, uv_pos = 0; pos + 32 + 1 <= len; pos += 32, uv_pos += 16) {   \
    Upsample32Pixels(top_u + uv_pos, cur_u + uv_pos, r_u);                    \
    Upsample32Pixels(top_v + uv_pos, cur_v + uv_pos, r_v);                    \
    FUNC32(top_y + pos, r_u, r_v, top_dst + pos * 4);                         \
    if (bottom_y != NULL) {                                                   \
      FUNC32(bottom_y + pos, r_u + 64, r_v + 64, bottom_dst + pos * 4);       \
    }                                                                         \
  }                                                                           \
  if (len > 1) {                                                              \
    const int left_over = ((len + 1) >> 1) - (pos >> 1);                      \
    uint8_t* const tmp_top_dst = r_u + 4 * 32;                                \
    uint8_t* const tmp_bottom_dst = tmp_top_dst + 4 * 32;                     \
    uint8_t* const tmp_top = tmp_bottom_dst + 4 * 32;                         \
    uint8_t* const tmp_bottom = (bottom_y == NULL) ? NULL : tmp_top + 32;     \
    uint8_t r1[17], r2[17];                                                   \
    assert(left_over > 0);                                                    \
    memcpy(r1, top_u + uv_pos, left_over);                                    \
    memcpy(r2, cur_u + uv_pos, left_over);                                    \
    memset(r1 + left_over, r1[left_over - 1], 17 - left_over);                \
    memset(r2 + left_over, r2[left_over - 1], 17 - left_over);                \
    Upsample32Pixels(r1, r2, r_u);                                            \
    memcpy(r1, top_v + uv_pos, left_over);                                    \
    memcpy(r2, cur_v + uv_pos, left_over);                                    \
    memset(r1 + left_over, r1[left_over - 1], 17 - left_over);                \
    memset(r2 + left_over, r2[left_over - 1], 17 - left_over);                \
    Upsample32Pixels(r1, r2, r_v);                                            \
    memcpy(tmp_top, top_y + pos, len - pos);                                  \
    if (bottom_y != NULL) memcpy(tmp_bottom, bottom_y + pos, len - pos);      \
    FUNC32(tmp_top, r_u, r_v, tmp_top_dst);                                   \
    if (bottom_y != NULL) FUNC32(tmp_bottom, r_u + 64, r_v + 64,             \
                                 tmp_bottom_dst);                            \
    memcpy(top_dst + pos * 4, tmp_top_dst, (len - pos) * 4);                  \
    if (bottom_y != NULL) {                                                   \
      memcpy(bottom_dst + pos * 4, tmp_bottom_dst, (len - pos) * 4);          \
    }                                                                         \
  }                                                                           \
}

UPSAMPLE_FUNC(UpsampleRgbaLinePair_VSX, VP8YuvToRgba, VP8YuvToRgba32_VSX)
UPSAMPLE_FUNC(UpsampleBgraLinePair_VSX, VP8YuvToBgra, VP8YuvToBgra32_VSX)
UPSAMPLE_FUNC(UpsampleArgbLinePair_VSX, VP8YuvToArgb, VP8YuvToArgb32_VSX)

extern WebPUpsampleLinePairFunc WebPUpsamplers[/* MODE_LAST */];

extern void WebPInitUpsamplersVSX(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitUpsamplersVSX(void) {
  WebPUpsamplers[MODE_RGBA] = UpsampleRgbaLinePair_VSX;
  WebPUpsamplers[MODE_BGRA] = UpsampleBgraLinePair_VSX;
  WebPUpsamplers[MODE_rgbA] = UpsampleRgbaLinePair_VSX;
  WebPUpsamplers[MODE_bgrA] = UpsampleBgraLinePair_VSX;
#if !defined(WEBP_REDUCE_CSP)
  WebPUpsamplers[MODE_ARGB] = UpsampleArgbLinePair_VSX;
  WebPUpsamplers[MODE_Argb] = UpsampleArgbLinePair_VSX;
#endif
}

extern void WebPInitYUV444ConvertersVSX(void);

// YUV444 point converters stay on the C path for now.
WEBP_TSAN_IGNORE_FUNCTION void WebPInitYUV444ConvertersVSX(void) {}

#else  // !WEBP_USE_VSX

WEBP_DSP_INIT_STUB(WebPInitYUV444ConvertersVSX)

WEBP_DSP_INIT_STUB(WebPInitUpsamplersVSX)

#endif  // WEBP_USE_VSX
