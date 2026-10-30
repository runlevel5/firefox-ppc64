/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// This file should only be compiled if the Power ISA target baseline has VSX.

#include <altivec.h>

#include <cstdint>

#include "CharacterDataBufferImpl.h"

namespace mozilla {
namespace VSX {

static inline vector unsigned short Load8(const char16_t* p) {
  return vec_xl(0, reinterpret_cast<const unsigned short*>(p));
}

int32_t FirstNon8Bit(const char16_t* str, const char16_t* end) {
  const uint32_t len = end - str;
  uint32_t i = 0;

  // A unichar is 8-bit iff it is not greater than 0xFF; OR-ing vectors keeps
  // that property, so four unaligned loads need one test.
  const vector unsigned short limit = vec_splats((unsigned short)0xFF);

  const uint32_t quadWalkEnd = (len / 32) * 32;
  for (; i < quadWalkEnd; i += 32) {
    vector unsigned short v0 = Load8(str + i);
    vector unsigned short v1 = Load8(str + i + 8);
    vector unsigned short v2 = Load8(str + i + 16);
    vector unsigned short v3 = Load8(str + i + 24);
    if (vec_any_gt(vec_or(vec_or(v0, v1), vec_or(v2, v3)), limit)) {
      return int32_t(i);
    }
  }

  const uint32_t vectWalkEnd = (len / 8) * 8;
  for (; i < vectWalkEnd; i += 8) {
    if (vec_any_gt(Load8(str + i), limit)) {
      return int32_t(i);
    }
  }

  if (i < len) {
    if (len >= 8) {
      // Fewer than 8 unichars remain: re-check the last 8 of the string, which
      // overlap already-checked ones. The result is only required to be a
      // lower bound of the first non-8-bit index.
      if (vec_any_gt(Load8(end - 8), limit)) {
        return int32_t(len - 8);
      }
    } else {
      for (; i < len; i++) {
        if (str[i] > 255) {
          return int32_t(i);
        }
      }
    }
  }

  return -1;
}

}  // namespace VSX
}  // namespace mozilla
