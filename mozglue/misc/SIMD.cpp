/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "mozilla/SIMD.h"

#include <bit>
#include <algorithm>
#include <cstring>
#include <stdint.h>
#include <type_traits>

#include "mozilla/Assertions.h"
#include "mozilla/SSE.h"

#ifdef MOZILLA_PRESUME_SSE2
#  include <immintrin.h>
#elif defined(__ALTIVEC__)
#  include <altivec.h>
#  undef vector
#  undef pixel
#  undef bool
#endif

namespace mozilla {

template <typename TValue>
const TValue* FindInBufferNaive(const TValue* ptr, TValue value,
                                size_t length) {
  const TValue* end = ptr + length;
  while (ptr < end) {
    if (*ptr == value) {
      return ptr;
    }
    ptr++;
  }
  return nullptr;
}

template <typename T>
T GetAs(uintptr_t ptr) {
  return *reinterpret_cast<const T*>(ptr);
}

// Akin to ceil/floor, AlignDown/AlignUp will return the original pointer if it
// is already aligned.
uintptr_t AlignDown16(uintptr_t ptr) { return ptr & ~0xf; }

uintptr_t AlignUp16(uintptr_t ptr) { return AlignDown16(ptr + 0xf); }

enum class HaystackOverlap {
  Overlapping,
  Sequential,
};

#ifdef MOZILLA_PRESUME_SSE2

const __m128i* Cast128(uintptr_t ptr) {
  return reinterpret_cast<const __m128i*>(ptr);
}

template <typename TValue>
__m128i CmpEq128(__m128i a, __m128i b) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2);
  if (sizeof(TValue) == 1) {
    return _mm_cmpeq_epi8(a, b);
  }
  return _mm_cmpeq_epi16(a, b);
}

#  ifdef __GNUC__

// Earlier versions of GCC are missing the _mm_loadu_si32 instruction. This
// workaround from Peter Cordes (https://stackoverflow.com/a/72837992) compiles
// down to the same instructions. We could just replace _mm_loadu_si32
__m128i Load32BitsIntoXMM(uintptr_t ptr) {
  int tmp;
  memcpy(&tmp, reinterpret_cast<const void*>(ptr),
         sizeof(tmp));            // unaligned aliasing-safe load
  return _mm_cvtsi32_si128(tmp);  // efficient on GCC/clang/MSVC
}

#  else

__m128i Load32BitsIntoXMM(uintptr_t ptr) {
  return _mm_loadu_si32(Cast128(ptr));
}

#  endif

const char* Check4x4Chars(__m128i needle, uintptr_t a, uintptr_t b, uintptr_t c,
                          uintptr_t d) {
  __m128i haystackA = Load32BitsIntoXMM(a);
  __m128i cmpA = CmpEq128<char>(needle, haystackA);
  __m128i haystackB = Load32BitsIntoXMM(b);
  __m128i cmpB = CmpEq128<char>(needle, haystackB);
  __m128i haystackC = Load32BitsIntoXMM(c);
  __m128i cmpC = CmpEq128<char>(needle, haystackC);
  __m128i haystackD = Load32BitsIntoXMM(d);
  __m128i cmpD = CmpEq128<char>(needle, haystackD);
  __m128i or_ab = _mm_or_si128(cmpA, cmpB);
  __m128i or_cd = _mm_or_si128(cmpC, cmpD);
  __m128i or_abcd = _mm_or_si128(or_ab, or_cd);
  int orMask = _mm_movemask_epi8(or_abcd);
  if (orMask & 0xf) {
    int cmpMask;
    cmpMask = _mm_movemask_epi8(cmpA);
    if (cmpMask & 0xf) {
      return reinterpret_cast<const char*>(a + __builtin_ctz(cmpMask));
    }
    cmpMask = _mm_movemask_epi8(cmpB);
    if (cmpMask & 0xf) {
      return reinterpret_cast<const char*>(b + __builtin_ctz(cmpMask));
    }
    cmpMask = _mm_movemask_epi8(cmpC);
    if (cmpMask & 0xf) {
      return reinterpret_cast<const char*>(c + __builtin_ctz(cmpMask));
    }
    cmpMask = _mm_movemask_epi8(cmpD);
    if (cmpMask & 0xf) {
      return reinterpret_cast<const char*>(d + __builtin_ctz(cmpMask));
    }
  }

  return nullptr;
}

template <typename TValue>
const TValue* Check4x16Bytes(__m128i needle, uintptr_t a, uintptr_t b,
                             uintptr_t c, uintptr_t d) {
  __m128i haystackA = _mm_loadu_si128(Cast128(a));
  __m128i cmpA = CmpEq128<TValue>(needle, haystackA);
  __m128i haystackB = _mm_loadu_si128(Cast128(b));
  __m128i cmpB = CmpEq128<TValue>(needle, haystackB);
  __m128i haystackC = _mm_loadu_si128(Cast128(c));
  __m128i cmpC = CmpEq128<TValue>(needle, haystackC);
  __m128i haystackD = _mm_loadu_si128(Cast128(d));
  __m128i cmpD = CmpEq128<TValue>(needle, haystackD);
  __m128i or_ab = _mm_or_si128(cmpA, cmpB);
  __m128i or_cd = _mm_or_si128(cmpC, cmpD);
  __m128i or_abcd = _mm_or_si128(or_ab, or_cd);
  int orMask = _mm_movemask_epi8(or_abcd);
  if (orMask) {
    int cmpMask;
    cmpMask = _mm_movemask_epi8(cmpA);
    if (cmpMask) {
      return reinterpret_cast<const TValue*>(a + __builtin_ctz(cmpMask));
    }
    cmpMask = _mm_movemask_epi8(cmpB);
    if (cmpMask) {
      return reinterpret_cast<const TValue*>(b + __builtin_ctz(cmpMask));
    }
    cmpMask = _mm_movemask_epi8(cmpC);
    if (cmpMask) {
      return reinterpret_cast<const TValue*>(c + __builtin_ctz(cmpMask));
    }
    cmpMask = _mm_movemask_epi8(cmpD);
    if (cmpMask) {
      return reinterpret_cast<const TValue*>(d + __builtin_ctz(cmpMask));
    }
  }

  return nullptr;
}

// Check two 16-byte chunks for the two-byte sequence loaded into needle1
// followed by needle1. `carryOut` is an optional pointer which we will
// populate based on whether the last character of b matches needle1. This
// should be provided on subsequent calls via `carryIn` so we can detect cases
// where the last byte of b's 16-byte chunk is needle1 and the first byte of
// the next a's 16-byte chunk is needle2. `overlap` and whether
// `carryIn`/`carryOut` are NULL should be knowable at compile time to avoid
// branching.
template <typename TValue>
const TValue* Check2x2x16Bytes(__m128i needle1, __m128i needle2, uintptr_t a,
                               uintptr_t b, __m128i* carryIn, __m128i* carryOut,
                               HaystackOverlap overlap) {
  const int shiftRightAmount = 16 - sizeof(TValue);
  const int shiftLeftAmount = sizeof(TValue);
  __m128i haystackA = _mm_loadu_si128(Cast128(a));
  __m128i cmpA1 = CmpEq128<TValue>(needle1, haystackA);
  __m128i cmpA2 = CmpEq128<TValue>(needle2, haystackA);
  __m128i cmpA;
  if (carryIn) {
    cmpA = _mm_and_si128(
        _mm_or_si128(_mm_bslli_si128(cmpA1, shiftLeftAmount), *carryIn), cmpA2);
  } else {
    cmpA = _mm_and_si128(_mm_bslli_si128(cmpA1, shiftLeftAmount), cmpA2);
  }
  __m128i haystackB = _mm_loadu_si128(Cast128(b));
  __m128i cmpB1 = CmpEq128<TValue>(needle1, haystackB);
  __m128i cmpB2 = CmpEq128<TValue>(needle2, haystackB);
  __m128i cmpB;
  if (overlap == HaystackOverlap::Overlapping) {
    cmpB = _mm_and_si128(_mm_bslli_si128(cmpB1, shiftLeftAmount), cmpB2);
  } else {
    MOZ_ASSERT(overlap == HaystackOverlap::Sequential);
    __m128i carryAB = _mm_bsrli_si128(cmpA1, shiftRightAmount);
    cmpB = _mm_and_si128(
        _mm_or_si128(_mm_bslli_si128(cmpB1, shiftLeftAmount), carryAB), cmpB2);
  }
  __m128i or_ab = _mm_or_si128(cmpA, cmpB);
  int orMask = _mm_movemask_epi8(or_ab);
  if (orMask) {
    int cmpMask;
    cmpMask = _mm_movemask_epi8(cmpA);
    if (cmpMask) {
      return reinterpret_cast<const TValue*>(a + __builtin_ctz(cmpMask) -
                                             shiftLeftAmount);
    }
    cmpMask = _mm_movemask_epi8(cmpB);
    if (cmpMask) {
      return reinterpret_cast<const TValue*>(b + __builtin_ctz(cmpMask) -
                                             shiftLeftAmount);
    }
  }

  if (carryOut) {
    _mm_store_si128(carryOut, _mm_bsrli_si128(cmpB1, shiftRightAmount));
  }

  return nullptr;
}

template <typename TValue>
const TValue* FindInBuffer(const TValue* ptr, TValue value, size_t length) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2);
  static_assert(std::is_unsigned<TValue>::value);
  uint64_t splat64;
  if (sizeof(TValue) == 1) {
    splat64 = 0x0101010101010101llu;
  } else {
    splat64 = 0x0001000100010001llu;
  }

  // Load our needle into a 16-byte register
  uint64_t u64_value = static_cast<uint64_t>(value) * splat64;
  int64_t i64_value = *reinterpret_cast<int64_t*>(&u64_value);
  __m128i needle = _mm_set_epi64x(i64_value, i64_value);

  size_t numBytes = length * sizeof(TValue);
  uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = cur + numBytes;

  if ((sizeof(TValue) > 1 && numBytes < 16) || numBytes < 4) {
    while (cur < end) {
      if (GetAs<TValue>(cur) == value) {
        return reinterpret_cast<const TValue*>(cur);
      }
      cur += sizeof(TValue);
    }
    return nullptr;
  }

  if (numBytes < 16) {
    // NOTE: here and below, we have some bit fiddling which could look a
    // little weird. The important thing to note though is it's just a trick
    // for getting the number 4 if numBytes is greater than or equal to 8,
    // and 0 otherwise. This lets us fully cover the range without any
    // branching for the case where numBytes is in [4,8), and [8,16). We get
    // four ranges from this - if numbytes > 8, we get:
    //   [0,4), [4,8], [end - 8), [end - 4)
    // and if numbytes < 8, we get
    //   [0,4), [0,4), [end - 4), [end - 4)
    uintptr_t a = cur;
    uintptr_t b = cur + ((numBytes & 8) >> 1);
    uintptr_t c = end - 4 - ((numBytes & 8) >> 1);
    uintptr_t d = end - 4;
    const char* charResult = Check4x4Chars(needle, a, b, c, d);
    // Note: we ensure above that sizeof(TValue) == 1 here, so this is
    // either char to char or char to something like a uint8_t.
    return reinterpret_cast<const TValue*>(charResult);
  }

  if (numBytes < 64) {
    // NOTE: see the above explanation of the similar chunk of code, but in
    // this case, replace 8 with 32 and 4 with 16.
    uintptr_t a = cur;
    uintptr_t b = cur + ((numBytes & 32) >> 1);
    uintptr_t c = end - 16 - ((numBytes & 32) >> 1);
    uintptr_t d = end - 16;
    return Check4x16Bytes<TValue>(needle, a, b, c, d);
  }

  // Get the initial unaligned load out of the way. This will overlap with the
  // aligned stuff below, but the overlapped part should effectively be free
  // (relative to a mispredict from doing a byte-by-byte loop).
  __m128i haystack = _mm_loadu_si128(Cast128(cur));
  __m128i cmp = CmpEq128<TValue>(needle, haystack);
  int cmpMask = _mm_movemask_epi8(cmp);
  if (cmpMask) {
    return reinterpret_cast<const TValue*>(cur + __builtin_ctz(cmpMask));
  }

  // Now we're working with aligned memory. Hooray! \o/
  cur = AlignUp16(cur);

  // The address of the final 48-63 bytes. We overlap this with what we check in
  // our hot loop below to avoid branching. Again, the overlap should be
  // negligible compared with a branch mispredict.
  uintptr_t tailStartPtr = AlignDown16(end - 48);
  uintptr_t tailEndPtr = end - 16;

  while (cur < tailStartPtr) {
    uintptr_t a = cur;
    uintptr_t b = cur + 16;
    uintptr_t c = cur + 32;
    uintptr_t d = cur + 48;
    const TValue* result = Check4x16Bytes<TValue>(needle, a, b, c, d);
    if (result) {
      return result;
    }
    cur += 64;
  }

  uintptr_t a = tailStartPtr;
  uintptr_t b = tailStartPtr + 16;
  uintptr_t c = tailStartPtr + 32;
  uintptr_t d = tailEndPtr;
  return Check4x16Bytes<TValue>(needle, a, b, c, d);
}

template <typename TValue>
const TValue* TwoElementLoop(uintptr_t start, uintptr_t end, TValue v1,
                             TValue v2) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2);

  const TValue* cur = reinterpret_cast<const TValue*>(start);
  const TValue* preEnd = reinterpret_cast<const TValue*>(end - sizeof(TValue));

  uint32_t expected = static_cast<uint32_t>(v1) |
                      (static_cast<uint32_t>(v2) << (sizeof(TValue) * 8));
  while (cur < preEnd) {
    // NOTE: this should only ever be called on little endian architectures.
    static_assert(std::endian::native == std::endian::little);
    // We or cur[0] and cur[1] together explicitly and compare to expected,
    // in order to avoid UB from just loading them as a uint16_t/uint32_t.
    // However, it will compile down the same code after optimizations on
    // little endian systems which support unaligned loads. Comparing them
    // value-by-value, however, will not, and seems to perform worse in local
    // microbenchmarking. Even after bitwise or'ing the comparison values
    // together to avoid the short circuit, the compiler doesn't seem to get
    // the hint and creates two branches, the first of which might be
    // frequently mispredicted.
    uint32_t actual = static_cast<uint32_t>(cur[0]) |
                      (static_cast<uint32_t>(cur[1]) << (sizeof(TValue) * 8));
    if (actual == expected) {
      return cur;
    }
    cur++;
  }
  return nullptr;
}

template <typename TValue>
const TValue* FindTwoInBuffer(const TValue* ptr, TValue v1, TValue v2,
                              size_t length) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2);
  static_assert(std::is_unsigned<TValue>::value);
  uint64_t splat64;
  if (sizeof(TValue) == 1) {
    splat64 = 0x0101010101010101llu;
  } else {
    splat64 = 0x0001000100010001llu;
  }

  // Load our needle into a 16-byte register
  uint64_t u64_v1 = static_cast<uint64_t>(v1) * splat64;
  int64_t i64_v1 = *reinterpret_cast<int64_t*>(&u64_v1);
  __m128i needle1 = _mm_set_epi64x(i64_v1, i64_v1);
  uint64_t u64_v2 = static_cast<uint64_t>(v2) * splat64;
  int64_t i64_v2 = *reinterpret_cast<int64_t*>(&u64_v2);
  __m128i needle2 = _mm_set_epi64x(i64_v2, i64_v2);

  size_t numBytes = length * sizeof(TValue);
  uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = cur + numBytes;

  if (numBytes < 16) {
    return TwoElementLoop<TValue>(cur, end, v1, v2);
  }

  if (numBytes < 32) {
    uintptr_t a = cur;
    uintptr_t b = end - 16;
    return Check2x2x16Bytes<TValue>(needle1, needle2, a, b, nullptr, nullptr,
                                    HaystackOverlap::Overlapping);
  }

  // Get the initial unaligned load out of the way. This will likely overlap
  // with the aligned stuff below, but the overlapped part should effectively
  // be free.
  __m128i haystack = _mm_loadu_si128(Cast128(cur));
  __m128i cmp1 = CmpEq128<TValue>(needle1, haystack);
  __m128i cmp2 = CmpEq128<TValue>(needle2, haystack);
  int cmpMask1 = _mm_movemask_epi8(cmp1);
  int cmpMask2 = _mm_movemask_epi8(cmp2);
  int cmpMask = (cmpMask1 << sizeof(TValue)) & cmpMask2;
  if (cmpMask) {
    return reinterpret_cast<const TValue*>(cur + __builtin_ctz(cmpMask) -
                                           sizeof(TValue));
  }

  // Now we're working with aligned memory. Hooray! \o/
  cur = AlignUp16(cur);

  // The address of the final 48-63 bytes. We overlap this with what we check in
  // our hot loop below to avoid branching. Again, the overlap should be
  // negligible compared with a branch mispredict.
  uintptr_t tailEndPtr = end - 16;
  uintptr_t tailStartPtr = AlignDown16(tailEndPtr);

  __m128i cmpMaskCarry = _mm_set1_epi32(0);
  while (cur < tailStartPtr) {
    uintptr_t a = cur;
    uintptr_t b = cur + 16;
    const TValue* result =
        Check2x2x16Bytes<TValue>(needle1, needle2, a, b, &cmpMaskCarry,
                                 &cmpMaskCarry, HaystackOverlap::Sequential);
    if (result) {
      return result;
    }
    cur += 32;
  }

  uint32_t carry = (cur == tailStartPtr) ? 0xffffffff : 0;
  __m128i wideCarry = Load32BitsIntoXMM(reinterpret_cast<uintptr_t>(&carry));
  cmpMaskCarry = _mm_and_si128(cmpMaskCarry, wideCarry);
  uintptr_t a = tailStartPtr;
  uintptr_t b = tailEndPtr;
  return Check2x2x16Bytes<TValue>(needle1, needle2, a, b, &cmpMaskCarry,
                                  nullptr, HaystackOverlap::Overlapping);
}

const char* SIMD::memchr8SSE2(const char* ptr, char value, size_t length) {
  // Signed chars are just really annoying to do bit logic with. Convert to
  // unsigned at the outermost scope so we don't have to worry about it.
  const unsigned char* uptr = reinterpret_cast<const unsigned char*>(ptr);
  unsigned char uvalue = static_cast<unsigned char>(value);
  const unsigned char* uresult =
      FindInBuffer<unsigned char>(uptr, uvalue, length);
  return reinterpret_cast<const char*>(uresult);
}

// So, this is a bit awkward. It generally simplifies things if we can just
// assume all the AVX2 code is 64-bit, so we have this preprocessor guard
// in SIMD_avx2 over all of its actual code, and it also defines versions
// of its endpoints that just assert false if the guard is not satisfied.
// A 32 bit processor could implement the AVX2 instruction set though, which
// would result in it passing the supports_avx2() check and landing in an
// assertion failure. Accordingly, we just don't allow that to happen. We
// are not particularly concerned about ensuring that newer 32 bit processors
// get access to the AVX2 functions exposed here.
#  if defined(MOZILLA_MAY_SUPPORT_AVX2) && defined(__x86_64__)

bool SupportsAVX2() { return supports_avx2(); }

#  else

bool SupportsAVX2() { return false; }

#  endif

const char* SIMD::memchr8(const char* ptr, char value, size_t length) {
  if (SupportsAVX2()) {
    return memchr8AVX2(ptr, value, length);
  }
  return memchr8SSE2(ptr, value, length);
}

const char16_t* SIMD::memchr16SSE2(const char16_t* ptr, char16_t value,
                                   size_t length) {
  return FindInBuffer<char16_t>(ptr, value, length);
}

const char16_t* SIMD::memchr16(const char16_t* ptr, char16_t value,
                               size_t length) {
  if (SupportsAVX2()) {
    return memchr16AVX2(ptr, value, length);
  }
  return memchr16SSE2(ptr, value, length);
}

const uint32_t* SIMD::memchr32(const uint32_t* ptr, uint32_t value,
                               size_t length) {
  if (SupportsAVX2()) {
    return memchr32AVX2(ptr, value, length);
  }
  return FindInBufferNaive<uint32_t>(ptr, value, length);
}

const uint64_t* SIMD::memchr64(const uint64_t* ptr, uint64_t value,
                               size_t length) {
  if (SupportsAVX2()) {
    return memchr64AVX2(ptr, value, length);
  }
  return FindInBufferNaive<uint64_t>(ptr, value, length);
}

const char* SIMD::memchr2x8(const char* ptr, char v1, char v2, size_t length) {
  // Signed chars are just really annoying to do bit logic with. Convert to
  // unsigned at the outermost scope so we don't have to worry about it.
  const unsigned char* uptr = reinterpret_cast<const unsigned char*>(ptr);
  unsigned char uv1 = static_cast<unsigned char>(v1);
  unsigned char uv2 = static_cast<unsigned char>(v2);
  const unsigned char* uresult =
      FindTwoInBuffer<unsigned char>(uptr, uv1, uv2, length);
  return reinterpret_cast<const char*>(uresult);
}

const char16_t* SIMD::memchr2x16(const char16_t* ptr, char16_t v1, char16_t v2,
                                 size_t length) {
  return FindTwoInBuffer<char16_t>(ptr, v1, v2, length);
}

#elif defined(__ALTIVEC__)

// Power ISA. The structure follows the SSE2 path. Which instructions back it
// is decided by the compile-time baseline:
//  - POWER9 and POWER10 (_ARCH_PWR9): vclzlsbb gives the index of the first
//    matching element in one instruction.
//  - POWER8 (__POWER8_VECTOR__): the compare result is moved to GPRs with
//    mfvsrd and scanned with cnttzd/cntlzd; 64-bit lanes are available.
//  - VMX only (PPC970/G5, big-endian): no unaligned vector loads, so the
//    unaligned head and tail of a buffer are handled with scalar code and only
//    the aligned middle is vectorized; no 64-bit lanes.
// Lane 0 of a vector holds the lowest address on both endiannesses, but the
// lowest address is the least significant byte of a 64-bit lane on LE and the
// most significant on BE, and vec_sld moves bytes in opposite directions.

typedef __vector unsigned char VecU8;

#  if defined(__VSX__)
#    define MOZ_PPC_UNALIGNED_LOADS 1
#  endif
#  if defined(__POWER8_VECTOR__)
#    define MOZ_PPC_VEC64 1
#  endif

template <typename TValue>
VecU8 SplatNeedle(TValue value) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2 ||
                sizeof(TValue) == 4 || sizeof(TValue) == 8);
  if constexpr (sizeof(TValue) == 1) {
    return vec_splats(static_cast<unsigned char>(value));
  } else if constexpr (sizeof(TValue) == 2) {
    return (VecU8)vec_splats(static_cast<unsigned short>(value));
  } else if constexpr (sizeof(TValue) == 4) {
    return (VecU8)vec_splats(static_cast<unsigned int>(value));
  } else {
#  ifdef MOZ_PPC_VEC64
    return (VecU8)vec_splats(static_cast<unsigned long long>(value));
#  else
    MOZ_CRASH("64-bit lanes need VSX");
#  endif
  }
}

template <typename TValue>
VecU8 CmpEq128(VecU8 a, VecU8 b) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2 ||
                sizeof(TValue) == 4 || sizeof(TValue) == 8);
  if constexpr (sizeof(TValue) == 1) {
    return (VecU8)vec_cmpeq(a, b);
  } else if constexpr (sizeof(TValue) == 2) {
    return (VecU8)vec_cmpeq((__vector unsigned short)a,
                            (__vector unsigned short)b);
  } else if constexpr (sizeof(TValue) == 4) {
    return (VecU8)vec_cmpeq((__vector unsigned int)a, (__vector unsigned int)b);
  } else {
#  ifdef MOZ_PPC_VEC64
    return (VecU8)vec_cmpeq((__vector unsigned long long)a,
                            (__vector unsigned long long)b);
#  else
    MOZ_CRASH("64-bit lanes need VSX");
#  endif
  }
}

// On VMX-only targets every caller passes a 16-byte aligned address.
VecU8 LoadVec(uintptr_t ptr) {
#  ifdef MOZ_PPC_UNALIGNED_LOADS
  return vec_xl(0, reinterpret_cast<const unsigned char*>(ptr));
#  else
  MOZ_ASSERT((ptr & 0xf) == 0);
  return vec_ld(0, reinterpret_cast<const unsigned char*>(ptr));
#  endif
}

bool AnyMatch(VecU8 cmp) {
  return vec_any_ne(cmp, vec_splats(static_cast<unsigned char>(0)));
}

// Index of the lowest-address byte that is set in two 64-bit lanes holding
// bytes 0-7 and 8-15, or 16 if none is.
int FirstMatchIndex64(uint64_t lo, uint64_t hi) {
  uint64_t lane = lo ? lo : hi;
  if (!lane) {
    return 16;
  }
  int base = lo ? 0 : 8;
#  ifdef __LITTLE_ENDIAN__
  return base + (__builtin_ctzll(lane) >> 3);
#  else
  return base + (__builtin_clzll(lane) >> 3);
#  endif
}

// Index of the lowest-address byte of the first match in a compare result, or
// 16 if there is none.
int FirstMatchIndex(VecU8 cmp) {
#  if defined(_ARCH_PWR9)
  return vec_cntlz_lsbb(cmp);
#  elif defined(MOZ_PPC_VEC64)
  // vbpermq gathers the top bit of each byte, lowest address first, into the
  // low 16 bits of the big-endian first doubleword; the sentinel makes the
  // count 16 when nothing matched.
  const VecU8 bitIndex = {0,  8,  16, 24, 32, 40,  48,  56,
                          64, 72, 80, 88, 96, 104, 112, 120};
  __vector unsigned long long mask = vec_vbpermq(cmp, bitIndex);
#    ifdef __LITTLE_ENDIAN__
  uint32_t bits = static_cast<uint32_t>(vec_extract(mask, 1));
#    else
  uint32_t bits = static_cast<uint32_t>(vec_extract(mask, 0));
#    endif
  return __builtin_clz((bits << 16) | 0x8000);
#  else
  uint64_t lanes[2] __attribute__((aligned(16)));
  vec_st(cmp, 0, reinterpret_cast<unsigned char*>(lanes));
  return FirstMatchIndex64(lanes[0], lanes[1]);
#  endif
}

// Equivalents of _mm_bslli_si128 and _mm_bsrli_si128 in memory order: shift
// bytes toward higher (respectively lower) addresses, filling with zero.
template <int N>
VecU8 BSlliN(VecU8 x) {
  static_assert(N > 0 && N < 16);
  VecU8 zero = vec_splats(static_cast<unsigned char>(0));
#  ifdef __LITTLE_ENDIAN__
  return vec_sld(x, zero, N);
#  else
  return vec_sld(zero, x, 16 - N);
#  endif
}

template <int N>
VecU8 BSrliN(VecU8 x) {
  static_assert(N > 0 && N < 16);
  VecU8 zero = vec_splats(static_cast<unsigned char>(0));
#  ifdef __LITTLE_ENDIAN__
  return vec_sld(zero, x, 16 - N);
#  else
  return vec_sld(x, zero, N);
#  endif
}

// GCC on little-endian POWER8 picks lvx for loads it can prove aligned and
// then spills every vector through the stack to recover memory order for
// FirstMatchIndex. Hiding the alignment keeps it on lxvd2x and in registers.
MOZ_ALWAYS_INLINE uintptr_t HideAlignment(uintptr_t ptr) {
#  if defined(__GNUC__) && !defined(__clang__) && \
      defined(__LITTLE_ENDIAN__) && !defined(_ARCH_PWR9)
  __asm__("" : "+r"(ptr));
#  endif
  return ptr;
}

// Index of the first zero byte of v (lowest address first), or 8 if none.
int FirstZeroByte(uint64_t v) {
  constexpr uint64_t k7f = 0x7f7f7f7f7f7f7f7fllu;
  uint64_t t = ~(((v & k7f) + k7f) | v | k7f);
  if (!t) {
    return 8;
  }
#  ifdef __LITTLE_ENDIAN__
  return __builtin_ctzll(t) >> 3;
#  else
  return __builtin_clzll(t) >> 3;
#  endif
}

// Two possibly overlapping 8-byte windows, for 8-byte to 15-byte buffers.
const unsigned char* Check2x8Bytes(uint64_t needle64, uintptr_t a,
                                   uintptr_t b) {
  uint64_t chunk;
  memcpy(&chunk, reinterpret_cast<const void*>(a), sizeof(chunk));
  int idx = FirstZeroByte(chunk ^ needle64);
  if (idx != 8) {
    return reinterpret_cast<const unsigned char*>(a + idx);
  }
  memcpy(&chunk, reinterpret_cast<const void*>(b), sizeof(chunk));
  idx = FirstZeroByte(chunk ^ needle64);
  if (idx != 8) {
    return reinterpret_cast<const unsigned char*>(b + idx);
  }
  return nullptr;
}

template <typename TValue>
MOZ_ALWAYS_INLINE const TValue* Check16Bytes(VecU8 needle, uintptr_t a) {
  VecU8 cmp = CmpEq128<TValue>(needle, LoadVec(a));
  if (AnyMatch(cmp)) {
    return reinterpret_cast<const TValue*>(a + FirstMatchIndex(cmp));
  }
  return nullptr;
}

template <typename TValue>
MOZ_ALWAYS_INLINE const TValue* Check4x16Bytes(VecU8 needle, uintptr_t a,
                                               uintptr_t b, uintptr_t c,
                                               uintptr_t d) {
  VecU8 cmpA = CmpEq128<TValue>(needle, LoadVec(a));
  VecU8 cmpB = CmpEq128<TValue>(needle, LoadVec(b));
  VecU8 cmpC = CmpEq128<TValue>(needle, LoadVec(c));
  VecU8 cmpD = CmpEq128<TValue>(needle, LoadVec(d));
  VecU8 or_ab = vec_or(cmpA, cmpB);
  VecU8 or_cd = vec_or(cmpC, cmpD);
  if (AnyMatch(vec_or(or_ab, or_cd))) {
    int idx = FirstMatchIndex(cmpA);
    if (idx != 16) {
      return reinterpret_cast<const TValue*>(a + idx);
    }
    idx = FirstMatchIndex(cmpB);
    if (idx != 16) {
      return reinterpret_cast<const TValue*>(b + idx);
    }
    idx = FirstMatchIndex(cmpC);
    if (idx != 16) {
      return reinterpret_cast<const TValue*>(c + idx);
    }
    return reinterpret_cast<const TValue*>(d + FirstMatchIndex(cmpD));
  }
  return nullptr;
}

// See the SSE2 Check2x2x16Bytes for an explanation of the carry handling.
template <typename TValue>
MOZ_ALWAYS_INLINE const TValue* Check2x2x16Bytes(VecU8 needle1, VecU8 needle2,
                                                 uintptr_t a, uintptr_t b,
                                                 VecU8* carryIn,
                                                 VecU8* carryOut,
                                                 HaystackOverlap overlap) {
  constexpr int shiftRightAmount = 16 - sizeof(TValue);
  constexpr int shiftLeftAmount = sizeof(TValue);
  VecU8 haystackA = LoadVec(a);
  VecU8 cmpA1 = CmpEq128<TValue>(needle1, haystackA);
  VecU8 cmpA2 = CmpEq128<TValue>(needle2, haystackA);
  VecU8 cmpA;
  if (carryIn) {
    cmpA = vec_and(vec_or(BSlliN<shiftLeftAmount>(cmpA1), *carryIn), cmpA2);
  } else {
    cmpA = vec_and(BSlliN<shiftLeftAmount>(cmpA1), cmpA2);
  }
  VecU8 haystackB = LoadVec(b);
  VecU8 cmpB1 = CmpEq128<TValue>(needle1, haystackB);
  VecU8 cmpB2 = CmpEq128<TValue>(needle2, haystackB);
  VecU8 cmpB;
  if (overlap == HaystackOverlap::Overlapping) {
    cmpB = vec_and(BSlliN<shiftLeftAmount>(cmpB1), cmpB2);
  } else {
    MOZ_ASSERT(overlap == HaystackOverlap::Sequential);
    VecU8 carryAB = BSrliN<shiftRightAmount>(cmpA1);
    cmpB = vec_and(vec_or(BSlliN<shiftLeftAmount>(cmpB1), carryAB), cmpB2);
  }
  if (AnyMatch(vec_or(cmpA, cmpB))) {
    int idx = FirstMatchIndex(cmpA);
    if (idx != 16) {
      return reinterpret_cast<const TValue*>(a + idx - shiftLeftAmount);
    }
    return reinterpret_cast<const TValue*>(b + FirstMatchIndex(cmpB) -
                                           shiftLeftAmount);
  }

  if (carryOut) {
    *carryOut = BSrliN<shiftRightAmount>(cmpB1);
  }

  return nullptr;
}

template <typename TValue>
const TValue* TwoElementLoop(uintptr_t start, uintptr_t end, TValue v1,
                             TValue v2) {
  const TValue* cur = reinterpret_cast<const TValue*>(start);
  const TValue* preEnd = reinterpret_cast<const TValue*>(end - sizeof(TValue));
  while (cur < preEnd) {
    if (((cur[0] ^ v1) | (cur[1] ^ v2)) == 0) {
      return cur;
    }
    cur++;
  }
  return nullptr;
}

template <typename TValue>
const TValue* FindInBuffer(const TValue* ptr, TValue value, size_t length) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2 ||
                sizeof(TValue) == 4 || sizeof(TValue) == 8);
  static_assert(std::is_unsigned_v<TValue>);

  size_t numBytes = length * sizeof(TValue);
  uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = cur + numBytes;

  if ((sizeof(TValue) > 1 && numBytes < 16) || numBytes < 8) {
    while (cur < end) {
      if (GetAs<TValue>(cur) == value) {
        return reinterpret_cast<const TValue*>(cur);
      }
      cur += sizeof(TValue);
    }
    return nullptr;
  }

  if (numBytes < 16) {
    uint64_t needle64 = static_cast<uint64_t>(value) * 0x0101010101010101llu;
    return reinterpret_cast<const TValue*>(
        Check2x8Bytes(needle64, cur, end - 8));
  }

  VecU8 needle = SplatNeedle<TValue>(value);

#  ifdef MOZ_PPC_UNALIGNED_LOADS
  if (numBytes < 64) {
    // See the SSE2 path for an explanation of the bit-fiddling.
    uintptr_t a = cur;
    uintptr_t b = cur + ((numBytes & 32) >> 1);
    uintptr_t c = end - 16 - ((numBytes & 32) >> 1);
    uintptr_t d = end - 16;
    return Check4x16Bytes<TValue>(needle, a, b, c, d);
  }

  const TValue* result = Check16Bytes<TValue>(needle, cur);
  if (result) {
    return result;
  }

  cur = HideAlignment(AlignUp16(cur));
  uintptr_t tailStartPtr = HideAlignment(AlignDown16(end - 48));
  uintptr_t tailEndPtr = end - 16;

  while (cur < tailStartPtr) {
    result = Check4x16Bytes<TValue>(needle, cur, cur + 16, cur + 32, cur + 48);
    if (result) {
      return result;
    }
    cur += 64;
  }

  return Check4x16Bytes<TValue>(needle, tailStartPtr, tailStartPtr + 16,
                                tailStartPtr + 32, tailEndPtr);
#  else
  uintptr_t alignedStart = AlignUp16(cur);
  while (cur < alignedStart) {
    if (GetAs<TValue>(cur) == value) {
      return reinterpret_cast<const TValue*>(cur);
    }
    cur += sizeof(TValue);
  }

  uintptr_t alignedEnd = AlignDown16(end);
  while (cur + 64 <= alignedEnd) {
    const TValue* result =
        Check4x16Bytes<TValue>(needle, cur, cur + 16, cur + 32, cur + 48);
    if (result) {
      return result;
    }
    cur += 64;
  }
  while (cur + 16 <= alignedEnd) {
    const TValue* result = Check16Bytes<TValue>(needle, cur);
    if (result) {
      return result;
    }
    cur += 16;
  }

  while (cur < end) {
    if (GetAs<TValue>(cur) == value) {
      return reinterpret_cast<const TValue*>(cur);
    }
    cur += sizeof(TValue);
  }
  return nullptr;
#  endif
}

template <typename TValue>
const TValue* FindTwoInBuffer(const TValue* ptr, TValue v1, TValue v2,
                              size_t length) {
  static_assert(sizeof(TValue) == 1 || sizeof(TValue) == 2);
  static_assert(std::is_unsigned_v<TValue>);

  size_t numBytes = length * sizeof(TValue);
  uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = cur + numBytes;

  if (numBytes < 32) {
    return TwoElementLoop<TValue>(cur, end, v1, v2);
  }

  VecU8 needle1 = SplatNeedle<TValue>(v1);
  VecU8 needle2 = SplatNeedle<TValue>(v2);
  VecU8 zero = vec_splats(static_cast<unsigned char>(0));

#  ifdef MOZ_PPC_UNALIGNED_LOADS
  VecU8 haystack = LoadVec(cur);
  VecU8 cmp1 = CmpEq128<TValue>(needle1, haystack);
  VecU8 cmp2 = CmpEq128<TValue>(needle2, haystack);
  VecU8 cmp = vec_and(BSlliN<sizeof(TValue)>(cmp1), cmp2);
  if (AnyMatch(cmp)) {
    return reinterpret_cast<const TValue*>(cur + FirstMatchIndex(cmp) -
                                           sizeof(TValue));
  }

  cur = HideAlignment(AlignUp16(cur));
  uintptr_t tailEndPtr = end - 16;
  uintptr_t tailStartPtr = HideAlignment(AlignDown16(tailEndPtr));

  VecU8 carry = zero;
  while (cur < tailStartPtr) {
    const TValue* result =
        Check2x2x16Bytes<TValue>(needle1, needle2, cur, cur + 16, &carry,
                                 &carry, HaystackOverlap::Sequential);
    if (result) {
      return result;
    }
    cur += 32;
  }

  // The carry only applies if the loop stopped right before the tail block.
  if (cur != tailStartPtr) {
    carry = zero;
  }
  return Check2x2x16Bytes<TValue>(needle1, needle2, tailStartPtr, tailEndPtr,
                                  &carry, nullptr,
                                  HaystackOverlap::Overlapping);
#  else
  // Pairs starting before the first aligned block, including the one that
  // straddles its start.
  uintptr_t alignedStart = AlignUp16(cur);
  const TValue* result = TwoElementLoop<TValue>(
      cur, std::min(alignedStart + sizeof(TValue), end), v1, v2);
  if (result) {
    return result;
  }

  cur = alignedStart;
  uintptr_t alignedEnd = AlignDown16(end);
  VecU8 carry = zero;
  while (cur + 32 <= alignedEnd) {
    result = Check2x2x16Bytes<TValue>(needle1, needle2, cur, cur + 16, &carry,
                                      &carry, HaystackOverlap::Sequential);
    if (result) {
      return result;
    }
    cur += 32;
  }
  if (cur + 16 <= alignedEnd) {
    VecU8 haystack = LoadVec(cur);
    VecU8 cmp1 = CmpEq128<TValue>(needle1, haystack);
    VecU8 cmp2 = CmpEq128<TValue>(needle2, haystack);
    VecU8 cmp = vec_and(vec_or(BSlliN<sizeof(TValue)>(cmp1), carry), cmp2);
    if (AnyMatch(cmp)) {
      return reinterpret_cast<const TValue*>(cur + FirstMatchIndex(cmp) -
                                             sizeof(TValue));
    }
    cur += 16;
  }

  // Pairs from the one straddling the last block's end onward.
  uintptr_t tailStart = cur > alignedStart ? cur - sizeof(TValue) : cur;
  return TwoElementLoop<TValue>(tailStart, end, v1, v2);
#  endif
}

// glibc's hand-written memchr for POWER8 pulls ahead of FindInBuffer on long
// buffers, while FindInBuffer avoids its call and dispatch overhead on short
// ones.
const char* SIMD::memchr8(const char* ptr, char value, size_t length) {
#  ifdef __GLIBC__
  if (length >= 512) {
    const void* result = ::memchr(reinterpret_cast<const void*>(ptr),
                                  static_cast<int>(value), length);
    return reinterpret_cast<const char*>(result);
  }
#  endif
  const unsigned char* uptr = reinterpret_cast<const unsigned char*>(ptr);
  unsigned char uvalue = static_cast<unsigned char>(value);
  const unsigned char* uresult =
      FindInBuffer<unsigned char>(uptr, uvalue, length);
  return reinterpret_cast<const char*>(uresult);
}

const char* SIMD::memchr8SSE2(const char* ptr, char value, size_t length) {
  return memchr8(ptr, value, length);
}

const char16_t* SIMD::memchr16(const char16_t* ptr, char16_t value,
                               size_t length) {
  return FindInBuffer<char16_t>(ptr, value, length);
}

const char16_t* SIMD::memchr16SSE2(const char16_t* ptr, char16_t value,
                                   size_t length) {
  return memchr16(ptr, value, length);
}

const uint32_t* SIMD::memchr32(const uint32_t* ptr, uint32_t value,
                               size_t length) {
  return FindInBuffer<uint32_t>(ptr, value, length);
}

const uint64_t* SIMD::memchr64(const uint64_t* ptr, uint64_t value,
                               size_t length) {
#  ifdef MOZ_PPC_VEC64
  return FindInBuffer<uint64_t>(ptr, value, length);
#  else
  return FindInBufferNaive<uint64_t>(ptr, value, length);
#  endif
}

const char* SIMD::memchr2x8(const char* ptr, char v1, char v2, size_t length) {
  const unsigned char* uptr = reinterpret_cast<const unsigned char*>(ptr);
  unsigned char uv1 = static_cast<unsigned char>(v1);
  unsigned char uv2 = static_cast<unsigned char>(v2);
  const unsigned char* uresult =
      FindTwoInBuffer<unsigned char>(uptr, uv1, uv2, length);
  return reinterpret_cast<const char*>(uresult);
}

const char16_t* SIMD::memchr2x16(const char16_t* ptr, char16_t v1, char16_t v2,
                                 size_t length) {
  return FindTwoInBuffer<char16_t>(ptr, v1, v2, length);
}

#else

const char* SIMD::memchr8(const char* ptr, char value, size_t length) {
  const void* result = ::memchr(reinterpret_cast<const void*>(ptr),
                                static_cast<int>(value), length);
  return reinterpret_cast<const char*>(result);
}

const char* SIMD::memchr8SSE2(const char* ptr, char value, size_t length) {
  return memchr8(ptr, value, length);
}

const char16_t* SIMD::memchr16(const char16_t* ptr, char16_t value,
                               size_t length) {
  return FindInBufferNaive<char16_t>(ptr, value, length);
}

const char16_t* SIMD::memchr16SSE2(const char16_t* ptr, char16_t value,
                                   size_t length) {
  return memchr16(ptr, value, length);
}

const uint32_t* SIMD::memchr32(const uint32_t* ptr, uint32_t value,
                               size_t length) {
  return FindInBufferNaive<uint32_t>(ptr, value, length);
}

const uint64_t* SIMD::memchr64(const uint64_t* ptr, uint64_t value,
                               size_t length) {
  return FindInBufferNaive<uint64_t>(ptr, value, length);
}

const char* SIMD::memchr2x8(const char* ptr, char v1, char v2, size_t length) {
  const char* end = ptr + length - 1;
  while (ptr < end) {
    ptr = memchr8(ptr, v1, end - ptr);
    if (!ptr) {
      return nullptr;
    }
    if (ptr[1] == v2) {
      return ptr;
    }
    ptr++;
  }
  return nullptr;
}

const char16_t* SIMD::memchr2x16(const char16_t* ptr, char16_t v1, char16_t v2,
                                 size_t length) {
  const char16_t* end = ptr + length - 1;
  while (ptr < end) {
    ptr = memchr16(ptr, v1, end - ptr);
    if (!ptr) {
      return nullptr;
    }
    if (ptr[1] == v2) {
      return ptr;
    }
    ptr++;
  }
  return nullptr;
}

#endif

}  // namespace mozilla
