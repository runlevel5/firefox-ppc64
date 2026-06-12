/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_Architecture_ppc64_h
#define jit_ppc64_Architecture_ppc64_h

#include <algorithm>
#include <bit>

#include "jit/shared/Architecture-shared.h"

#include "js/Utility.h"

namespace js {
namespace jit {

// PPC64 has 32 64-bit general purpose registers, r0 through r31.
// The program counter is not directly accessible as a register.
// The link register (LR) and count register (CTR) are SPRs.

// PPC64 ELFv2 GPR Convention:
//  Name    Usage
//  r0      Volatile, cannot be base register in load/store
//  r1      Stack pointer (callee-saved)
//  r2      TOC pointer (reserved)
//  r3      Return value / first argument
//  r4-r10  Arguments 2-8
//  r11     Environment pointer / scratch
//  r12     Branch target / scratch
//  r13     Thread pointer (reserved, TLS)
//  r14-r31 Callee-saved

// PPC64 ELFv2 FPR Convention:
//  f0      Scratch
//  f1-f13  Arguments / volatile
//  f14-f31 Callee-saved

class Registers {
 public:
  enum RegisterID {
    r0 = 0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    r8,
    r9,
    r10,
    r11,
    r12,
    r13,
    r14,
    r15,
    r16,
    r17,
    r18,
    r19,
    r20,
    r21,
    r22,
    r23,
    r24,
    r25,
    r26,
    r27,
    r28,
    r29,
    r30,
    r31,
    sp = r1,
    invalid_reg,
  };
  typedef uint8_t Code;
  typedef RegisterID Encoding;
  typedef uint32_t SetType;

  static const Encoding StackPointer = sp;
  static const Encoding Invalid = invalid_reg;

  union RegisterContent {
    uintptr_t r;
  };

  static uint32_t SetSize(SetType x) { return std::popcount(x); }
  static uint32_t FirstBit(SetType x) {
    MOZ_ASSERT(x);
    return std::countr_zero(x);
  }
  static uint32_t LastBit(SetType x) {
    MOZ_ASSERT(x);
    return std::bit_width(x) - 1;
  }

  static const char* GetName(uint32_t code) {
    static const char* const Names[] = {
        "r0",  "sp",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",
        "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
        "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
        "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"};
    static_assert(Total == std::size(Names), "Table is the correct size");
    if (code >= Total) {
      return "invalid";
    }
    return Names[code];
  }

  static Code FromName(const char* name);

  static const uint32_t Total = 32;
  static const uint32_t TotalPhys = 32;
  static const uint32_t Allocatable = 24;

  static const SetType AllMask = 0xFFFFFFFF;
  static const SetType NoneMask = 0x0;

  static const SetType ArgRegMask =
      (1U << Registers::r3) | (1U << Registers::r4) | (1U << Registers::r5) |
      (1U << Registers::r6) | (1U << Registers::r7) | (1U << Registers::r8) |
      (1U << Registers::r9) | (1U << Registers::r10);

  // r0, r11, r12 are also volatile but handled separately.
  static const SetType VolatileMask = ArgRegMask;

  // ELFv2 callee-saved GPRs are r14..r31. r2 (TOC) and r13 (TLS) are
  // dedicated registers, NOT general callee-saved: r2 is restored by the
  // PLT-call linkage convention (`ld r2, 24(r1)` after every cross-module
  // call); r13 is the thread pointer and must NEVER be written. Including
  // them here previously made `PushRegsInMask(NonVolatileMask)` save and
  // restore them — wasted 16 bytes per wasm-stub frame at best, latent
  // TLS corruption if save/restore were ever misordered. Verified that
  // no JIT-emitted code writes r2 or r13 (both are NonAllocatable, and
  // grep across js/src/jit/ppc64/ finds no `as_*` site assigning to
  // them), so they're preserved across the JIT body for free.
  static const SetType NonVolatileMask =
      (1U << Registers::r14) |
      (1U << Registers::r15) | (1U << Registers::r16) | (1U << Registers::r17) |
      (1U << Registers::r18) | (1U << Registers::r19) | (1U << Registers::r20) |
      (1U << Registers::r21) | (1U << Registers::r22) | (1U << Registers::r23) |
      (1U << Registers::r24) | (1U << Registers::r25) | (1U << Registers::r26) |
      (1U << Registers::r27) | (1U << Registers::r28) | (1U << Registers::r29) |
      (1U << Registers::r30) | (1U << Registers::r31);

  static const SetType NonAllocatableMask =
      (1U << Registers::r0) |   // Cannot be base in load/store.
      (1U << Registers::sp) |   // Stack pointer.
      (1U << Registers::r2) |   // TOC pointer (ELFv2).
      (1U << Registers::r11) |  // Third scratch.
      (1U << Registers::r12) |  // Second scratch / addressTempRegister.
      (1U << Registers::r13) |  // Thread-local storage (ELFv2).
      (1U << Registers::r16) |  // Saved scratch register.
      (1U << Registers::r31);   // Frame pointer.

  static const SetType WrapperMask = VolatileMask;

  // Registers returned from a JS -> JS call.
  static const SetType JSCallMask = (1U << Registers::r5);

  // Registers returned from a JS -> C call.
  static const SetType CallMask = (1U << Registers::r3);

  static const SetType AllocatableMask = AllMask & ~NonAllocatableMask;
};

typedef uint32_t PackedRegisterMask;

template <typename T>
class TypedRegisterSet;

class FloatRegisters {
 public:
  enum FPRegisterID {
    f0 = 0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f25,
    f26,
    f27,
    f28,
    f29,
    f30,
    f31,
  };

  // Eight bits: (invalid << 7) | (kind << 5) | encoding
  typedef uint8_t Code;
  typedef FPRegisterID Encoding;
  // 3 kinds × 32 regs = 96 bits needed. Use __uint128_t.
  typedef __uint128_t SetType;

  enum Kind : uint8_t { Double, Single, Simd128, NumTypes };

  static constexpr Code Invalid = 0x80;

  static const char* GetName(uint32_t code) {
    static const char* const Names[] = {
        "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
        "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
        "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
        "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31"};
    static_assert(TotalPhys == std::size(Names), "Table is the correct size");
    if (code >= Total) {
      return "invalid";
    }
    return Names[code % TotalPhys];
  }

  static Code FromName(const char* name);

  static const uint32_t TotalPhys = 32;
  static const uint32_t Total = TotalPhys * NumTypes;
  static const uint32_t Allocatable = 31;  // Without f0, the scratch register.

  static_assert(sizeof(SetType) * 8 >= Total,
                "SetType should be large enough to enumerate all registers.");

  static const SetType SpreadSingle = SetType(1)
                                      << (uint32_t(Single) * TotalPhys);
  static const SetType SpreadDouble = SetType(1)
                                      << (uint32_t(Double) * TotalPhys);
  static const SetType SpreadSimd128 = SetType(1)
                                       << (uint32_t(Simd128) * TotalPhys);
  static const SetType Spread = SpreadSingle | SpreadDouble | SpreadSimd128;

  static const SetType AllPhysMask = ((SetType(1) << TotalPhys) - 1);
  static const SetType AllMask = AllPhysMask * Spread;
  static const SetType AllSingleMask = AllPhysMask * SpreadSingle;
  static const SetType AllDoubleMask = AllPhysMask * SpreadDouble;
  static const SetType AllSimd128Mask = AllPhysMask * SpreadSimd128;
  static const SetType NoneMask = SetType(0);

  // ELFv2: f14-f31 are non-volatile (callee-saved) for scalar FP.
  // The upper 64 bits of VSR 0-31 are volatile, so Simd128 view is all-volatile.
  static const SetType NonVolatilePhysMask =
      SetType((1U << FloatRegisters::f14) | (1U << FloatRegisters::f15) |
              (1U << FloatRegisters::f16) | (1U << FloatRegisters::f17) |
              (1U << FloatRegisters::f18) | (1U << FloatRegisters::f19) |
              (1U << FloatRegisters::f20) | (1U << FloatRegisters::f21) |
              (1U << FloatRegisters::f22) | (1U << FloatRegisters::f23) |
              (1U << FloatRegisters::f24) | (1U << FloatRegisters::f25) |
              (1U << FloatRegisters::f26) | (1U << FloatRegisters::f27) |
              (1U << FloatRegisters::f28) | (1U << FloatRegisters::f29) |
              (1U << FloatRegisters::f30) | (1U << FloatRegisters::f31));
  // Simd128 lives in VR-namespace (VSR32-63 = VR0-VR31). Per ELFv2 ABI,
  // VR20-VR31 are non-volatile (callee-saved). Encoding storage is 20-31
  // with kind=Simd128.
  static const SetType SimdNonVolatilePhysMask =
      SetType((1U << 20) | (1U << 21) | (1U << 22) | (1U << 23) |
              (1U << 24) | (1U << 25) | (1U << 26) | (1U << 27) |
              (1U << 28) | (1U << 29) | (1U << 30) | (1U << 31));
  static const SetType NonVolatileMask =
      NonVolatilePhysMask * (SpreadSingle | SpreadDouble) |
      SimdNonVolatilePhysMask * SpreadSimd128;

  static const SetType VolatileMask = AllMask & ~NonVolatileMask;

  static const SetType WrapperMask = VolatileMask;

  // f0 is the scratch register (all three views: single, double, simd128).
  static const SetType NonAllocatableMask =
      (SetType(1) << FloatRegisters::f0) * Spread;

  static const SetType AllocatableMask = AllMask & ~NonAllocatableMask;

  union RegisterContent {
    float s;
    double d;
    // No v128 here. Simd128 lives in physically-distinct VRs (VSR32-63)
    // and never reaches RegisterDump (asserted by SafepointState; bailout
    // AllRegs excludes Simd128). With v128 in the union, sizeof was 16,
    // forcing PushRegsInMask to a 16-byte stride that mismatched
    // addressOfRegister's 8-byte walk via (*iter).size().
  };

  static constexpr Encoding encoding(Code c) { return Encoding(c & 31); }

  static constexpr Kind kind(Code c) { return Kind((c >> 5) & 3); }

  static constexpr Code fromParts(uint32_t encoding, uint32_t kind,
                                  uint32_t invalid) {
    return Code((invalid << 7) | (kind << 5) | encoding);
  }
};

// SpillSlotSize must fit the widest register class (Simd128 = 16 bytes).
// We can't derive from sizeof(FloatRegisters::RegisterContent) — that
// union is sized for FPRs only (8 bytes since v128 lives in distinct
// VRs, not in the FPR union), so deriving would under-reserve for
// Simd128 cycle breaks. SpillSlotSize is consumed only by MoveEmitter
// and is not part of the JIT frame layout.
static const uint32_t SpillSlotSize = 16;

// PPC64 ELFv2 ABI: the callee saves LR at [caller_SP+16], CR at
// [caller_SP+8], and may save TOC at [caller_SP+24]. Reserve 32 bytes
// (the minimum ELFv2 stack frame) as a shadow area for every ABI call.
static constexpr uint32_t ShadowStackSpace = 32;
static const uint32_t SizeOfReturnAddressAfterCall = 0;

// PPC64 branch instructions have a 26-bit signed offset field, giving a
// range of +/- 32MB. We reduce this to leave room for jump island insertion.
static constexpr uint32_t JumpImmediateRange = (32 * 1024 * 1024) - 32;

// Size of each bailout table entry (a single bl instruction).
static const uint32_t BAILOUT_TABLE_ENTRY_SIZE = 4;

// PPC64 special purpose registers (not exposed to the allocator).
enum SPRegisterID {
  spr_xer = 1,
  spr_lr = 8,
  spr_ctr = 9,
  spr_vrsave = 256,
  invalid_spreg
};

// PPC64 condition registers.
enum CRegisterID { cr0 = 0, cr1, cr5 = 5, cr6, cr7, invalid_creg };

struct FloatRegister {
  typedef FloatRegisters Codes;
  typedef size_t Code;
  typedef Codes::Encoding Encoding;
  typedef Codes::SetType SetType;

  static uint32_t SetSize(SetType x) {
    // Fold all 3 kinds (Single, Double, Simd128) down to physical mask.
    SetType phys = (x & FloatRegisters::AllPhysMask) |
                   ((x >> FloatRegisters::TotalPhys) & FloatRegisters::AllPhysMask) |
                   ((x >> (2 * FloatRegisters::TotalPhys)) & FloatRegisters::AllPhysMask);
    return std::popcount(static_cast<uint64_t>(phys));
  }

  // __uint128_t helpers for FirstBit/LastBit.
  static uint32_t FirstBit(SetType x) {
    MOZ_ASSERT(x);
    uint64_t lo = static_cast<uint64_t>(x);
    if (lo) {
      return std::countr_zero(lo);
    }
    return 64 + std::countr_zero(static_cast<uint64_t>(x >> 64));
  }
  static uint32_t LastBit(SetType x) {
    MOZ_ASSERT(x);
    uint64_t hi = static_cast<uint64_t>(x >> 64);
    if (hi) {
      return 64 + (std::bit_width(hi) - 1);
    }
    return std::bit_width(static_cast<uint64_t>(x)) - 1;
  }

 private:
  uint8_t encoding_;
  uint8_t kind_;
  bool invalid_;

  typedef Codes::Kind Kind;

 public:
  constexpr FloatRegister(Encoding encoding, Kind kind)
      : encoding_(encoding), kind_(kind), invalid_(false) {}

  constexpr FloatRegister()
      : encoding_(0), kind_(FloatRegisters::Double), invalid_(true) {}

  static FloatRegister FromCode(uint32_t i) {
    MOZ_ASSERT(i < Codes::Total);
    return FloatRegister(FloatRegisters::encoding(i), FloatRegisters::kind(i));
  }

  bool isSingle() const {
    MOZ_ASSERT(!invalid_);
    return kind_ == FloatRegisters::Single;
  }
  bool isDouble() const {
    MOZ_ASSERT(!invalid_);
    return kind_ == FloatRegisters::Double;
  }
  bool isSimd128() const {
    MOZ_ASSERT(!invalid_);
    return kind_ == FloatRegisters::Simd128;
  }
  bool isInvalid() const { return invalid_; }

  FloatRegister asSingle() const {
    MOZ_ASSERT(!invalid_);
    return FloatRegister(Encoding(encoding_), FloatRegisters::Single);
  }
  FloatRegister asDouble() const {
    MOZ_ASSERT(!invalid_);
    return FloatRegister(Encoding(encoding_), FloatRegisters::Double);
  }
  FloatRegister asSimd128() const {
    MOZ_ASSERT(!invalid_);
    return FloatRegister(Encoding(encoding_), FloatRegisters::Simd128);
  }

  constexpr uint32_t size() const {
    MOZ_ASSERT(!invalid_);
    if (kind_ == FloatRegisters::Double) {
      return sizeof(double);
    }
    if (kind_ == FloatRegisters::Single) {
      return sizeof(float);
    }
    MOZ_ASSERT(kind_ == FloatRegisters::Simd128);
    return 16;
  }

  constexpr Code code() const {
    return Codes::fromParts(encoding_, kind_, invalid_);
  }

  constexpr Encoding encoding() const {
    MOZ_ASSERT(!invalid_);
    // Simd128 lives in VR-namespace at VSR32-63 (= VR0-31). Single/Double
    // share FPR namespace at VSR0-31. The unified XX-form encoders split
    // the result into low-5-bit VRT/VRA/VRB + TX/AX/BX bits; VMX
    // FloatRegister-taking encoders mask with `& 31` for the raw VR
    // field. So 32+E flows correctly through both paths.
    return Encoding(encoding_ +
                    (kind_ == FloatRegisters::Simd128 ? 32 : 0));
  }

  const char* name() const { return FloatRegisters::GetName(code()); }
  bool volatile_() const {
    MOZ_ASSERT(!invalid_);
    return !!((SetType(1) << code()) & FloatRegisters::VolatileMask);
  }
  constexpr bool operator!=(FloatRegister other) const {
    return code() != other.code();
  }
  constexpr bool operator==(FloatRegister other) const {
    return code() == other.code();
  }

  bool aliases(FloatRegister other) const {
    // Register-class partition: {Single, Double} share FPRs (VSR0-31);
    // Simd128 lives in VR-namespace (VSR32-63). FPR f5 (Single/Double
    // encoding 5) and VR v5 (Simd128 encoding 5) are distinct physical
    // registers.
    if (encoding_ != other.encoding_) return false;
    bool selfSimd = (kind_ == FloatRegisters::Simd128);
    bool otherSimd = (other.kind_ == FloatRegisters::Simd128);
    return selfSimd == otherSimd;
  }
  bool equiv(FloatRegister other) const {
    MOZ_ASSERT(!invalid_);
    return kind_ == other.kind_;
  }

  uint32_t numAliased() const {
    return (kind_ == FloatRegisters::Simd128) ? 1 : 2;
  }
  uint32_t numAlignedAliased() { return numAliased(); }

  FloatRegister aliased(uint32_t aliasIdx) {
    MOZ_ASSERT(!invalid_);
    MOZ_ASSERT(aliasIdx < numAliased());
    if (kind_ == FloatRegisters::Simd128) {
      return *this;
    }
    Kind otherKind = (kind_ == FloatRegisters::Single)
                         ? FloatRegisters::Double
                         : FloatRegisters::Single;
    Kind selectedKind = (aliasIdx == 0) ? Kind(kind_) : otherKind;
    return FloatRegister(Encoding(encoding_), selectedKind);
  }
  FloatRegister alignedAliased(uint32_t aliasIdx) {
    MOZ_ASSERT(aliasIdx < numAliased());
    return aliased(aliasIdx);
  }
  SetType alignedOrDominatedAliasedSet() const {
    if (kind_ == FloatRegisters::Simd128) {
      return SetType(1) << ((uint32_t(FloatRegisters::Simd128) *
                             FloatRegisters::TotalPhys) +
                            encoding_);
    }
    return (Codes::SpreadSingle | Codes::SpreadDouble) << encoding_;
  }

  static constexpr RegTypeName DefaultType = RegTypeName::Float64;

  template <RegTypeName Name = DefaultType>
  static SetType LiveAsIndexableSet(SetType s) {
    return SetType(0);
  }

  template <RegTypeName Name = DefaultType>
  static SetType AllocatableAsIndexableSet(SetType s) {
    static_assert(Name != RegTypeName::Any, "Allocatable set are not iterable");
    return LiveAsIndexableSet<Name>(s);
  }

  static TypedRegisterSet<FloatRegister> ReduceSetForPush(
      const TypedRegisterSet<FloatRegister>& s);
  static uint32_t GetPushSizeInBytes(const TypedRegisterSet<FloatRegister>& s);
  uint32_t getRegisterDumpOffsetInBytes();
};

template <>
inline FloatRegister::SetType
FloatRegister::LiveAsIndexableSet<RegTypeName::Float32>(SetType set) {
  return set & FloatRegisters::AllSingleMask;
}

template <>
inline FloatRegister::SetType
FloatRegister::LiveAsIndexableSet<RegTypeName::Float64>(SetType set) {
  return set & FloatRegisters::AllDoubleMask;
}

template <>
inline FloatRegister::SetType
FloatRegister::LiveAsIndexableSet<RegTypeName::Vector128>(SetType set) {
  return set & FloatRegisters::AllSimd128Mask;
}

template <>
inline FloatRegister::SetType
FloatRegister::LiveAsIndexableSet<RegTypeName::Any>(SetType set) {
  return set;
}

inline bool hasUnaliasedDouble() { return false; }
inline bool hasMultiAlias() { return false; }

// PPC64 feature bits packed into the value GetPPC64Flags() returns,
// which feeds wasm/WasmCompile.cpp's per-architecture code signature.
// Defined as enum constants (not enum class) so callers can OR/AND
// freely. New bits should remain backward-compatible — older signatures
// must keep meaning the same set of features.
enum PPC64FeatureFlags : uint32_t {
  PPC64Flag_POWER9 = 1u << 0,
  // Future: PPC64Flag_POWER10 = 1u << 1, PPC64Flag_VSX4 = 1u << 2, ...
};

uint32_t GetPPC64Flags();

class PPC64Flags final {
 public:
  PPC64Flags() = delete;

  // PPC64Flags::Init is called from the JitContext constructor to read the
  // hardware capabilities (via getauxval(AT_HWCAP2)). It must be called
  // exactly once, before HasPOWER9()/HasPOWER10() are used.
  static void Init();
};

bool HasPOWER9();
bool HasPOWER10();

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_Architecture_ppc64_h */
