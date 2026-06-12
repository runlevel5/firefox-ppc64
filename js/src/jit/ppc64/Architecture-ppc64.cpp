/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/Architecture-ppc64.h"

#ifndef JS_SIMULATOR
#  include <sys/auxv.h>
#endif

#include "jit/FlushICache.h"  // js::jit::FlushICache
#include "jit/RegisterSets.h"

namespace js {
namespace jit {

Registers::Code Registers::FromName(const char* name) {
  for (size_t i = 0; i < Total; i++) {
    if (strcmp(GetName(i), name) == 0) {
      return Code(i);
    }
  }

  return Invalid;
}

FloatRegisters::Code FloatRegisters::FromName(const char* name) {
  for (size_t i = 0; i < Total; i++) {
    if (strcmp(GetName(i), name) == 0) {
      return Code(i);
    }
  }

  return Invalid;
}

FloatRegisterSet FloatRegister::ReduceSetForPush(const FloatRegisterSet& s) {
  SetType all = s.bits();
  SetType simd128Set =
      (all >> (uint32_t(FloatRegisters::Simd128) * FloatRegisters::TotalPhys)) &
      FloatRegisters::AllPhysMask;
  SetType doubleSet =
      (all >> (uint32_t(FloatRegisters::Double) * FloatRegisters::TotalPhys)) &
      FloatRegisters::AllPhysMask;
  SetType singleSet =
      (all >> (uint32_t(FloatRegisters::Single) * FloatRegisters::TotalPhys)) &
      FloatRegisters::AllPhysMask;

  // Single+Double share physical FPRs (push as Double, 8-byte slot);
  // Simd128 lives in its own physical VRs (push as Simd128, 16-byte
  // slot). Different physical pools — no dedup. Note that
  // sizeof(FloatRegisters::RegisterContent) is 8 bytes (no v128 in the
  // union), so RegisterDump::FPUArray is 32 × 8 = 256 bytes, matching
  // the Float-only layout PushRegsInMask produces.
  SetType set64 = singleSet | doubleSet;

  SetType reduced =
      (simd128Set << (uint32_t(FloatRegisters::Simd128) *
                      FloatRegisters::TotalPhys)) |
      (set64 << (uint32_t(FloatRegisters::Double) * FloatRegisters::TotalPhys));
  return FloatRegisterSet(reduced);
}

uint32_t FloatRegister::GetPushSizeInBytes(const FloatRegisterSet& s) {
  SetType all = s.bits();
  SetType simd128Set =
      (all >> (uint32_t(FloatRegisters::Simd128) * FloatRegisters::TotalPhys)) &
      FloatRegisters::AllPhysMask;
  SetType doubleSet =
      (all >> (uint32_t(FloatRegisters::Double) * FloatRegisters::TotalPhys)) &
      FloatRegisters::AllPhysMask;
  SetType singleSet =
      (all >> (uint32_t(FloatRegisters::Single) * FloatRegisters::TotalPhys)) &
      FloatRegisters::AllPhysMask;

  // Natural per-kind slot sizes. See ReduceSetForPush comment.
  SetType set64 = singleSet | doubleSet;

  uint32_t count64 = std::popcount(static_cast<uint64_t>(set64));
  uint32_t count128 = std::popcount(static_cast<uint64_t>(simd128Set));

  return count64 * sizeof(double) + count128 * 16;
}

uint32_t FloatRegister::getRegisterDumpOffsetInBytes() {
  // Simd128 encoding is 32-63 — mask back to 0-31 for the FPUArray-
  // relative offset. (FPUArray has 32 slots; Simd128 should never be in
  // a SafepointState/BailoutState anyway.)
  return (encoding() & 31) * sizeof(FloatRegisters::RegisterContent);
}

static bool sPOWER9Detected = false;
static bool sPOWER10Detected = false;
static bool sCPUFlagsComputed = false;

#ifndef JS_SIMULATOR
// Cache line sizes, detected at startup from ELF auxiliary vector.
// Fallback to 32 bytes (safe minimum per LuaJIT/LLVM compiler-rt).
static size_t sDCacheLineSize = 0;
static size_t sICacheLineSize = 0;
#endif

void PPC64Flags::Init() {
  if (sCPUFlagsComputed) {
    return;
  }
#ifndef JS_SIMULATOR
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  // PPC_FEATURE2_ARCH_3_00 = 0x00800000 (ISA 3.0 / POWER9)
  sPOWER9Detected = (hwcap2 & 0x00800000) != 0;
  // PPC_FEATURE2_ARCH_3_1 = 0x00040000 (ISA 3.1 / POWER10)
  sPOWER10Detected = (hwcap2 & 0x00040000) != 0;
  // Allow forcing POWER8 mode for testing: MOZ_PPC64_FORCE_POWER8=1.
  // P10 implies P9; downgrade clears both.
  const char* forceP8 = getenv("MOZ_PPC64_FORCE_POWER8");
  if (forceP8 && forceP8[0] == '1') {
    sPOWER9Detected = false;
    sPOWER10Detected = false;
  }

  size_t dcache = getauxval(AT_DCACHEBSIZE);
  size_t icache = getauxval(AT_ICACHEBSIZE);
  sDCacheLineSize = dcache ? dcache : 32;
  sICacheLineSize = icache ? icache : 32;
#endif
  // FORCE_POWER9/10 opt into the corresponding ISA fast paths. Useful under
  // the simulator; on real silicon below the gated level they are foot-guns
  // because the CPU will trap on undefined ops. Outside the JS_SIMULATOR
  // guard so the sim can opt in via env.
  //
  // FORCE_POWER10 also implies FORCE_POWER9 — this matches what real-P10
  // silicon advertises in hwcap2 (both ARCH_3_00 and ARCH_3_1 bits set), so
  // we don't ask sim users to pass both vars separately.
  const char* forceP9 = getenv("MOZ_PPC64_FORCE_POWER9");
  if (forceP9 && forceP9[0] == '1') {
    sPOWER9Detected = true;
  }
  const char* forceP10 = getenv("MOZ_PPC64_FORCE_POWER10");
  if (forceP10 && forceP10[0] == '1') {
    sPOWER10Detected = true;
    sPOWER9Detected = true;
  }
  sCPUFlagsComputed = true;
}

bool HasPOWER9() {
  MOZ_ASSERT(sCPUFlagsComputed);
  return sPOWER9Detected;
}

bool HasPOWER10() {
  MOZ_ASSERT(sCPUFlagsComputed);
  return sPOWER10Detected;
}

bool CPUFlagsHaveBeenComputed() { return sCPUFlagsComputed; }

// Per-bit feature flags packed into the wasm code signature. Adding a
// new bit (e.g., POWER10, VSX4) should be a 1-line change here plus a
// corresponding HasPOWER10()/IsVSX4Available() probe above. The value
// is also assert-checked into a fixed-width field in
// js/src/wasm/WasmCompile.cpp — if that field ever overflows, widen
// it there before landing more bits here.
uint32_t GetPPC64Flags() {
  uint32_t flags = 0;
  if (sPOWER9Detected) {
    flags |= PPC64Flag_POWER9;
  }
  return flags;
}

void FlushICache(void* code, size_t size) {
#if defined(JS_SIMULATOR)
  js::jit::SimulatorProcess::FlushICache(code, size);
#else
  // PPC64 has incoherent I/D caches. GCC's __builtin___clear_cache is a
  // no-op on PPC64 Linux, so we implement the flush explicitly.
  // This follows the same approach as QEMU (util/cacheflush.c) and the
  // Linux kernel (arch/powerpc/mm/cacheflush.c):
  //   dcbst loop -> sync -> icbi loop -> sync -> isync
  if (!size) {
    return;
  }
  MOZ_ASSERT(sCPUFlagsComputed,
             "PPC64Flags::Init must run before any FlushICache call");

  uintptr_t start = reinterpret_cast<uintptr_t>(code);
  uintptr_t end = start + size;

  // Step 1: Write back data cache to memory.
  for (uintptr_t addr = start & ~(sDCacheLineSize - 1); addr < end;
       addr += sDCacheLineSize) {
    asm volatile("dcbst 0, %0" : : "r"(addr) : "memory");
  }
  asm volatile("sync" ::: "memory");

  // Step 2: Invalidate instruction cache.
  for (uintptr_t addr = start & ~(sICacheLineSize - 1); addr < end;
       addr += sICacheLineSize) {
    asm volatile("icbi 0, %0" : : "r"(addr) : "memory");
  }
  // The extra sync before isync matches the Linux kernel and QEMU.
  // It ensures all icbi operations complete before the pipeline flush.
  asm volatile("sync" ::: "memory");
  asm volatile("isync" ::: "memory");
#endif
}

void FlushExecutionContext() {
#if !defined(JS_SIMULATOR)
  // PPC64's isync flushes the instruction pipeline on the current core,
  // ensuring any previously invalidated icache entries are discarded and
  // instructions are re-fetched from coherent memory.
  asm volatile("isync" ::: "memory");
#endif
}

}  // namespace jit
}  // namespace js
