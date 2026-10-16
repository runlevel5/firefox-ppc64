/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_JitCommon_h
#define jit_JitCommon_h

// Various macros used by all JITs.

#include "jit/Simulator.h"

#ifdef JS_SIMULATOR
// Call into cross-jitted code by following the ABI of the simulated
// architecture.
#  define CALL_GENERATED_CODE(entry, p0, p1, p2, p3, p4, p5, p6, p7)          \
    (js::jit::Simulator::Current()->call(                                     \
        JS_FUNC_TO_DATA_PTR(uint8_t*, entry), 8, intptr_t(p0), intptr_t(p1),  \
        intptr_t(p2), intptr_t(p3), intptr_t(p4), intptr_t(p5), intptr_t(p6), \
        intptr_t(p7)))

#  define CALL_GENERATED_0(entry)                                              \
    (js::jit::Simulator::Current()->call(JS_FUNC_TO_DATA_PTR(uint8_t*, entry), \
                                         0))

#  define CALL_GENERATED_1(entry, p0)                                          \
    (js::jit::Simulator::Current()->call(JS_FUNC_TO_DATA_PTR(uint8_t*, entry), \
                                         1, intptr_t(p0)))

#  define CALL_GENERATED_2(entry, p0, p1)                                      \
    (js::jit::Simulator::Current()->call(JS_FUNC_TO_DATA_PTR(uint8_t*, entry), \
                                         2, intptr_t(p0), intptr_t(p1)))

#  define CALL_GENERATED_3(entry, p0, p1, p2)                                  \
    (js::jit::Simulator::Current()->call(JS_FUNC_TO_DATA_PTR(uint8_t*, entry), \
                                         3, intptr_t(p0), intptr_t(p1),        \
                                         intptr_t(p2)))

#else

// Call into jitted code by following the ABI of the native architecture.
#  define CALL_GENERATED_CODE(entry, p0, p1, p2, p3, p4, p5, p6, p7) \
    entry(p0, p1, p2, p3, p4, p5, p6, p7)

#  define CALL_GENERATED_0(entry) entry()
#  define CALL_GENERATED_1(entry, p0) entry(p0)
#  define CALL_GENERATED_2(entry, p0, p1) entry(p0, p1)
#  define CALL_GENERATED_3(entry, p0, p1, p2) entry(p0, p1, p2)

#endif

#if defined(JS_CODEGEN_PPC64) && defined(_CALL_ELF) && _CALL_ELF == 1
namespace js {
namespace jit {

// PPC64 ELFv1 function descriptor: {entry, toc, env}. See MakeELFv1Call.
struct ELFv1FunctionDescriptor {
  void* entry;
  void* toc;
  void* env;
};

// Fill |desc| with a synthetic ELFv1 descriptor for calling raw JIT code
// |entry| as a C function pointer, and return it typed as |Fn|. On ELFv1 a
// function pointer is the address of such a descriptor, not a code entry, and
// the compiler emits every indirect call as a descriptor dereference
// (ld r12,0(p); ld r2,8(p); mtctr r12; bctrl); a raw entry would be read as
// descriptor data. The captured TOC (r2) is libmozjs's, shared by all its
// functions, so the callee can reach C++ data. The trailing inline-asm memory
// clobber forces the descriptor stores to retire before the compiler emits the
// indirect call from the returned pointer. |desc| must outlive the call.
template <typename Fn>
inline Fn MakeELFv1Call(void* entry, ELFv1FunctionDescriptor* desc) {
  void* toc;
  asm volatile("mr %0, 2" : "=r"(toc));
  desc->entry = entry;
  desc->toc = toc;
  desc->env = nullptr;
  asm volatile("" : : "r"(desc) : "memory");
  return reinterpret_cast<Fn>(desc);
}

}  // namespace jit
}  // namespace js
#endif

#endif  // jit_JitCommon_h
