/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// The x86_64 libgcc_s.so.1 the Widevine CDM is linked against, for box64 to
// load in place of the system's (there is none on ppc64le). The CDM imports
// only these GCC_3.0 symbols.
//
// The CDM is built without exceptions, so nothing in it catches one: with the
// real unwinder a throw finds no handler and the C++ runtime calls
// std::terminate. _Unwind_RaiseException reports that outcome directly, and
// the remaining unwinder entry points are only reached while unwinding to a
// handler, which never happens.

#include <stddef.h>
#include <stdint.h>

#define EXPORT __attribute__((visibility("default")))

typedef enum {
  _URC_NO_REASON = 0,
  _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
  _URC_END_OF_STACK = 5,
} _Unwind_Reason_Code;

struct _Unwind_Exception;
typedef void (*_Unwind_Exception_Cleanup_Fn)(_Unwind_Reason_Code,
                                             struct _Unwind_Exception*);
struct _Unwind_Exception {
  uint64_t exception_class;
  _Unwind_Exception_Cleanup_Fn exception_cleanup;
  uint64_t private_1;
  uint64_t private_2;
};
struct _Unwind_Context;

EXPORT _Unwind_Reason_Code
_Unwind_RaiseException(struct _Unwind_Exception* exc) {
  return _URC_END_OF_STACK;
}

EXPORT void _Unwind_Resume(struct _Unwind_Exception* exc) { __builtin_trap(); }

EXPORT void _Unwind_DeleteException(struct _Unwind_Exception* exc) {
  if (exc->exception_cleanup) {
    exc->exception_cleanup(_URC_FOREIGN_EXCEPTION_CAUGHT, exc);
  }
}

EXPORT uintptr_t _Unwind_GetIP(struct _Unwind_Context* ctx) { return 0; }

EXPORT uintptr_t _Unwind_GetLanguageSpecificData(struct _Unwind_Context* ctx) {
  return 0;
}

EXPORT uintptr_t _Unwind_GetRegionStart(struct _Unwind_Context* ctx) {
  return 0;
}

EXPORT void _Unwind_SetGR(struct _Unwind_Context* ctx, int reg,
                          uintptr_t value) {}

EXPORT void _Unwind_SetIP(struct _Unwind_Context* ctx, uintptr_t value) {}

// Shift-and-subtract, so that the compiler has no reason to call __udivti3.
EXPORT unsigned __int128 __udivti3(unsigned __int128 n, unsigned __int128 d) {
  if (d == 0) {
    __builtin_trap();
  }
  unsigned __int128 q = 0, r = 0;
  for (int i = 127; i >= 0; i--) {
    r = (r << 1) | ((n >> i) & 1);
    if (r >= d) {
      r -= d;
      q |= (unsigned __int128)1 << i;
    }
  }
  return q;
}
