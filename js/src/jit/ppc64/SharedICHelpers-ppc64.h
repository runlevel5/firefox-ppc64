/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_SharedICHelpers_ppc64_h
#define jit_ppc64_SharedICHelpers_ppc64_h

#include "jit/BaselineIC.h"
#include "jit/JitFrames.h"
#include "jit/MacroAssembler.h"
#include "jit/SharedICRegisters.h"

namespace js {
namespace jit {

// Distance from sp to the top Value inside an IC stub (no return address on
// the stack on PPC64).
static const size_t ICStackValueOffset = 0;

struct BaselineStubFrame {
  uintptr_t savedFrame;
  uintptr_t savedStub;
  uintptr_t returnAddress;
  uintptr_t descriptor;
};

inline void EmitRestoreTailCallReg(MacroAssembler& masm) {
  // On PPC64, LR always holds the return address after a bl/bctrl call.
  // No-op: LR is the hardware link register, not a GPR on the stack.
}

inline void EmitRepushTailCallReg(MacroAssembler& masm) {
  // No-op: LR already holds the return address.
}

inline void EmitCallIC(MacroAssembler& masm, CodeOffset* callOffset) {
  // The stub pointer must already be in ICStubReg.
  // Load stubcode pointer from the ICStub.
  // R2 won't be active when we call ICs, so we can use it as scratch.
  masm.loadPtr(Address(ICStubReg, ICStub::offsetOfStubCode()), R2.scratchReg());

  // Call the stubcode. On PPC64 call(Register) emits mtctr + bctrl,
  // which sets LR to the address after bctrl.
  masm.call(R2.scratchReg());
  *callOffset = CodeOffset(masm.currentOffset());
}

inline void EmitReturnFromIC(MacroAssembler& masm) {
  // Return via hardware LR (set by the original bl/bctrl call).
  masm.as_blr();
}

inline void EmitBaselineLeaveStubFrame(MacroAssembler& masm) {
  masm.loadPtr(
      Address(FramePointer, BaselineStubFrameLayout::ICStubOffsetFromFP),
      ICStubReg);

  masm.movePtr(FramePointer, StackPointer);
  masm.Pop(FramePointer);

  // Load the return address and restore it to LR.
  masm.Pop(ICTailCallReg);
  masm.xs_mtlr(ICTailCallReg);

  // Discard the frame descriptor.
  {
    UseScratchRegisterScope temps(masm);
    Register scratch = temps.Acquire();
    masm.Pop(scratch);
  }
}

template <typename AddrType>
inline void EmitPreBarrier(MacroAssembler& masm, const AddrType& addr,
                           MIRType type) {
  // On PPC64, LR is clobbered by guardedCallPreBarrier. Save it first.
  masm.xs_mflr(r0);
  masm.push(r0);
  masm.guardedCallPreBarrier(addr, type);
  masm.pop(r0);
  masm.xs_mtlr(r0);
}

inline void EmitStubGuardFailure(MacroAssembler& masm) {
  // Load next stub into ICStubReg.
  masm.loadPtr(Address(ICStubReg, ICCacheIRStub::offsetOfNext()), ICStubReg);

  // Return address is in LR. Jump to the next stubcode.
  masm.jump(Address(ICStubReg, ICStub::offsetOfStubCode()));
}

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_SharedICHelpers_ppc64_h */
