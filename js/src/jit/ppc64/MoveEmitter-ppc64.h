/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_MoveEmitter_ppc64_h
#define jit_ppc64_MoveEmitter_ppc64_h

#include "jit/MacroAssembler.h"
#include "jit/MoveResolver.h"

namespace js {
namespace jit {

class MoveEmitterPPC64 {
  void emitDoubleMove(const MoveOperand& from, const MoveOperand& to);
  void emitSimd128Move(const MoveOperand& from, const MoveOperand& to);
  void breakCycle(const MoveOperand& from, const MoveOperand& to,
                  MoveOp::Type type, uint32_t slot);
  void completeCycle(const MoveOperand& from, const MoveOperand& to,
                     MoveOp::Type type, uint32_t slot);

 protected:
  uint32_t inCycle_;
  MacroAssembler& masm;

  uint32_t pushedAtStart_;

  int32_t pushedAtCycle_;

  void assertDone();
  Address cycleSlot(uint32_t slot, uint32_t subslot = 0) const;
  int32_t getAdjustedOffset(const MoveOperand& operand);
  Address getAdjustedAddress(const MoveOperand& operand);

  void emitMove(const MoveOperand& from, const MoveOperand& to);
  void emitInt32Move(const MoveOperand& from, const MoveOperand& to);
  void emitFloat32Move(const MoveOperand& from, const MoveOperand& to);
  void emit(const MoveOp& move);

 public:
  explicit MoveEmitterPPC64(MacroAssembler& masm)
      : inCycle_(0),
        masm(masm),
        pushedAtStart_(masm.framePushed()),
        pushedAtCycle_(-1) {}

  ~MoveEmitterPPC64() { assertDone(); }

  void emit(const MoveResolver& moves);
  void finish();
  // setScratchRegister is part of the cross-arch MoveEmitter interface
  // but we never spill, so there's no scratch to set. No-op kept for
  // shared-code compatibility.
  void setScratchRegister(Register reg) {}
};

typedef MoveEmitterPPC64 MoveEmitter;

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_MoveEmitter_ppc64_h */
