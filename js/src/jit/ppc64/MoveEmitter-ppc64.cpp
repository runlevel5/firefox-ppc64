/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/MoveEmitter-ppc64.h"

#include "jit/MacroAssembler-inl.h"

using namespace js;
using namespace js::jit;

void MoveEmitterPPC64::breakCycle(const MoveOperand& from,
                                  const MoveOperand& to, MoveOp::Type type,
                                  uint32_t slotId) {
  switch (type) {
    case MoveOp::FLOAT32:
      if (to.isMemory()) {
        ScratchFloat32Scope fpscratch32(masm);
        masm.loadFloat32(getAdjustedAddress(to), fpscratch32);
        masm.storeFloat32(fpscratch32, cycleSlot(slotId));
      } else {
        masm.storeFloat32(to.floatReg(), cycleSlot(slotId));
      }
      break;
    case MoveOp::DOUBLE:
      if (to.isMemory()) {
        ScratchDoubleScope fpscratch64(masm);
        masm.loadDouble(getAdjustedAddress(to), fpscratch64);
        masm.storeDouble(fpscratch64, cycleSlot(slotId));
      } else {
        masm.storeDouble(to.floatReg(), cycleSlot(slotId));
      }
      break;
    case MoveOp::INT32:
      if (to.isMemory()) {
        UseScratchRegisterScope temps(masm);
        Register scratch = temps.Acquire();
        masm.load32(getAdjustedAddress(to), scratch);
        masm.store32(scratch, cycleSlot(0));
      } else {
        masm.store32(to.reg(), cycleSlot(0));
      }
      break;
    case MoveOp::GENERAL:
      if (to.isMemory()) {
        UseScratchRegisterScope temps(masm);
        Register scratch = temps.Acquire();
        masm.loadPtr(getAdjustedAddress(to), scratch);
        masm.storePtr(scratch, cycleSlot(0));
      } else {
        masm.storePtr(to.reg(), cycleSlot(0));
      }
      break;
    case MoveOp::SIMD128:
      if (to.isMemory()) {
        ScratchSimd128Scope scratch(masm);
        masm.loadUnalignedSimd128(getAdjustedAddress(to), scratch);
        masm.storeUnalignedSimd128(scratch, cycleSlot(slotId));
      } else {
        masm.storeUnalignedSimd128(to.floatReg(), cycleSlot(slotId));
      }
      break;
    default:
      MOZ_CRASH("Unexpected move type");
  }
}

void MoveEmitterPPC64::completeCycle(const MoveOperand& from,
                                     const MoveOperand& to, MoveOp::Type type,
                                     uint32_t slotId) {
  switch (type) {
    case MoveOp::FLOAT32:
      if (to.isMemory()) {
        ScratchFloat32Scope fpscratch32(masm);
        masm.loadFloat32(cycleSlot(slotId), fpscratch32);
        masm.storeFloat32(fpscratch32, getAdjustedAddress(to));
      } else {
        masm.loadFloat32(cycleSlot(slotId), to.floatReg());
      }
      break;
    case MoveOp::DOUBLE:
      if (to.isMemory()) {
        ScratchDoubleScope fpscratch64(masm);
        masm.loadDouble(cycleSlot(slotId), fpscratch64);
        masm.storeDouble(fpscratch64, getAdjustedAddress(to));
      } else {
        masm.loadDouble(cycleSlot(slotId), to.floatReg());
      }
      break;
    case MoveOp::INT32:
      MOZ_ASSERT(slotId == 0);
      if (to.isMemory()) {
        UseScratchRegisterScope temps(masm);
        Register scratch = temps.Acquire();
        masm.load32(cycleSlot(0), scratch);
        masm.store32(scratch, getAdjustedAddress(to));
      } else {
        masm.load32(cycleSlot(0), to.reg());
      }
      break;
    case MoveOp::GENERAL:
      MOZ_ASSERT(slotId == 0);
      if (to.isMemory()) {
        UseScratchRegisterScope temps(masm);
        Register scratch = temps.Acquire();
        masm.loadPtr(cycleSlot(0), scratch);
        masm.storePtr(scratch, getAdjustedAddress(to));
      } else {
        masm.loadPtr(cycleSlot(0), to.reg());
      }
      break;
    case MoveOp::SIMD128:
      if (to.isMemory()) {
        ScratchSimd128Scope scratch(masm);
        masm.loadUnalignedSimd128(cycleSlot(slotId), scratch);
        masm.storeUnalignedSimd128(scratch, getAdjustedAddress(to));
      } else {
        masm.loadUnalignedSimd128(cycleSlot(slotId), to.floatReg());
      }
      break;
    default:
      MOZ_CRASH("Unexpected move type");
  }
}

void MoveEmitterPPC64::emit(const MoveResolver& moves) {
  if (moves.numCycles()) {
    // SpillSlotSize must be wide enough for the widest cycled value
    // (SIMD128 = 16 bytes). The stride below assumes the same. See
    // Architecture-ppc64.h for the rationale.
    static_assert(SpillSlotSize == 16);
    masm.reserveStack(moves.numCycles() * SpillSlotSize);
    pushedAtCycle_ = masm.framePushed();
  }

  for (size_t i = 0; i < moves.numMoves(); i++) {
    emit(moves.getMove(i));
  }
}

Address MoveEmitterPPC64::cycleSlot(uint32_t slot, uint32_t subslot) const {
  int32_t offset = masm.framePushed() - pushedAtCycle_;
  // Stride must match the per-cycle reservation in emit(); using a
  // narrower stride causes adjacent SIMD128 slots to overlap.
  return Address(StackPointer, offset + slot * SpillSlotSize + subslot);
}

int32_t MoveEmitterPPC64::getAdjustedOffset(const MoveOperand& operand) {
  MOZ_ASSERT(operand.isMemoryOrEffectiveAddress());
  if (operand.base() != StackPointer) {
    return operand.disp();
  }

  return operand.disp() + masm.framePushed() - pushedAtStart_;
}

Address MoveEmitterPPC64::getAdjustedAddress(const MoveOperand& operand) {
  return Address(operand.base(), getAdjustedOffset(operand));
}

void MoveEmitterPPC64::emitMove(const MoveOperand& from,
                                const MoveOperand& to) {
  if (from.isGeneralReg()) {
    if (to.isGeneralReg()) {
      masm.movePtr(from.reg(), to.reg());
    } else if (to.isMemory()) {
      masm.storePtr(from.reg(), getAdjustedAddress(to));
    } else {
      MOZ_CRASH("Invalid emitMove arguments.");
    }
  } else if (from.isMemory()) {
    if (to.isGeneralReg()) {
      masm.loadPtr(getAdjustedAddress(from), to.reg());
    } else if (to.isMemory()) {
      UseScratchRegisterScope temps(masm);
      Register scratch = temps.Acquire();
      masm.loadPtr(getAdjustedAddress(from), scratch);
      masm.storePtr(scratch, getAdjustedAddress(to));
    } else {
      MOZ_CRASH("Invalid emitMove arguments.");
    }
  } else if (from.isEffectiveAddress()) {
    if (to.isGeneralReg()) {
      masm.computeEffectiveAddress(getAdjustedAddress(from), to.reg());
    } else if (to.isMemory()) {
      UseScratchRegisterScope temps(masm);
      Register scratch = temps.Acquire();
      masm.computeEffectiveAddress(getAdjustedAddress(from), scratch);
      masm.storePtr(scratch, getAdjustedAddress(to));
    } else {
      MOZ_CRASH("Invalid emitMove arguments.");
    }
  } else {
    MOZ_CRASH("Invalid emitMove arguments.");
  }
}

void MoveEmitterPPC64::emitInt32Move(const MoveOperand& from,
                                     const MoveOperand& to) {
  if (from.isGeneralReg()) {
    if (to.isGeneralReg()) {
      masm.move32(from.reg(), to.reg());
    } else if (to.isMemory()) {
      masm.store32(from.reg(), getAdjustedAddress(to));
    } else {
      MOZ_CRASH("Invalid emitInt32Move arguments.");
    }
  } else if (from.isMemory()) {
    if (to.isGeneralReg()) {
      masm.load32(getAdjustedAddress(from), to.reg());
    } else if (to.isMemory()) {
      UseScratchRegisterScope temps(masm);
      Register scratch = temps.Acquire();
      masm.load32(getAdjustedAddress(from), scratch);
      masm.store32(scratch, getAdjustedAddress(to));
    } else {
      MOZ_CRASH("Invalid emitInt32Move arguments.");
    }
  } else if (from.isEffectiveAddress()) {
    if (to.isGeneralReg()) {
      masm.computeEffectiveAddress(getAdjustedAddress(from), to.reg());
    } else if (to.isMemory()) {
      UseScratchRegisterScope temps(masm);
      Register scratch = temps.Acquire();
      masm.computeEffectiveAddress(getAdjustedAddress(from), scratch);
      masm.store32(scratch, getAdjustedAddress(to));
    } else {
      MOZ_CRASH("Invalid emitInt32Move arguments.");
    }
  } else {
    MOZ_CRASH("Invalid emitInt32Move arguments.");
  }
}

void MoveEmitterPPC64::emitFloat32Move(const MoveOperand& from,
                                       const MoveOperand& to) {
  if (from.isFloatReg()) {
    if (to.isFloatReg()) {
      masm.moveFloat32(from.floatReg(), to.floatReg());
    } else {
      MOZ_ASSERT(to.isMemory());
      masm.storeFloat32(from.floatReg(), getAdjustedAddress(to));
    }
  } else if (to.isFloatReg()) {
    MOZ_ASSERT(from.isMemory());
    masm.loadFloat32(getAdjustedAddress(from), to.floatReg());
  } else {
    MOZ_ASSERT(from.isMemory());
    MOZ_ASSERT(to.isMemory());
    ScratchFloat32Scope fpscratch32(masm);
    masm.loadFloat32(getAdjustedAddress(from), fpscratch32);
    masm.storeFloat32(fpscratch32, getAdjustedAddress(to));
  }
}

void MoveEmitterPPC64::emitDoubleMove(const MoveOperand& from,
                                      const MoveOperand& to) {
  if (from.isFloatReg()) {
    if (to.isFloatReg()) {
      masm.moveDouble(from.floatReg(), to.floatReg());
    } else if (to.isGeneralReg()) {
      // FPR -> GPR: use mfvsrd directly.
      masm.as_mfvsrd(to.reg(), from.floatReg());
    } else {
      MOZ_ASSERT(to.isMemory());
      masm.storeDouble(from.floatReg(), getAdjustedAddress(to));
    }
  } else if (to.isFloatReg()) {
    if (from.isMemory()) {
      masm.loadDouble(getAdjustedAddress(from), to.floatReg());
    } else {
      // GPR -> FPR: use mtvsrd directly.
      masm.as_mtvsrd(to.floatReg(), from.reg());
    }
  } else {
    MOZ_ASSERT(from.isMemory());
    MOZ_ASSERT(to.isMemory());
    ScratchDoubleScope fpscratch64(masm);
    masm.loadDouble(getAdjustedAddress(from), fpscratch64);
    masm.storeDouble(fpscratch64, getAdjustedAddress(to));
  }
}

void MoveEmitterPPC64::emitSimd128Move(const MoveOperand& from,
                                       const MoveOperand& to) {
  if (from.isFloatReg()) {
    if (to.isFloatReg()) {
      masm.moveSimd128(from.floatReg(), to.floatReg());
    } else {
      MOZ_ASSERT(to.isMemory());
      masm.storeUnalignedSimd128(from.floatReg(), getAdjustedAddress(to));
    }
  } else if (to.isFloatReg()) {
    MOZ_ASSERT(from.isMemory());
    masm.loadUnalignedSimd128(getAdjustedAddress(from), to.floatReg());
  } else {
    MOZ_ASSERT(from.isMemory());
    MOZ_ASSERT(to.isMemory());
    ScratchSimd128Scope scratch(masm);
    masm.loadUnalignedSimd128(getAdjustedAddress(from), scratch);
    masm.storeUnalignedSimd128(scratch, getAdjustedAddress(to));
  }
}

void MoveEmitterPPC64::emit(const MoveOp& move) {
  const MoveOperand& from = move.from();
  const MoveOperand& to = move.to();

  if (move.isCycleEnd() && move.isCycleBegin()) {
    breakCycle(from, to, move.endCycleType(), move.cycleBeginSlot());
    completeCycle(from, to, move.type(), move.cycleEndSlot());
    return;
  }

  if (move.isCycleEnd()) {
    MOZ_ASSERT(inCycle_);
    completeCycle(from, to, move.type(), move.cycleEndSlot());
    MOZ_ASSERT(inCycle_ > 0);
    inCycle_--;
    return;
  }

  if (move.isCycleBegin()) {
    breakCycle(from, to, move.endCycleType(), move.cycleBeginSlot());
    inCycle_++;
  }

  switch (move.type()) {
    case MoveOp::FLOAT32:
      emitFloat32Move(from, to);
      break;
    case MoveOp::DOUBLE:
      emitDoubleMove(from, to);
      break;
    case MoveOp::SIMD128:
      emitSimd128Move(from, to);
      break;
    case MoveOp::INT32:
      emitInt32Move(from, to);
      break;
    case MoveOp::GENERAL:
      emitMove(from, to);
      break;
    default:
      MOZ_CRASH("Unexpected move type");
  }
}

void MoveEmitterPPC64::assertDone() { MOZ_ASSERT(inCycle_ == 0); }

void MoveEmitterPPC64::finish() {
  assertDone();

  masm.freeStack(masm.framePushed() - pushedAtStart_);
}
