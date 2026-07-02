/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/MacroAssembler-ppc64.h"

#include "jit/Bailouts.h"
#include "jit/BaselineFrame.h"
#include "jit/FlushICache.h"
#include "jit/JitFrames.h"
#include "jit/JitRuntime.h"
#include "jit/MacroAssembler.h"
#include "jit/MoveEmitter.h"
#include "jit/ppc64/SharedICRegisters-ppc64.h"
#include "vm/JitActivation.h"
#include "vm/JSContext.h"
#include "wasm/WasmBuiltins.h"
#include "wasm/WasmStubs.h"

#include "jit/MacroAssembler-inl.h"

namespace js {
namespace jit {

MacroAssembler& MacroAssemblerPPC64::asMasm() {
  return *static_cast<MacroAssembler*>(this);
}

const MacroAssembler& MacroAssemblerPPC64::asMasm() const {
  return *static_cast<const MacroAssembler*>(this);
}

// ===============================================================
// Out-of-line fake exit frame

bool MacroAssemblerPPC64Compat::buildOOLFakeExitFrame(void* fakeReturnAddr) {
  asMasm().Push(FrameDescriptor(FrameType::IonJS));
  asMasm().Push(ImmPtr(fakeReturnAddr));
  asMasm().Push(FramePointer);
  return true;
}

// ===============================================================
// Load int32 or double from memory

void MacroAssemblerPPC64Compat::loadInt32OrDouble(const Address& src,
                                                  FloatRegister dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  Label end;

  // Load the boxed value and stash in the FPR immediately, then reuse the
  // GPR for the tag test.  Only one scratch GPR is held here so that
  // branchTestInt32 can acquire the second one for the ImmTag constant.
  loadPtr(Address(src.base, src.offset), scratch);
  as_mtvsrd(dest, scratch);
  x_srdi(scratch, scratch, JSVAL_TAG_SHIFT);
  asMasm().branchTestInt32(Assembler::NotEqual, scratch, &end);
  // It was an int32.  Recover the boxed value from the FPR, sign-extend
  // the low 32 bits, and convert to double.
  as_mfvsrd(scratch, dest);
  as_extsw(scratch, scratch);
  as_mtvsrd(dest, scratch);
  as_fcfid(dest, dest);

  bind(&end);
}

void MacroAssemblerPPC64Compat::loadInt32OrDouble(const BaseIndex& addr,
                                                  FloatRegister dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  Label end;

  computeScaledAddress(addr, scratch);
  loadPtr(Address(scratch, addr.offset), scratch);
  as_mtvsrd(dest, scratch);
  x_srdi(scratch, scratch, JSVAL_TAG_SHIFT);
  asMasm().branchTestInt32(Assembler::NotEqual, scratch, &end);
  as_mfvsrd(scratch, dest);
  as_extsw(scratch, scratch);
  as_mtvsrd(dest, scratch);
  as_fcfid(dest, dest);

  bind(&end);
}

// ===============================================================
// Conversion functions

void MacroAssemblerPPC64Compat::convertUInt32ToDouble(Register src,
                                                      FloatRegister dest) {
  // mtvsrwz: VSR[dest].dw0 = zero_ext_64(src[32:63]); P8+ (ISA 2.07).
  // Replaces rldicl + mtvsrd (2 insns + scratch) with 1 insn.
  as_mtvsrwz(dest, src);
  as_fcfid(dest, dest);
}

void MacroAssemblerPPC64Compat::convertUInt32ToFloat32(Register src,
                                                       FloatRegister dest) {
  // mtvsrwz + fcfids; same recipe as convertUInt32ToDouble.
  as_mtvsrwz(dest, src);
  as_fcfids(dest, dest);
}

// Helper for the negative-zero check after a successful round-trip.
// Precondition: `dest` holds the integer round-trip result; if it equals
// zero, then `src` was either +0.0 or -0.0 (those are the only doubles
// that round-trip to int 0). Distinguish them by inspecting src's sign
// bit: -0.0 has its MSB set, so an mfvsrd-then-signed-cmp-against-zero
// branches to `fail` only for -0.0. Non-zero `dest` values (including
// every negative integer) skip the check entirely.
static void EmitNegativeZeroCheck(MacroAssemblerPPC64Compat& masm,
                                  FloatRegister src, Register dest,
                                  Label* fail) {
  Label notZero;
  masm.as_cmpdi(dest, 0);
  masm.ma_b(Assembler::NotEqual, &notZero);
  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();
  masm.as_mfvsrd(scratch, src);
  masm.as_cmpdi(scratch, 0);
  masm.ma_b(Assembler::LessThan, fail);
  masm.bind(&notZero);
}

void MacroAssemblerPPC64Compat::convertDoubleToInt32(FloatRegister src,
                                                     Register dest, Label* fail,
                                                     bool negativeZeroCheck) {
  // Truncate to int32 (round toward zero), sign-extend, and verify
  // exactness via round-trip compare. fctiwz writes the int32 to BE
  // bits 32..63 of the FPR; mfvsrd extracts and extsw sign-extends.
  // The compare also catches NaN (unordered) and Inf (saturated to
  // INT32_{MIN,MAX}, won't round-trip equal).
  as_fctiwz(ScratchDoubleReg, src);
  as_mfvsrd(dest, ScratchDoubleReg);
  as_extsw(dest, dest);
  as_mtvsrd(ScratchDoubleReg, dest);
  as_fcfid(ScratchDoubleReg, ScratchDoubleReg);
  as_fcmpu(ScratchDoubleReg, src);
  ma_b(Assembler::DoubleNotEqualOrUnordered, fail);

  if (negativeZeroCheck) {
    EmitNegativeZeroCheck(*this, src, dest, fail);
  }
}

void MacroAssemblerPPC64Compat::convertDoubleToPtr(FloatRegister src,
                                                   Register dest, Label* fail,
                                                   bool negativeZeroCheck) {
  // Same pattern as convertDoubleToInt32 but to int64 (no sign-extend
  // needed since fctidz already produces a 64-bit result).
  as_fctidz(ScratchDoubleReg, src);
  as_mfvsrd(dest, ScratchDoubleReg);
  as_mtvsrd(ScratchDoubleReg, dest);
  as_fcfid(ScratchDoubleReg, ScratchDoubleReg);
  as_fcmpu(ScratchDoubleReg, src);
  ma_b(Assembler::DoubleNotEqualOrUnordered, fail);

  if (negativeZeroCheck) {
    EmitNegativeZeroCheck(*this, src, dest, fail);
  }
}

void MacroAssemblerPPC64Compat::convertFloat32ToInt32(FloatRegister src,
                                                      Register dest,
                                                      Label* fail,
                                                      bool negativeZeroCheck) {
  // Same as convertDoubleToInt32 but the round-trip uses fcfids so the
  // comparison happens at single precision (matches src's actual width).
  as_fctiwz(ScratchDoubleReg, src);
  as_mfvsrd(dest, ScratchDoubleReg);
  as_extsw(dest, dest);
  as_mtvsrd(ScratchDoubleReg, dest);
  as_fcfids(ScratchDoubleReg, ScratchDoubleReg);
  as_fcmpu(ScratchDoubleReg, src);
  ma_b(Assembler::DoubleNotEqualOrUnordered, fail);

  if (negativeZeroCheck) {
    EmitNegativeZeroCheck(*this, src, dest, fail);
  }
}

CodeOffset MacroAssemblerPPC64Compat::toggledCall(JitCode* target,
                                                  bool enabled) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // stanza(8) + mtctr/bctrl(2) = 10 instructions.
  m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
  BufferOffset boLoad =
      emitLoad64Stanza(scratch, (uint64_t)uintptr_t(target->raw()));
  CodeOffset offset(boLoad.getOffset());
  addPendingJump(boLoad, ImmPtr(target->raw()), RelocationKind::JITCODE);
  if (enabled) {
    xs_mtctr(scratch);
    as_bctr(LinkBit::LinkB);
  } else {
    writeInst(NopInst);
    writeInst(NopInst);
  }
  m_buffer.leaveNoPool();
  MOZ_ASSERT_IF(!oom(), nextOffset().getOffset() - offset.offset() ==
                            ToggledCallSize(nullptr));
  return offset;
}

// ===============================================================
// Exception handling

void MacroAssemblerPPC64Compat::handleFailureWithHandlerTail(
    Label* profilerExitTail, Label* bailoutTail,
    uint32_t* returnValueCheckOffset) {
  // Round sizeof(ResumeFromException) up to ABIStackAlignment. The
  // canonical (sz + align - 1) & ~(align - 1) form is exact: when sz
  // is already a multiple of `align` the rounding is a no-op. The
  // previous (sz + align) & ~(align - 1) over-allocated by `align`
  // bytes whenever sz was already aligned.
  int size = (sizeof(ResumeFromException) + ABIStackAlignment - 1) &
             ~(ABIStackAlignment - 1);
  asMasm().subPtr(Imm32(size), StackPointer);
  // Use r3 (first argument register).
  mov(StackPointer, r3);

  using Fn = void (*)(ResumeFromException* rfe);
  asMasm().setupUnalignedABICall(r4);
  asMasm().passABIArg(r3);
  asMasm().callWithABI<Fn, HandleException>(
      ABIType::General, CheckUnsafeCallWithABI::DontCheckHasExitFrame);

  *returnValueCheckOffset = asMasm().currentOffset();

  Label entryFrame;
  Label catch_;
  Label finally;
  Label returnBaseline;
  Label returnIon;
  Label bailout;
  Label wasmInterpEntry;
  Label wasmCatch;

  load32(Address(StackPointer, ResumeFromException::offsetOfKind()), r3);
  asMasm().branch32(Assembler::Equal, r3,
                    Imm32(ExceptionResumeKind::EntryFrame), &entryFrame);
  asMasm().branch32(Assembler::Equal, r3, Imm32(ExceptionResumeKind::Catch),
                    &catch_);
  asMasm().branch32(Assembler::Equal, r3, Imm32(ExceptionResumeKind::Finally),
                    &finally);
  asMasm().branch32(Assembler::Equal, r3,
                    Imm32(ExceptionResumeKind::ForcedReturnBaseline),
                    &returnBaseline);
  asMasm().branch32(Assembler::Equal, r3,
                    Imm32(ExceptionResumeKind::ForcedReturnIon), &returnIon);
  asMasm().branch32(Assembler::Equal, r3, Imm32(ExceptionResumeKind::Bailout),
                    &bailout);
  asMasm().branch32(Assembler::Equal, r3,
                    Imm32(ExceptionResumeKind::WasmInterpEntry),
                    &wasmInterpEntry);
  asMasm().branch32(Assembler::Equal, r3, Imm32(ExceptionResumeKind::WasmCatch),
                    &wasmCatch);

  breakpoint();  // Invalid kind.

  // No exception handler. Return error from entry frame.
  bind(&entryFrame);
  asMasm().moveValue(MagicValue(JS_ION_ERROR), JSReturnOperand);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfFramePointer()),
          FramePointer);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);
  ret();

  // Catch handler.
  bind(&catch_);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfTarget()), r3);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfFramePointer()),
          FramePointer);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);
  jump(r3);

  // Finally block.
  bind(&finally);
  ValueOperand exception = ValueOperand(r4);
  loadValue(Address(StackPointer, ResumeFromException::offsetOfException()),
            exception);

  ValueOperand exceptionStack = ValueOperand(r5);
  loadValue(
      Address(StackPointer, ResumeFromException::offsetOfExceptionStack()),
      exceptionStack);

  loadPtr(Address(StackPointer, ResumeFromException::offsetOfTarget()), r3);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfFramePointer()),
          FramePointer);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);

  pushValue(exception);
  pushValue(exceptionStack);
  pushValue(BooleanValue(true));
  jump(r3);

  // Forced return from baseline.
  Label profilingInstrumentation;
  bind(&returnBaseline);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfFramePointer()),
          FramePointer);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);
  loadValue(Address(FramePointer, BaselineFrame::reverseOffsetOfReturnValue()),
            JSReturnOperand);
  jump(&profilingInstrumentation);

  // Forced return from Ion.
  bind(&returnIon);
  loadValue(Address(StackPointer, ResumeFromException::offsetOfException()),
            JSReturnOperand);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfFramePointer()),
          FramePointer);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);

  bind(&profilingInstrumentation);
  {
    Label skipProfilingInstrumentation;
    AbsoluteAddress addressOfEnabled(
        asMasm().runtime()->geckoProfiler().addressOfEnabled());
    asMasm().branch32(Assembler::Equal, addressOfEnabled, Imm32(0),
                      &skipProfilingInstrumentation);
    jump(profilerExitTail);
    bind(&skipProfilingInstrumentation);
  }

  xs_mr(StackPointer, FramePointer);
  // Pop FP from stack, then return (pop LR + blr).
  loadPtr(Address(StackPointer, 0), FramePointer);
  asMasm().addPtr(Imm32(sizeof(void*)), StackPointer);
  ret();

  // Bailout.
  bind(&bailout);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfBailoutInfo()),
          r5);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);
  xs_li(ReturnReg, 1);
  jump(bailoutTail);

  // Wasm interp entry.
  bind(&wasmInterpEntry);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfFramePointer()),
          FramePointer);
  loadPtr(Address(StackPointer, ResumeFromException::offsetOfStackPointer()),
          StackPointer);
  movePtr(ImmWord(wasm::InterpFailInstanceReg), InstanceReg);
  ret();

  // Wasm catch.
  bind(&wasmCatch);
  wasm::GenerateJumpToCatchHandler(asMasm(), StackPointer, r4, r5, r6);
}

void MacroAssembler::clampDoubleToUint8(FloatRegister input, Register output) {
  ScratchDoubleScope fpscratch(asMasm());

  if (HasPOWER9()) {
    // P9 xsmaxjdp uses Java/JS semantics (ISA v3.0B): any NaN
    // is treated as "less than any number that is not a NaN", so
    // xsmaxjdp(input, 0) collapses {NaN, -Inf, ≤ 0} to 0 in one insn —
    // the "≤ 0 or NaN → 0" branch dance disappears.
    //
    // After the max, fctid (round-to-nearest-even per FPSCR default,
    // matches ECMA Uint8ClampedArray's round-half-to-even) saturates
    // out-of-int64 values to INT64_MAX. Remaining upper clamp
    // (output > 255 → 255) is one cmpdi + isel.
    zeroDouble(fpscratch);
    as_xsmaxjdp(fpscratch, input, fpscratch);
    as_fctid(fpscratch, fpscratch);
    as_mfvsrd(output, fpscratch);
    UseScratchRegisterScope temps(asMasm());
    Register max255 = temps.Acquire();
    xs_li(max255, 255);
    as_cmpdi(output, 255);
    as_isel(output, max255, output, GreaterThan);
    return;
  }

  // POWER8 fallback: xsmaxjdp is unavailable, so filter NaN explicitly
  // before fctid. Per Power ISA, fctid maps NaN to INT64_MAX, which
  // would clamp to 255 instead of the spec-required 0.
  Label positive, below255, done;
  zeroDouble(fpscratch);
  branchDouble(DoubleGreaterThan, input, fpscratch, &positive);
  {
    move32(Imm32(0), output);
    jump(&done);
  }

  bind(&positive);

  loadConstantDouble(255.0, fpscratch);
  branchDouble(DoubleLessThan, input, fpscratch, &below255);
  {
    move32(Imm32(255), output);
    jump(&done);
  }

  bind(&below255);

  as_fctid(fpscratch, input);
  as_mfvsrd(output, fpscratch);
  bind(&done);
}

void MacroAssembler::subFromStackPtr(Imm32 imm32) {
  if (imm32.value) {
    asMasm().subPtr(imm32, StackPointer);
  }
}

//{{{ check_macroassembler_style

void MacroAssembler::widenInt32(Register r) {
  move32To64SignExtend(r, Register64(r));
}

// Stack operations.
void MacroAssembler::Push(Register reg) {
  push(reg);
  adjustFrame(int32_t(sizeof(intptr_t)));
}
void MacroAssembler::Push(const Imm32 imm) {
  push(imm);
  adjustFrame(int32_t(sizeof(intptr_t)));
}

void MacroAssembler::Push(const ImmWord imm) {
  push(imm);
  adjustFrame(int32_t(sizeof(intptr_t)));
}

void MacroAssembler::Push(const ImmPtr imm) {
  Push(ImmWord(uintptr_t(imm.value)));
}

void MacroAssembler::Push(const ImmGCPtr ptr) {
  push(ptr);
  adjustFrame(int32_t(sizeof(intptr_t)));
}

void MacroAssembler::PushBoxed(FloatRegister reg) {
  subFromStackPtr(Imm32(sizeof(double)));
  boxDouble(reg, Address(getStackPointer(), 0));
  adjustFrame(sizeof(double));
}

void MacroAssembler::Pop(Register reg) {
  pop(reg);
  adjustFrame(-int32_t(sizeof(intptr_t)));
}
void MacroAssembler::PushRegsInMask(LiveRegisterSet set) {
  int32_t diff =
      set.gprs().size() * sizeof(intptr_t) + set.fpus().getPushSizeInBytes();
  const int32_t reserved = diff;

  reserveStack(reserved);
  for (GeneralRegisterBackwardIterator iter(set.gprs()); iter.more(); ++iter) {
    diff -= sizeof(intptr_t);
    storePtr(*iter, Address(StackPointer, diff));
  }

  // Natural per-kind slot — 8 bytes for Single/Double via stfd, 16 bytes
  // for Simd128 via stxvx. RegisterDump::FPUArray is sized 32 × 8 = 256
  // bytes (sizeof(RegisterContent) is 8 — no v128 in the union), so
  // f_K's stfd slot lands at the right offset. Bailout AllRegs excludes
  // Simd128 (Ion has no SIMD live), so the FP region in bailout frames
  // is strictly Float-only.
  for (FloatRegisterBackwardIterator iter(set.fpus().reduceSetForPush());
       iter.more(); ++iter) {
    FloatRegister reg = *iter;
    diff -= reg.size();
    if (reg.isSimd128()) {
      storeUnalignedSimd128(reg, Address(StackPointer, diff));
    } else {
      storeDouble(reg.asDouble(), Address(StackPointer, diff));
    }
  }
  MOZ_ASSERT(diff == 0);
}
void MacroAssembler::PopRegsInMaskIgnore(LiveRegisterSet set,
                                         LiveRegisterSet ignore) {
  int32_t diff =
      set.gprs().size() * sizeof(intptr_t) + set.fpus().getPushSizeInBytes();
  const int32_t reserved = diff;

  for (GeneralRegisterBackwardIterator iter(set.gprs()); iter.more(); ++iter) {
    diff -= sizeof(intptr_t);
    if (!ignore.has(*iter)) {
      loadPtr(Address(StackPointer, diff), *iter);
    }
  }

  // Natural per-kind slot. See PushRegsInMask comment.
  for (FloatRegisterBackwardIterator iter(set.fpus().reduceSetForPush());
       iter.more(); ++iter) {
    FloatRegister reg = *iter;
    diff -= reg.size();
    if (!ignore.has(reg)) {
      if (reg.isSimd128()) {
        loadUnalignedSimd128(Address(StackPointer, diff), reg);
      } else {
        loadDouble(Address(StackPointer, diff), reg.asDouble());
      }
    }
  }
  MOZ_ASSERT(diff == 0);
  freeStack(reserved);
}

// Call operations.
CodeOffset MacroAssembler::call(Register reg) {
  // ELFv2 ABI: r12 must hold the target address at function entry
  // so the callee can compute its TOC pointer from r12.
  if (reg != CallReg) {
    movePtr(reg, CallReg);
  }
  xs_mtctr(CallReg);
  as_bctr(LinkB);
  return CodeOffset(currentOffset());
}
CodeOffset MacroAssembler::call(Label* label) {
  if (label->bound()) {
    // Open the no-pool window BEFORE computing the displacement.
    // enterNoPool() can itself trigger a pending pool flush, advancing
    // currentOffset(). A pre-flush displacement emitted at the post-flush
    // position would overshoot the target by poolSize bytes.
    m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
    int32_t offset = label->offset() - currentOffset();
    // Call instruction goes at inst[9] in the 10-word stanza.
    int32_t callOffset = offset - 9 * (int32_t)sizeof(uint32_t);
    if (JOffImm26::IsInRange(callOffset)) {
      // Short: 9 nops + bl = 10 instructions.
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      writeInst(NopInst);
      as_b(JOffImm26(callOffset), RelativeBranch, LinkB);
      m_buffer.leaveNoPool();
      return CodeOffset(currentOffset());
    }
    // Long call to bound label: stanza(8) + mtctr + bctrl = 10 instructions.
    BufferOffset bo =
        emitLoad64Stanza(SecondScratchReg, LabelBase::INVALID_OFFSET);
    xs_mtctr(SecondScratchReg);
    as_bctr(LinkB);
    m_buffer.leaveNoPool();
    addLongJump(bo, BufferOffset(label->offset()));
    return CodeOffset(currentOffset());
  }
  // Emit a CallTag stanza: trap + chain + 8 nops (10 instructions total).
  m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
  BufferOffset bo = xs_trap_tagged(CallTag);
  writeInst(label->used() ? label->offset() : LabelBase::INVALID_OFFSET);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  m_buffer.leaveNoPool();
  if (!oom()) {
    label->use(bo.getOffset());
  }
  return CodeOffset(currentOffset());
}
CodeOffset MacroAssembler::call(const Address& addr) {
  loadPtr(addr, CallReg);
  return call(CallReg);
}

void MacroAssembler::call(ImmPtr target) {
  uint64_t addr = uintptr_t(target.value);
  // stanza(8) + mtctr + bctrl = 10 instructions.
  m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
  BufferOffset bo = emitLoad64Stanza(CallReg, addr);
  addPendingJump(bo, target, RelocationKind::HARDCODED);
  xs_mtctr(CallReg);
  as_bctr(LinkB);
  m_buffer.leaveNoPool();
}

CodeOffset MacroAssembler::call(wasm::SymbolicAddress target) {
  movePtr(target, CallReg);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // ELFv1: a non-thunked SymbolicAddress (a C function) resolves to a
  // {entry,toc,env} function descriptor, not a raw code entry, so it must be
  // dereferenced like the callWithABI sites do. Thunked symbols are patched
  // to the builtin thunk, which is raw JIT code; the dance would jump to its
  // first eight instruction bytes read as an address.
  if (!wasm::NeedsBuiltinThunk(target)) {
    return callABIDescriptorELFv1(CallReg);
  }
#endif
  return call(CallReg);
}

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
CodeOffset MacroAssemblerPPC64Compat::callABIDescriptorELFv1(
    Register descriptor) {
  // On ELFv1 a C function pointer is a 24-byte descriptor {entry@0, toc@8,
  // env@16}, not a code entry (the ELFv2 convention call(Register) assumes).
  // Allocate the ELFv1 call frame: a 48-byte linkage area (the callee's LR
  // (+16) / TOC saves land in scratch space; we park our r2, the JIT's
  // libmozjs TOC, in the reserved +24 slot) plus a 64-byte parameter save area
  // (8 doublewords). The parameter save area is mandatory: a GCC-compiled
  // callee may spill its incoming register arguments to [SP+48 ..), and without
  // it those spills land on the JIT's outparameter slot sitting just above SP
  // and corrupt it (e.g. CreateThisFromICWithAllocSite's MutableHandleValue
  // result). Load the callee entry and TOC from the descriptor, call, restore
  // r2 and pop. Load TOC before entry so a descriptor==r12 alias is safe. Note:
  // register-arg calls only; stack-passed args would need the area folded into
  // callWithABIPre.
  constexpr int32_t kELFv1FrameSize = 48 + 64;
  as_stdu(StackPointer, StackPointer, -kELFv1FrameSize);
  as_std(r2, StackPointer, 24);
  as_ld(r2, descriptor, 8);
  as_ld(r12, descriptor, 0);
  xs_mtctr(r12);
  as_bctr(LinkB);
  // Return address (where the callee returns to) is the instruction after bctr.
  CodeOffset callOffset(currentOffset());
  as_ld(r2, StackPointer, 24);
  as_addi(StackPointer, StackPointer, kELFv1FrameSize);
  return callOffset;
}
#endif

// The wasm-module SymbolicAddress call (the call(CallSiteDesc, SymbolicAddress)
// path). StaticallyLink patches this access to SymbolicAddressTarget(): a symbol
// that NeedsBuiltinThunk resolves to the builtin thunk's raw wasm-ABI code
// entry, while a non-thunk symbol resolves to a C function pointer, which on
// ELFv1 big-endian is a {entry,toc,env} descriptor. So call the thunk straight
// and dereference the C function. (The bare call(SymbolicAddress) used by stubs
// and the process-global thunks always targets a C function, so it always
// dereferences.)
CodeOffset MacroAssemblerPPC64Compat::callWasmSymbolic(wasm::SymbolicAddress imm) {
  asMasm().movePtr(imm, CallReg);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (wasm::NeedsBuiltinThunk(imm)) {
    return asMasm().call(CallReg);
  }
  return callABIDescriptorELFv1(CallReg);
#else
  return asMasm().call(CallReg);
#endif
}

void MacroAssembler::callWithABINoProfiler(const Address& fun, ABIType result) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(fun, scratch);

  uint32_t stackAdjust;
  callWithABIPre(&stackAdjust);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  callABIDescriptorELFv1(scratch);
#else
  call(scratch);
#endif
  callWithABIPost(stackAdjust, result);
}

void MacroAssembler::callWithABIPre(uint32_t* stackAdjust, bool callFromWasm) {
  MOZ_ASSERT(inCall_);
  uint32_t stackForCall = abiArgs_.stackBytesConsumedSoFar();

  // Reserve place for LR save.
  stackForCall += sizeof(intptr_t);

  if (dynamicAlignment_) {
    stackForCall += ComputeByteAlignment(stackForCall, ABIStackAlignment);
  } else {
    uint32_t alignmentAtPrologue = callFromWasm ? sizeof(wasm::Frame) : 0;
    stackForCall += ComputeByteAlignment(
        stackForCall + framePushed() + alignmentAtPrologue, ABIStackAlignment);
  }

  *stackAdjust = stackForCall;
  reserveStack(stackForCall);

  // Save LR. Restore it in callWithABIPost.
  {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    xs_mflr(scratch);
    storePtr(scratch, Address(StackPointer, stackForCall - sizeof(intptr_t)));
  }

  // Position all arguments.
  {
    enoughMemory_ &= moveResolver_.resolve();
    if (!enoughMemory_) {
      return;
    }

    MoveEmitter emitter(*this);
    emitter.emit(moveResolver_);
    emitter.finish();
  }

  assertStackAlignment(ABIStackAlignment);
}

void MacroAssembler::callWithABIPost(uint32_t stackAdjust, ABIType result) {
  {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    loadPtr(Address(StackPointer, stackAdjust - sizeof(intptr_t)), scratch);
    xs_mtlr(scratch);
  }

  if (dynamicAlignment_) {
    // Restore SP from stack (as stored in setupUnalignedABICall).
    loadPtr(Address(StackPointer, stackAdjust), StackPointer);
    adjustFrame(-stackAdjust);
  } else {
    freeStack(stackAdjust);
  }

#ifdef DEBUG
  MOZ_ASSERT(inCall_);
  inCall_ = false;
#endif
}

// Value operations.
void MacroAssembler::moveValue(const ValueOperand& src,
                               const ValueOperand& dest) {
  if (src.valueReg() != dest.valueReg()) {
    movePtr(src.valueReg(), dest.valueReg());
  }
}
void MacroAssembler::moveValue(const Value& src, const ValueOperand& dest) {
  if (!src.isGCThing()) {
    movePtr(ImmWord(src.asRawBits()), dest.valueReg());
    return;
  }
  CodeOffset off = movWithPatch(ImmWord(src.asRawBits()), dest.valueReg());
  writeDataRelocation(off, src);
}

// Branch operations.
void MacroAssembler::branchTestValue(Condition cond, const ValueOperand& lhs,
                                     const Value& rhs, Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  MOZ_ASSERT(!rhs.isNaN());

  if (!rhs.isGCThing()) {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    MOZ_ASSERT(lhs.valueReg() != scratch);
    movePtr(ImmWord(rhs.asRawBits()), scratch);
    branchPtr(cond, lhs.valueReg(), scratch, label);
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    MOZ_ASSERT(lhs.valueReg() != scratch);
    moveValue(rhs, ValueOperand(scratch));
    branchPtr(cond, lhs.valueReg(), scratch, label);
  }
}
void MacroAssembler::branchTestNaNValue(Condition cond, const ValueOperand& val,
                                        Register temp, Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(val.valueReg() != scratch);

  // Strip the IEEE sign bit (LSB-numbering bit 63 = PPC-numbering bit 0)
  // with rldicl SH=0, MB=1: rotate by zero (no-op) then keep bits 1..63 of
  // PPC-numbering, clearing bit 0. Rotating by 1 instead would also shift
  // the quiet-NaN bit out of position and cause 1.5 (0x3FF8...) and NaN
  // (0x7FF8...) to collide after masking — bug 1943704 PPC64 regression.
  as_rldicl(temp, val.valueReg(), 0, 1);

  // Load canonical NaN (with sign bit 0) and strip its sign bit too.
  static_assert(JS::detail::CanonicalizedNaNSignBit == 0);
  moveValue(DoubleValue(JS::GenericNaN()), ValueOperand(scratch));
  as_rldicl(scratch, scratch, 0, 1);

  branchPtr(cond, temp, scratch, label);
}

void MacroAssembler::branchPtrInNurseryChunk(Condition cond, Register ptr,
                                             Register temp, Label* label) {
  MOZ_ASSERT(cond == Assembler::Equal || cond == Assembler::NotEqual);
  MOZ_ASSERT(ptr != temp);
  MOZ_ASSERT(temp != InvalidReg);

  andPtr(Imm32(int32_t(~gc::ChunkMask)), ptr, temp);
  branchPtr(InvertCondition(cond), Address(temp, gc::ChunkStoreBufferOffset),
            ImmWord(0), label);
}
void MacroAssembler::branchValueIsNurseryCell(Condition cond,
                                              ValueOperand value, Register temp,
                                              Label* label) {
  branchValueIsNurseryCellImpl(cond, value, temp, label);
}

// Patching / near address operations.
CodeOffset MacroAssembler::nopPatchableToCall() {
  // Emit 10 nops that can be patched to a call stanza:
  // 8 load64 nops + mtctr nop + bctrl nop
  // Return offset AFTER the stanza (= the return address).
  m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  m_buffer.leaveNoPool();
  return CodeOffset(currentOffset());
}
CodeOffset MacroAssembler::moveNearAddressWithPatch(Register dest) {
  CodeOffset offset(currentOffset());
  emitLoad64Stanza(dest, 0);
  return offset;
}
// static
void MacroAssembler::patchNearAddressMove(CodeLocationLabel loc,
                                          CodeLocationLabel target) {
  Instruction* inst = (Instruction*)loc.raw();
  UpdateLoad64Value(inst, (uint64_t)target.raw());
}

// Return address operations (link register architectures).
//
// Note: these MUST decrement SP by exactly 8 bytes. wasm::Frame is 16 bytes
// (callerFP_ + returnAddress_) and GenerateCallablePrologue pairs this with
// push(FramePointer) to match that layout exactly — a 16-byte decrement here
// would insert 8 bytes of padding and break FP-chain unwinding. The 8-byte
// intermediate misalignment between this save and the following push(FP) is
// never observed by a C call (no intervening transition), and any caller that
// does make a C call after pushReturnAddress routes through
// setupUnalignedABICall which re-aligns.
void MacroAssembler::pushReturnAddress() {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  xs_mflr(scratch);
  push(scratch);
}
void MacroAssembler::popReturnAddress() {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  pop(scratch);
  xs_mtlr(scratch);
}

// ABI setup.
void MacroAssembler::setupUnalignedABICall(Register scratch) {
  MOZ_ASSERT(!IsCompilingWasm(), "wasm should only use aligned ABI calls");
  setupNativeABICall();
  dynamicAlignment_ = true;

  movePtr(StackPointer, scratch);

  // Force sp to be aligned.
  subPtr(Imm32(sizeof(uintptr_t)), StackPointer);
  andPtr(Imm32(~(ABIStackAlignment - 1)), StackPointer);
  storePtr(scratch, Address(StackPointer, 0));
}

// ===============================================================
// Arithmetic helpers.

void MacroAssembler::flexibleDivMod32(Register lhs, Register rhs,
                                      Register divOutput, Register remOutput,
                                      bool isUnsigned, const LiveRegisterSet&) {
  MOZ_ASSERT(lhs != divOutput && lhs != remOutput, "lhs is preserved");
  MOZ_ASSERT(rhs != divOutput && rhs != remOutput, "rhs is preserved");

  // PPC64 has no modulus instruction. Compute: rem = lhs - (lhs/rhs)*rhs
  // PPC64 divw(INT32_MIN, -1) is undefined; quotient=INT32_MIN, remainder=0.
  Label done;
  if (!isUnsigned) {
    Label notMinOverflow;
    branchPtr(Assembler::NotEqual, lhs, ImmWord(INT32_MIN), &notMinOverflow);
    branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinOverflow);
    move32(Imm32(INT32_MIN), divOutput);
    move32(Imm32(0), remOutput);
    jump(&done);
    bind(&notMinOverflow);
  }
  if (isUnsigned) {
    as_divwu(divOutput, lhs, rhs);
  } else {
    as_divw(divOutput, lhs, rhs);
  }
  as_extsw(divOutput, divOutput);
  if (HasPOWER9()) {
    if (isUnsigned) {
      as_moduw(remOutput, lhs, rhs);
    } else {
      as_modsw(remOutput, lhs, rhs);
    }
  } else {
    as_mullw(remOutput, divOutput, rhs);
    as_subf(remOutput, remOutput, lhs);
  }
  as_extsw(remOutput, remOutput);
  bind(&done);
}

void MacroAssembler::shiftIndex32AndAdd(Register indexTemp32, int shift,
                                        Register pointer) {
  if (IsShiftInScaleRange(shift)) {
    computeEffectiveAddress(
        BaseIndex(pointer, indexTemp32, ShiftToScale(shift)), pointer);
    return;
  }
  lshift32(Imm32(shift), indexTemp32);
  addPtr(indexTemp32, pointer);
}

void MacroAssembler::convertInt64ToDouble(Register64 src, FloatRegister dest) {
  as_mtvsrd(dest, src.reg);
  as_fcfid(dest, dest);
}

void MacroAssembler::nearbyIntDouble(RoundingMode mode, FloatRegister src,
                                     FloatRegister dest) {
  switch (mode) {
    case RoundingMode::NearestTiesToEven: {
      // PPC64's frin rounds ties away from zero, NOT to even (ISA v3.1).
      // Use fctid+fcfid which uses FPSCR RN (default = round-to-nearest-even).
      // Guard: if |src| >= 2^52, value is already integral (or NaN/Inf) —
      // just copy src. This preserves NaN, Inf, and -0.
      // Check via integer exponent extraction to avoid FP temp conflicts.
      Label done;
      UseScratchRegisterScope temps(*this);
      Register scratch = temps.Acquire();
      moveDouble(src, ScratchDoubleReg);
      if (src != dest) {
        moveDouble(src, dest);
      }
      if (HasPOWER9()) {
        // xsxexpdp lays the 11-bit biased exponent in XT.dw0 with the
        // rest zeroed, so mfvsrd reads it directly — drops the
        // srdi+andi. masking pair.
        ScratchSimd128Scope expScratch(*this);
        as_xsxexpdp(expScratch, ScratchDoubleReg);
        as_mfvsrd(scratch, expScratch);
      } else {
        as_mfvsrd(scratch, ScratchDoubleReg);
        x_srdi(scratch, scratch, 52);
        as_andi_rc(scratch, scratch, 0x7FF);
      }
      // Biased exponent >= 1075 (= 1023+52) means |val| >= 2^52.
      // Also catches Inf (exp=2047) and NaN (exp=2047).
      ma_cmp(scratch, Imm32(1075), Assembler::GreaterThanOrEqual);
      ma_b(Assembler::GreaterThanOrEqual, &done);
      as_fctid(dest, ScratchDoubleReg);
      as_fcfid(dest, dest);
      as_fcpsgn(dest, ScratchDoubleReg, dest);
      bind(&done);
      break;
    }
    case RoundingMode::TowardsZero:
      as_friz(dest, src);
      break;
    case RoundingMode::Up:
      as_frip(dest, src);
      break;
    case RoundingMode::Down:
      as_frim(dest, src);
      break;
    default:
      MOZ_CRASH("Unexpected rounding mode");
  }
}

void MacroAssembler::nearbyIntFloat32(RoundingMode mode, FloatRegister src,
                                      FloatRegister dest) {
  // PPC FP rounding instructions operate on double-precision.
  // For single-precision, we round as double then round back to single.
  // The frsp instruction handles the double->single conversion.
  nearbyIntDouble(mode, src, dest);
  as_frsp(dest, dest);
}

// ===============================================================
// Far jump support.

CodeOffset MacroAssembler::farJumpWithPatch() {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  // stanza(8) + mtctr + bctr = 10 instructions.
  CodeOffset loadOffset(currentOffset());
  m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
  emitLoad64Stanza(scratch, 0);
  xs_mtctr(scratch);
  as_bctr();
  m_buffer.leaveNoPool();

  return loadOffset;
}

// ===============================================================
void MacroAssembler::flush() { Assembler::flush(); }

// Wasm support.

FaultingCodeOffset MacroAssembler::wasmTrapInstruction() {
  m_buffer.flushPool();  // see comment in wasmLoadImpl
  FaultingCodeOffset fco = FaultingCodeOffset(currentOffset());
  xs_trap();
  return fco;
}

// PPC64 SlowCallMarker: `ori r0, r0, 0` -- a NOP-like instruction
// that won't appear in normal code generation.
// ori r0, r0, 0 = 0x60000000 -- that's actually PPC_nop.
// Use a distinguishable encoding: `ori r12, r12, 0` = 0x618C0000
static const int32_t SlowCallMarker = 0x618C0000;

void MacroAssembler::wasmMarkCallAsSlow() {
  // Emit: ori r12, r12, 0
  as_ori(CallReg, CallReg, 0);
}

void MacroAssembler::wasmCheckSlowCallsite(Register ra_, Label* notSlow,
                                           Register temp1, Register temp2) {
  MOZ_ASSERT(ra_ != temp2);
  load32(Address(ra_, 0), temp2);
  branch32(Assembler::NotEqual, temp2, Imm32(SlowCallMarker), notSlow);
}

CodeOffset MacroAssembler::wasmMarkedSlowCall(const wasm::CallSiteDesc& desc,
                                              const Register reg) {
  CodeOffset offset = call(desc, reg);
  wasmMarkCallAsSlow();
  return offset;
}

// ===============================================================
// Additional stack operations.

void MacroAssembler::Push(FloatRegister f) {
  push(f);
  adjustFrame(int32_t(sizeof(double)));
}
void MacroAssembler::Pop(FloatRegister f) {
  pop(f);
  adjustFrame(-int32_t(sizeof(double)));
}
void MacroAssembler::Pop(const ValueOperand& val) {
  popValue(val);
  adjustFrame(-int32_t(sizeof(Value)));
}

// static
size_t MacroAssembler::PushRegsInMaskSizeInBytes(LiveRegisterSet set) {
  return set.gprs().size() * sizeof(intptr_t) + set.fpus().getPushSizeInBytes();
}

void MacroAssembler::storeRegsInMask(LiveRegisterSet set, Address dest,
                                     Register scratch) {
  FloatRegisterSet fpuSet(set.fpus().reduceSetForPush());
  mozilla::DebugOnly<unsigned> numFpu = fpuSet.size();
  mozilla::DebugOnly<int32_t> diffF = fpuSet.getPushSizeInBytes();
  mozilla::DebugOnly<int32_t> diffG = set.gprs().size() * sizeof(intptr_t);

  MOZ_ASSERT(dest.offset >= diffG + diffF);

  for (GeneralRegisterBackwardIterator iter(set.gprs()); iter.more(); ++iter) {
    diffG -= sizeof(intptr_t);
    dest.offset -= sizeof(intptr_t);
    storePtr(*iter, dest);
  }
  MOZ_ASSERT(diffG == 0);

  // Natural per-kind slot. See PushRegsInMask comment.
  for (FloatRegisterBackwardIterator iter(fpuSet); iter.more(); ++iter) {
    FloatRegister reg = *iter;
    diffF -= reg.size();
    numFpu -= 1;
    dest.offset -= reg.size();
    if (reg.isSimd128()) {
      storeUnalignedSimd128(reg, dest);
    } else {
      storeDouble(reg.asDouble(), dest);
    }
  }
  MOZ_ASSERT(diffF == 0);
}

void MacroAssembler::freeStackTo(uint32_t framePushed) {
  MOZ_ASSERT(framePushed <= framePushed_);
  // SP = FP - framePushed
  movePtr(FramePointer, StackPointer);
  if (framePushed) {
    subPtr(Imm32(framePushed), StackPointer);
  }
  framePushed_ = framePushed;
}

// ===============================================================
// Additional call / patch operations.

void MacroAssembler::call(JitCode* c) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  uint64_t addr = uintptr_t(c->raw());
  BufferOffset bo = emitLoad64Stanza(scratch, addr);
  addPendingJump(bo, ImmPtr(c->raw()), RelocationKind::JITCODE);

  callJitNoProfiler(scratch);
}

CodeOffset MacroAssembler::callWithPatch() {
  // Emit a CallTag-sized stanza of nops. Will be patched by patchCall.
  // Return offset AFTER the stanza (= the return address when bl executes).
  m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  writeInst(NopInst);
  m_buffer.leaveNoPool();
  return CodeOffset(currentOffset());
}

void MacroAssembler::patchCall(uint32_t callerOffset, uint32_t calleeOffset) {
  // callerOffset points AFTER the 10-instruction stanza (the return address).
  // Subtract to find the stanza start. The `bl` goes at inst[9].
  uint32_t stanzaStart = callerOffset - 10 * sizeof(uint32_t);
  Instruction* i0 = (Instruction*)(m_buffer.getInst(BufferOffset(stanzaStart)));
  // bl offset is relative to inst[9], which is at stanzaStart + 36.
  intptr_t blAddr = (intptr_t)stanzaStart + 9 * (intptr_t)sizeof(uint32_t);
  intptr_t callOffset = (intptr_t)calleeOffset - blAddr;
  if (JOffImm26::IsInRange(callOffset)) {
    i0[0].makeNop();
    i0[1].makeNop();
    i0[2].makeNop();
    i0[3].makeNop();
    i0[4].makeNop();
    i0[5].makeNop();
    i0[6].makeNop();
    i0[7].makeNop();
    i0[8].makeNop();
    i0[9].setData(PPC_b | JOffImm26(callOffset).encode() | LinkB);
  } else {
    addLongJump(BufferOffset(stanzaStart), BufferOffset(calleeOffset));
    WriteLoad64Instructions(i0, SecondScratchReg, LabelBase::INVALID_OFFSET);
    i0[8].makeOp_mtctr(SecondScratchReg);
    i0[9].makeOp_bctr(LinkB);
  }
}

void MacroAssembler::patchFarJump(CodeOffset farJump, uint32_t targetOffset) {
  Instruction* inst =
      (Instruction*)m_buffer.getInst(BufferOffset(farJump.offset()));
  // Extract the destination register from the existing stanza. Both shapes
  // encode rD at LE bits [21..25] of their first "register-touching" slot:
  // P8 = mflr rD at [2], P9+ = addpcis rD at [0]. Major opcode of slot [0]
  // distinguishes (31 = mfspr, 19 = addpcis).
  uint32_t i0 = inst[0].encode();
  uint32_t regCode = (((i0 >> 26) & 0x3f) == 19)
                         ? ((i0 >> 21) & 0x1f)
                         : ((inst[2].encode() >> 21) & 0x1f);
  Register reg = Register::FromCode(regCode);
  WriteLoad64Instructions(inst, reg, LabelBase::INVALID_OFFSET);
  addLongJump(BufferOffset(farJump.offset()), BufferOffset(targetOffset));
}

// static
void MacroAssembler::patchFarJump(uint8_t* farJump, uint8_t* target) {
  UpdateLoad64Value((Instruction*)farJump, (uint64_t)(uintptr_t)target);
  FlushICache(farJump, 8 * sizeof(Instruction));
}

// static
void MacroAssembler::patchNopToCall(uint8_t* callsite, uint8_t* target) {
  // callsite points AFTER the 10-instruction stanza. Subtract to find start.
  Instruction* inst = (Instruction*)callsite - 10;
  WriteLoad64Instructions(inst, SecondScratchReg, (uint64_t)(uintptr_t)target);
  inst[8].makeOp_mtctr(SecondScratchReg);
  inst[9].makeOp_bctr(LinkB);
  FlushICache(inst, 10 * sizeof(Instruction));
}

// static
void MacroAssembler::patchCallToNop(uint8_t* callsite) {
  // callsite points AFTER the 10-instruction stanza. Subtract to find start.
  Instruction* inst = (Instruction*)callsite - 10;
  for (int i = 0; i < 10; i++) {
    inst[i].makeNop();
  }
  FlushICache(inst, 10 * sizeof(Instruction));
}

void MacroAssembler::patchMove32(CodeOffset offset, Imm32 n) {
  // Patch an 8-instruction load64 sequence with a 32-bit value.
  Instruction* inst =
      (Instruction*)m_buffer.getInst(BufferOffset(offset.offset()));
  UpdateLoad64Value(inst, uint64_t(int64_t(n.value)));
}

uint32_t MacroAssembler::pushFakeReturnAddress(Register scratch) {
  CodeLabel cl;

  // Use mov(CodeLabel*, Register) which always emits a full 8-instruction
  // load64 sequence (via NOPs + WriteLoad64Instructions). This is critical
  // because movePtr(ImmWord(0)) would optimize to a single li instruction,
  // but processCodeLabels->Bind->UpdateLoad64Value expects the full
  // 8-instruction literal pool sequence at the patchAt offset.
  mov(&cl, scratch);

  Push(scratch);

  bind(&cl);
  uint32_t retAddr = currentOffset();

  addCodeLabel(cl);
  return retAddr;
}

void MacroAssembler::callWithABINoProfiler(Register fun, ABIType result) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // Save fun to scratch since fun might be clobbered by callWithABIPre.
  movePtr(fun, scratch);

  uint32_t stackAdjust;
  callWithABIPre(&stackAdjust);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  callABIDescriptorELFv1(scratch);
#else
  call(scratch);
#endif
  callWithABIPost(stackAdjust, result);
}

// ===============================================================
// Additional arithmetic helpers.

void MacroAssembler::flexibleRemainder32(Register lhs, Register rhs,
                                         Register dest, bool isUnsigned,
                                         const LiveRegisterSet&) {
  // rem = lhs - (lhs/rhs)*rhs
  // PPC64 divw(INT32_MIN, -1) is undefined; result is 0.
  Label done;
  if (!isUnsigned) {
    Label notMinOverflow;
    branchPtr(Assembler::NotEqual, lhs, ImmWord(INT32_MIN), &notMinOverflow);
    branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinOverflow);
    move32(Imm32(0), dest);
    jump(&done);
    bind(&notMinOverflow);
  }
  if (HasPOWER9()) {
    if (isUnsigned) {
      as_moduw(dest, lhs, rhs);
    } else {
      as_modsw(dest, lhs, rhs);
    }
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    if (isUnsigned) {
      as_divwu(scratch, lhs, rhs);
    } else {
      as_divw(scratch, lhs, rhs);
    }
    as_mullw(scratch, scratch, rhs);
    as_subf(dest, scratch, lhs);
  }
  as_extsw(dest, dest);
  bind(&done);
}

void MacroAssembler::flexibleQuotientPtr(Register lhs, Register rhs,
                                         Register dest, bool isUnsigned,
                                         const LiveRegisterSet&) {
  // PPC64 divd(INT64_MIN, -1) is undefined; return INT64_MIN to match
  // ARM64/LoongArch64 hardware sdiv behavior.
  Label done;
  if (!isUnsigned) {
    Label notMinOverflow;
    branchPtr(Assembler::NotEqual, lhs, ImmWord(INT64_MIN), &notMinOverflow);
    branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinOverflow);
    movePtr(ImmWord(INT64_MIN), dest);
    jump(&done);
    bind(&notMinOverflow);
  }
  if (isUnsigned) {
    as_divdu(dest, lhs, rhs);
  } else {
    as_divd(dest, lhs, rhs);
  }
  bind(&done);
}

void MacroAssembler::flexibleRemainderPtr(Register lhs, Register rhs,
                                          Register dest, bool isUnsigned,
                                          const LiveRegisterSet&) {
  // rem = lhs - (lhs/rhs)*rhs
  // PPC64 divd(INT64_MIN, -1) is undefined; result is 0.
  Label done;
  if (!isUnsigned) {
    Label notMinOverflow;
    branchPtr(Assembler::NotEqual, lhs, ImmWord(INT64_MIN), &notMinOverflow);
    branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinOverflow);
    movePtr(ImmWord(0), dest);
    jump(&done);
    bind(&notMinOverflow);
  }
  if (HasPOWER9()) {
    if (isUnsigned) {
      as_modud(dest, lhs, rhs);
    } else {
      as_modsd(dest, lhs, rhs);
    }
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    if (isUnsigned) {
      as_divdu(scratch, lhs, rhs);
    } else {
      as_divd(scratch, lhs, rhs);
    }
    as_mulld(scratch, scratch, rhs);
    as_subf(dest, scratch, lhs);
  }
  bind(&done);
}

// ===============================================================
// Rounding helpers.

void MacroAssembler::floorDoubleToInt32(FloatRegister src, Register dest,
                                        Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  // Round toward negative infinity, then convert to int64.
  as_frim(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  as_mfvsrd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    // If top 2 bits of src are set, it's negative or NaN.
    as_mfvsrd(dest, src);
    // rldicl. = x_srdi + record form: dest = top 2 bits, CR0[eq]=(dest==0).
    // Folds the explicit cmpdi src,0 that would otherwise drive the branch.
    as_rldicl_rc(dest, dest, 2, 62);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::floorFloat32ToInt32(FloatRegister src, Register dest,
                                         Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  // PPC FP rounding works on doubles. Single-precision FPRs are
  // already in double-width registers, so frim works fine.
  as_frim(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  as_mfvsrd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    // src is held in the FPR as a 64-bit double (lfs widens float32 to
    // double on load), so the same top-2-bits check used for doubles
    // applies: bit 63 = sign, bit 62 = exponent MSB. Nonzero means -0,
    // ±Inf, NaN, or a large magnitude — none of which is +0.
    as_mfvsrd(dest, src);
    // rldicl. = x_srdi + record form: dest = top 2 bits, CR0[eq]=(dest==0).
    // Folds the explicit cmpdi src,0 that would otherwise drive the branch.
    as_rldicl_rc(dest, dest, 2, 62);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::ceilDoubleToInt32(FloatRegister src, Register dest,
                                       Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  as_frip(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  as_mfvsrd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for (-1, -0] and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    // If binary value is not zero, input was not 0 (could be -0 or NaN).
    as_mfvsrd(dest, src);
    as_cmpdi(dest, 0);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::ceilFloat32ToInt32(FloatRegister src, Register dest,
                                        Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  as_frip(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  as_mfvsrd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for (-1, -0] and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    as_mfvsrd(dest, src);
    as_cmpdi(dest, 0);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::truncDoubleToInt32(FloatRegister src, Register dest,
                                        Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  as_fctidz(fpscratch, src);
  as_mfvsrd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    as_mfvsrd(dest, src);
    // rldicl. = x_srdi + record form: dest = top 2 bits, CR0[eq]=(dest==0).
    // Folds the explicit cmpdi src,0 that would otherwise drive the branch.
    as_rldicl_rc(dest, dest, 2, 62);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::truncFloat32ToInt32(FloatRegister src, Register dest,
                                         Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  as_fctidz(fpscratch, src);
  as_mfvsrd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    as_mfvsrd(dest, src);
    // rldicl. = x_srdi + record form: dest = top 2 bits, CR0[eq]=(dest==0).
    // Folds the explicit cmpdi src,0 that would otherwise drive the branch.
    as_rldicl_rc(dest, dest, 2, 62);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::roundDoubleToInt32(FloatRegister src, Register dest,
                                        FloatRegister temp, Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  Label negative, end, performRound;

  // Branch for negative inputs.
  zeroDouble(fpscratch);
  branchDouble(DoubleGreaterThanOrEqual, src, fpscratch, &performRound);

  // Input is negative.
  loadConstantDouble(-0.5, fpscratch);
  branchDouble(DoubleGreaterThanOrEqual, src, fpscratch, fail);
  jump(&performRound);

  bind(&performRound);
  {
    loadConstantDouble(GetBiggestNumberLessThan(0.5), temp);
    as_fadd(fpscratch, src, temp);
    as_frim(fpscratch, fpscratch);
    as_fctidz(fpscratch, fpscratch);
    as_mfvsrd(dest, fpscratch);

    // Check if result fits in int32.
    as_extsw(scratch, dest);
    as_cmpd(dest, scratch);
    ma_b(NotEqual, fail);
  }
  bind(&end);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    as_mfvsrd(dest, src);
    as_cmpdi(dest, 0);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

void MacroAssembler::roundFloat32ToInt32(FloatRegister src, Register dest,
                                         FloatRegister temp, Label* fail) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();

  Label negative, end, performRound;

  // Branch for non-negative inputs.
  loadConstantFloat32(0.0f, fpscratch);
  branchFloat(DoubleGreaterThanOrEqual, src, fpscratch, &performRound);

  // Input is negative.
  loadConstantFloat32(-0.5f, fpscratch);
  branchFloat(DoubleGreaterThanOrEqual, src, fpscratch, fail);
  jump(&performRound);

  bind(&performRound);
  {
    loadConstantFloat32(float(GetBiggestNumberLessThan(0.5)), temp);
    as_fadds(fpscratch, src, temp);
    as_frim(fpscratch, fpscratch);
    as_fctidz(fpscratch, fpscratch);
    as_mfvsrd(dest, fpscratch);

    // Check if result fits in int32.
    as_extsw(scratch, dest);
    as_cmpd(dest, scratch);
    ma_b(NotEqual, fail);
  }
  bind(&end);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    as_mfvsrd(dest, src);
    as_cmpdi(dest, 0);
    ma_b(NotEqual, fail);
  }
  bind(&notZero);
}

// ===============================================================
// FP conversion / copy-sign.

void MacroAssembler::convertIntPtrToDouble(Register src, FloatRegister dest) {
  convertInt64ToDouble(Register64(src), dest);
}

void MacroAssembler::copySignDouble(FloatRegister lhs, FloatRegister rhs,
                                    FloatRegister output) {
  // fcpsgn frt, fra, frb: copies sign of fra to magnitude of frb.
  // lhs = magnitude source, rhs = sign source.
  as_fcpsgn(output, rhs, lhs);
}

void MacroAssembler::copySignFloat32(FloatRegister lhs, FloatRegister rhs,
                                     FloatRegister output) {
  as_fcpsgn(output, rhs, lhs);
}

// ===============================================================
// GC / nursery helpers.

void MacroAssembler::loadStoreBuffer(Register ptr, Register buffer) {
  andPtr(Imm32(int32_t(~gc::ChunkMask)), ptr, buffer);
  loadPtr(Address(buffer, gc::ChunkStoreBufferOffset), buffer);
}

void MacroAssembler::branchValueIsNurseryCell(Condition cond,
                                              const Address& address,
                                              Register temp, Label* label) {
  branchValueIsNurseryCellImpl(cond, address, temp, label);
}

template <typename T>
void MacroAssembler::branchValueIsNurseryCellImpl(Condition cond,
                                                  const T& value, Register temp,
                                                  Label* label) {
  MOZ_ASSERT(cond == Assembler::Equal || cond == Assembler::NotEqual);
  MOZ_ASSERT(temp != InvalidReg);
  Label done;
  branchTestGCThing(Assembler::NotEqual, value,
                    cond == Assembler::Equal ? &done : label);

  getGCThingValueChunk(value, temp);
  loadPtr(Address(temp, gc::ChunkStoreBufferOffset), temp);
  branchPtr(InvertCondition(cond), temp, ImmWord(0), label);

  bind(&done);
}

// ===============================================================
// Template instantiations.

template <typename T>
void MacroAssembler::storeUnboxedValue(const ConstantOrRegister& value,
                                       MIRType valueType, const T& dest) {
  MOZ_ASSERT(valueType < MIRType::Value);

  if (valueType == MIRType::Double) {
    boxDouble(value.reg().typedReg().fpu(), dest);
    return;
  }

  if (value.constant()) {
    storeValue(value.value(), dest);
  } else {
    storeValue(ValueTypeFromMIRType(valueType), value.reg().typedReg().gpr(),
               dest);
  }
}

template void MacroAssembler::storeUnboxedValue(const ConstantOrRegister& value,
                                                MIRType valueType,
                                                const Address& dest);
template void MacroAssembler::storeUnboxedValue(
    const ConstantOrRegister& value, MIRType valueType,
    const BaseObjectElementIndex& dest);

// ===============================================================
// Misc stubs.

void MacroAssembler::comment(const char* msg) {}

void MacroAssembler::speculationBarrier() {
  // isync provides execution synchronization: discards prefetched
  // instructions and forces a refetch+reexecute past the barrier.
  // No instruction following isync may begin (architecturally) until
  // isync completes, blocking speculative bypass — exactly the
  // Spectre v1 guarantee needed after a C call returns a value that
  // may influence subsequent loads. Reachable from shared
  // CodeGenerator under JitOptions.spectreJitToCxxCalls.
  as_isync();
}

void MacroAssembler::atomicPause() { nop(); }

void MacroAssembler::enterFakeExitFrameForWasm(Register cxreg, Register scratch,
                                               ExitFrameType type) {
  enterFakeExitFrame(cxreg, scratch, type);
}

void MacroAssembler::wasmBoundsCheck32(Condition cond, Register index,
                                       Register boundsCheckLimit,
                                       Label* label) {
  ma_cmp(index, boundsCheckLimit, cond);
  ma_b(cond, label);
}

void MacroAssembler::wasmBoundsCheck32(Condition cond, Register index,
                                       Address boundsCheckLimit, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(boundsCheckLimit, scratch);
  ma_cmp(index, scratch, cond);
  ma_b(cond, label);
}

void MacroAssembler::wasmBoundsCheck64(Condition cond, Register64 index,
                                       Register64 boundsCheckLimit,
                                       Label* label) {
  ma_cmp(index.reg, boundsCheckLimit.reg, cond);
  ma_b(cond, label);
}

void MacroAssembler::wasmBoundsCheck64(Condition cond, Register64 index,
                                       Address boundsCheckLimit, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(boundsCheckLimit, scratch);
  ma_cmp(index.reg, scratch, cond);
  ma_b(cond, label);
}

CodeOffset MacroAssembler::move32WithPatch(Register dest) {
  CodeOffset offset(currentOffset());
  emitLoad64Stanza(dest, 0);
  return offset;
}

CodeOffset MacroAssembler::sub32FromMemAndBranchIfNegativeWithPatch(
    Address address, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(scratch != address.base);
  load32(address, scratch);
  // Subtract a placeholder value (will be patched).
  // Use addi with positive placeholder (128), which will be patched to
  // addi with negative value. The immediate is in the addi instruction.
  as_addi(scratch, scratch, 128);
  CodeOffset patchPoint = CodeOffset(currentOffset());
  store32(scratch, address);
  // Branch if result is negative (signed).
  as_cmpwi(scratch, 0);
  ma_b(LessThan, label);
  return patchPoint;
}

bool MacroAssembler::convertUInt64ToDoubleNeedsTemp() { return false; }

void MacroAssembler::call(ImmWord imm) { call(ImmPtr((void*)imm.value)); }

void MacroAssembler::convertUInt64ToDouble(Register64 src, FloatRegister dest,
                                           Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  // POWER7+ has fcfidu (unsigned i64 → f64) as a single instruction; no
  // sign-split / branch / GPR scratch needed.
  as_mtvsrd(dest, src.reg);
  as_fcfidu(dest, dest);
}

void MacroAssembler::convertInt64ToFloat32(Register64 src, FloatRegister dest) {
  as_mtvsrd(dest, src.reg);
  as_fcfids(dest, dest);
}

void MacroAssembler::convertUInt64ToFloat32(Register64 src, FloatRegister dest,
                                            Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  // POWER7+ has fcfidus (unsigned i64 → f32) as a single instruction.
  as_mtvsrd(dest, src.reg);
  as_fcfidus(dest, dest);
}

void MacroAssembler::flexibleQuotient32(
    Register lhs, Register rhs, Register dest, bool isUnsigned,
    const LiveRegisterSet& volatileLiveRegs) {
  // PPC64 divw(INT32_MIN, -1) is undefined; return INT32_MIN to match
  // ARM64/LoongArch64 hardware sdiv behavior.
  Label done;
  if (!isUnsigned) {
    Label notMinOverflow;
    branchPtr(Assembler::NotEqual, lhs, ImmWord(INT32_MIN), &notMinOverflow);
    branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinOverflow);
    move32(Imm32(INT32_MIN), dest);
    jump(&done);
    bind(&notMinOverflow);
  }
  if (isUnsigned) {
    as_divwu(dest, lhs, rhs);
  } else {
    as_divw(dest, lhs, rhs);
  }
  as_extsw(dest, dest);
  bind(&done);
}

void MacroAssembler::oolWasmTruncateCheckF32ToI32(
    FloatRegister input, Register output, TruncFlags flags,
    const wasm::TrapSiteDesc& trapSiteDesc, Label* rejoin) {
  outOfLineWasmTruncateToInt32Check(input, output, MIRType::Float32, flags,
                                    rejoin, trapSiteDesc);
}

void MacroAssembler::oolWasmTruncateCheckF32ToI64(
    FloatRegister input, Register64 output, TruncFlags flags,
    const wasm::TrapSiteDesc& trapSiteDesc, Label* rejoin) {
  outOfLineWasmTruncateToInt64Check(input, output, MIRType::Float32, flags,
                                    rejoin, trapSiteDesc);
}

void MacroAssembler::oolWasmTruncateCheckF64ToI32(
    FloatRegister input, Register output, TruncFlags flags,
    const wasm::TrapSiteDesc& trapSiteDesc, Label* rejoin) {
  outOfLineWasmTruncateToInt32Check(input, output, MIRType::Double, flags,
                                    rejoin, trapSiteDesc);
}

void MacroAssembler::oolWasmTruncateCheckF64ToI64(
    FloatRegister input, Register64 output, TruncFlags flags,
    const wasm::TrapSiteDesc& trapSiteDesc, Label* rejoin) {
  outOfLineWasmTruncateToInt64Check(input, output, MIRType::Double, flags,
                                    rejoin, trapSiteDesc);
}

void MacroAssemblerPPC64Compat::outOfLineWasmTruncateToInt32Check(
    FloatRegister input, Register output, MIRType fromType, TruncFlags flags,
    Label* rejoin, const wasm::TrapSiteDesc& trapSiteDesc) {
  bool isUnsigned = flags & TRUNC_UNSIGNED;
  bool isSaturating = flags & TRUNC_SATURATING;

  if (isSaturating) {
    ScratchDoubleScope fpscratch(asMasm());
    if (fromType == MIRType::Double) {
      asMasm().loadConstantDouble(0.0, fpscratch);
    } else {
      asMasm().loadConstantFloat32(0.0f, fpscratch);
    }

    if (isUnsigned) {
      // If input < 0 or NaN, output = 0; else output = UINT32_MAX.
      Label notNegOrNaN;
      if (fromType == MIRType::Double) {
        asMasm().branchDouble(Assembler::DoubleGreaterThanOrEqual, input,
                              fpscratch, &notNegOrNaN);
      } else {
        asMasm().branchFloat(Assembler::DoubleGreaterThanOrEqual, input,
                             fpscratch, &notNegOrNaN);
      }
      asMasm().move32(Imm32(0), output);
      asMasm().jump(rejoin);
      asMasm().bind(&notNegOrNaN);
      asMasm().move32(Imm32(UINT32_MAX), output);
    } else {
      // Signed: NaN -> 0, negative overflow -> INT32_MIN,
      // positive overflow already saturated to INT32_MAX.
      Label notNaN, done;
      if (fromType == MIRType::Double) {
        asMasm().branchDouble(Assembler::DoubleOrdered, input, input, &notNaN);
      } else {
        asMasm().branchFloat(Assembler::DoubleOrdered, input, input, &notNaN);
      }
      asMasm().move32(Imm32(0), output);
      asMasm().jump(rejoin);

      asMasm().bind(&notNaN);
      if (fromType == MIRType::Double) {
        asMasm().branchDouble(Assembler::DoubleGreaterThanOrEqual, input,
                              fpscratch, rejoin);
      } else {
        asMasm().branchFloat(Assembler::DoubleGreaterThanOrEqual, input,
                             fpscratch, rejoin);
      }
      asMasm().move32(Imm32(INT32_MIN), output);
    }

    MOZ_ASSERT(rejoin->bound());
    asMasm().jump(rejoin);
    return;
  }

  Label inputIsNaN;
  if (fromType == MIRType::Double) {
    asMasm().branchDouble(Assembler::DoubleUnordered, input, input,
                          &inputIsNaN);
  } else {
    asMasm().branchFloat(Assembler::DoubleUnordered, input, input, &inputIsNaN);
  }

  asMasm().wasmTrap(wasm::Trap::IntegerOverflow, trapSiteDesc);
  asMasm().bind(&inputIsNaN);
  asMasm().wasmTrap(wasm::Trap::InvalidConversionToInteger, trapSiteDesc);
}

void MacroAssemblerPPC64Compat::outOfLineWasmTruncateToInt64Check(
    FloatRegister input, Register64 output_, MIRType fromType, TruncFlags flags,
    Label* rejoin, const wasm::TrapSiteDesc& trapSiteDesc) {
  bool isUnsigned = flags & TRUNC_UNSIGNED;
  bool isSaturating = flags & TRUNC_SATURATING;

  if (isSaturating) {
    ScratchDoubleScope fpscratch(asMasm());
    Register output = output_.reg;

    if (fromType == MIRType::Double) {
      asMasm().loadConstantDouble(0.0, fpscratch);
    } else {
      asMasm().loadConstantFloat32(0.0f, fpscratch);
    }

    if (isUnsigned) {
      Label notNegOrNaN;
      if (fromType == MIRType::Double) {
        asMasm().branchDouble(Assembler::DoubleGreaterThanOrEqual, input,
                              fpscratch, &notNegOrNaN);
      } else {
        asMasm().branchFloat(Assembler::DoubleGreaterThanOrEqual, input,
                             fpscratch, &notNegOrNaN);
      }
      asMasm().movePtr(ImmWord(0), output);
      asMasm().jump(rejoin);
      asMasm().bind(&notNegOrNaN);
      asMasm().movePtr(ImmWord(UINT64_MAX), output);
    } else {
      Label notNaN;
      if (fromType == MIRType::Double) {
        asMasm().branchDouble(Assembler::DoubleOrdered, input, input, &notNaN);
      } else {
        asMasm().branchFloat(Assembler::DoubleOrdered, input, input, &notNaN);
      }
      asMasm().movePtr(ImmWord(0), output);
      asMasm().jump(rejoin);

      asMasm().bind(&notNaN);
      if (fromType == MIRType::Double) {
        asMasm().branchDouble(Assembler::DoubleGreaterThanOrEqual, input,
                              fpscratch, rejoin);
      } else {
        asMasm().branchFloat(Assembler::DoubleGreaterThanOrEqual, input,
                             fpscratch, rejoin);
      }
      asMasm().movePtr(ImmWord(INT64_MIN), output);
    }

    MOZ_ASSERT(rejoin->bound());
    asMasm().jump(rejoin);
    return;
  }

  Label inputIsNaN;
  if (fromType == MIRType::Double) {
    asMasm().branchDouble(Assembler::DoubleUnordered, input, input,
                          &inputIsNaN);
  } else {
    asMasm().branchFloat(Assembler::DoubleUnordered, input, input, &inputIsNaN);
  }

  asMasm().wasmTrap(wasm::Trap::IntegerOverflow, trapSiteDesc);
  asMasm().bind(&inputIsNaN);
  asMasm().wasmTrap(wasm::Trap::InvalidConversionToInteger, trapSiteDesc);
}

void MacroAssembler::PopStackPtr() {
  loadPtr(Address(StackPointer, 0), StackPointer);
  adjustFrame(-int32_t(sizeof(intptr_t)));
}

void MacroAssembler::patchSub32FromMemAndBranchIfNegative(CodeOffset offset,
                                                          Imm32 imm) {
  int32_t val = imm.value;
  MOZ_RELEASE_ASSERT(val >= 1 && val <= 127);
  // Patch the addi instruction that's right before patchPoint.
  // addi is 1 instruction before the CodeOffset (which is after the addi).
  Instruction* inst =
      (Instruction*)m_buffer.getInst(BufferOffset(offset.offset() - 4));
  // Rewrite the immediate field to -val.
  // PPC addi: opcode(6) | RT(5) | RA(5) | SI(16)
  uint32_t instWord = inst->encode();
  uint32_t base = instWord & 0xffff0000;
  inst->setData(base | (uint16_t)(-val & 0xffff));
}

void MacroAssembler::wasmTruncateDoubleToInt32(FloatRegister input,
                                               Register output,
                                               bool isSaturating,
                                               Label* oolEntry) {
  ScratchDoubleScope fpscratch(asMasm());
  // Clear VXCVI (bit 23) before the conversion so we can detect overflow.
  as_mtfsb0(23);
  as_fctiwz(fpscratch, input);
  as_mfvsrd(output, fpscratch);
  as_extsw(output, output);
  // Move FPSCR field 5 (which contains VXCVI) to CR0.
  // If the conversion was invalid (NaN or out-of-range), VXCVI=1 → SO set.
  as_mcrfs(cr0, 5);
  ma_b(SOBit, oolEntry);
}

void MacroAssembler::wasmTruncateDoubleToUInt32(FloatRegister input,
                                                Register output,
                                                bool isSaturating,
                                                Label* oolEntry) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // Always check for NaN — the ool handler clamps for saturating mode.
  as_fcmpu(input, input);
  ma_b(DoubleUnordered, oolEntry);
  as_fctidz(fpscratch, input);
  as_mfvsrd(output, fpscratch);
  x_srdi(scratch, output, 32);
  as_extsw(output, output);
  as_cmpdi(scratch, 0);
  ma_b(NotEqual, oolEntry);
}

void MacroAssembler::wasmTruncateFloat32ToInt32(FloatRegister input,
                                                Register output,
                                                bool isSaturating,
                                                Label* oolEntry) {
  ScratchDoubleScope fpscratch(asMasm());
  as_mtfsb0(23);
  as_fctiwz(fpscratch, input);
  as_mfvsrd(output, fpscratch);
  as_extsw(output, output);
  as_mcrfs(cr0, 5);
  ma_b(SOBit, oolEntry);
}

void MacroAssembler::wasmTruncateFloat32ToUInt32(FloatRegister input,
                                                 Register output,
                                                 bool isSaturating,
                                                 Label* oolEntry) {
  ScratchDoubleScope fpscratch(asMasm());
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  as_fcmpu(input, input);
  ma_b(DoubleUnordered, oolEntry);
  as_fctidz(fpscratch, input);
  as_mfvsrd(output, fpscratch);
  x_srdi(scratch, output, 32);
  as_extsw(output, output);
  as_cmpdi(scratch, 0);
  ma_b(NotEqual, oolEntry);
}

void MacroAssembler::wasmTruncateDoubleToInt64(
    FloatRegister input, Register64 output, bool isSaturating, Label* oolEntry,
    Label* oolRejoin, FloatRegister tempDouble) {
  MOZ_ASSERT(tempDouble.isInvalid());
  ScratchDoubleScope fpscratch(asMasm());
  as_mtfsb0(23);
  as_fctidz(fpscratch, input);
  as_mfvsrd(output.reg, fpscratch);
  as_mcrfs(cr0, 5);
  ma_b(SOBit, oolEntry);
  if (isSaturating) {
    bind(oolRejoin);
  }
}

void MacroAssembler::wasmTruncateFloat32ToInt64(
    FloatRegister input, Register64 output, bool isSaturating, Label* oolEntry,
    Label* oolRejoin, FloatRegister tempFloat) {
  MOZ_ASSERT(tempFloat.isInvalid());
  ScratchDoubleScope fpscratch(asMasm());
  as_mtfsb0(23);
  as_fctidz(fpscratch, input);
  as_mfvsrd(output.reg, fpscratch);
  as_mcrfs(cr0, 5);
  ma_b(SOBit, oolEntry);
  if (isSaturating) {
    bind(oolRejoin);
  }
}

void MacroAssembler::wasmTruncateDoubleToUInt64(
    FloatRegister input, Register64 output, bool isSaturating, Label* oolEntry,
    Label* oolRejoin, FloatRegister tempDouble) {
  MOZ_ASSERT(tempDouble.isInvalid());
  ScratchDoubleScope fpscratch(asMasm());
  as_mtfsb0(23);
  as_fctiduz(fpscratch, input);
  as_mfvsrd(output.reg, fpscratch);
  as_mcrfs(cr0, 5);
  ma_b(SOBit, oolEntry);
  if (isSaturating) {
    bind(oolRejoin);
  }
}

void MacroAssembler::wasmTruncateFloat32ToUInt64(
    FloatRegister input, Register64 output, bool isSaturating, Label* oolEntry,
    Label* oolRejoin, FloatRegister tempFloat) {
  MOZ_ASSERT(tempFloat.isInvalid());
  ScratchDoubleScope fpscratch(asMasm());
  as_mtfsb0(23);
  as_fctiduz(fpscratch, input);
  as_mfvsrd(output.reg, fpscratch);
  as_mcrfs(cr0, 5);
  ma_b(SOBit, oolEntry);
  if (isSaturating) {
    bind(oolRejoin);
  }
}

void MacroAssemblerPPC64Compat::profilerEnterFrame(Register framePtr,
                                                   Register scratch) {
  asMasm().loadJSContext(scratch);
  loadPtr(Address(scratch, offsetof(JSContext, profilingActivation_)), scratch);
  storePtr(framePtr,
           Address(scratch, JitActivation::offsetOfLastProfilingFrame()));
  storePtr(ImmPtr(nullptr),
           Address(scratch, JitActivation::offsetOfLastProfilingCallSite()));
}

void MacroAssemblerPPC64Compat::profilerExitFrame() {
  jump(asMasm().runtime()->jitRuntime()->getProfilerExitFrameTail());
}

void MacroAssemblerPPC64Compat::ma_mod_mask(Register src, Register dest,
                                            Register hold, Register remain,
                                            int32_t shift, Label* negZero) {
  // Compute x % ((1<<shift) - 1) by digit-summing in base b = 1<<shift.
  // Since b % (b-1) == 1, x % (b-1) == sum of base-b digits of x, mod (b-1).
  int32_t mask = (1 << shift) - 1;
  Label head, negative, sumSigned, done;

  as_or_(remain, src, src);  // move src -> remain
  xs_li(dest, 0);

  // Check sign (32-bit signed comparison)
  as_cmpwi(remain, 0);
  ma_b(Assembler::LessThan, &negative);
  xs_li(hold, 1);
  jump(&head);

  bind(&negative);
  xs_li(hold, -1);
  as_neg(remain, remain);
  as_rldicl(remain, remain, 0, 32);

  bind(&head);
  {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();

    // Extract bottom 'shift' bits: scratch = remain & mask
    move32(Imm32(mask), scratch);
    as_and_(scratch, remain, scratch);

    // Add to accumulator
    as_add(dest, dest, scratch);

    // Trial subtraction: scratch = dest - mask
    move32(Imm32(mask), scratch);
    as_subf(scratch, scratch, dest);  // scratch = dest - scratch

    // If (dest - mask) > 0, keep the subtracted value
    as_cmpwi(scratch, 0);
    ma_b(Assembler::LessThan, &sumSigned);
    as_or_(dest, scratch, scratch);  // dest = scratch
    bind(&sumSigned);

    // Shift out the bits we just processed
    x_srwi(remain, remain, shift);

    // Continue if remain != 0
    as_cmpwi(remain, 0);
    ma_b(Assembler::NotEqual, &head);
  }

  // If input was negative, negate result
  as_cmpwi(hold, 0);
  ma_b(Assembler::GreaterThanOrEqual, &done);

  if (negZero != nullptr) {
    as_cmpwi(dest, 0);
    ma_b(Assembler::Equal, negZero);
  }

  as_neg(dest, dest);
  as_extsw(dest, dest);

  bind(&done);
}

// ========================================================================
// Atomic operations.

template <typename T>
static void CompareExchange(MacroAssembler& masm,
                            const wasm::MemoryAccessDesc* access,
                            Scalar::Type type, Synchronization sync,
                            const T& mem, Register oldval, Register newval,
                            Register valueTemp, Register offsetTemp,
                            Register maskTemp, Register output) {
  UseScratchRegisterScope temps(masm);
  bool signExtend = Scalar::isSignedIntType(type);
  unsigned nbytes = Scalar::byteSize(type);

  switch (nbytes) {
    case 1:
    case 2:
      break;
    case 4:
      MOZ_ASSERT(valueTemp == InvalidReg);
      MOZ_ASSERT(offsetTemp == InvalidReg);
      MOZ_ASSERT(maskTemp == InvalidReg);
      break;
    default:
      MOZ_CRASH();
  }

  Label again, end;

  Register scratch = temps.Acquire();
  masm.computeEffectiveAddress(mem, scratch);

  if (nbytes == 4) {
    masm.memoryBarrierBefore(sync);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // wasm atomic memory is little-endian. Byte-reverse the replacement value
    // to native once before the loop, and the loaded value to LE inside the
    // loop (it is compared against the little-endian oldval and returned). JS
    // atomics (access == nullptr) are native byte order, so no swap.
    if (access) {
      masm.as_rldicl(newval, newval, 0, 32);
      masm.as_mtvsrd(ScratchDoubleReg, newval);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(newval, ScratchDoubleReg);
      masm.x_srdi(newval, newval, 32);
    }
#endif

    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(output, r0, scratch);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access) {
      masm.as_mtvsrd(ScratchDoubleReg, output);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(output, ScratchDoubleReg);
      masm.x_srdi(output, output, 32);
    }
#endif
    // ma_cmp(..., is32bit=true) emits cmpw, which compares only bits
    // 32:63 (low 32) of both operands per ISA v3.0B. The upper
    // 32 bits of oldval are ignored, so no canonicalising extsw needed.
    masm.ma_cmp(output, oldval, Assembler::NotEqual, /* is32bit */ true);
    masm.ma_b(Assembler::NotEqual, &end);
    masm.as_stwcx(newval, r0, scratch);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);
    masm.bind(&end);
    // lwarx zero-extends; sign-extend for 32-bit canonical form.
    masm.as_extsw(output, output);

    return;
  }

  // Sub-word (1 or 2 byte) compare-exchange via native lbarx/lharx +
  // stbcx./sthcx. POWER7+ (well below our POWER8 baseline). Replaces the prior
  // round-down-to-word
  // + mask + RMW dance. lXarx zero-extends the loaded byte/half; stXcx. stores
  // only the low 8/16 bits of RS, so no pre-masking is needed on the store
  // side. offsetTemp / maskTemp are still allocated by the lowering but unused
  // here.
  (void)offsetTemp;
  (void)maskTemp;

  masm.memoryBarrierBefore(sync);

  masm.bind(&again);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  switch (nbytes) {
    case 1:
      masm.as_lbarx(output, r0, scratch);
      if (signExtend) {
        masm.as_extsb(valueTemp, oldval);
        masm.as_extsb(output, output);
      } else {
        masm.as_andi_rc(valueTemp, oldval, 0xff);
      }
      break;
    case 2:
      masm.as_lharx(output, r0, scratch);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      // wasm atomic memory is little-endian; byte-reverse the loaded halfword
      // to LE so it compares against the (little-endian) oldval and is returned
      // correctly. offsetTemp is unused in the sub-word path. JS atomics
      // (access == nullptr) are native byte order, so no swap.
      if (access) {
        masm.as_rlwinm(offsetTemp, output, 8, 16, 23);
        masm.as_rlwinm(output, output, 24, 24, 31);
        masm.as_or_(output, output, offsetTemp);
      }
#endif
      if (signExtend) {
        masm.as_extsh(valueTemp, oldval);
        masm.as_extsh(output, output);
      } else {
        masm.as_rlwinm(valueTemp, oldval, 0, 16, 31);
      }
      break;
  }

  masm.ma_cmp(output, valueTemp, Assembler::NotEqual, /* is32bit */ true);
  masm.ma_b(Assembler::NotEqual, &end);

  if (nbytes == 1) {
    masm.as_stbcx(newval, r0, scratch);
  } else {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // Byte-reverse the replacement to native (rlwinm+rlwimi keeps it in
    // offsetTemp, leaving newval intact for retries); sthcx stores low 16 bits.
    // JS atomics (access == nullptr) are native byte order, so store as-is.
    if (access) {
      masm.as_rlwinm(offsetTemp, newval, 8, 16, 23);
      masm.as_rlwimi(offsetTemp, newval, 24, 24, 31);
      masm.as_sthcx(offsetTemp, r0, scratch);
    } else {
      masm.as_sthcx(newval, r0, scratch);
    }
#else
    masm.as_sthcx(newval, r0, scratch);
#endif
  }
  masm.ma_b(Assembler::NotEqual, &again);

  masm.memoryBarrierAfter(sync);

  masm.bind(&end);
}

template <typename T>
static void CompareExchange64(MacroAssembler& masm,
                              const wasm::MemoryAccessDesc* access,
                              Synchronization sync, const T& mem,
                              Register64 expect, Register64 replace,
                              Register64 output) {
  MOZ_ASSERT(expect != output && replace != output);
  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();
  masm.computeEffectiveAddress(mem, scratch);

  Label tryAgain;
  Label exit;

  masm.memoryBarrierBefore(sync);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // wasm atomic memory is little-endian. Byte-reverse the replacement to native
  // once before the loop, and the loaded value to LE inside the loop (compared
  // against the little-endian expect and returned). VSX scratch swap. JS atomics
  // (access == nullptr) are native byte order, so no swap.
  if (access) {
    masm.as_mtvsrd(ScratchDoubleReg, replace.reg);
    masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    masm.as_mfvsrd(replace.reg, ScratchDoubleReg);
  }
#endif

  masm.bind(&tryAgain);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  masm.as_ldarx(output.reg, r0, scratch);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (access) {
    masm.as_mtvsrd(ScratchDoubleReg, output.reg);
    masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    masm.as_mfvsrd(output.reg, ScratchDoubleReg);
  }
#endif

  masm.ma_cmp(output.reg, expect.reg, Assembler::NotEqual);
  masm.ma_b(Assembler::NotEqual, &exit);
  masm.as_stdcx(replace.reg, r0, scratch);
  masm.ma_b(Assembler::NotEqual, &tryAgain);

  masm.memoryBarrierAfter(sync);

  masm.bind(&exit);
}

template <typename T>
static void AtomicExchange(MacroAssembler& masm,
                           const wasm::MemoryAccessDesc* access,
                           Scalar::Type type, Synchronization sync,
                           const T& mem, Register value, Register valueTemp,
                           Register offsetTemp, Register maskTemp,
                           Register output) {
  UseScratchRegisterScope temps(masm);
  bool signExtend = Scalar::isSignedIntType(type);
  unsigned nbytes = Scalar::byteSize(type);

  switch (nbytes) {
    case 1:
    case 2:
      break;
    case 4:
      MOZ_ASSERT(valueTemp == InvalidReg);
      MOZ_ASSERT(offsetTemp == InvalidReg);
      MOZ_ASSERT(maskTemp == InvalidReg);
      break;
    default:
      MOZ_CRASH();
  }

  Label again;

  Register memTemp = temps.Acquire();
  masm.computeEffectiveAddress(mem, memTemp);

  if (nbytes == 4) {
    masm.memoryBarrierBefore(sync);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // wasm atomic memory is little-endian. Byte-reverse the to-be-stored value
    // to native once before the loop (it is constant across retries) and the
    // loaded old value to LE after the loop. Swap via the VSX scratch. JS
    // atomics (access == nullptr) are native byte order, so no swap.
    if (access) {
      masm.as_rldicl(value, value, 0, 32);
      masm.as_mtvsrd(ScratchDoubleReg, value);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(value, ScratchDoubleReg);
      masm.x_srdi(value, value, 32);
    }
#endif

    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(output, r0, memTemp);
    masm.as_stwcx(value, r0, memTemp);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access) {
      masm.as_mtvsrd(ScratchDoubleReg, output);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(output, ScratchDoubleReg);
      masm.x_srdi(output, output, 32);
    }
#endif
    // lwarx zero-extends; sign-extend for 32-bit canonical form.
    masm.as_extsw(output, output);

    return;
  }

  // Sub-word exchange via native lbarx/lharx + stbcx./sthcx. (POWER7+).
  // valueTemp / offsetTemp / maskTemp are still allocated by the lowering but
  // unused here.
  (void)valueTemp;
  (void)offsetTemp;
  (void)maskTemp;

  masm.memoryBarrierBefore(sync);

  masm.bind(&again);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  if (nbytes == 1) {
    masm.as_lbarx(output, r0, memTemp);
    masm.as_stbcx(value, r0, memTemp);
  } else {
    masm.as_lharx(output, r0, memTemp);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // wasm atomic memory is little-endian: store value byte-reversed to native
    // (rlwinm+rlwimi keeps it in offsetTemp, leaving value intact for retries)
    // and byte-reverse the loaded old value to LE. JS atomics (access ==
    // nullptr) are native byte order, so store/return as-is.
    if (access) {
      masm.as_rlwinm(offsetTemp, value, 8, 16, 23);
      masm.as_rlwimi(offsetTemp, value, 24, 24, 31);
      masm.as_sthcx(offsetTemp, r0, memTemp);
      masm.as_rlwinm(offsetTemp, output, 8, 16, 23);
      masm.as_rlwimi(offsetTemp, output, 24, 24, 31);
      masm.as_or_(output, offsetTemp, offsetTemp);
    } else {
      masm.as_sthcx(value, r0, memTemp);
    }
#else
    masm.as_sthcx(value, r0, memTemp);
#endif
  }
  masm.ma_b(Assembler::NotEqual, &again);

  if (signExtend) {
    if (nbytes == 1) {
      masm.as_extsb(output, output);
    } else {
      masm.as_extsh(output, output);
    }
  }
  // Unsigned: lbarx/lharx already zero-extend; output is canonical.

  masm.memoryBarrierAfter(sync);
}

template <typename T>
static void AtomicExchange64(MacroAssembler& masm,
                             const wasm::MemoryAccessDesc* access,
                             Synchronization sync, const T& mem,
                             Register64 value, Register64 output) {
  MOZ_ASSERT(value != output);
  UseScratchRegisterScope temps(masm);

  Register scratch = temps.Acquire();
  masm.computeEffectiveAddress(mem, scratch);

  Label tryAgain;

  masm.memoryBarrierBefore(sync);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // wasm atomic memory is little-endian. Byte-reverse the to-be-stored value to
  // native once before the loop and the loaded old value to LE after. VSX swap.
  // JS atomics (access == nullptr) are native byte order, so no swap.
  if (access) {
    masm.as_mtvsrd(ScratchDoubleReg, value.reg);
    masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    masm.as_mfvsrd(value.reg, ScratchDoubleReg);
  }
#endif

  masm.bind(&tryAgain);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  masm.as_ldarx(output.reg, r0, scratch);

  masm.as_stdcx(value.reg, r0, scratch);
  masm.ma_b(Assembler::NotEqual, &tryAgain);

  masm.memoryBarrierAfter(sync);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (access) {
    masm.as_mtvsrd(ScratchDoubleReg, output.reg);
    masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    masm.as_mfvsrd(output.reg, ScratchDoubleReg);
  }
#endif
}

template <typename T>
static void AtomicFetchOp(MacroAssembler& masm,
                          const wasm::MemoryAccessDesc* access,
                          Scalar::Type type, Synchronization sync, AtomicOp op,
                          const T& mem, Register value, Register valueTemp,
                          Register offsetTemp, Register maskTemp,
                          Register output) {
  UseScratchRegisterScope temps(masm);
  bool signExtend = Scalar::isSignedIntType(type);
  unsigned nbytes = Scalar::byteSize(type);

  switch (nbytes) {
    case 1:
    case 2:
      break;
    case 4:
      MOZ_ASSERT(valueTemp == InvalidReg);
      MOZ_ASSERT(offsetTemp == InvalidReg);
      MOZ_ASSERT(maskTemp == InvalidReg);
      break;
    default:
      MOZ_CRASH();
  }

  Label again;

  Register memTemp = temps.Acquire();
  masm.computeEffectiveAddress(mem, memTemp);

  Register scratch = temps.Acquire();

  if (nbytes == 4) {
    masm.memoryBarrierBefore(sync);
    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(output, r0, memTemp);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // wasm atomic memory is little-endian, but lwarx read the value natively.
    // Byte-reverse it to LE before the op (and the result before stwcx).
    // Swap via the VSX scratch: no GPR temp (the scratch pool is exhausted by
    // memTemp/scratch) and no memory access (which would clear the reservation).
    // output then holds the LE old value to return. JS atomics
    // (access == nullptr) are native byte order, so no swap.
    if (access) {
      masm.as_mtvsrd(ScratchDoubleReg, output);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(output, ScratchDoubleReg);
      masm.x_srdi(output, output, 32);
    }
#endif

    switch (op) {
      case AtomicOp::Add:
        masm.as_add(scratch, output, value);
        break;
      case AtomicOp::Sub:
        masm.as_subf(scratch, value, output);
        break;
      case AtomicOp::And:
        masm.as_and_(scratch, output, value);
        break;
      case AtomicOp::Or:
        masm.as_or_(scratch, output, value);
        break;
      case AtomicOp::Xor:
        masm.as_xor_(scratch, output, value);
        break;
      default:
        MOZ_CRASH();
    }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // Byte-reverse the LE result back to native for the store. Mask the high
    // 32 bits first (an Add/Sub may have carried into bit 32) so the swap
    // operates only on the 32-bit result.
    if (access) {
      masm.as_rldicl(scratch, scratch, 0, 32);
      masm.as_mtvsrd(ScratchDoubleReg, scratch);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(scratch, ScratchDoubleReg);
      masm.x_srdi(scratch, scratch, 32);
    }
#endif

    masm.as_stwcx(scratch, r0, memTemp);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);
    // output already holds the (byte-reversed) little-endian old value;
    // sign-extend for 32-bit canonical form.
    masm.as_extsw(output, output);

    return;
  }

  // Sub-word fetch-and-op via native lbarx/lharx + stbcx./sthcx. (POWER7+).
  // `output` holds the pre-op loaded value (returned to caller); `valueTemp`
  // is the post-op value we condition-store. stXcx. only stores low 8/16 bits
  // of RS, so no pre-mask of valueTemp is needed.
  // offsetTemp / maskTemp are still allocated by the lowering but unused; the
  // local `scratch` is only used in the 4-byte branch above.
  (void)offsetTemp;
  (void)maskTemp;

  masm.memoryBarrierBefore(sync);

  masm.bind(&again);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  if (nbytes == 1) {
    masm.as_lbarx(output, r0, memTemp);
  } else {
    masm.as_lharx(output, r0, memTemp);
  }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (nbytes == 2 && access) {
    // wasm atomic memory is little-endian; byte-reverse the loaded halfword
    // before the op (and the result before sthcx). offsetTemp is unused here.
    // JS atomics (access == nullptr) are native byte order, so no swap.
    masm.as_rlwinm(offsetTemp, output, 8, 16, 23);
    masm.as_rlwinm(output, output, 24, 24, 31);
    masm.as_or_(output, output, offsetTemp);
  }
#endif

  switch (op) {
    case AtomicOp::Add:
      masm.as_add(valueTemp, output, value);
      break;
    case AtomicOp::Sub:
      masm.as_subf(valueTemp, value, output);
      break;
    case AtomicOp::And:
      masm.as_and_(valueTemp, output, value);
      break;
    case AtomicOp::Or:
      masm.as_or_(valueTemp, output, value);
      break;
    case AtomicOp::Xor:
      masm.as_xor_(valueTemp, output, value);
      break;
    default:
      MOZ_CRASH();
  }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (nbytes == 2 && access) {
    // Byte-reverse the little-endian result back to native; sthcx stores the
    // low 16 bits.
    masm.as_rlwinm(offsetTemp, valueTemp, 8, 16, 23);
    masm.as_rlwinm(valueTemp, valueTemp, 24, 24, 31);
    masm.as_or_(valueTemp, valueTemp, offsetTemp);
  }
#endif

  if (nbytes == 1) {
    masm.as_stbcx(valueTemp, r0, memTemp);
  } else {
    masm.as_sthcx(valueTemp, r0, memTemp);
  }
  masm.ma_b(Assembler::NotEqual, &again);

  if (signExtend) {
    if (nbytes == 1) {
      masm.as_extsb(output, output);
    } else {
      masm.as_extsh(output, output);
    }
  }
  // Unsigned: lbarx/lharx already zero-extend; output is canonical.

  masm.memoryBarrierAfter(sync);
}

template <typename T>
static void AtomicFetchOp64(MacroAssembler& masm,
                            const wasm::MemoryAccessDesc* access,
                            Synchronization sync, AtomicOp op, Register64 value,
                            const T& mem, Register64 temp, Register64 output) {
  MOZ_ASSERT(value != output);
  MOZ_ASSERT(value != temp);
  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();
  masm.computeEffectiveAddress(mem, scratch);

  Label tryAgain;

  masm.memoryBarrierBefore(sync);

  masm.bind(&tryAgain);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  masm.as_ldarx(output.reg, r0, scratch);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // wasm atomic memory is little-endian; byte-reverse the natively-loaded
  // value to LE before the op (and the result before stdcx). Swap via the VSX
  // scratch: no GPR temp, no memory access (which would clear the reservation).
  // JS atomics (access == nullptr) are native byte order, so no swap.
  if (access) {
    masm.as_mtvsrd(ScratchDoubleReg, output.reg);
    masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    masm.as_mfvsrd(output.reg, ScratchDoubleReg);
  }
#endif

  switch (op) {
    case AtomicOp::Add:
      masm.as_add(temp.reg, output.reg, value.reg);
      break;
    case AtomicOp::Sub:
      masm.as_subf(temp.reg, value.reg, output.reg);
      break;
    case AtomicOp::And:
      masm.as_and_(temp.reg, output.reg, value.reg);
      break;
    case AtomicOp::Or:
      masm.as_or_(temp.reg, output.reg, value.reg);
      break;
    case AtomicOp::Xor:
      masm.as_xor_(temp.reg, output.reg, value.reg);
      break;
    default:
      MOZ_CRASH();
  }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (access) {
    masm.as_mtvsrd(ScratchDoubleReg, temp.reg);
    masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    masm.as_mfvsrd(temp.reg, ScratchDoubleReg);
  }
#endif

  masm.as_stdcx(temp.reg, r0, scratch);
  masm.ma_b(Assembler::NotEqual, &tryAgain);

  masm.memoryBarrierAfter(sync);
}

template <typename T>
static void AtomicEffectOp(MacroAssembler& masm,
                           const wasm::MemoryAccessDesc* access,
                           Scalar::Type type, Synchronization sync, AtomicOp op,
                           const T& mem, Register value, Register valueTemp,
                           Register offsetTemp, Register maskTemp) {
  UseScratchRegisterScope temps(masm);
  unsigned nbytes = Scalar::byteSize(type);

  switch (nbytes) {
    case 1:
    case 2:
      break;
    case 4:
      MOZ_ASSERT(valueTemp == InvalidReg);
      MOZ_ASSERT(offsetTemp == InvalidReg);
      MOZ_ASSERT(maskTemp == InvalidReg);
      break;
    default:
      MOZ_CRASH();
  }

  Label again;

  Register scratch = temps.Acquire();
  masm.computeEffectiveAddress(mem, scratch);

  Register scratch2 = temps.Acquire();

  if (nbytes == 4) {
    masm.memoryBarrierBefore(sync);
    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(scratch2, r0, scratch);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // wasm atomic memory is little-endian; byte-reverse the loaded value to LE
    // before the op and the result back to native before stwcx (VSX scratch).
    // JS atomics (access == nullptr) are native byte order, so no swap.
    if (access) {
      masm.as_mtvsrd(ScratchDoubleReg, scratch2);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(scratch2, ScratchDoubleReg);
      masm.x_srdi(scratch2, scratch2, 32);
    }
#endif

    switch (op) {
      case AtomicOp::Add:
        masm.as_add(scratch2, scratch2, value);
        break;
      case AtomicOp::Sub:
        masm.as_subf(scratch2, value, scratch2);
        break;
      case AtomicOp::And:
        masm.as_and_(scratch2, scratch2, value);
        break;
      case AtomicOp::Or:
        masm.as_or_(scratch2, scratch2, value);
        break;
      case AtomicOp::Xor:
        masm.as_xor_(scratch2, scratch2, value);
        break;
      default:
        MOZ_CRASH();
    }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access) {
      masm.as_rldicl(scratch2, scratch2, 0, 32);
      masm.as_mtvsrd(ScratchDoubleReg, scratch2);
      masm.as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
      masm.as_mfvsrd(scratch2, ScratchDoubleReg);
      masm.x_srdi(scratch2, scratch2, 32);
    }
#endif

    masm.as_stwcx(scratch2, r0, scratch);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);

    return;
  }

  // Sub-word effect-only op via native lbarx/lharx + stbcx./sthcx. (POWER7+).
  // No output to return; scratch2 holds the load+op+store value.
  // valueTemp / offsetTemp / maskTemp are still allocated by the lowering but
  // unused here.
  (void)valueTemp;
  (void)offsetTemp;
  (void)maskTemp;

  masm.memoryBarrierBefore(sync);

  masm.bind(&again);

  if (access) {
    masm.flushBuffer();  // see comment in wasmLoadImpl
    masm.append(*access, wasm::TrapMachineInsn::Atomic,
                FaultingCodeOffset(masm.currentOffset()));
  }

  if (nbytes == 1) {
    masm.as_lbarx(scratch2, r0, scratch);
  } else {
    masm.as_lharx(scratch2, r0, scratch);
  }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (nbytes == 2 && access) {
    // wasm atomic memory is little-endian; byte-reverse the loaded halfword
    // before the op and the result before sthcx. offsetTemp is unused here.
    // JS atomics (access == nullptr) are native byte order, so no swap.
    masm.as_rlwinm(offsetTemp, scratch2, 8, 16, 23);
    masm.as_rlwinm(scratch2, scratch2, 24, 24, 31);
    masm.as_or_(scratch2, scratch2, offsetTemp);
  }
#endif

  switch (op) {
    case AtomicOp::Add:
      masm.as_add(scratch2, scratch2, value);
      break;
    case AtomicOp::Sub:
      masm.as_subf(scratch2, value, scratch2);
      break;
    case AtomicOp::And:
      masm.as_and_(scratch2, scratch2, value);
      break;
    case AtomicOp::Or:
      masm.as_or_(scratch2, scratch2, value);
      break;
    case AtomicOp::Xor:
      masm.as_xor_(scratch2, scratch2, value);
      break;
    default:
      MOZ_CRASH();
  }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (nbytes == 2 && access) {
    masm.as_rlwinm(offsetTemp, scratch2, 8, 16, 23);
    masm.as_rlwinm(scratch2, scratch2, 24, 24, 31);
    masm.as_or_(scratch2, scratch2, offsetTemp);
  }
#endif

  if (nbytes == 1) {
    masm.as_stbcx(scratch2, r0, scratch);
  } else {
    masm.as_sthcx(scratch2, r0, scratch);
  }
  masm.ma_b(Assembler::NotEqual, &again);

  masm.memoryBarrierAfter(sync);
}

// Public MacroAssembler methods.

void MacroAssembler::compareExchange(Scalar::Type type, Synchronization sync,
                                     const Address& mem, Register oldval,
                                     Register newval, Register valueTemp,
                                     Register offsetTemp, Register maskTemp,
                                     Register output) {
  CompareExchange(*this, nullptr, type, sync, mem, oldval, newval, valueTemp,
                  offsetTemp, maskTemp, output);
}

void MacroAssembler::compareExchange(Scalar::Type type, Synchronization sync,
                                     const BaseIndex& mem, Register oldval,
                                     Register newval, Register valueTemp,
                                     Register offsetTemp, Register maskTemp,
                                     Register output) {
  CompareExchange(*this, nullptr, type, sync, mem, oldval, newval, valueTemp,
                  offsetTemp, maskTemp, output);
}

void MacroAssembler::compareExchange64(Synchronization sync, const Address& mem,
                                       Register64 expect, Register64 replace,
                                       Register64 output) {
  CompareExchange64(*this, nullptr, sync, mem, expect, replace, output);
}

void MacroAssembler::compareExchange64(Synchronization sync,
                                       const BaseIndex& mem, Register64 expect,
                                       Register64 replace, Register64 output) {
  CompareExchange64(*this, nullptr, sync, mem, expect, replace, output);
}

void MacroAssembler::wasmCompareExchange(const wasm::MemoryAccessDesc& access,
                                         const Address& mem, Register oldval,
                                         Register newval, Register valueTemp,
                                         Register offsetTemp, Register maskTemp,
                                         Register output) {
  CompareExchange(*this, &access, access.type(), access.sync(), mem, oldval,
                  newval, valueTemp, offsetTemp, maskTemp, output);
}

void MacroAssembler::wasmCompareExchange(const wasm::MemoryAccessDesc& access,
                                         const BaseIndex& mem, Register oldval,
                                         Register newval, Register valueTemp,
                                         Register offsetTemp, Register maskTemp,
                                         Register output) {
  CompareExchange(*this, &access, access.type(), access.sync(), mem, oldval,
                  newval, valueTemp, offsetTemp, maskTemp, output);
}

void MacroAssembler::wasmCompareExchange64(const wasm::MemoryAccessDesc& access,
                                           const Address& mem,
                                           Register64 expect,
                                           Register64 replace,
                                           Register64 output) {
  CompareExchange64(*this, &access, access.sync(), mem, expect, replace,
                    output);
}

void MacroAssembler::wasmCompareExchange64(const wasm::MemoryAccessDesc& access,
                                           const BaseIndex& mem,
                                           Register64 expect,
                                           Register64 replace,
                                           Register64 output) {
  CompareExchange64(*this, &access, access.sync(), mem, expect, replace,
                    output);
}

void MacroAssembler::atomicExchange(Scalar::Type type, Synchronization sync,
                                    const Address& mem, Register value,
                                    Register valueTemp, Register offsetTemp,
                                    Register maskTemp, Register output) {
  AtomicExchange(*this, nullptr, type, sync, mem, value, valueTemp, offsetTemp,
                 maskTemp, output);
}

void MacroAssembler::atomicExchange(Scalar::Type type, Synchronization sync,
                                    const BaseIndex& mem, Register value,
                                    Register valueTemp, Register offsetTemp,
                                    Register maskTemp, Register output) {
  AtomicExchange(*this, nullptr, type, sync, mem, value, valueTemp, offsetTemp,
                 maskTemp, output);
}

void MacroAssembler::atomicExchange64(Synchronization sync, const Address& mem,
                                      Register64 value, Register64 output) {
  AtomicExchange64(*this, nullptr, sync, mem, value, output);
}

void MacroAssembler::atomicExchange64(Synchronization sync,
                                      const BaseIndex& mem, Register64 value,
                                      Register64 output) {
  AtomicExchange64(*this, nullptr, sync, mem, value, output);
}

void MacroAssembler::wasmAtomicExchange(const wasm::MemoryAccessDesc& access,
                                        const Address& mem, Register value,
                                        Register valueTemp, Register offsetTemp,
                                        Register maskTemp, Register output) {
  AtomicExchange(*this, &access, access.type(), access.sync(), mem, value,
                 valueTemp, offsetTemp, maskTemp, output);
}

void MacroAssembler::wasmAtomicExchange(const wasm::MemoryAccessDesc& access,
                                        const BaseIndex& mem, Register value,
                                        Register valueTemp, Register offsetTemp,
                                        Register maskTemp, Register output) {
  AtomicExchange(*this, &access, access.type(), access.sync(), mem, value,
                 valueTemp, offsetTemp, maskTemp, output);
}

template <typename T>
static void WasmAtomicExchange64(MacroAssembler& masm,
                                 const wasm::MemoryAccessDesc& access,
                                 const T& mem, Register64 value,
                                 Register64 output) {
  AtomicExchange64(masm, &access, access.sync(), mem, value, output);
}

void MacroAssembler::wasmAtomicExchange64(const wasm::MemoryAccessDesc& access,
                                          const Address& mem, Register64 src,
                                          Register64 output) {
  WasmAtomicExchange64(*this, access, mem, src, output);
}

void MacroAssembler::wasmAtomicExchange64(const wasm::MemoryAccessDesc& access,
                                          const BaseIndex& mem, Register64 src,
                                          Register64 output) {
  WasmAtomicExchange64(*this, access, mem, src, output);
}

void MacroAssembler::atomicFetchOp(Scalar::Type type, Synchronization sync,
                                   AtomicOp op, Register value,
                                   const Address& mem, Register valueTemp,
                                   Register offsetTemp, Register maskTemp,
                                   Register output) {
  AtomicFetchOp(*this, nullptr, type, sync, op, mem, value, valueTemp,
                offsetTemp, maskTemp, output);
}

void MacroAssembler::atomicFetchOp(Scalar::Type type, Synchronization sync,
                                   AtomicOp op, Register value,
                                   const BaseIndex& mem, Register valueTemp,
                                   Register offsetTemp, Register maskTemp,
                                   Register output) {
  AtomicFetchOp(*this, nullptr, type, sync, op, mem, value, valueTemp,
                offsetTemp, maskTemp, output);
}

void MacroAssembler::atomicFetchOp64(Synchronization sync, AtomicOp op,
                                     Register64 value, const Address& mem,
                                     Register64 temp, Register64 output) {
  AtomicFetchOp64(*this, nullptr, sync, op, value, mem, temp, output);
}

void MacroAssembler::atomicFetchOp64(Synchronization sync, AtomicOp op,
                                     Register64 value, const BaseIndex& mem,
                                     Register64 temp, Register64 output) {
  AtomicFetchOp64(*this, nullptr, sync, op, value, mem, temp, output);
}

void MacroAssembler::atomicEffectOp64(Synchronization sync, AtomicOp op,
                                      Register64 value, const Address& mem,
                                      Register64 temp) {
  AtomicFetchOp64(*this, nullptr, sync, op, value, mem, temp, temp);
}

void MacroAssembler::atomicEffectOp64(Synchronization sync, AtomicOp op,
                                      Register64 value, const BaseIndex& mem,
                                      Register64 temp) {
  AtomicFetchOp64(*this, nullptr, sync, op, value, mem, temp, temp);
}

void MacroAssembler::wasmAtomicFetchOp(const wasm::MemoryAccessDesc& access,
                                       AtomicOp op, Register value,
                                       const Address& mem, Register valueTemp,
                                       Register offsetTemp, Register maskTemp,
                                       Register output) {
  AtomicFetchOp(*this, &access, access.type(), access.sync(), op, mem, value,
                valueTemp, offsetTemp, maskTemp, output);
}

void MacroAssembler::wasmAtomicFetchOp(const wasm::MemoryAccessDesc& access,
                                       AtomicOp op, Register value,
                                       const BaseIndex& mem, Register valueTemp,
                                       Register offsetTemp, Register maskTemp,
                                       Register output) {
  AtomicFetchOp(*this, &access, access.type(), access.sync(), op, mem, value,
                valueTemp, offsetTemp, maskTemp, output);
}

void MacroAssembler::wasmAtomicFetchOp64(const wasm::MemoryAccessDesc& access,
                                         AtomicOp op, Register64 value,
                                         const Address& mem, Register64 temp,
                                         Register64 output) {
  AtomicFetchOp64(*this, &access, access.sync(), op, value, mem, temp, output);
}

void MacroAssembler::wasmAtomicFetchOp64(const wasm::MemoryAccessDesc& access,
                                         AtomicOp op, Register64 value,
                                         const BaseIndex& mem, Register64 temp,
                                         Register64 output) {
  AtomicFetchOp64(*this, &access, access.sync(), op, value, mem, temp, output);
}

void MacroAssembler::wasmAtomicEffectOp(const wasm::MemoryAccessDesc& access,
                                        AtomicOp op, Register value,
                                        const Address& mem, Register valueTemp,
                                        Register offsetTemp,
                                        Register maskTemp) {
  AtomicEffectOp(*this, &access, access.type(), access.sync(), op, mem, value,
                 valueTemp, offsetTemp, maskTemp);
}

void MacroAssembler::wasmAtomicEffectOp(const wasm::MemoryAccessDesc& access,
                                        AtomicOp op, Register value,
                                        const BaseIndex& mem,
                                        Register valueTemp, Register offsetTemp,
                                        Register maskTemp) {
  AtomicEffectOp(*this, &access, access.type(), access.sync(), op, mem, value,
                 valueTemp, offsetTemp, maskTemp);
}

// ========================================================================
// JS atomic operations.

template <typename T>
static void CompareExchangeJS(MacroAssembler& masm, Scalar::Type arrayType,
                              Synchronization sync, const T& mem,
                              Register oldval, Register newval,
                              Register valueTemp, Register offsetTemp,
                              Register maskTemp, Register temp,
                              AnyRegister output) {
  if (arrayType == Scalar::Uint32) {
    masm.compareExchange(arrayType, sync, mem, oldval, newval, valueTemp,
                         offsetTemp, maskTemp, temp);
    masm.convertUInt32ToDouble(temp, output.fpu());
  } else {
    masm.compareExchange(arrayType, sync, mem, oldval, newval, valueTemp,
                         offsetTemp, maskTemp, output.gpr());
  }
}

template <typename T>
static void AtomicExchangeJS(MacroAssembler& masm, Scalar::Type arrayType,
                             Synchronization sync, const T& mem, Register value,
                             Register valueTemp, Register offsetTemp,
                             Register maskTemp, Register temp,
                             AnyRegister output) {
  if (arrayType == Scalar::Uint32) {
    masm.atomicExchange(arrayType, sync, mem, value, valueTemp, offsetTemp,
                        maskTemp, temp);
    masm.convertUInt32ToDouble(temp, output.fpu());
  } else {
    masm.atomicExchange(arrayType, sync, mem, value, valueTemp, offsetTemp,
                        maskTemp, output.gpr());
  }
}

template <typename T>
static void AtomicFetchOpJS(MacroAssembler& masm, Scalar::Type arrayType,
                            Synchronization sync, AtomicOp op, Register value,
                            const T& mem, Register valueTemp,
                            Register offsetTemp, Register maskTemp,
                            Register temp, AnyRegister output) {
  if (arrayType == Scalar::Uint32) {
    masm.atomicFetchOp(arrayType, sync, op, value, mem, valueTemp, offsetTemp,
                       maskTemp, temp);
    masm.convertUInt32ToDouble(temp, output.fpu());
  } else {
    masm.atomicFetchOp(arrayType, sync, op, value, mem, valueTemp, offsetTemp,
                       maskTemp, output.gpr());
  }
}

void MacroAssembler::compareExchangeJS(Scalar::Type arrayType,
                                       Synchronization sync, const Address& mem,
                                       Register oldval, Register newval,
                                       Register valueTemp, Register offsetTemp,
                                       Register maskTemp, Register temp,
                                       AnyRegister output) {
  CompareExchangeJS(*this, arrayType, sync, mem, oldval, newval, valueTemp,
                    offsetTemp, maskTemp, temp, output);
}

void MacroAssembler::compareExchangeJS(Scalar::Type arrayType,
                                       Synchronization sync,
                                       const BaseIndex& mem, Register oldval,
                                       Register newval, Register valueTemp,
                                       Register offsetTemp, Register maskTemp,
                                       Register temp, AnyRegister output) {
  CompareExchangeJS(*this, arrayType, sync, mem, oldval, newval, valueTemp,
                    offsetTemp, maskTemp, temp, output);
}

void MacroAssembler::atomicExchangeJS(Scalar::Type arrayType,
                                      Synchronization sync, const Address& mem,
                                      Register value, Register valueTemp,
                                      Register offsetTemp, Register maskTemp,
                                      Register temp, AnyRegister output) {
  AtomicExchangeJS(*this, arrayType, sync, mem, value, valueTemp, offsetTemp,
                   maskTemp, temp, output);
}

void MacroAssembler::atomicExchangeJS(Scalar::Type arrayType,
                                      Synchronization sync,
                                      const BaseIndex& mem, Register value,
                                      Register valueTemp, Register offsetTemp,
                                      Register maskTemp, Register temp,
                                      AnyRegister output) {
  AtomicExchangeJS(*this, arrayType, sync, mem, value, valueTemp, offsetTemp,
                   maskTemp, temp, output);
}

void MacroAssembler::atomicFetchOpJS(Scalar::Type arrayType,
                                     Synchronization sync, AtomicOp op,
                                     Register value, const Address& mem,
                                     Register valueTemp, Register offsetTemp,
                                     Register maskTemp, Register temp,
                                     AnyRegister output) {
  AtomicFetchOpJS(*this, arrayType, sync, op, value, mem, valueTemp, offsetTemp,
                  maskTemp, temp, output);
}

void MacroAssembler::atomicFetchOpJS(Scalar::Type arrayType,
                                     Synchronization sync, AtomicOp op,
                                     Register value, const BaseIndex& mem,
                                     Register valueTemp, Register offsetTemp,
                                     Register maskTemp, Register temp,
                                     AnyRegister output) {
  AtomicFetchOpJS(*this, arrayType, sync, op, value, mem, valueTemp, offsetTemp,
                  maskTemp, temp, output);
}

void MacroAssembler::atomicEffectOpJS(Scalar::Type arrayType,
                                      Synchronization sync, AtomicOp op,
                                      Register value, const BaseIndex& mem,
                                      Register valueTemp, Register offsetTemp,
                                      Register maskTemp) {
  AtomicEffectOp(*this, nullptr, arrayType, sync, op, mem, value, valueTemp,
                 offsetTemp, maskTemp);
}

void MacroAssembler::atomicEffectOpJS(Scalar::Type arrayType,
                                      Synchronization sync, AtomicOp op,
                                      Register value, const Address& mem,
                                      Register valueTemp, Register offsetTemp,
                                      Register maskTemp) {
  AtomicEffectOp(*this, nullptr, arrayType, sync, op, mem, value, valueTemp,
                 offsetTemp, maskTemp);
}

// ========================================================================
// Wasm address offset carry tests.

void MacroAssemblerPPC64Compat::ma_add32TestCarry(Condition cond, Register rd,
                                                  Register rs, Imm32 imm,
                                                  Label* overflow) {
  MOZ_ASSERT(cond == Assembler::CarrySet || cond == Assembler::CarryClear);
  if (rd != rs) {
    asMasm().move32(rs, rd);
    asMasm().add32(imm, rd);
    as_cmplw(rd, rs);
  } else {
    // visitWasmAddOffset uses useRegisterAtStart, so the LIR allocator may
    // collapse rd onto rs. move32 + add32 would clobber rs before the
    // compare; save rs to a scratch first.
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    asMasm().move32(rs, scratch);
    asMasm().add32(imm, rd);
    as_cmplw(rd, scratch);
  }
  ma_b(cond == Assembler::CarrySet ? LessThan : GreaterThanOrEqual, overflow);
}

void MacroAssemblerPPC64Compat::ma_addPtrTestCarry(Condition cond, Register rd,
                                                   Register rs, ImmWord imm,
                                                   Label* overflow) {
  MOZ_ASSERT(cond == Assembler::CarrySet || cond == Assembler::CarryClear);
  if (rd != rs) {
    asMasm().movePtr(rs, rd);
    asMasm().addPtr(ImmWord(imm.value), rd);
    as_cmpld(rd, rs);
  } else {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    asMasm().movePtr(rs, scratch);
    asMasm().addPtr(ImmWord(imm.value), rd);
    as_cmpld(rd, scratch);
  }
  ma_b(cond == Assembler::CarrySet ? LessThan : GreaterThanOrEqual, overflow);
}

// ========================================================================
// Wasm load/store helpers.

void MacroAssemblerPPC64Compat::wasmProbeLastByte(
    const wasm::MemoryAccessDesc& access, Register memoryBase, Register ptr) {
  if (HasPOWER9()) {
    return;
  }
  const unsigned size = Scalar::byteSize(access.type());
  if (size <= 1) {
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register probeAddr = temps.Acquire();
  // size is at most 16 (Simd128), well within the int16_t range of as_addi.
  as_addi(probeAddr, ptr, int16_t(size - 1));
  // Record the probe as a wasm trap site so its SIGSEGV dispatches
  // through the wasm signal handler the same way the real access would.
  m_buffer.flushPool();
  append(access, wasm::TrapMachineInsn::Load8,
         FaultingCodeOffset(currentOffset()));
  // Probing 1-byte load; result discarded.
  as_lbzx(probeAddr, memoryBase, probeAddr);
}

void MacroAssemblerPPC64Compat::wasmLoadImpl(
    const wasm::MemoryAccessDesc& access, Register memoryBase, Register ptr,
    Register ptrScratch, AnyRegister output) {
  access.assertOffsetInGuardPages();
  uint32_t offset = access.offset32();
  MOZ_ASSERT_IF(offset, ptrScratch != InvalidReg);

  if (offset) {
    asMasm().addPtr(ImmWord(offset), ptrScratch);
    ptr = ptrScratch;
  }

  wasmProbeLastByte(access, memoryBase, ptr);

  asMasm().memoryBarrierBefore(access.sync());
  // Flush any pending constant pool entries before recording the trap site,
  // otherwise a pool body inserted between the recorded offset and the
  // emitted load shifts the load and leaves the pool guard branch at the
  // recorded offset (SummarizeTrapInstruction then rejects the trap site).
  m_buffer.flushPool();
  append(access, wasm::TrapMachineInsnForLoad(Scalar::byteSize(access.type())),
         FaultingCodeOffset(currentOffset()));

  switch (access.type()) {
    case Scalar::Int8:
      as_lbzx(output.gpr(), memoryBase, ptr);
      as_extsb(output.gpr(), output.gpr());
      break;
    case Scalar::Uint8:
      as_lbzx(output.gpr(), memoryBase, ptr);
      break;
    case Scalar::Int16:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      // wasm memory is little-endian; load byte-reversed then sign-extend.
      as_lhbrx(output.gpr(), memoryBase, ptr);
      as_extsh(output.gpr(), output.gpr());
#else
      as_lhax(output.gpr(), memoryBase, ptr);
#endif
      break;
    case Scalar::Uint16:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_lhbrx(output.gpr(), memoryBase, ptr);
#else
      as_lhzx(output.gpr(), memoryBase, ptr);
#endif
      break;
    case Scalar::Int32:
    case Scalar::Uint32:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_lwbrx(output.gpr(), memoryBase, ptr);
#else
      as_lwzx(output.gpr(), memoryBase, ptr);
#endif
      as_extsw(output.gpr(), output.gpr());
      break;
    case Scalar::Float64:
      if (access.isZeroExtendSimd128Load() || access.isSplatSimd128Load() ||
          access.isWidenSimd128Load()) {
        // lfdx is X-form scalar FP — encodes only 5-bit FRT, so a
        // Simd128 dest (encoding 32+) corrupts the opcode. Bridge
        // through ScratchDoubleReg (FPR f0, encoding 0).
        ScratchDoubleScope dscratch(asMasm());
        as_lfdx(dscratch, memoryBase, ptr);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // wasm memory is little-endian, so reload the 8 bytes byte-reversed
        // (the lfdx above is overwritten on big-endian but keeps the trap
        // site). mtvsrd places the value in dw0 exactly like lfdx, so the
        // xxpermdi shuffles below are unchanged.
        {
          UseScratchRegisterScope temps(asMasm());
          Register tmp = temps.Acquire();
          as_ldbrx(tmp, memoryBase, ptr);
          as_mtvsrd(dscratch, tmp);
        }
#endif
        if (access.isZeroExtendSimd128Load()) {
          // Loaded value goes to BE dw1 (= LE dw0 = lane 0); BE dw0 = 0.
          as_xxlxor(ScratchSimd128Reg, ScratchSimd128Reg, ScratchSimd128Reg);
          as_xxpermdi(output.fpu(), ScratchSimd128Reg, dscratch, 0);
        } else if (access.isSplatSimd128Load()) {
          as_xxpermdi(output.fpu(), dscratch, dscratch, 0);
        } else {
          // widen: place loaded 64 bits in LE dw0 (= BE dw1) for widenLow.
          as_xxpermdi(output.fpu(), dscratch, dscratch, 2);
          switch (access.widenSimdOp()) {
            case wasm::SimdOp::V128Load8x8S:
              asMasm().widenLowInt8x16(output.fpu(), output.fpu());
              break;
            case wasm::SimdOp::V128Load8x8U:
              asMasm().unsignedWidenLowInt8x16(output.fpu(), output.fpu());
              break;
            case wasm::SimdOp::V128Load16x4S:
              asMasm().widenLowInt16x8(output.fpu(), output.fpu());
              break;
            case wasm::SimdOp::V128Load16x4U:
              asMasm().unsignedWidenLowInt16x8(output.fpu(), output.fpu());
              break;
            case wasm::SimdOp::V128Load32x2S:
              asMasm().widenLowInt32x4(output.fpu(), output.fpu());
              break;
            case wasm::SimdOp::V128Load32x2U:
              asMasm().unsignedWidenLowInt32x4(output.fpu(), output.fpu());
              break;
            default:
              MOZ_CRASH("Unexpected widen op");
          }
        }
      } else {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // wasm memory is little-endian: load 8 bytes byte-reversed into a GPR,
        // then move into the FPR (matches what lfdx would place in dw0).
        UseScratchRegisterScope temps(asMasm());
        Register tmp = temps.Acquire();
        as_ldbrx(tmp, memoryBase, ptr);
        as_mtvsrd(output.fpu(), tmp);
#else
        as_lfdx(output.fpu(), memoryBase, ptr);
#endif
      }
      break;
    case Scalar::Float32:
      if (access.isZeroExtendSimd128Load()) {
        // v128.load32_zero: load 32 bits into lane 0, zero the rest. wasm
        // memory is little-endian, so the scalar load is byte-reversed.
        UseScratchRegisterScope temps(asMasm());
        Register tmp = temps.Acquire();
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        as_lwbrx(tmp, memoryBase, ptr);
#else
        as_lwzx(tmp, memoryBase, ptr);
#endif
        as_xxlxor(output.fpu(), output.fpu(), output.fpu());
        if (HasPOWER9()) {
          as_mtvsrws(ScratchSimd128Reg, tmp);
          as_xxinsertw(output.fpu(), ScratchSimd128Reg, 12);
        } else {
          // POWER8: mtvsrd puts value in BE dw0 low 32 bits.
          // xxpermdi(dest, zero, scratch, 0) = {zero[dw0], scratch[dw0]}
          // in BE, placing the value in LE word 0 with the rest zero.
          as_mtvsrd(ScratchSimd128Reg, tmp);
          as_xxpermdi(output.fpu(), output.fpu(), ScratchSimd128Reg, 0);
        }
      } else {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // wasm memory is little-endian: load the 4 single-precision bytes
        // byte-reversed into a GPR, then reinterpret into the FPR (matching
        // what lfsx would place, including the single->double conversion).
        UseScratchRegisterScope temps(asMasm());
        Register tmp = temps.Acquire();
        as_lwbrx(tmp, memoryBase, ptr);
        asMasm().moveGPRToFloat32(tmp, output.fpu());
#else
        as_lfsx(output.fpu(), memoryBase, ptr);
#endif
      }
      break;
    case Scalar::Simd128:
      if (HasPOWER9()) {
        as_lxvx(output.fpu(), memoryBase, ptr);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // wasm v128 memory is little-endian: byte-reverse the 16 bytes so the
        // register matches the canonical LE layout the VSX lane/element ops
        // use. (lxvb16x is endian-normalizing and a no-op on BE; xxbrq is an
        // absolute register byte-reverse.)
        as_xxbrq(output.fpu(), output.fpu());
#endif
      } else {
        as_lxvd2x(output.fpu(), memoryBase, ptr);
        as_xxpermdi(output.fpu(), output.fpu(), output.fpu(), 2);
      }
      break;
    default:
      MOZ_CRASH("unexpected array type");
  }

  asMasm().memoryBarrierAfter(access.sync());
}

void MacroAssemblerPPC64Compat::wasmStoreImpl(
    const wasm::MemoryAccessDesc& access, AnyRegister value,
    Register memoryBase, Register ptr, Register ptrScratch) {
  access.assertOffsetInGuardPages();
  uint32_t offset = access.offset32();
  MOZ_ASSERT_IF(offset, ptrScratch != InvalidReg);

  if (offset) {
    asMasm().addPtr(ImmWord(offset), ptrScratch);
    ptr = ptrScratch;
  }

  wasmProbeLastByte(access, memoryBase, ptr);

  asMasm().memoryBarrierBefore(access.sync());
  // Record trap site at the faulting memory instruction. For P8 Simd128
  // store, the faulting instruction (stxvd2x) is after a byte-swap
  // (xxpermdi), so we defer the trap site recording.
  // Flush pool first; see comment in wasmLoadImpl.
  bool deferTrapSite = (access.type() == Scalar::Simd128 && !HasPOWER9());
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // On BE, an f32/f64 store moves FP->GPR before the byte-reversed store, so
  // the faulting store is not the first emitted instruction; defer the
  // trap-site record to just before the store (as the P8 Simd128 path does).
  if (access.type() == Scalar::Float64 || access.type() == Scalar::Float32) {
    deferTrapSite = true;
  }
  // On BE the P9 Simd128 store byte-reverses (xxbrq) into a scratch before the
  // faulting stxvx, so its trap site is deferred too (like the P8 path).
  if (access.type() == Scalar::Simd128) {
    deferTrapSite = true;
  }
#endif
  if (!deferTrapSite) {
    m_buffer.flushPool();
    append(access,
           wasm::TrapMachineInsnForStore(Scalar::byteSize(access.type())),
           FaultingCodeOffset(currentOffset()));
  }

  switch (access.type()) {
    case Scalar::Int8:
    case Scalar::Uint8:
      as_stbx(value.gpr(), memoryBase, ptr);
      break;
    case Scalar::Int16:
    case Scalar::Uint16:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      // wasm memory is little-endian; store byte-reversed.
      as_sthbrx(value.gpr(), memoryBase, ptr);
#else
      as_sthx(value.gpr(), memoryBase, ptr);
#endif
      break;
    case Scalar::Int32:
    case Scalar::Uint32:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_stwbrx(value.gpr(), memoryBase, ptr);
#else
      as_stwx(value.gpr(), memoryBase, ptr);
#endif
      break;
    case Scalar::Int64:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_stdbrx(value.gpr(), memoryBase, ptr);
#else
      as_stdx(value.gpr(), memoryBase, ptr);
#endif
      break;
    case Scalar::Float64:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    {
      // wasm memory is little-endian: move the f64 bits to a GPR and store
      // byte-reversed. Record the trap site at the faulting stdbrx.
      UseScratchRegisterScope temps(asMasm());
      Register tmp = temps.Acquire();
      as_mfvsrd(tmp, value.fpu());
      m_buffer.flushPool();
      append(access, wasm::TrapMachineInsnForStore(8),
             FaultingCodeOffset(currentOffset()));
      as_stdbrx(tmp, memoryBase, ptr);
    }
#else
      as_stfdx(value.fpu(), memoryBase, ptr);
#endif
      break;
    case Scalar::Float32:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    {
      // wasm memory is little-endian: reinterpret the f32 into its single
      // bits in a GPR and store byte-reversed. Trap site at the faulting store.
      UseScratchRegisterScope temps(asMasm());
      Register tmp = temps.Acquire();
      asMasm().moveFloat32ToGPR(value.fpu(), tmp);
      m_buffer.flushPool();
      append(access, wasm::TrapMachineInsnForStore(4),
             FaultingCodeOffset(currentOffset()));
      as_stwbrx(tmp, memoryBase, ptr);
    }
#else
      as_stfsx(value.fpu(), memoryBase, ptr);
#endif
      break;
    case Scalar::Simd128:
      if (HasPOWER9()) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // wasm v128 memory is little-endian: byte-reverse (xxbrq) into a
        // scratch, then store. Record the trap site at the faulting stxvx.
        as_xxbrq(ScratchSimd128Reg, value.fpu());
        m_buffer.flushPool();
        append(access,
               wasm::TrapMachineInsnForStore(Scalar::byteSize(access.type())),
               FaultingCodeOffset(currentOffset()));
        as_stxvx(ScratchSimd128Reg, memoryBase, ptr);
#else
        as_stxvx(value.fpu(), memoryBase, ptr);
#endif
      } else {
        as_xxpermdi(ScratchSimd128Reg, value.fpu(), value.fpu(), 2);
        m_buffer.flushPool();  // see comment in wasmLoadImpl
        append(access,
               wasm::TrapMachineInsnForStore(Scalar::byteSize(access.type())),
               FaultingCodeOffset(currentOffset()));
        as_stxvd2x(ScratchSimd128Reg, memoryBase, ptr);
      }
      break;
    default:
      MOZ_CRASH("unexpected array type");
  }

  asMasm().memoryBarrierAfter(access.sync());
}

void MacroAssemblerPPC64Compat::wasmLoadI64Impl(
    const wasm::MemoryAccessDesc& access, Register memoryBase, Register ptr,
    Register ptrScratch, Register64 output) {
  uint32_t offset = access.offset32();
  MOZ_ASSERT_IF(offset, ptrScratch != InvalidReg);

  if (offset) {
    asMasm().addPtr(ImmWord(offset), ptrScratch);
    ptr = ptrScratch;
  }

  wasmProbeLastByte(access, memoryBase, ptr);

  asMasm().memoryBarrierBefore(access.sync());
  m_buffer.flushPool();  // see comment in wasmLoadImpl
  append(access, wasm::TrapMachineInsnForLoad(Scalar::byteSize(access.type())),
         FaultingCodeOffset(currentOffset()));

  switch (access.type()) {
    case Scalar::Int8:
      as_lbzx(output.reg, memoryBase, ptr);
      as_extsb(output.reg, output.reg);
      break;
    case Scalar::Uint8:
      as_lbzx(output.reg, memoryBase, ptr);
      break;
    case Scalar::Int16:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_lhbrx(output.reg, memoryBase, ptr);
      as_extsh(output.reg, output.reg);
#else
      as_lhax(output.reg, memoryBase, ptr);
#endif
      break;
    case Scalar::Uint16:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_lhbrx(output.reg, memoryBase, ptr);
#else
      as_lhzx(output.reg, memoryBase, ptr);
#endif
      break;
    case Scalar::Int32:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_lwbrx(output.reg, memoryBase, ptr);
#else
      as_lwzx(output.reg, memoryBase, ptr);
#endif
      as_extsw(output.reg, output.reg);
      break;
    case Scalar::Uint32:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_lwbrx(output.reg, memoryBase, ptr);  // zero-extended
#else
      as_lwzx(output.reg, memoryBase, ptr);
      // Zero-extended by lwzx already
#endif
      break;
    case Scalar::Int64:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_ldbrx(output.reg, memoryBase, ptr);
#else
      as_ldx(output.reg, memoryBase, ptr);
#endif
      break;
    default:
      MOZ_CRASH("unexpected array type");
  }

  asMasm().memoryBarrierAfter(access.sync());
}

void MacroAssemblerPPC64Compat::wasmStoreI64Impl(
    const wasm::MemoryAccessDesc& access, Register64 value, Register memoryBase,
    Register ptr, Register ptrScratch) {
  uint32_t offset = access.offset32();
  MOZ_ASSERT_IF(offset, ptrScratch != InvalidReg);

  if (offset) {
    asMasm().addPtr(ImmWord(offset), ptrScratch);
    ptr = ptrScratch;
  }

  wasmProbeLastByte(access, memoryBase, ptr);

  asMasm().memoryBarrierBefore(access.sync());
  m_buffer.flushPool();  // see comment in wasmLoadImpl
  append(access, wasm::TrapMachineInsnForStore(Scalar::byteSize(access.type())),
         FaultingCodeOffset(currentOffset()));

  switch (access.type()) {
    case Scalar::Int8:
    case Scalar::Uint8:
      as_stbx(value.reg, memoryBase, ptr);
      break;
    case Scalar::Int16:
    case Scalar::Uint16:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_sthbrx(value.reg, memoryBase, ptr);
#else
      as_sthx(value.reg, memoryBase, ptr);
#endif
      break;
    case Scalar::Int32:
    case Scalar::Uint32:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_stwbrx(value.reg, memoryBase, ptr);
#else
      as_stwx(value.reg, memoryBase, ptr);
#endif
      break;
    case Scalar::Int64:
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      as_stdbrx(value.reg, memoryBase, ptr);
#else
      as_stdx(value.reg, memoryBase, ptr);
#endif
      break;
    default:
      MOZ_CRASH("unexpected array type");
  }

  asMasm().memoryBarrierAfter(access.sync());
}

void MacroAssembler::wasmLoad(const wasm::MemoryAccessDesc& access,
                              Register memoryBase, Register ptr,
                              Register ptrScratch, AnyRegister output) {
  wasmLoadImpl(access, memoryBase, ptr, ptrScratch, output);
}

void MacroAssembler::wasmLoadI64(const wasm::MemoryAccessDesc& access,
                                 Register memoryBase, Register ptr,
                                 Register ptrScratch, Register64 output) {
  wasmLoadI64Impl(access, memoryBase, ptr, ptrScratch, output);
}

void MacroAssembler::wasmStore(const wasm::MemoryAccessDesc& access,
                               AnyRegister value, Register memoryBase,
                               Register ptr, Register ptrScratch) {
  wasmStoreImpl(access, value, memoryBase, ptr, ptrScratch);
}

void MacroAssembler::wasmStoreI64(const wasm::MemoryAccessDesc& access,
                                  Register64 value, Register memoryBase,
                                  Register ptr, Register ptrScratch) {
  wasmStoreI64Impl(access, value, memoryBase, ptr, ptrScratch);
}

//}}} check_macroassembler_style

}  // namespace jit
}  // namespace js

#ifdef ENABLE_WASM_SIMD
// static
bool MacroAssembler::MustMaskShiftCountSimd128(wasm::SimdOp op, int32_t* mask) {
  return false;
}
#endif
