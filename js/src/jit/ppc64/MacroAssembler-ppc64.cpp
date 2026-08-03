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
  ma_mtfprd(dest, scratch);
  x_srdi(scratch, scratch, JSVAL_TAG_SHIFT);
  asMasm().branchTestInt32(Assembler::NotEqual, scratch, &end);
  // It was an int32.  Recover the boxed value from the FPR, sign-extend
  // the low 32 bits, and convert to double.
  ma_mffprd(scratch, dest);
  as_extsw(scratch, scratch);
  ma_mtfprd(dest, scratch);
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
  ma_mtfprd(dest, scratch);
  x_srdi(scratch, scratch, JSVAL_TAG_SHIFT);
  asMasm().branchTestInt32(Assembler::NotEqual, scratch, &end);
  ma_mffprd(scratch, dest);
  as_extsw(scratch, scratch);
  ma_mtfprd(dest, scratch);
  as_fcfid(dest, dest);

  bind(&end);
}

// ===============================================================
// Conversion functions

void MacroAssemblerPPC64Compat::convertUInt32ToDouble(Register src,
                                                      FloatRegister dest) {
  // mtvsrwz: VSR[dest].dw0 = zero_ext_64(src[32:63]); P8+ (ISA 2.07).
  // Replaces rldicl + mtvsrd (2 insns + scratch) with 1 insn.
  ma_mtfprwz(dest, src);
  as_fcfid(dest, dest);
}

void MacroAssemblerPPC64Compat::convertUInt32ToFloat32(Register src,
                                                       FloatRegister dest) {
  // mtvsrwz + fcfids; same recipe as convertUInt32ToDouble.
  ma_mtfprwz(dest, src);
  ma_fcfids(dest, dest);
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
  masm.ma_mffprd(scratch, src);
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
  ma_mffprd(dest, ScratchDoubleReg);
  as_extsw(dest, dest);
  ma_mtfprd(ScratchDoubleReg, dest);
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
  ma_mffprd(dest, ScratchDoubleReg);
  ma_mtfprd(ScratchDoubleReg, dest);
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
  ma_mffprd(dest, ScratchDoubleReg);
  as_extsw(dest, dest);
  ma_mtfprd(ScratchDoubleReg, dest);
  ma_fcfids(ScratchDoubleReg, ScratchDoubleReg);
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
    ma_mffprd(output, fpscratch);
    UseScratchRegisterScope temps(asMasm());
    Register max255 = temps.Acquire();
    xs_li(max255, 255);
    as_cmpdi(output, 255);
    ma_isel(output, max255, output, GreaterThan);
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
  ma_mffprd(output, fpscratch);
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
      // Pre-VSX tier: no wasm SIMD, so no live v128 state exists. Keep the
      // slot (offset math must match getPushSizeInBytes) but skip the store.
      if (HasVSX()) {
        storeUnalignedSimd128(reg, Address(StackPointer, diff));
      }
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
        if (HasVSX()) {
          loadUnalignedSimd128(Address(StackPointer, diff), reg);
        }
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
#if defined(_CALL_ELF) && _CALL_ELF == 1
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

#if defined(_CALL_ELF) && _CALL_ELF == 1
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
// ELFv1 is a {entry,toc,env} descriptor. So call the thunk straight
// and dereference the C function. (The bare call(SymbolicAddress) used by stubs
// and the process-global thunks always targets a C function, so it always
// dereferences.)
CodeOffset MacroAssemblerPPC64Compat::callWasmSymbolic(wasm::SymbolicAddress imm) {
  asMasm().movePtr(imm, CallReg);
#if defined(_CALL_ELF) && _CALL_ELF == 1
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
#if defined(_CALL_ELF) && _CALL_ELF == 1
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
  // The offset must be taken from emitLoad64Stanza's returned BufferOffset:
  // its enterNoPool can dump a pending constant pool first, so
  // currentOffset() sampled before the call may point at pool data rather
  // than the stanza.
  BufferOffset bo = emitLoad64Stanza(dest, 0);
  return CodeOffset(bo.getOffset());
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
  ma_mtfprd(dest, src.reg);
  as_fcfid(dest, dest);
}

void MacroAssemblerPPC64Compat::ma_fcpsgn(FloatRegister dest, FloatRegister signSrc,
                                    FloatRegister magSrc) {
  if (HasVSX()) {
    as_fcpsgn(dest, signSrc, magSrc);
    return;
  }
  // Read the sign via the red zone, then negate |magSrc| when it is set.
  // Works for -0.0 and NaN: the stored bit pattern's MSB is the sign, and
  // the signed 64-bit compare against zero tests exactly that bit.
  Label positive;
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_stfd(signSrc, StackPointer, -8);
  as_ld(scratch, StackPointer, -8);
  as_fabs(dest, magSrc);
  as_cmpdi(scratch, 0);
  ma_b(Assembler::GreaterThanOrEqual, &positive);
  as_fneg(dest, dest);
  bind(&positive);
}

void MacroAssemblerPPC64Compat::ma_fri970(uint8_t rn, FloatRegister dest,
                                    FloatRegister src) {
  // Pre-VSX round-to-integral: swap the FPSCR RN field around fctid + fcfid
  // (all 970-legal). Values with |src| >= 2^52 (incl. NaN/Inf) are already
  // integral and must pass through untouched -- fctid would saturate them.
  // The original bit pattern stays in the red-zone slot for the whole
  // sequence (no calls intervene), so the final sign fixup -- needed when
  // the result collapses to zero, e.g. frim(-0.3) = -0.0 -- rereads it
  // rather than requiring a second FPR. dest == src and dest ==
  // ScratchDoubleReg are both safe.
  Label done, passthrough;
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_stfd(src, StackPointer, -8);
  as_ld(scratch, StackPointer, -8);
  if (src != dest) {
    as_fmr(dest, src);
  }
  x_srdi(scratch, scratch, 52);
  as_andi_rc(scratch, scratch, 0x7FF);
  ma_cmp(scratch, Imm32(1075), Assembler::GreaterThanOrEqual);
  ma_b(Assembler::GreaterThanOrEqual, &passthrough);
  as_mtfsfi(7, rn);
  as_fctid(dest, dest);
  as_fcfid(dest, dest);
  as_mtfsfi(7, 0);
  as_ld(scratch, StackPointer, -8);
  as_cmpdi(scratch, 0);
  ma_b(Assembler::GreaterThanOrEqual, &done);
  as_fabs(dest, dest);
  as_fneg(dest, dest);
  jump(&done);

  // The guard path passes already-integral values, infinities, and NaN
  // through unchanged -- but hardware fri* quiets a signaling NaN, so
  // match that. Only NaN is unordered with itself; adding it to itself
  // sets the quiet bit and leaves everything else untouched.
  bind(&passthrough);
  as_fcmpu(dest, dest);
  ma_b(Assembler::DoubleOrdered, &done);
  as_fadd(dest, dest, dest);
  bind(&done);
}

void MacroAssemblerPPC64Compat::ma_popcnt64_970(Register out, Register in,
                                          Register work) {
  MOZ_ASSERT(out != in && out != work && in != work);
  x_srdi(out, in, 1);
  movePtr(ImmWord(0x5555555555555555), work);
  as_and_(out, out, work);
  as_subf(in, out, in);
  movePtr(ImmWord(0x3333333333333333), work);
  as_and_(out, in, work);
  x_srdi(in, in, 2);
  as_and_(in, in, work);
  as_add(in, in, out);
  x_srdi(out, in, 4);
  as_add(in, in, out);
  movePtr(ImmWord(0x0F0F0F0F0F0F0F0F), work);
  as_and_(in, in, work);
  movePtr(ImmWord(0x0101010101010101), work);
  as_mulld(out, in, work);
  x_srdi(out, out, 56);
}

void MacroAssemblerPPC64Compat::ma_fcfid_rto(FloatRegister dest,
                                             FloatRegister src) {
  // Truncate-mode conversion, then mark inexactness in the mantissa lsb.
  // FPSCR FI is bit 14 (field 3); after mcrfs it lands in the EQ slot of
  // cr0, so Equal == inexact. The lsb OR is monotone away from zero on the
  // sign-magnitude double format, which is exactly round-to-odd. A mantissa
  // of all-ones is odd and never incremented, so no carry into the exponent.
  Label exact;
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_mtfsfi(7, 1);
  as_fcfid(dest, src);
  as_mtfsfi(7, 0);
  as_mcrfs(cr0, 3);
  ma_b(Assembler::NotEqual, &exact);
  as_stfd(dest, StackPointer, -8);
  as_ld(scratch, StackPointer, -8);
  as_ori(scratch, scratch, 1);
  as_std(scratch, StackPointer, -8);
  as_lfd(dest, StackPointer, -8);
  bind(&exact);
}

void MacroAssemblerPPC64Compat::ma_friz(FloatRegister dest, FloatRegister src) {
  if (HasVSX()) {
    as_friz(dest, src);
  } else {
    ma_fri970(1, dest, src);
  }
}

void MacroAssemblerPPC64Compat::ma_frip(FloatRegister dest, FloatRegister src) {
  if (HasVSX()) {
    as_frip(dest, src);
  } else {
    ma_fri970(2, dest, src);
  }
}

void MacroAssemblerPPC64Compat::ma_frim(FloatRegister dest, FloatRegister src) {
  if (HasVSX()) {
    as_frim(dest, src);
  } else {
    ma_fri970(3, dest, src);
  }
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
        ma_mffprd(scratch, ScratchDoubleReg);
        x_srdi(scratch, scratch, 52);
        as_andi_rc(scratch, scratch, 0x7FF);
      }
      // Biased exponent >= 1075 (= 1023+52) means |val| >= 2^52.
      // Also catches Inf (exp=2047) and NaN (exp=2047).
      ma_cmp(scratch, Imm32(1075), Assembler::GreaterThanOrEqual);
      ma_b(Assembler::GreaterThanOrEqual, &done);
      as_fctid(dest, ScratchDoubleReg);
      as_fcfid(dest, dest);
      ma_fcpsgn(dest, ScratchDoubleReg, dest);
      bind(&done);
      break;
    }
    case RoundingMode::TowardsZero:
      ma_friz(dest, src);
      break;
    case RoundingMode::Up:
      ma_frip(dest, src);
      break;
    case RoundingMode::Down:
      ma_frim(dest, src);
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
      if (HasVSX()) {
        storeUnalignedSimd128(reg, dest);
      }
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
#if defined(_CALL_ELF) && _CALL_ELF == 1
  callABIDescriptorELFv1(scratch);
#else
  call(scratch);
#endif
  callWithABIPost(stackAdjust, result);
}

void MacroAssembler::callWithABIJitCode(Register fun, ABIType result) {
  AutoProfilerCallInstrumentation profiler(*this);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(fun, scratch);

  uint32_t stackAdjust;
  callWithABIPre(&stackAdjust);
  // The target is JIT code, never an ELFv1 descriptor, on either endianness.
  call(scratch);
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
  ma_frim(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  ma_mffprd(dest, fpscratch);

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
    ma_mffprd(dest, src);
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
  ma_frim(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  ma_mffprd(dest, fpscratch);

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
    ma_mffprd(dest, src);
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

  ma_frip(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  ma_mffprd(dest, fpscratch);

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
    ma_mffprd(dest, src);
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

  ma_frip(fpscratch, src);
  as_fctidz(fpscratch, fpscratch);
  ma_mffprd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for (-1, -0] and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    ma_mffprd(dest, src);
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
  ma_mffprd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    ma_mffprd(dest, src);
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
  ma_mffprd(dest, fpscratch);

  // Check if result fits in int32.
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  ma_b(NotEqual, fail);

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    ma_mffprd(dest, src);
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

  Label performRound;

  // Branch for negative inputs.
  zeroDouble(fpscratch);
  branchDouble(DoubleGreaterThanOrEqual, src, fpscratch, &performRound);

  // Input is negative.
  loadConstantDouble(-0.5, fpscratch);
  branchDouble(DoubleGreaterThanOrEqual, src, fpscratch, fail);

  bind(&performRound);
  {
    loadConstantDouble(GetBiggestNumberLessThan(0.5), temp);
    as_fadd(fpscratch, src, temp);
    ma_frim(fpscratch, fpscratch);
    as_fctidz(fpscratch, fpscratch);
    ma_mffprd(dest, fpscratch);

    // Check if result fits in int32.
    as_extsw(scratch, dest);
    as_cmpd(dest, scratch);
    ma_b(NotEqual, fail);
  }

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    ma_mffprd(dest, src);
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

  Label performRound;

  // Branch for non-negative inputs.
  loadConstantFloat32(0.0f, fpscratch);
  branchFloat(DoubleGreaterThanOrEqual, src, fpscratch, &performRound);

  // Input is negative.
  loadConstantFloat32(-0.5f, fpscratch);
  branchFloat(DoubleGreaterThanOrEqual, src, fpscratch, fail);

  bind(&performRound);
  {
    loadConstantFloat32(float(GetBiggestNumberLessThan(0.5)), temp);
    as_fadds(fpscratch, src, temp);
    ma_frim(fpscratch, fpscratch);
    as_fctidz(fpscratch, fpscratch);
    ma_mffprd(dest, fpscratch);

    // Check if result fits in int32.
    as_extsw(scratch, dest);
    as_cmpd(dest, scratch);
    ma_b(NotEqual, fail);
  }

  // Check for -0 and NaN when result is zero.
  Label notZero;
  as_cmpdi(dest, 0);
  ma_b(NotEqual, &notZero);
  {
    ma_mffprd(dest, src);
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
  ma_fcpsgn(output, rhs, lhs);
}

void MacroAssembler::copySignFloat32(FloatRegister lhs, FloatRegister rhs,
                                     FloatRegister output) {
  ma_fcpsgn(output, rhs, lhs);
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
  if (HasVSX()) {
    // POWER7+ has fcfidu (unsigned i64 → f64) as a single instruction; no
    // sign-split / branch / GPR scratch needed.
    ma_mtfprd(dest, src.reg);
    as_fcfidu(dest, dest);
    return;
  }
  // Pre-VSX: values with the top bit clear convert directly via the signed
  // fcfid. Otherwise halve with a sticky low bit (round-to-odd, which
  // preserves the correctly rounded result), convert, and double.
  Label big, even, done;
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_cmpdi(src.reg, 0);
  ma_b(Assembler::LessThan, &big);
  ma_mtfprd(dest, src.reg);
  as_fcfid(dest, dest);
  jump(&done);
  bind(&big);
  as_rldicl_rc(scratch, src.reg, 0, 63);  // scratch = src & 1, sets CR0
  as_rldicl(scratch, src.reg, 63, 1);     // scratch = src >> 1
  ma_b(Assembler::Equal, &even);
  as_ori(scratch, scratch, 1);
  bind(&even);
  ma_mtfprd(dest, scratch);
  as_fcfid(dest, dest);
  as_fadd(dest, dest, dest);
  bind(&done);
}

void MacroAssembler::convertInt64ToFloat32(Register64 src, FloatRegister dest) {
  ma_mtfprd(dest, src.reg);
  ma_fcfids(dest, dest);
}

void MacroAssembler::convertUInt64ToFloat32(Register64 src, FloatRegister dest,
                                            Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  if (HasVSX()) {
    // POWER7+ has fcfidus (unsigned i64 → f32) as a single instruction.
    ma_mtfprd(dest, src.reg);
    as_fcfidus(dest, dest);
    return;
  }
  // Pre-VSX: same sign-split as convertUInt64ToDouble, but the i64->f64
  // step converts with round-to-odd so the final frsp single-rounds
  // correctly (no double rounding).
  Label big, even, done;
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_cmpdi(src.reg, 0);
  ma_b(Assembler::LessThan, &big);
  ma_mtfprd(dest, src.reg);
  ma_fcfid_rto(dest, dest);
  jump(&done);
  bind(&big);
  as_rldicl_rc(scratch, src.reg, 0, 63);  // scratch = src & 1, sets CR0
  as_rldicl(scratch, src.reg, 63, 1);     // scratch = src >> 1
  ma_b(Assembler::Equal, &even);
  as_ori(scratch, scratch, 1);
  bind(&even);
  ma_mtfprd(dest, scratch);
  ma_fcfid_rto(dest, dest);
  as_fadd(dest, dest, dest);
  bind(&done);
  as_frsp(dest, dest);
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
  ma_mffprd(output, fpscratch);
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
  ma_mffprd(output, fpscratch);
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
  ma_mffprd(output, fpscratch);
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
  ma_mffprd(output, fpscratch);
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
  ma_mffprd(output.reg, fpscratch);
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
  ma_mffprd(output.reg, fpscratch);
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
  if (HasVSX()) {
    as_fctiduz(fpscratch, input);
    ma_mffprd(output.reg, fpscratch);
  } else {
    // Pre-VSX: no fctiduz. Split at 2^63 by comparing the raw bit pattern
    // (positive doubles order like their bits): below it the signed fctidz
    // covers the unsigned range [0, 2^63); at or above it convert
    // input - 2^63 (exact: ulp >= 2^11 there) and set the top bit. The big
    // path traps for x >= 2^64 and positive NaN via fctidz's VXCVI. The
    // below path must additionally trap x <= -1 explicitly: signed fctidz
    // accepts those (valid signed results) but they are out of unsigned
    // range. Negative NaN also lands here and fctidz maps it to INT64_MIN,
    // which the same negative test rejects.
    Label big, join;
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    ma_mffprd(output.reg, input);
    xs_li(scratch, 0x43E);
    x_sldi(scratch, scratch, 52);  // 0x43E0000000000000 = 2^63 as double
    as_cmpd(output.reg, scratch);
    ma_b(Assembler::GreaterThanOrEqual, &big);
    as_fctidz(fpscratch, input);
    ma_mffprd(output.reg, fpscratch);
    as_cmpdi(output.reg, 0);
    ma_b(Assembler::LessThan, oolEntry);
    jump(&join);
    bind(&big);
    as_std(scratch, StackPointer, -8);
    as_lfd(fpscratch, StackPointer, -8);
    as_fsub(fpscratch, input, fpscratch);
    as_fctidz(fpscratch, fpscratch);
    ma_mffprd(output.reg, fpscratch);
    xs_li(scratch, 1);
    x_sldi(scratch, scratch, 63);
    as_or_(output.reg, output.reg, scratch);
    bind(&join);
  }
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
  if (HasVSX()) {
    as_fctiduz(fpscratch, input);
    ma_mffprd(output.reg, fpscratch);
  } else {
    // Pre-VSX: no fctiduz. Split at 2^63 by comparing the raw bit pattern
    // (positive doubles order like their bits): below it the signed fctidz
    // covers the unsigned range [0, 2^63); at or above it convert
    // input - 2^63 (exact: ulp >= 2^11 there) and set the top bit. The big
    // path traps for x >= 2^64 and positive NaN via fctidz's VXCVI. The
    // below path must additionally trap x <= -1 explicitly: signed fctidz
    // accepts those (valid signed results) but they are out of unsigned
    // range. Negative NaN also lands here and fctidz maps it to INT64_MIN,
    // which the same negative test rejects.
    Label big, join;
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    ma_mffprd(output.reg, input);
    xs_li(scratch, 0x43E);
    x_sldi(scratch, scratch, 52);  // 0x43E0000000000000 = 2^63 as double
    as_cmpd(output.reg, scratch);
    ma_b(Assembler::GreaterThanOrEqual, &big);
    as_fctidz(fpscratch, input);
    ma_mffprd(output.reg, fpscratch);
    as_cmpdi(output.reg, 0);
    ma_b(Assembler::LessThan, oolEntry);
    jump(&join);
    bind(&big);
    as_std(scratch, StackPointer, -8);
    as_lfd(fpscratch, StackPointer, -8);
    as_fsub(fpscratch, input, fpscratch);
    as_fctidz(fpscratch, fpscratch);
    ma_mffprd(output.reg, fpscratch);
    xs_li(scratch, 1);
    x_sldi(scratch, scratch, 63);
    as_or_(output.reg, output.reg, scratch);
    bind(&join);
  }
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

// Byte-reverse the 32-bit value in the low half of `reg` (upper 32 bits are
// left unspecified) for the wasm-BE atomic RMW loops below, at call sites
// where the 2-register scratch pool (r11/r12) is already fully acquired.
// Uses r0 as a working register instead of acquiring a scratch: r0 is
// excluded from the allocatable set and, at these call sites, only ever
// appears as the hardwired-zero RA operand of indexed loads/stores, so its
// content is never read as live state and is safe to clobber. Emits no
// memory access, so it is also safe between an lwarx/lharx and its paired
// stwcx/sthcx (an intervening store could clear the reservation).
static void ReverseWordBytesForAtomics(MacroAssembler& masm, Register reg) {
  if (HasPOWER10()) {
    masm.as_brw(reg, reg);
    masm.as_extsw(reg, reg);
  } else {
    // Byte-reverse into r0 (rather than an acquired scratch), then sign-extend.
    ByteReverseLow32(masm, r0, reg);
    masm.as_extsw(reg, r0);
  }
}

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
// Byte-reverse a 16-bit value in place (value order <-> wasm's little-endian
// memory order) using `tmp` as a pair register. Result is clean in the low
// 16 bits regardless of junk above them.
static void SwapHalf(MacroAssembler& masm, Register reg, Register tmp) {
  masm.as_rlwinm(tmp, reg, 8, 16, 23);
  masm.as_rlwinm(reg, reg, 24, 24, 31);
  masm.as_or_(reg, reg, tmp);
}
#endif

// Pre-VSX (970/G5) sub-word atomics: lbarx/lharx/stbcx./sthcx. are ISA 2.06,
// so operate on the containing aligned word with lwarx/stwcx. and a shifted
// mask. Converts the byte EA in `mem` to the word EA in place, computes the
// sub-word's big-endian bit position into `shift`, and the shifted width
// mask into `mask`. Sub-word atomics are naturally aligned (JS typed arrays
// by construction, wasm by its alignment trap), so a halfword never
// straddles words.
static void SubwordAtomicPrologue970(MacroAssembler& masm, unsigned nbytes,
                                     Register mem, Register shift,
                                     Register mask) {
  masm.as_andi_rc(shift, mem, 3);
  masm.as_xori(shift, shift, nbytes == 1 ? 3 : 2);
  masm.x_slwi(shift, shift, 3);
  masm.as_rldicr(mem, mem, 0, 61);
  masm.xs_li(mask, 0);
  masm.as_ori(mask, mask, nbytes == 1 ? 0xff : 0xffff);
  masm.as_slw(mask, mask, shift);
}

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
    // loop (it is compared against the little-endian oldval and returned). The
    // byte-reversal mutates newval in place, but the register allocator hands
    // us newval as a plain read-only input that may be shared with a value
    // that is still live after this op (a coalesced copy). Preserve the
    // caller's register across the mutation via the ABI protected zone below
    // SP (no call happens here, so it is safe). JS atomics (access == nullptr)
    // are native byte order, so no swap and nothing to preserve.
    if (access) {
      masm.storePtr(newval, Address(StackPointer, -8));
      masm.as_rldicl(newval, newval, 0, 32);
      masm.byteSwap32(newval);
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
      masm.byteSwap32(output);
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

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access) {
      masm.loadPtr(Address(StackPointer, -8), newval);
    }
#endif
    return;
  }

  if (!HasVSX()) {
    // Pre-VSX masked-word compare-exchange. Both shifted constants (expected
    // and replacement, in memory byte order) are parked in the protected
    // zone below SP so the loop needs only one temp; the in-loop reloads are
    // plain loads, which do not clear the lwarx reservation.
    SubwordAtomicPrologue970(masm, nbytes, scratch, offsetTemp, maskTemp);
    masm.as_rlwinm(valueTemp, oldval, 0, nbytes == 1 ? 24 : 16, 31);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    masm.as_slw(valueTemp, valueTemp, offsetTemp);
    masm.as_std(valueTemp, StackPointer, -16);
    masm.as_rlwinm(valueTemp, newval, 0, nbytes == 1 ? 24 : 16, 31);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    masm.as_slw(valueTemp, valueTemp, offsetTemp);
    masm.as_std(valueTemp, StackPointer, -24);

    masm.memoryBarrierBefore(sync);
    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(output, r0, scratch);
    masm.as_ld(valueTemp, StackPointer, -16);
    masm.as_and_(r0, output, maskTemp);
    masm.ma_cmp(r0, valueTemp, Assembler::NotEqual, /* is32bit */ true);
    masm.ma_b(Assembler::NotEqual, &end);
    masm.as_andc(r0, output, maskTemp);
    masm.as_ld(valueTemp, StackPointer, -24);
    masm.as_or_(r0, r0, valueTemp);
    masm.as_stwcx(r0, r0, scratch);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);
    masm.bind(&end);
    masm.as_srw(output, output, offsetTemp);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, output, r0);
    }
#endif
    if (signExtend) {
      if (nbytes == 1) {
        masm.as_extsb(output, output);
      } else {
        masm.as_extsh(output, output);
      }
    } else {
      masm.as_rlwinm(output, output, 0, nbytes == 1 ? 24 : 16, 31);
    }
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
  // against the little-endian expect and returned). The byte-reversal mutates
  // replace in place; preserve the caller's register across it via the ABI
  // protected zone below SP (this input may be a coalesced copy of a value
  // that is still live after the op; no call happens here). JS atomics
  // (access == nullptr) are native byte order, so no swap.
  if (access) {
    masm.storePtr(replace.reg, Address(StackPointer, -8));
    masm.byteSwap64(replace);
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
  // In-loop swap between ldarx and stdcx: byteSwap64 is register-only on
  // every ISA level, so the reservation is preserved.
  if (access) {
    masm.byteSwap64(output);
  }
#endif

  masm.ma_cmp(output.reg, expect.reg, Assembler::NotEqual);
  masm.ma_b(Assembler::NotEqual, &exit);
  masm.as_stdcx(replace.reg, r0, scratch);
  masm.ma_b(Assembler::NotEqual, &tryAgain);

  masm.memoryBarrierAfter(sync);

  masm.bind(&exit);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  if (access) {
    masm.loadPtr(Address(StackPointer, -8), replace.reg);
  }
#endif
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
    // loaded old value to LE after the loop. The byte-reversal mutates value
    // in place; preserve the caller's register across it via the ABI
    // protected zone below SP (this input may be a coalesced copy of a value
    // that is still live after the op; no call happens here). JS atomics
    // (access == nullptr) are native byte order, so no swap.
    if (access) {
      masm.storePtr(value, Address(StackPointer, -8));
      masm.as_rldicl(value, value, 0, 32);
      masm.byteSwap32(value);
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
      masm.byteSwap32(output);
      masm.loadPtr(Address(StackPointer, -8), value);
    }
#endif
    // lwarx zero-extends; sign-extend for 32-bit canonical form.
    masm.as_extsw(output, output);

    return;
  }

  if (!HasVSX()) {
    // Pre-VSX masked-word exchange: the shifted replacement is loop-invariant
    // and lives in valueTemp.
    SubwordAtomicPrologue970(masm, nbytes, memTemp, offsetTemp, maskTemp);
    masm.as_rlwinm(valueTemp, value, 0, nbytes == 1 ? 24 : 16, 31);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    masm.as_slw(valueTemp, valueTemp, offsetTemp);

    masm.memoryBarrierBefore(sync);
    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(output, r0, memTemp);
    masm.as_andc(r0, output, maskTemp);
    masm.as_or_(r0, r0, valueTemp);
    masm.as_stwcx(r0, r0, memTemp);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);
    masm.as_srw(output, output, offsetTemp);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, output, r0);
    }
#endif
    if (signExtend) {
      if (nbytes == 1) {
        masm.as_extsb(output, output);
      } else {
        masm.as_extsh(output, output);
      }
    } else {
      masm.as_rlwinm(output, output, 0, nbytes == 1 ? 24 : 16, 31);
    }
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
  // native once before the loop and the loaded old value to LE after. The
  // byte-reversal mutates value in place; preserve the caller's register
  // across it via the ABI protected zone below SP (this input may be a
  // coalesced copy of a value that is still live after the op; no call
  // happens here). JS atomics (access == nullptr) are native byte order, so
  // no swap.
  if (access) {
    masm.storePtr(value.reg, Address(StackPointer, -8));
    masm.byteSwap64(value);
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
    masm.byteSwap64(output);
    masm.loadPtr(Address(StackPointer, -8), value.reg);
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
    // Byte-reverse it to LE before the op (and the result before stwcx). The
    // scratch pool is exhausted by memTemp/scratch, so use the r0-based
    // helper (no GPR acquisition, no memory access, which would clear the
    // reservation). output then holds the LE old value to return. JS atomics
    // (access == nullptr) are native byte order, so no swap.
    if (access) {
      ReverseWordBytesForAtomics(masm, output);
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
      ReverseWordBytesForAtomics(masm, scratch);
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

  if (!HasVSX()) {
    // Pre-VSX masked-word fetch-and-op: extract the sub-word each iteration,
    // apply the op in value order, and reinsert under the mask. Junk above
    // the sub-word after the extract shift is harmless (add/sub carries only
    // propagate upward; the reinsert masks) and the wasm halfword swap
    // cleans its own high bits.
    SubwordAtomicPrologue970(masm, nbytes, memTemp, offsetTemp, maskTemp);

    masm.memoryBarrierBefore(sync);
    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(output, r0, memTemp);
    masm.as_srw(valueTemp, output, offsetTemp);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    switch (op) {
      case AtomicOp::Add:
        masm.as_add(valueTemp, valueTemp, value);
        break;
      case AtomicOp::Sub:
        masm.as_subf(valueTemp, value, valueTemp);
        break;
      case AtomicOp::And:
        masm.as_and_(valueTemp, valueTemp, value);
        break;
      case AtomicOp::Or:
        masm.as_or_(valueTemp, valueTemp, value);
        break;
      case AtomicOp::Xor:
        masm.as_xor_(valueTemp, valueTemp, value);
        break;
      default:
        MOZ_CRASH();
    }
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    masm.as_slw(valueTemp, valueTemp, offsetTemp);
    masm.as_and_(valueTemp, valueTemp, maskTemp);
    masm.as_andc(scratch, output, maskTemp);
    masm.as_or_(valueTemp, valueTemp, scratch);
    masm.as_stwcx(valueTemp, r0, memTemp);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.as_srw(output, output, offsetTemp);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, output, r0);
    }
#endif
    if (signExtend) {
      if (nbytes == 1) {
        masm.as_extsb(output, output);
      } else {
        masm.as_extsh(output, output);
      }
    } else {
      masm.as_rlwinm(output, output, 0, nbytes == 1 ? 24 : 16, 31);
    }

    masm.memoryBarrierAfter(sync);
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
  // value to LE before the op (and the result before stdcx). byteSwap64 is
  // register-only on every ISA level, so the reservation is preserved. JS
  // atomics (access == nullptr) are native byte order, so no swap.
  if (access) {
    masm.byteSwap64(output);
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
    masm.byteSwap64(temp);
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
      ReverseWordBytesForAtomics(masm, scratch2);
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
      ReverseWordBytesForAtomics(masm, scratch2);
    }
#endif

    masm.as_stwcx(scratch2, r0, scratch);
    masm.ma_b(Assembler::NotEqual, &again);

    masm.memoryBarrierAfter(sync);

    return;
  }

  if (!HasVSX()) {
    // Pre-VSX masked-word effect-only op; same dance as AtomicFetchOp minus
    // the old-value extraction.
    SubwordAtomicPrologue970(masm, nbytes, scratch, offsetTemp, maskTemp);

    masm.memoryBarrierBefore(sync);
    masm.bind(&again);

    if (access) {
      masm.flushBuffer();  // see comment in wasmLoadImpl
      masm.append(*access, wasm::TrapMachineInsn::Atomic,
                  FaultingCodeOffset(masm.currentOffset()));
    }

    masm.as_lwarx(scratch2, r0, scratch);
    masm.as_srw(valueTemp, scratch2, offsetTemp);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    switch (op) {
      case AtomicOp::Add:
        masm.as_add(valueTemp, valueTemp, value);
        break;
      case AtomicOp::Sub:
        masm.as_subf(valueTemp, value, valueTemp);
        break;
      case AtomicOp::And:
        masm.as_and_(valueTemp, valueTemp, value);
        break;
      case AtomicOp::Or:
        masm.as_or_(valueTemp, valueTemp, value);
        break;
      case AtomicOp::Xor:
        masm.as_xor_(valueTemp, valueTemp, value);
        break;
      default:
        MOZ_CRASH();
    }
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    if (access && nbytes == 2) {
      SwapHalf(masm, valueTemp, r0);
    }
#endif
    masm.as_slw(valueTemp, valueTemp, offsetTemp);
    masm.as_and_(valueTemp, valueTemp, maskTemp);
    masm.as_andc(r0, scratch2, maskTemp);
    masm.as_or_(valueTemp, valueTemp, r0);
    masm.as_stwcx(valueTemp, r0, scratch);
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

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
void MacroAssemblerPPC64Compat::byteReverseSimd128(FloatRegister src,
                                                   FloatRegister dest) {
  if (HasPOWER9()) {
    as_xxbrq(dest, src);
    return;
  }
  // POWER8: vperm with a full-reverse control vector. Expressed in wasm
  // lanes the control is the identity {0..15}: the canonical constant load
  // places lane K at big-endian register byte 15-K, which is exactly the
  // descending byte selector the reversal needs. vperm reads all of its
  // inputs before writing, so |dest| may alias |src| or the control (v0).
  MOZ_ASSERT(src != ScratchSimd128Reg, "control in v0 would clobber src");
  static const int8_t lanes[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                   8, 9, 10, 11, 12, 13, 14, 15};
  ScratchSimd128Scope ctl(asMasm());
  asMasm().loadConstantSimd128(SimdConstant::CreateX16(lanes), ctl);
  as_vperm(dest.encoding() & 31, src.encoding() & 31, src.encoding() & 31,
           ctl.encoding() & 31);
}
#endif

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
          ma_mtfprd(dscratch, tmp);
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
        if (HasVSX()) {
          as_ldbrx(tmp, memoryBase, ptr);
        } else {
          as_ldx(tmp, memoryBase, ptr);
          ma_byteSwap64_970(tmp);
        }
        ma_mtfprd(output.fpu(), tmp);
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
          ma_mtfprd(ScratchSimd128Reg, tmp);
          as_xxpermdi(output.fpu(), output.fpu(), ScratchSimd128Reg, 0);
        }
      } else {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // wasm memory is little-endian: load the 4 single-precision bytes
        // byte-reversed into a GPR, then reinterpret into the FPR (matching
        // what lfsx would place, including the single->double conversion).
        // Inline the reinterpret with the one scratch already held here:
        // moveGPRToFloat32's POWER8 branch would acquire a second scratch
        // and empty the pool when the outer codegen holds one too.
        UseScratchRegisterScope temps(asMasm());
        Register tmp = temps.Acquire();
        as_lwbrx(tmp, memoryBase, ptr);
        if (HasVSX()) {
          x_sldi(tmp, tmp, 32);
          ma_mtfprd(output.fpu(), tmp);
          as_xscvspdpn(output.fpu(), output.fpu());
        } else {
          // Pre-VSX reinterpret: red-zone stw + lfs (lfs expands single to
          // the FPR's double format).
          as_stw(tmp, StackPointer, -8);
          as_lfs(output.fpu(), StackPointer, -8);
        }
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
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // lxvd2x is an identity load on big-endian; byte-reverse to the
        // canonical LE register layout. The pool load this emits comes
        // after the faulting lxvd2x, so the trap site is unaffected.
        byteReverseSimd128(output.fpu(), output.fpu());
#else
        as_xxpermdi(output.fpu(), output.fpu(), output.fpu(), 2);
#endif
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
  // Pre-VSX Int64 byte-swaps in registers before the faulting stdx.
  if (access.type() == Scalar::Int64 && !HasVSX()) {
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
      if (HasVSX()) {
        as_stdbrx(value.gpr(), memoryBase, ptr);
      } else {
        UseScratchRegisterScope temps(asMasm());
        Register tmp = temps.Acquire();
        xs_mr(tmp, value.gpr());
        ma_byteSwap64_970(tmp);
        m_buffer.flushPool();
        append(access, wasm::TrapMachineInsnForStore(8),
               FaultingCodeOffset(currentOffset()));
        as_stdx(tmp, memoryBase, ptr);
      }
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
      ma_mffprd(tmp, value.fpu());
      if (!HasVSX()) {
        ma_byteSwap64_970(tmp);
      }
      m_buffer.flushPool();
      append(access, wasm::TrapMachineInsnForStore(8),
             FaultingCodeOffset(currentOffset()));
      if (HasVSX()) {
        as_stdbrx(tmp, memoryBase, ptr);
      } else {
        as_stdx(tmp, memoryBase, ptr);
      }
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
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // Byte-reverse the canonical value into the scratch and store it
        // raw (stxvd2x is an identity store on big-endian). The reversal's
        // pool load precedes the flushPool, keeping the trap site at the
        // faulting stxvd2x.
        byteReverseSimd128(value.fpu(), ScratchSimd128Reg);
#else
        as_xxpermdi(ScratchSimd128Reg, value.fpu(), value.fpu(), 2);
#endif
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
      if (HasVSX()) {
        as_ldbrx(output.reg, memoryBase, ptr);
      } else {
        // Pre-VSX: native load (same trap pc), then red-zone swap (the
        // scratch pool may be exhausted here).
        as_ldx(output.reg, memoryBase, ptr);
        ma_byteSwap64_970(output.reg);
      }
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
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // Pre-VSX Int64 byte-swaps in registers before the faulting stdx, so its
  // trap-site record is deferred into the case (like wasmStoreImpl's BE FP
  // paths).
  bool deferTrapSite = (access.type() == Scalar::Int64 && !HasVSX());
#else
  bool deferTrapSite = false;
#endif
  if (!deferTrapSite) {
    m_buffer.flushPool();  // see comment in wasmLoadImpl
    append(access,
           wasm::TrapMachineInsnForStore(Scalar::byteSize(access.type())),
           FaultingCodeOffset(currentOffset()));
  }

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
      if (HasVSX()) {
        as_stdbrx(value.reg, memoryBase, ptr);
      } else {
        UseScratchRegisterScope temps(asMasm());
        Register tmp = temps.Acquire();
        xs_mr(tmp, value.reg);
        ma_byteSwap64_970(tmp);
        m_buffer.flushPool();
        append(access, wasm::TrapMachineInsnForStore(8),
               FaultingCodeOffset(currentOffset()));
        as_stdx(tmp, memoryBase, ptr);
      }
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
