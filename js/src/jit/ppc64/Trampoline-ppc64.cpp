/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/Bailouts.h"
#include "jit/BaselineFrame.h"
#include "jit/CalleeToken.h"
#include "jit/JitFrames.h"
#include "jit/JitRuntime.h"
#include "jit/PerfSpewer.h"
#include "jit/ppc64/SharedICHelpers-ppc64.h"
#include "jit/VMFunctions.h"
#include "vm/JitActivation.h"
#include "vm/JSContext.h"

#include "jit/MacroAssembler-inl.h"

using namespace js;
using namespace js::jit;

// Float (Single+Double) and all GPRs. Simd128 excluded — Ion compiles JS
// (no v128 type), so SIMD regs are never live at bailout / invalidator /
// preBarrier entry. Including them would force the bailout frame's
// FPUArray to hold v128 slots that Ion never writes.
static const LiveRegisterSet AllRegs = LiveRegisterSet(
    GeneralRegisterSet(Registers::AllMask),
    FloatRegisterSet(FloatRegisters::AllSingleMask |
                     FloatRegisters::AllDoubleMask));

static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "Not 64-bit clean.");

// PPC64 ELFv2 callee-saved: GPRs r14-r31, FPRs f14-f31, VRs VR20-VR31, LR.
// We also save reg_vp (r10 / IntArgReg7) so we can use it after the JIT call.
//
// Layout is alignas(16) so that after `reserveStack(sizeof(EnterJITRegs))`
// the SP-relative offset of every VR slot is 16-byte aligned, satisfying
// the 16-byte alignment requirement of stxvd2x / stvx (stvx is technically
// alignment-tolerant, but we'd rather align by construction). Padding at
// the end keeps sizeof a multiple of 16 so SP stays quadword-aligned per
// the ELFv2 stack-pointer rule.
struct alignas(16) EnterJITRegs {
  // VR20-VR31 first so their SP-relative offsets are 0, 16, 32, ... — all
  // 16-byte aligned regardless of what follows.
  uint8_t vr20[16];
  uint8_t vr21[16];
  uint8_t vr22[16];
  uint8_t vr23[16];
  uint8_t vr24[16];
  uint8_t vr25[16];
  uint8_t vr26[16];
  uint8_t vr27[16];
  uint8_t vr28[16];
  uint8_t vr29[16];
  uint8_t vr30[16];
  uint8_t vr31[16];

  double f31;
  double f30;
  double f29;
  double f28;
  double f27;
  double f26;
  double f25;
  double f24;
  double f23;
  double f22;
  double f21;
  double f20;
  double f19;
  double f18;
  double f17;
  double f16;
  double f15;
  double f14;

  uint64_t r31;  // FramePointer
  uint64_t r30;
  uint64_t r29;
  uint64_t r28;
  uint64_t r27;
  uint64_t r26;
  uint64_t r25;
  uint64_t r24;
  uint64_t r23;
  uint64_t r22;
  uint64_t r21;
  uint64_t r20;
  uint64_t r19;
  uint64_t r18;
  uint64_t r17;
  uint64_t r16;
  uint64_t r15;
  uint64_t r14;
  uint64_t r2;  // TOC pointer
  uint64_t lr;
  // Save reg_vp (r10) on stack so we can use it after the JIT call returns.
  uint64_t r10;
};
// alignas(16) on the struct ensures sizeof is a multiple of 16, which keeps
// SP quadword-aligned after `reserveStack(sizeof(EnterJITRegs))`. The
// existing fields total 312 bytes; with the 192 bytes of VR slots we are
// at 504, which alignas(16) bumps to 512.
static_assert((sizeof(EnterJITRegs) % 16) == 0,
              "EnterJITRegs must be 16-byte aligned to keep SP aligned");

static void GenerateReturn(MacroAssembler& masm) {
  MOZ_ASSERT(masm.framePushed() == sizeof(EnterJITRegs));

  // Restore non-volatile GPRs.
  masm.as_ld(r14, StackPointer, offsetof(EnterJITRegs, r14));
  masm.as_ld(r15, StackPointer, offsetof(EnterJITRegs, r15));
  masm.as_ld(r16, StackPointer, offsetof(EnterJITRegs, r16));
  masm.as_ld(r17, StackPointer, offsetof(EnterJITRegs, r17));
  masm.as_ld(r18, StackPointer, offsetof(EnterJITRegs, r18));
  masm.as_ld(r19, StackPointer, offsetof(EnterJITRegs, r19));
  masm.as_ld(r20, StackPointer, offsetof(EnterJITRegs, r20));
  masm.as_ld(r21, StackPointer, offsetof(EnterJITRegs, r21));
  masm.as_ld(r22, StackPointer, offsetof(EnterJITRegs, r22));
  masm.as_ld(r23, StackPointer, offsetof(EnterJITRegs, r23));
  masm.as_ld(r24, StackPointer, offsetof(EnterJITRegs, r24));
  masm.as_ld(r25, StackPointer, offsetof(EnterJITRegs, r25));
  masm.as_ld(r26, StackPointer, offsetof(EnterJITRegs, r26));
  masm.as_ld(r27, StackPointer, offsetof(EnterJITRegs, r27));
  masm.as_ld(r28, StackPointer, offsetof(EnterJITRegs, r28));
  masm.as_ld(r29, StackPointer, offsetof(EnterJITRegs, r29));
  masm.as_ld(r30, StackPointer, offsetof(EnterJITRegs, r30));
  masm.as_ld(r31, StackPointer, offsetof(EnterJITRegs, r31));
  masm.as_ld(r2, StackPointer, offsetof(EnterJITRegs, r2));

  // Restore LR.
  masm.as_ld(r0, StackPointer, offsetof(EnterJITRegs, lr));
  masm.xs_mtlr(r0);

  // Restore non-volatile FPRs.
  masm.as_lfd(f14, StackPointer, offsetof(EnterJITRegs, f14));
  masm.as_lfd(f15, StackPointer, offsetof(EnterJITRegs, f15));
  masm.as_lfd(f16, StackPointer, offsetof(EnterJITRegs, f16));
  masm.as_lfd(f17, StackPointer, offsetof(EnterJITRegs, f17));
  masm.as_lfd(f18, StackPointer, offsetof(EnterJITRegs, f18));
  masm.as_lfd(f19, StackPointer, offsetof(EnterJITRegs, f19));
  masm.as_lfd(f20, StackPointer, offsetof(EnterJITRegs, f20));
  masm.as_lfd(f21, StackPointer, offsetof(EnterJITRegs, f21));
  masm.as_lfd(f22, StackPointer, offsetof(EnterJITRegs, f22));
  masm.as_lfd(f23, StackPointer, offsetof(EnterJITRegs, f23));
  masm.as_lfd(f24, StackPointer, offsetof(EnterJITRegs, f24));
  masm.as_lfd(f25, StackPointer, offsetof(EnterJITRegs, f25));
  masm.as_lfd(f26, StackPointer, offsetof(EnterJITRegs, f26));
  masm.as_lfd(f27, StackPointer, offsetof(EnterJITRegs, f27));
  masm.as_lfd(f28, StackPointer, offsetof(EnterJITRegs, f28));
  masm.as_lfd(f29, StackPointer, offsetof(EnterJITRegs, f29));
  masm.as_lfd(f30, StackPointer, offsetof(EnterJITRegs, f30));
  masm.as_lfd(f31, StackPointer, offsetof(EnterJITRegs, f31));

  // Restore callee-saved VR20-VR31 (ELFv2). lvx uses indexed addressing
  // (RA + RB), and r0's value is used here as RB (RA = StackPointer is
  // non-zero, so its value is added). r0 is non-allocatable.
#define RESTORE_VR(N)                                                 \
  masm.xs_li(r0, offsetof(EnterJITRegs, vr##N));                      \
  masm.as_lvx(N, StackPointer, r0)
  RESTORE_VR(20); RESTORE_VR(21); RESTORE_VR(22); RESTORE_VR(23);
  RESTORE_VR(24); RESTORE_VR(25); RESTORE_VR(26); RESTORE_VR(27);
  RESTORE_VR(28); RESTORE_VR(29); RESTORE_VR(30); RESTORE_VR(31);
#undef RESTORE_VR

  masm.freeStack(sizeof(EnterJITRegs));

  masm.as_blr();
}

static void GeneratePrologue(MacroAssembler& masm) {
  // Save LR first (PPC64 LR is SPR, not GPR).
  masm.xs_mflr(r0);

  // ELFv2 prologue convention: save LR at caller's frame [SP+16] BEFORE
  // decrementing SP. External unwinders (gdb, perf, libunwind) walk the
  // stack by reading LR-save slots at [SP+16] of every frame; without
  // this write they'd find junk at our caller's slot. Costs 1 extra
  // instruction; we still keep the in-frame save below for clean
  // restore symmetry.
  masm.as_std(r0, StackPointer, 16);

  masm.reserveStack(sizeof(EnterJITRegs));

  // Save LR (also kept in our own frame for the clean restore in
  // GenerateReturn — see comment there).
  masm.as_std(r0, StackPointer, offsetof(EnterJITRegs, lr));

  // Save non-volatile GPRs.
  masm.as_std(r2, StackPointer, offsetof(EnterJITRegs, r2));
  masm.as_std(r14, StackPointer, offsetof(EnterJITRegs, r14));
  masm.as_std(r15, StackPointer, offsetof(EnterJITRegs, r15));
  masm.as_std(r16, StackPointer, offsetof(EnterJITRegs, r16));
  masm.as_std(r17, StackPointer, offsetof(EnterJITRegs, r17));
  masm.as_std(r18, StackPointer, offsetof(EnterJITRegs, r18));
  masm.as_std(r19, StackPointer, offsetof(EnterJITRegs, r19));
  masm.as_std(r20, StackPointer, offsetof(EnterJITRegs, r20));
  masm.as_std(r21, StackPointer, offsetof(EnterJITRegs, r21));
  masm.as_std(r22, StackPointer, offsetof(EnterJITRegs, r22));
  masm.as_std(r23, StackPointer, offsetof(EnterJITRegs, r23));
  masm.as_std(r24, StackPointer, offsetof(EnterJITRegs, r24));
  masm.as_std(r25, StackPointer, offsetof(EnterJITRegs, r25));
  masm.as_std(r26, StackPointer, offsetof(EnterJITRegs, r26));
  masm.as_std(r27, StackPointer, offsetof(EnterJITRegs, r27));
  masm.as_std(r28, StackPointer, offsetof(EnterJITRegs, r28));
  masm.as_std(r29, StackPointer, offsetof(EnterJITRegs, r29));
  masm.as_std(r30, StackPointer, offsetof(EnterJITRegs, r30));
  masm.as_std(r31, StackPointer, offsetof(EnterJITRegs, r31));

  // Save reg_vp (r10) so we can retrieve it after the JIT call.
  masm.as_std(r10, StackPointer, offsetof(EnterJITRegs, r10));

  // Save non-volatile FPRs.
  masm.as_stfd(f14, StackPointer, offsetof(EnterJITRegs, f14));
  masm.as_stfd(f15, StackPointer, offsetof(EnterJITRegs, f15));
  masm.as_stfd(f16, StackPointer, offsetof(EnterJITRegs, f16));
  masm.as_stfd(f17, StackPointer, offsetof(EnterJITRegs, f17));
  masm.as_stfd(f18, StackPointer, offsetof(EnterJITRegs, f18));
  masm.as_stfd(f19, StackPointer, offsetof(EnterJITRegs, f19));
  masm.as_stfd(f20, StackPointer, offsetof(EnterJITRegs, f20));
  masm.as_stfd(f21, StackPointer, offsetof(EnterJITRegs, f21));
  masm.as_stfd(f22, StackPointer, offsetof(EnterJITRegs, f22));
  masm.as_stfd(f23, StackPointer, offsetof(EnterJITRegs, f23));
  masm.as_stfd(f24, StackPointer, offsetof(EnterJITRegs, f24));
  masm.as_stfd(f25, StackPointer, offsetof(EnterJITRegs, f25));
  masm.as_stfd(f26, StackPointer, offsetof(EnterJITRegs, f26));
  masm.as_stfd(f27, StackPointer, offsetof(EnterJITRegs, f27));
  masm.as_stfd(f28, StackPointer, offsetof(EnterJITRegs, f28));
  masm.as_stfd(f29, StackPointer, offsetof(EnterJITRegs, f29));
  masm.as_stfd(f30, StackPointer, offsetof(EnterJITRegs, f30));
  masm.as_stfd(f31, StackPointer, offsetof(EnterJITRegs, f31));

  // Save callee-saved VR20-VR31 (ELFv2). The JIT freely uses VMX registers
  // via EmitVmxBinary etc.; without this save the C caller's VR20-VR31
  // contents would be trashed on return. stvx uses indexed addressing —
  // r0 holds the offset (non-allocatable in JIT regalloc; safe to use as
  // a free temp here).
#define SAVE_VR(N)                                                    \
  masm.xs_li(r0, offsetof(EnterJITRegs, vr##N));                      \
  masm.as_stvx(N, StackPointer, r0)
  SAVE_VR(20); SAVE_VR(21); SAVE_VR(22); SAVE_VR(23);
  SAVE_VR(24); SAVE_VR(25); SAVE_VR(26); SAVE_VR(27);
  SAVE_VR(28); SAVE_VR(29); SAVE_VR(30); SAVE_VR(31);
#undef SAVE_VR
}

void JitRuntime::generateEnterJIT(JSContext* cx, MacroAssembler& masm) {
  AutoCreatedBy acb(masm, "JitRuntime::generateEnterJIT");

  enterJITOffset_ = startTrampolineCode(masm);

  // EnterJitCode signature: (void* code, unsigned argc, Value* argv,
  //                          InterpreterFrame* fp, CalleeToken calleeToken,
  //                          JSObject* envChain, size_t numStackValues,
  //                          Value* vp)
  const Register reg_code = IntArgReg0;                       // r3
  const Register reg_argc = IntArgReg1;                       // r4
  const Register reg_argv = IntArgReg2;                       // r5
  const mozilla::DebugOnly<Register> reg_frame = IntArgReg3;  // r6
  const Register reg_token = IntArgReg4;                      // r7
  const Register reg_chain = IntArgReg5;                      // r8
  const Register reg_values = IntArgReg6;                     // r9
  const Register reg_vp = IntArgReg7;                         // r10

  MOZ_ASSERT(OsrFrameReg == reg_frame);

  GeneratePrologue(masm);

  // Save stack pointer as baseline frame.
  masm.movePtr(StackPointer, FramePointer);

  // Use non-volatile scratch registers for generateEnterJitShared.
  // r14, r15, r17 are non-volatile and not special-purpose in JIT.
  generateEnterJitShared(masm, reg_argc, reg_argv, reg_token, r14, r15, r17);

  // Push the descriptor.
  masm.unboxInt32(Address(reg_vp, 0), r14);
  masm.pushFrameDescriptorForJitCall(FrameType::CppToJSJit, r14, r14);

  CodeLabel returnLabel;
  Label oomReturnLabel;
  {
    // Handle Interpreter -> Baseline OSR.
    AllocatableGeneralRegisterSet regs(GeneralRegisterSet::All());
    MOZ_ASSERT(!regs.has(FramePointer));
    regs.take(OsrFrameReg);
    regs.take(reg_code);
    MOZ_ASSERT(!regs.has(ReturnReg), "ReturnReg matches reg_code");

    Label notOsr;
    masm.branchTestPtr(Assembler::Zero, OsrFrameReg, OsrFrameReg, &notOsr);

    Register numStackValues = reg_values;
    regs.take(numStackValues);
    Register scratch = regs.takeAny();

    // Push return address.
    masm.subPtr(Imm32(sizeof(uintptr_t)), StackPointer);
    masm.mov(&returnLabel, scratch);
    masm.storePtr(scratch, Address(StackPointer, 0));

    // Push previous frame pointer.
    masm.subPtr(Imm32(sizeof(uintptr_t)), StackPointer);
    masm.storePtr(FramePointer, Address(StackPointer, 0));

    // Reserve frame.
    Register framePtr = FramePointer;
    masm.movePtr(StackPointer, framePtr);
    masm.subPtr(Imm32(BaselineFrame::Size()), StackPointer);

    Register framePtrScratch = regs.takeAny();
    masm.movePtr(StackPointer, framePtrScratch);

    // Reserve space for locals and stack values.
    masm.x_sldi(scratch, numStackValues, 3);
    masm.subPtr(scratch, StackPointer);

    // Enter exit frame.
    masm.reserveStack(3 * sizeof(uintptr_t));
    masm.storePtr(ImmWord(MakeFrameDescriptor(FrameType::BaselineJS)),
                  Address(StackPointer, 2 * sizeof(uintptr_t)));
    masm.storePtr(ImmPtr(nullptr), Address(StackPointer, sizeof(uintptr_t)));
    masm.storePtr(FramePointer, Address(StackPointer, 0));

    // No GC things to mark, push a bare token.
    masm.loadJSContext(scratch);
    masm.enterFakeExitFrame(scratch, scratch, ExitFrameType::Bare);

    masm.reserveStack(2 * sizeof(uintptr_t));
    masm.storePtr(framePtr, Address(StackPointer, sizeof(uintptr_t)));
    masm.storePtr(reg_code, Address(StackPointer, 0));

    using Fn = void (*)(BaselineFrame* frame, InterpreterFrame* interpFrame,
                        uint32_t numStackValues);
    masm.setupUnalignedABICall(scratch);
    masm.passABIArg(framePtrScratch);
    masm.passABIArg(OsrFrameReg);
    masm.passABIArg(numStackValues);
    masm.callWithABI<Fn, jit::InitBaselineFrameForOsr>(
        ABIType::General, CheckUnsafeCallWithABI::DontCheckHasExitFrame);

    regs.add(OsrFrameReg);
    Register jitcode = regs.takeAny();
    masm.loadPtr(Address(StackPointer, 0), jitcode);
    masm.loadPtr(Address(StackPointer, sizeof(uintptr_t)), framePtr);
    masm.freeStack(2 * sizeof(uintptr_t));

    masm.freeStack(ExitFrameLayout::SizeWithFooter());

    // If OSR-ing, then emit instrumentation for setting lastProfilerFrame
    // if profiler instrumentation is enabled.
    {
      Label skipProfilingInstrumentation;
      AbsoluteAddress addressOfEnabled(
          cx->runtime()->geckoProfiler().addressOfEnabled());
      masm.branch32(Assembler::Equal, addressOfEnabled, Imm32(0),
                    &skipProfilingInstrumentation);
      masm.profilerEnterFrame(framePtr, scratch);
      masm.bind(&skipProfilingInstrumentation);
    }

    masm.jump(jitcode);

    masm.bind(&notOsr);
    // Load the scope chain in R1.
    MOZ_ASSERT(R1.scratchReg() != reg_code);
    masm.movePtr(reg_chain, R1.scratchReg());
  }

  // The call will push the return address and frame pointer on the stack, thus
  // we check that the stack would be aligned once the call is complete.
  masm.assertStackAlignment(JitStackAlignment, 2 * sizeof(uintptr_t));

  // Call the function with pushing return address to stack.
  masm.callJitNoProfiler(reg_code);

  {
    // Interpreter -> Baseline OSR will return here.
    masm.bind(&returnLabel);
    masm.addCodeLabel(returnLabel);
    masm.bind(&oomReturnLabel);
  }

  // Discard arguments and padding. Set sp to the address of the EnterJITRegs
  // on the stack.
  masm.movePtr(FramePointer, StackPointer);

  // Store the returned value into the vp.
  masm.as_ld(reg_vp, StackPointer, offsetof(EnterJITRegs, r10));
  masm.storeValue(JSReturnOperand, Address(reg_vp, 0));

  // Restore non-volatile registers and return.
  GenerateReturn(masm);
}

// static
mozilla::Maybe<::JS::ProfilingFrameIterator::RegisterState>
JitRuntime::getCppEntryRegisters(JitFrameLayout* frameStackAddress) {
  return mozilla::Nothing{};
}

void JitRuntime::generateInvalidator(MacroAssembler& masm, Label* bailoutTail) {
  AutoCreatedBy acb(masm, "JitRuntime::generateInvalidator");

  invalidatorOffset_ = startTrampolineCode(masm);

  masm.checkStackAlignment();

  // Push all registers so we can access them from [base + code].
  masm.PushRegsInMask(AllRegs);

  // Pass pointer to InvalidationBailoutStack structure.
  masm.movePtr(StackPointer, IntArgReg0);

  // Reserve place for BailoutInfo pointer. Two words to ensure alignment for
  // setupAlignedABICall.
  masm.subPtr(Imm32(2 * sizeof(uintptr_t)), StackPointer);
  masm.movePtr(StackPointer, IntArgReg1);

  using Fn = bool (*)(InvalidationBailoutStack* sp, BaselineBailoutInfo** info);
  masm.setupAlignedABICall();
  masm.passABIArg(IntArgReg0);
  masm.passABIArg(IntArgReg1);
  masm.callWithABI<Fn, InvalidationBailout>(
      ABIType::General, CheckUnsafeCallWithABI::DontCheckOther);

  masm.pop(IntArgReg2);

  // Pop the machine state and the dead frame.
  masm.moveToStackPtr(FramePointer);

  // Jump to shared bailout tail. The BailoutInfo pointer has to be in
  // IntArgReg2 (r5).
  masm.jump(bailoutTail);
}

// When bailout is done via out of line code (lazy bailout).
// Frame size is stored in LR (look at
// CodeGeneratorPPC64::generateOutOfLineCode()) and thunk code should save it
// on stack.
static void PushBailoutFrame(MacroAssembler& masm, Register spArg) {
  // Push the frameSize_ stored in LR.
  // See: CodeGeneratorPPC64::generateOutOfLineCode()
  masm.pushReturnAddress();

  // Push registers such that we can access them from [base + code].
  masm.PushRegsInMask(AllRegs);

  // Put pointer to BailoutStack as first argument to the Bailout().
  masm.movePtr(StackPointer, spArg);
}

static void GenerateBailoutThunk(MacroAssembler& masm, Label* bailoutTail) {
  PushBailoutFrame(masm, IntArgReg0);

  // Make space for Bailout's bailoutInfo outparam.
  masm.reserveStack(sizeof(void*));
  masm.movePtr(StackPointer, IntArgReg1);

  // Call the bailout function.
  using Fn = bool (*)(BailoutStack* sp, BaselineBailoutInfo** info);
  masm.setupUnalignedABICall(IntArgReg2);
  masm.passABIArg(IntArgReg0);
  masm.passABIArg(IntArgReg1);
  masm.callWithABI<Fn, Bailout>(ABIType::General,
                                CheckUnsafeCallWithABI::DontCheckOther);

  // Get the bailoutInfo outparam.
  masm.pop(IntArgReg2);

  // Remove both the bailout frame and the topmost Ion frame's stack.
  masm.moveToStackPtr(FramePointer);

  // Jump to shared bailout tail. The BailoutInfo pointer has to be in
  // IntArgReg2 (r5).
  masm.jump(bailoutTail);
}

void JitRuntime::generateBailoutHandler(MacroAssembler& masm,
                                        Label* bailoutTail) {
  AutoCreatedBy acb(masm, "JitRuntime::generateBailoutHandler");

  bailoutHandlerOffset_ = startTrampolineCode(masm);

  GenerateBailoutThunk(masm, bailoutTail);
}

bool JitRuntime::generateVMWrapper(JSContext* cx, MacroAssembler& masm,
                                   VMFunctionId id, const VMFunctionData& f,
                                   DynFn nativeFun, uint32_t* wrapperOffset) {
  AutoCreatedBy acb(masm, "JitRuntime::generateVMWrapper");

  *wrapperOffset = startTrampolineCode(masm);

  // Avoid conflicts with argument registers while discarding the result after
  // the function call.
  AllocatableGeneralRegisterSet regs(Register::Codes::WrapperMask);

  static_assert(
      (Register::Codes::VolatileMask & ~Register::Codes::WrapperMask) == 0,
      "Wrapper register set should be a superset of Volatile register set.");

  // The context is the first argument; r3 is the first argument register.
  Register cxreg = IntArgReg0;
  regs.take(cxreg);

  // On link-register platforms, it is the responsibility of the VM *callee* to
  // push the return address, while the caller must ensure that the address
  // is stored in LR on entry. This allows the VM wrapper to work with both
  // direct calls and tail calls.
  masm.pushReturnAddress();

  // Push the frame pointer to finish the exit frame, then link it up.
  masm.Push(FramePointer);
  masm.moveStackPtrTo(FramePointer);
  masm.loadJSContext(cxreg);
  masm.enterExitFrame(cxreg, regs.getAny(), id);

  // Reserve space for the outparameter.
  masm.reserveVMFunctionOutParamSpace(f);

  masm.setupUnalignedABICallDontSaveRestoreSP();
  masm.passABIArg(cxreg);

  size_t argDisp = ExitFrameLayout::Size();

  // Copy any arguments.
  for (uint32_t explicitArg = 0; explicitArg < f.explicitArgs; explicitArg++) {
    switch (f.argProperties(explicitArg)) {
      case VMFunctionData::WordByValue:
        if (f.argPassedInFloatReg(explicitArg)) {
          masm.passABIArg(MoveOperand(FramePointer, argDisp), ABIType::Float64);
        } else {
          masm.passABIArg(MoveOperand(FramePointer, argDisp), ABIType::General);
        }
        argDisp += sizeof(void*);
        break;
      case VMFunctionData::WordByRef:
        masm.passABIArg(MoveOperand(FramePointer, argDisp,
                                    MoveOperand::Kind::EffectiveAddress),
                        ABIType::General);
        argDisp += sizeof(void*);
        break;
      case VMFunctionData::DoubleByValue:
      case VMFunctionData::DoubleByRef:
        MOZ_CRASH("NYI: PPC64 callVM should not be used with 128bits values.");
        break;
    }
  }

  // Copy the implicit outparam, if any.
  const int32_t outParamOffset =
      -int32_t(ExitFooterFrame::Size()) - f.sizeOfOutParamStackSlot();
  if (f.outParam != Type_Void) {
    masm.passABIArg(MoveOperand(FramePointer, outParamOffset,
                                MoveOperand::Kind::EffectiveAddress),
                    ABIType::General);
  }

  masm.callWithABI(nativeFun, ABIType::General,
                   CheckUnsafeCallWithABI::DontCheckHasExitFrame);

  // Test for failure.
  switch (f.failType()) {
    case Type_Cell:
      masm.branchTestPtr(Assembler::Zero, IntArgReg0, IntArgReg0,
                         masm.failureLabel());
      break;
    case Type_Bool:
      masm.branchIfFalseBool(IntArgReg0, masm.failureLabel());
      break;
    case Type_Void:
      break;
    default:
      MOZ_CRASH("unknown failure kind");
  }

  // Load the outparam.
  masm.loadVMFunctionOutParam(f, Address(FramePointer, outParamOffset));

  // Pop frame and restore frame pointer.
  masm.moveToStackPtr(FramePointer);
  masm.pop(FramePointer);

  // Return. Subtract sizeof(void*) for the frame pointer.
  masm.retn(Imm32(sizeof(ExitFrameLayout) - sizeof(void*) +
                  f.explicitStackSlots() * sizeof(void*) +
                  f.extraValuesToPop * sizeof(Value)));

  return true;
}

uint32_t JitRuntime::generatePreBarrier(JSContext* cx, MacroAssembler& masm,
                                        MIRType type) {
  AutoCreatedBy acb(masm, "JitRuntime::generatePreBarrier");

  uint32_t offset = startTrampolineCode(masm);

  MOZ_ASSERT(PreBarrierReg == IntArgReg1);  // r4
  Register temp1 = IntArgReg0;              // r3
  Register temp2 = IntArgReg2;              // r5
  Register temp3 = IntArgReg3;              // r6
  masm.push(temp1);
  masm.push(temp2);
  masm.push(temp3);

  Label noBarrier;
  masm.emitPreBarrierFastPath(type, temp1, temp2, temp3, &noBarrier);

  // Call into C++ to mark this GC thing.
  masm.pop(temp3);
  masm.pop(temp2);
  masm.pop(temp1);

  LiveRegisterSet save;
  save.set() = RegisterSet(GeneralRegisterSet(Registers::VolatileMask),
                           FloatRegisterSet(FloatRegisters::VolatileMask));
  // On PPC64, save LR since we'll be making a call.
  masm.pushReturnAddress();
  masm.PushRegsInMask(save);

  masm.movePtr(ImmPtr(cx->runtime()), IntArgReg0);

  masm.setupUnalignedABICall(IntArgReg2);
  masm.passABIArg(IntArgReg0);
  masm.passABIArg(IntArgReg1);
  masm.callWithABI(JitPreWriteBarrier(type));

  masm.PopRegsInMask(save);
  masm.ret();

  masm.bind(&noBarrier);
  masm.pop(temp3);
  masm.pop(temp2);
  masm.pop(temp1);
  masm.abiret();

  return offset;
}

void JitRuntime::generateBailoutTailStub(MacroAssembler& masm,
                                         Label* bailoutTail) {
  AutoCreatedBy acb(masm, "JitRuntime::generateBailoutTailStub");

  masm.bind(bailoutTail);
  masm.generateBailoutTail(IntArgReg1, IntArgReg2);
}
