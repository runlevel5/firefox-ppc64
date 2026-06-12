/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_Assembler_ppc64_h
#define jit_ppc64_Assembler_ppc64_h

#include "jit/CompactBuffer.h"
#include "jit/JitCode.h"
#include "jit/JitSpewer.h"
#include "jit/ppc64/Architecture-ppc64.h"
#include "jit/shared/Assembler-shared.h"
#include "jit/shared/Disassembler-shared.h"
#include "jit/shared/IonAssemblerBuffer.h"
#include "jit/shared/IonAssemblerBufferWithConstantPools.h"
#include "wasm/WasmTypeDecls.h"

namespace js {
namespace jit {

// GPR register constants.
static constexpr Register r0{Registers::r0};
static constexpr Register r1{Registers::r1};
static constexpr Register r2{Registers::r2};
static constexpr Register r3{Registers::r3};
static constexpr Register r4{Registers::r4};
static constexpr Register r5{Registers::r5};
static constexpr Register r6{Registers::r6};
static constexpr Register r7{Registers::r7};
static constexpr Register r8{Registers::r8};
static constexpr Register r9{Registers::r9};
static constexpr Register r10{Registers::r10};
static constexpr Register r11{Registers::r11};
static constexpr Register r12{Registers::r12};
static constexpr Register r13{Registers::r13};
static constexpr Register r14{Registers::r14};
static constexpr Register r15{Registers::r15};
static constexpr Register r16{Registers::r16};
static constexpr Register r17{Registers::r17};
static constexpr Register r18{Registers::r18};
static constexpr Register r19{Registers::r19};
static constexpr Register r20{Registers::r20};
static constexpr Register r21{Registers::r21};
static constexpr Register r22{Registers::r22};
static constexpr Register r23{Registers::r23};
static constexpr Register r24{Registers::r24};
static constexpr Register r25{Registers::r25};
static constexpr Register r26{Registers::r26};
static constexpr Register r27{Registers::r27};
static constexpr Register r28{Registers::r28};
static constexpr Register r29{Registers::r29};
static constexpr Register r30{Registers::r30};
static constexpr Register r31{Registers::r31};

// FPR register constants.
static constexpr FloatRegister f0{FloatRegisters::f0, FloatRegisters::Double};
static constexpr FloatRegister f1{FloatRegisters::f1, FloatRegisters::Double};
static constexpr FloatRegister f2{FloatRegisters::f2, FloatRegisters::Double};
static constexpr FloatRegister f3{FloatRegisters::f3, FloatRegisters::Double};
static constexpr FloatRegister f4{FloatRegisters::f4, FloatRegisters::Double};
static constexpr FloatRegister f5{FloatRegisters::f5, FloatRegisters::Double};
static constexpr FloatRegister f6{FloatRegisters::f6, FloatRegisters::Double};
static constexpr FloatRegister f7{FloatRegisters::f7, FloatRegisters::Double};
static constexpr FloatRegister f8{FloatRegisters::f8, FloatRegisters::Double};
static constexpr FloatRegister f9{FloatRegisters::f9, FloatRegisters::Double};
static constexpr FloatRegister f10{FloatRegisters::f10, FloatRegisters::Double};
static constexpr FloatRegister f11{FloatRegisters::f11, FloatRegisters::Double};
static constexpr FloatRegister f12{FloatRegisters::f12, FloatRegisters::Double};
static constexpr FloatRegister f13{FloatRegisters::f13, FloatRegisters::Double};
static constexpr FloatRegister f14{FloatRegisters::f14, FloatRegisters::Double};
static constexpr FloatRegister f15{FloatRegisters::f15, FloatRegisters::Double};
static constexpr FloatRegister f16{FloatRegisters::f16, FloatRegisters::Double};
static constexpr FloatRegister f17{FloatRegisters::f17, FloatRegisters::Double};
static constexpr FloatRegister f18{FloatRegisters::f18, FloatRegisters::Double};
static constexpr FloatRegister f19{FloatRegisters::f19, FloatRegisters::Double};
static constexpr FloatRegister f20{FloatRegisters::f20, FloatRegisters::Double};
static constexpr FloatRegister f21{FloatRegisters::f21, FloatRegisters::Double};
static constexpr FloatRegister f22{FloatRegisters::f22, FloatRegisters::Double};
static constexpr FloatRegister f23{FloatRegisters::f23, FloatRegisters::Double};
static constexpr FloatRegister f24{FloatRegisters::f24, FloatRegisters::Double};
static constexpr FloatRegister f25{FloatRegisters::f25, FloatRegisters::Double};
static constexpr FloatRegister f26{FloatRegisters::f26, FloatRegisters::Double};
static constexpr FloatRegister f27{FloatRegisters::f27, FloatRegisters::Double};
static constexpr FloatRegister f28{FloatRegisters::f28, FloatRegisters::Double};
static constexpr FloatRegister f29{FloatRegisters::f29, FloatRegisters::Double};
static constexpr FloatRegister f30{FloatRegisters::f30, FloatRegisters::Double};
static constexpr FloatRegister f31{FloatRegisters::f31, FloatRegisters::Double};

static constexpr Register InvalidReg{Registers::Invalid};
static constexpr FloatRegister InvalidFloatReg;

static constexpr Register StackPointer = r1;
static constexpr Register FramePointer = r31;
static constexpr Register ReturnReg = r3;
static constexpr Register64 ReturnReg64(ReturnReg);
static constexpr FloatRegister ReturnFloat32Reg{FloatRegisters::f1,
                                                FloatRegisters::Single};
static constexpr FloatRegister ReturnDoubleReg = f1;
static constexpr FloatRegister ReturnSimd128Reg{FloatRegisters::f1,
                                                FloatRegisters::Simd128};

// r16 is non-volatile and non-allocatable, used as a saved scratch.
static constexpr Register SavedScratchRegister = r16;

static constexpr Register SecondScratchReg = r12;

static constexpr FloatRegister ScratchFloat32Reg{FloatRegisters::f0,
                                                 FloatRegisters::Single};
static constexpr FloatRegister ScratchDoubleReg = f0;
static constexpr FloatRegister ScratchSimd128Reg{FloatRegisters::f0,
                                                 FloatRegisters::Simd128};

struct ScratchFloat32Scope : public AutoFloatRegisterScope {
  explicit ScratchFloat32Scope(MacroAssembler& masm)
      : AutoFloatRegisterScope(masm, ScratchFloat32Reg) {}
};

struct ScratchDoubleScope : public AutoFloatRegisterScope {
  explicit ScratchDoubleScope(MacroAssembler& masm)
      : AutoFloatRegisterScope(masm, ScratchDoubleReg) {}
};

// PPC64: ScratchSimd128Scope is a simple register wrapper, NOT a scoped
// acquire/release. On PPC64, ScratchSimd128Reg is v0 (VSR32; encoded as
// {FloatRegisters::f0, Simd128} so encoding() = 0 + 32 = 32) — distinct
// from ScratchDoubleReg = f0 (VSR0). It is non-allocatable and always
// available. Many SIMD functions call other SIMD functions that also need
// v0, creating nested "scopes". Using AutoFloatRegisterScope would assert
// on double-acquire in debug builds. Since v0 is never allocated by the
// register allocator, nesting is safe.
struct ScratchSimd128Scope : public FloatRegister {
  explicit ScratchSimd128Scope(MacroAssembler&)
      : FloatRegister(ScratchSimd128Reg) {}
};

class Assembler;

class UseScratchRegisterScope {
 public:
  explicit UseScratchRegisterScope(Assembler& assembler);
  explicit UseScratchRegisterScope(Assembler* assembler);
  ~UseScratchRegisterScope();

  Register Acquire();
  void Release(const Register& reg);
  bool hasAvailable() const;
  void Include(const GeneralRegisterSet& list) {
    *available_ = GeneralRegisterSet::Union(*available_, list);
  }
  void Exclude(const GeneralRegisterSet& list) {
    *available_ = GeneralRegisterSet::Subtract(*available_, list);
  }

 private:
  GeneralRegisterSet* available_;
  GeneralRegisterSet old_available_;
};

static constexpr Register OsrFrameReg = r6;
static constexpr Register PreBarrierReg = r4;
static constexpr Register InterpreterPCReg = r17;

static constexpr Register CallTempReg0 = r4;
static constexpr Register CallTempReg1 = r9;
static constexpr Register CallTempReg2 = r10;
static constexpr Register CallTempReg3 = r7;
// CallTempReg4 must NOT be JSReturnReg (r5): LMegamorphicLoadSlotPermissive
// uses tempFixed(CallTempReg4) for a saved obj pointer AND defineReturn
// (JSReturnOperand=r5) for output. If they alias, the megamorphic cache
// lookup clobbers the saved obj, corrupting the 'this' pointer.
static constexpr Register CallTempReg4 = r8;
static constexpr Register CallTempReg5 = r6;

// PPC64 ELFv2 has no volatile non-arg GPRs (r3-r10 are all arg regs).
// Use allocatable non-volatile registers as overflow temps.
static constexpr Register CallTempNonArgRegs[] = {r14, r15};
static const uint32_t NumCallTempNonArgRegs = std::size(CallTempNonArgRegs);

static constexpr Register IntArgReg0 = r3;
static constexpr Register IntArgReg1 = r4;
static constexpr Register IntArgReg2 = r5;
static constexpr Register IntArgReg3 = r6;
static constexpr Register IntArgReg4 = r7;
static constexpr Register IntArgReg5 = r8;
static constexpr Register IntArgReg6 = r9;
static constexpr Register IntArgReg7 = r10;

// Registers used by RegExpMatcher and RegExpExecMatch stubs.
static constexpr Register RegExpMatcherRegExpReg = CallTempReg0;
static constexpr Register RegExpMatcherStringReg = CallTempReg1;
static constexpr Register RegExpMatcherLastIndexReg = CallTempReg2;

// Registers used by RegExpExecTest stub (do not use ReturnReg).
static constexpr Register RegExpExecTestRegExpReg = CallTempReg0;
static constexpr Register RegExpExecTestStringReg = CallTempReg1;

// Registers used by RegExpSearcher stub (do not use ReturnReg).
static constexpr Register RegExpSearcherRegExpReg = CallTempReg0;
static constexpr Register RegExpSearcherStringReg = CallTempReg1;
static constexpr Register RegExpSearcherLastIndexReg = CallTempReg2;

static constexpr Register JSReturnReg_Type = r6;
static constexpr Register JSReturnReg_Data = r5;
static constexpr Register JSReturnReg = r5;
static constexpr ValueOperand JSReturnOperand = ValueOperand(JSReturnReg);

static constexpr Register ABINonArgReg0 = r19;
static constexpr Register ABINonArgReg1 = r20;
static constexpr Register ABINonArgReg2 = r21;
static constexpr Register ABINonArgReg3 = r22;
static constexpr Register ABINonArgReturnReg0 = r29;
static constexpr Register ABINonArgReturnReg1 = r30;
static constexpr Register ABINonVolatileReg = r14;
static constexpr Register ABINonArgReturnVolatileReg = r11;

static constexpr FloatRegister ABINonArgDoubleReg{FloatRegisters::f14,
                                                  FloatRegisters::Double};

// Wasm instance pointer register. Preserved across wasm function calls.
static constexpr Register InstanceReg = r18;
static constexpr Register HeapReg = r24;
static constexpr Register GlobalReg = r23;

// Wasm table call registers.
static constexpr Register WasmTableCallScratchReg0 = ABINonArgReg0;
static constexpr Register WasmTableCallScratchReg1 = ABINonArgReg1;
static constexpr Register WasmTableCallSigReg = ABINonArgReg2;
static constexpr Register WasmTableCallIndexReg = ABINonArgReg3;

// Wasm ref call registers.
static constexpr Register WasmCallRefCallScratchReg0 = ABINonArgReg0;
static constexpr Register WasmCallRefCallScratchReg1 = ABINonArgReg1;
static constexpr Register WasmCallRefCallScratchReg2 = ABINonArgReg2;
static constexpr Register WasmCallRefReg = ABINonArgReg3;

// Wasm tail call scratch registers.
// WasmTailCallRAScratchReg must NOT be ABINonArgReg0: the shared tail-call
// code (wasmReturnCallImport, wasmReturnCallIndirect, wasmReturnCallRef)
// stores the callee address in ABINonArgReg0, and CollapseWasmFrame*
// overwrites tempForRA. On architectures with a GPR link register (ARM,
// MIPS, LA64, RISC-V) this is ra/lr. PPC64's LR is an SPR, so we use r14
// (ABINonVolatileReg) which is callee-saved and not used in call setup.
static constexpr Register WasmTailCallInstanceScratchReg = ABINonArgReg1;
static constexpr Register WasmTailCallRAScratchReg = ABINonVolatileReg;
static constexpr Register WasmTailCallFPScratchReg = ABINonArgReg3;

// Register used as a scratch along the return path in the fast js -> wasm stub
// code. Must not overlap ReturnReg, JSReturnOperand, or InstanceReg.
// Must be volatile.
static constexpr Register WasmJitEntryReturnScratch = r10;

static constexpr uint32_t ABIStackAlignment = 16;
static constexpr uint32_t CodeAlignment = 16;
static constexpr uint32_t JitStackAlignment = 16;

static constexpr uint32_t JitStackValueAlignment =
    JitStackAlignment / sizeof(Value);
static_assert(JitStackAlignment % sizeof(Value) == 0 &&
                  JitStackValueAlignment >= 1,
              "Stack alignment should be a non-zero multiple of sizeof(Value)");

static constexpr uint32_t SimdMemoryAlignment = 16;
static_assert(
    CodeAlignment % SimdMemoryAlignment == 0,
    "Code alignment should be larger than any of the alignments "
    "which are used for the constant sections of the code buffer. "
    "Thus it should be larger than the alignment for SIMD constants.");

static constexpr uint32_t WasmStackAlignment = SimdMemoryAlignment;
static const uint32_t WasmTrapInstructionLength = 4;

static constexpr uint32_t WasmCheckedCallEntryOffset = 0u;
static constexpr uint32_t WasmCheckedTailEntryOffset = 32u;

static constexpr Scale ScalePointer = TimesEight;

class ABIArgGenerator : public ABIArgGeneratorShared {
 public:
  explicit ABIArgGenerator(ABIKind kind)
      : ABIArgGeneratorShared(kind),
        intRegIndex_(0),
        floatRegIndex_(0),
        current_() {
    // PPC64 ELFv2 ABI: the callee saves LR, CR, TOC into the caller's
    // frame (offsets 8, 16, 24 from caller SP). Reserve 32 bytes so that
    // callWithABIPre always allocates enough space for this link area.
    stackOffset_ += ShadowStackSpace;
  }

  ABIArg next(MIRType argType);
  ABIArg& current() { return current_; }

 protected:
  unsigned intRegIndex_;
  unsigned floatRegIndex_;
  ABIArg current_;
};

static constexpr uint32_t NumIntArgRegs = 8;
static constexpr uint32_t NumFloatArgRegs = 13;

static inline bool GetIntArgReg(uint32_t usedIntArgs, Register* out) {
  if (usedIntArgs < NumIntArgRegs) {
    *out = Register::FromCode(r3.code() + usedIntArgs);
    return true;
  }
  return false;
}

static inline bool GetFloatArgReg(uint32_t usedFloatArgs, FloatRegister* out) {
  if (usedFloatArgs < NumFloatArgRegs) {
    *out = FloatRegister::FromCode(f1.code() + usedFloatArgs);
    return true;
  }
  return false;
}

static inline bool GetTempRegForIntArg(uint32_t usedIntArgs,
                                       uint32_t usedFloatArgs, Register* out) {
  MOZ_ASSERT(usedFloatArgs == 0);

  if (GetIntArgReg(usedIntArgs, out)) {
    return true;
  }

  usedIntArgs -= NumIntArgRegs;
  if (usedIntArgs >= NumCallTempNonArgRegs) {
    return false;
  }
  *out = CallTempNonArgRegs[usedIntArgs];
  return true;
}

// PPC64 instruction field positions.
// PPC uses big-endian bit numbering (bit 0 = MSB), but we store instructions
// in a uint32_t where bit 0 = LSB. The shifts below are in LSB-0 terms.
//
//   [0:5]  primary opcode   (OpcodeShift=26)
//   [6:10]  RT/RS/BF/TO     (RTShift=21, 5 bits)
//   [11:15] RA/BI           (RAShift=16, 5 bits)
//   [16:20] RB/SH           (RBShift=11, 5 bits)
//   [16:31] SI/UI/D         (Imm16Shift=0, 16 bits)
//   [21:25] subop bits      (varies)
//   [21:30] XO              (X-form; A/M/MD/MDS narrower)
//   [31]    Rc bit          (RcShift=0)

static const uint32_t OpcodeShift = 26;
static const uint32_t OpcodeBits = 6;

static const uint32_t RTShift = 21;
static const uint32_t RTBits = 5;
static const uint32_t RSShift = 21;
static const uint32_t RSBits = 5;
static const uint32_t RAShift = 16;
static const uint32_t RABits = 5;
static const uint32_t RBShift = 11;
static const uint32_t RBBits = 5;
static const uint32_t RCShift = 6;
static const uint32_t RCBits = 5;

static const uint32_t BOShift = 21;
static const uint32_t BOBits = 5;
static const uint32_t BIShift = 16;
static const uint32_t BIBits = 5;

static const uint32_t Imm16Shift = 0;
static const uint32_t Imm16Bits = 16;

static const uint32_t RcShift = 0;
static const uint32_t RcBit = 1;

static const uint32_t RTMask = ((1 << RTBits) - 1) << RTShift;
static const uint32_t RSMask = ((1 << RSBits) - 1) << RSShift;
static const uint32_t RAMask = ((1 << RABits) - 1) << RAShift;
static const uint32_t RBMask = ((1 << RBBits) - 1) << RBShift;
static const uint32_t Imm16Mask = (1 << Imm16Bits) - 1;
static const uint32_t RegMask = (1 << RTBits) - 1;

static inline uint32_t RT(Register r) { return (uint32_t)r.code() << RTShift; }
static inline uint32_t RT(FloatRegister r) {
  return (uint32_t)r.code() << RTShift;
}
static inline uint32_t RS(Register r) { return (uint32_t)r.code() << RSShift; }
static inline uint32_t RS(FloatRegister r) {
  return (uint32_t)r.code() << RSShift;
}
static inline uint32_t RA(Register r) { return (uint32_t)r.code() << RAShift; }
static inline uint32_t RA(FloatRegister r) {
  return (uint32_t)r.code() << RAShift;
}
static inline uint32_t RB(Register r) { return (uint32_t)r.code() << RBShift; }
static inline uint32_t RB(FloatRegister r) {
  return (uint32_t)r.code() << RBShift;
}

// SPR encoding: the SPR number is split across bits 11-15 and 16-20 in a
// swapped arrangement.  PPC_SPR(x) produces the value to OR into an
// mtspr/mfspr instruction at the RB+RA position (bits 11-20).
#define PPC_SPR(x) ((((int)(x) >> 5) & 0x1f) << 11 | ((int)(x) & 0x1f) << 16)

enum PPCOpcodes {
  PPC_add = 0x7C000214,
  PPC_addc = 0x7C000014,
  PPC_adde = 0x7C000114,
  PPC_addi = 0x38000000,
  PPC_addis = 0x3C000000,
  PPC_and_ = 0x7C000038,
  // andi. is always record form (no non-record andi exists).
  PPC_andi_dot = 0x70000000,
  PPC_b = 0x48000000,
  PPC_bc = 0x40000000,
  // Encoded "bcl 20, lt, $+4": PC-relative branch-and-link by 4 bytes
  // (land at the next instruction) with BO=20 (branch always); BI=0
  // (=lt) is don't-care because BO=20 forces the branch. Used by
  // PoolLoadFPR{32,64}'s POWER8 stanza and PoolLoadSimd128's stanza to
  // seed LR with the current PC for the subsequent mflr+ld base
  // computation. Used by patch sites that write raw instruction memory
  // (PatchConstantPoolLoad, WriteLoad64Instructions, etc.). Named for
  // grep-ability and to avoid magic-number copies.
  PPC_bcl_always_plus4 = 0x42800005,
  PPC_bctr = 0x4E800420,
  PPC_bcctr = 0x4C000420,
  PPC_blr = 0x4E800020,
  PPC_cmpd = 0x7C200000,
  PPC_cmpdi = 0x2C200000,
  PPC_cmpld = 0x7C200040,
  PPC_cmpldi = 0x28200000,
  PPC_cmpw = 0x7C000000,
  PPC_cmpwi = 0x2C000000,
  PPC_cmplw = 0x7C000040,
  PPC_cmplwi = 0x28000000,
  PPC_cntlzd = 0x7C000074,
  PPC_cntlzw = 0x7C000034,
  PPC_cnttzd = 0x7C000474,
  PPC_cnttzw = 0x7C000434,
  PPC_crandc = 0x4C000102,
  PPC_cror = 0x4C000382,
  PPC_crorc = 0x4C000342,
  PPC_divd = 0x7C0003D2,
  PPC_divdu = 0x7C000392,
  PPC_divw = 0x7C0003D6,
  PPC_divwu = 0x7C000396,
  // POWER9 (ISA 3.0) modulo instructions.
  PPC_modsd = 0x7C000612,
  PPC_modsw = 0x7C000616,
  PPC_modud = 0x7C000212,
  PPC_moduw = 0x7C000216,
  PPC_extsb = 0x7C000774,
  PPC_extsh = 0x7C000734,
  PPC_extsw = 0x7C0007B4,
  PPC_fabs = 0xFC000210,
  PPC_fadd = 0xFC00002A,
  PPC_fadds = 0xEC00002A,
  PPC_fcpsgn = 0xFC000010,
  PPC_fcfid = 0xFC00069C,
  PPC_fcfids = 0xEC00069C,
  PPC_fcfidu = 0xFC00079C,
  PPC_fcfidus = 0xEC00079C,
  PPC_fcmpu = 0xFC000000,
  PPC_fctid = 0xFC00065C,
  PPC_fctidz = 0xFC00065E,
  PPC_fctiduz = 0xFC00075E,
  PPC_fctiwz = 0xFC00001E,
  PPC_fdiv = 0xFC000024,
  PPC_fdivs = 0xEC000024,
  PPC_fmr = 0xFC000090,
  PPC_fmul = 0xFC000032,
  PPC_fmuls = 0xEC000032,
  PPC_fneg = 0xFC000050,
  PPC_frim = 0xFC0003D0,
  PPC_frip = 0xFC000390,
  PPC_friz = 0xFC000350,
  PPC_frsp = 0xFC000018,
  PPC_fsub = 0xFC000028,
  PPC_fsubs = 0xEC000028,
  PPC_fsqrt = 0xFC00002C,
  PPC_fsqrts = 0xEC00002C,
  PPC_isel = 0x7C00001E,
  // POWER10 (ISA 3.1). RT = (CR[BI]==1) ? 1 : 0. XO=384 at bits 21-30.
  PPC_setbc = 0x7C000300,
  // POWER10 (ISA 3.1). RT = (CR[BI]==0) ? 1 : 0. XO=416.
  PPC_setbcr = 0x7C000340,
  PPC_lbarx = 0x7C000068,
  PPC_lbz = 0x88000000,
  PPC_lbzx = 0x7C0000AE,
  PPC_ld = 0xE8000000,
  PPC_ldarx = 0x7C0000A8,
  PPC_ldx = 0x7C00002A,
  PPC_lfd = 0xC8000000,
  PPC_lfdx = 0x7C0004AE,
  PPC_lfiwax = 0x7C0006AE,
  PPC_lfiwzx = 0x7C0006EE,
  PPC_lfs = 0xC0000000,
  PPC_lfsx = 0x7C00042E,
  PPC_lha = 0xA8000000,
  PPC_lharx = 0x7C0000E8,
  PPC_lhax = 0x7C0002AE,
  PPC_lhz = 0xA0000000,
  PPC_lhzx = 0x7C00022E,
  PPC_lwa = 0xE8000002,
  PPC_lwarx = 0x7C000028,
  PPC_lwz = 0x80000000,
  // X-form sign-extending word load (opcode 31, XO=341). Single-insn
  // equivalent of lwzx + extsw.
  PPC_lwax = 0x7C0002AA,
  PPC_lwzx = 0x7C00002E,
  PPC_mcrxrx = 0x7C000480,
  PPC_mcrfs = 0xFC000080,
  PPC_mfocrf = 0x7C100026,
  PPC_mffs = 0xFC00048E,
  PPC_mfspr = 0x7C0002A6,
  PPC_mfvsrd = 0x7C000066,
  PPC_mtcrf = 0x7C000120,
  PPC_mtfsb0 = 0xFC00008C,
  PPC_mtvsrd = 0x7C000166,
  // POWER8+ (ISA 2.07). VSR[XT].dw[0] = sign_ext_64(RA[32:63]).
  // XO=211 at bits 21-30. Combines extsw + mtvsrd into one insn.
  PPC_mtvsrwa = 0x7C0001A6,
  PPC_mtvsrws = 0x7C000326,
  PPC_mtvsrwz = 0x7C0001E6,
  PPC_mtspr = 0x7C0003A6,
  PPC_mulhd = 0x7C000092,
  PPC_mulhdu = 0x7C000012,
  PPC_mulhwu = 0x7C000016,
  PPC_mulli = 0x1C000000,
  PPC_mulld = 0x7C0001D2,
  PPC_mulldo = 0x7C0005D2,
  PPC_mullw = 0x7C0001D6,
  PPC_neg = 0x7C0000D0,
  PPC_nor = 0x7C0000F8,
  PPC_or_ = 0x7C000378,
  PPC_ori = 0x60000000,
  PPC_oris = 0x64000000,
  PPC_popcntb = 0x7C0000F4,
  PPC_popcntd = 0x7C0003F4,
  PPC_popcntw = 0x7C0002F4,
  PPC_brd = 0x7C000176,  // POWER10: byte-reverse doubleword (X-form, XO=187)
  PPC_brh = 0x7C0001B6,  // POWER10: byte-reverse each halfword (X-form, XO=219)
  PPC_brw = 0x7C000136,  // POWER10: byte-reverse each word     (X-form, XO=155)
  PPC_rldcl = 0x78000010,
  PPC_rldicl = 0x78000000,
  PPC_rldcr = 0x78000012,
  PPC_rldicr = 0x78000004,
  PPC_rldimi = 0x7800000C,
  PPC_rlwimi = 0x50000000,
  PPC_rlwinm = 0x54000000,
  PPC_rlwnm = 0x5C000000,
  PPC_sld = 0x7C000036,
  PPC_slw = 0x7C000030,
  PPC_srad = 0x7C000634,
  PPC_sradi = 0x7C000674,
  PPC_sraw = 0x7C000630,
  PPC_srawi = 0x7C000670,
  PPC_srd = 0x7C000436,
  PPC_srw = 0x7C000430,
  PPC_stb = 0x98000000,
  PPC_stbcx = 0x7C00056D,
  PPC_stbx = 0x7C0001AE,
  PPC_std = 0xF8000000,
  PPC_stdcx = 0x7C0001AD,
  PPC_stdu = 0xF8000001,
  PPC_stdx = 0x7C00012A,
  PPC_stfd = 0xD8000000,
  PPC_stfdu = 0xDC000000,
  PPC_stfdx = 0x7C0005AE,
  PPC_stfs = 0xD0000000,
  PPC_stfsu = 0xD4000000,
  PPC_stfsx = 0x7C00052E,
  PPC_sth = 0xB0000000,
  PPC_sthcx = 0x7C0005AD,
  PPC_sthx = 0x7C00032E,
  PPC_stw = 0x90000000,
  PPC_stwx = 0x7C00012E,
  PPC_stwbrx = 0x7C00052C,
  PPC_stwcx = 0x7C00012D,
  PPC_subf = 0x7C000050,
  PPC_subfc = 0x7C000010,
  PPC_subfe = 0x7C000110,
  PPC_subfic = 0x20000000,
  PPC_sync = 0x7C0004AC,
  // isync — execution synchronization. Discards prefetched instructions and
  // forces a refetch+reexecute of everything past the barrier; prevents
  // speculative bypass. Used for Spectre v1 mitigation in speculationBarrier.
  // Encoding: bytes `2c 01 00 4c` (LE) = 0x4C00012C.
  PPC_isync = 0x4C00012C,
  PPC_trap = 0x7FE00008,
  PPC_tw = 0x7C000008,
  PPC_xor_ = 0x7C000278,
  PPC_xori = 0x68000000,
  PPC_xoris = 0x6C000000,
  // VMX register load/store (X-form, opcode 31, XO=103/231).
  // Operate on raw VR0-31 (the lvx/stvx mnemonics predate VSX, so the
  // assembler exposes them with a uint8_t VR index rather than via the
  // VSR-namespace FloatRegister overloads used for lxvx/stxvx.)
  PPC_lvx = 0x7C0000CE,
  PPC_lxvd2x = 0x7C000698,
  PPC_lxvx = 0x7C000218,
  PPC_mfvsrld = 0x7C000266,
  PPC_mtvsrdd = 0x7C000366,
  PPC_stvx = 0x7C0001CE,
  PPC_stxvd2x = 0x7C000798,
  PPC_stxvx = 0x7C000318,
  PPC_vaddubm = 0x10000000,
  PPC_vavgub = 0x10000402,
  PPC_vavguh = 0x10000442,
  PPC_vcmpequb = 0x10000006,
  PPC_vcmpequh = 0x10000046,
  PPC_vcmpequw = 0x10000086,
  PPC_vcmpequd = 0x100000C7,
  PPC_vcmpgtsb = 0x10000306,
  PPC_vcmpgtsh = 0x10000346,
  PPC_vcmpgtsw = 0x10000386,
  PPC_vcmpgtsd = 0x100003C7,
  PPC_vcmpgtub = 0x10000206,
  PPC_vcmpgtuh = 0x10000246,
  PPC_vcmpgtuw = 0x10000286,
  PPC_vcmpgtud = 0x100002C7,
  PPC_vcmpneb = 0x10000007,  // POWER9 (ISA 3.0)
  PPC_vcmpneh = 0x10000047,  // POWER9
  PPC_vcmpnew = 0x10000087,  // POWER9
  PPC_vadduhm = 0x10000040,
  PPC_vadduwm = 0x10000080,
  PPC_vaddudm = 0x100000C0,
  PPC_vaddubs = 0x10000200,
  PPC_vadduhs = 0x10000240,
  PPC_vaddsbs = 0x10000300,
  PPC_vaddshs = 0x10000340,
  PPC_vmaxsb = 0x10000102,
  PPC_vmaxsh = 0x10000142,
  PPC_vmaxsw = 0x10000182,
  PPC_vmaxsd = 0x100001C2,
  PPC_vmaxub = 0x10000002,
  PPC_vmaxuh = 0x10000042,
  PPC_vmaxuw = 0x10000082,
  PPC_vmhraddshs = 0x10000021,
  PPC_vmrghb = 0x1000000C,
  PPC_vmrghh = 0x1000004C,
  PPC_vmrghw = 0x1000008C,
  PPC_vmrglb = 0x1000010C,
  PPC_vmrglh = 0x1000014C,
  PPC_vmrglw = 0x1000018C,
  PPC_vminsb = 0x10000302,
  PPC_vminsh = 0x10000342,
  PPC_vminsw = 0x10000382,
  PPC_vminub = 0x10000202,
  PPC_vminuh = 0x10000242,
  PPC_vminuw = 0x10000282,
  // POWER9 (ISA 3.0) per-lane integer negate. VRA field carries the subop
  // code: 6 for vnegw, 7 for vnegd. Base XO is 0x602.
  PPC_vnegw = 0x10060602,
  PPC_vnegd = 0x10070602,
  PPC_vmladduhm = 0x10000022,
  PPC_vmuluwm = 0x10000089,
  PPC_vmulld = 0x100001C9,      // POWER10 (XO=457, vector i64x2 multiply low)
  PPC_vmulesb = 0x10000308,
  PPC_vmuleub = 0x10000208,
  PPC_vmulesh = 0x10000348,
  PPC_vmuleuh = 0x10000248,
  PPC_vmulesw = 0x10000388,
  PPC_vmuleuw = 0x10000288,
  PPC_vmulosb = 0x10000108,
  PPC_vmuloub = 0x10000008,
  PPC_vmulosh = 0x10000148,
  PPC_vmulouh = 0x10000048,
  PPC_vmulosw = 0x10000188,
  PPC_vmulouw = 0x10000088,
  PPC_vmsumshm = 0x10000028,
  PPC_vmsumuhm = 0x10000026,
  PPC_vperm = 0x1000002B,
  // VX-form, opcode 4, XO=0x54C. Per-byte bit-permute of a 128-bit value;
  // result 16-bit bitmap lands in dw0 low 16 bits, recoverable via mfvsrd.
  // Available on POWER8+ (ISA 2.07).
  PPC_vbpermq = 0x1000054C,
  // POWER10 (ISA 3.1) Vector Extract Mask. VX-form, opcode 4, XO=0x642,
  // with UIM at bits 11..15 selecting lane width: 8=byte, 9=halfword,
  // 10=word, 11=doubleword. RT is a GPR (low N bits = wasm bitmask).
  PPC_vextractbm = 0x10080642,
  PPC_vextracthm = 0x10090642,
  PPC_vextractwm = 0x100A0642,
  PPC_vextractdm = 0x100B0642,
  // POWER10 vector insert from GPR at immediate byte offset:
  //   vinsw VRT, RB, UIM   VRT[UIM*8:UIM*8+31] ← RB[32:63]
  //   vinsd VRT, RB, UIM   VRT[UIM*8:UIM*8+63] ← RB[0:63]
  // VX-form, opcode 4. RB at bits 16..20, UIM at bits 11..15.
  PPC_vinsw = 0x100000CF,  // POWER10 (XO=207)
  PPC_vinsd = 0x100001CF,  // POWER10 (XO=463)
  // POWER10 vector insert byte/halfword from GPR with register-supplied
  // (right-indexed = LE-natural) byte position:
  //   vinsbrx VRT, RA, RB   VRT.byte[RA & 0xF]  ← RB & 0xFF
  //   vinshrx VRT, RA, RB   VRT.hword[(RA & 0xE)/2] ← RB & 0xFFFF
  // VX-form, opcode 4. RA at bits 16..20, RB at bits 11..15.
  PPC_vinsbrx = 0x1000030F,  // POWER10 (XO=783)
  PPC_vinshrx = 0x1000034F,  // POWER10 (XO=847)
  // POWER9 (ISA 3.0) vector insert byte/halfword from VR at immediate
  // byte position:
  //   vinsertb VRT, VRB, UIM  VRT.byte[UIM]      ← VRB.byte[7]    (BE)
  //   vinserth VRT, VRB, UIM  VRT.hword[UIM..+1] ← VRB.byte[6..7] (BE)
  // V-form, opcode 4. VRB at bits 11..15, UIM at bits 16..20. Simd128
  // lives in VSR32-63 (= VR0-31), so the V-form VRT field addresses our
  // Simd128 storage via `encoding() & 31`.
  PPC_vinsertb = 0x1000030D,  // POWER9 (XO=781)
  PPC_vinserth = 0x1000034D,  // POWER9 (XO=845)
  PPC_vextractub = 0x1000020D,  // POWER9 (XO=525)
  PPC_vextractuh = 0x1000024D,  // POWER9 (XO=589)
  PPC_vspltisb = 0x1000030C,    // POWER7+ (XO=780, splat 5-bit SIMM to all 16 byte lanes)
  PPC_vspltish = 0x1000034C,    // POWER7+ (XO=844, splat 5-bit SIMM to all 8 i16 lanes)
  PPC_vspltisw = 0x1000038C,    // POWER7+ (XO=908, splat 5-bit SIMM to all 4 i32 lanes)
  PPC_vpopcntb = 0x10000703,
  PPC_vslb = 0x10000104,
  PPC_vsld = 0x100005C4,
  PPC_vsldoi = 0x1000002C,
  PPC_vslh = 0x10000144,
  PPC_vslo = 0x1000040C,
  PPC_vslw = 0x10000184,
  PPC_vspltb = 0x1000020C,
  PPC_vsplth = 0x1000024C,
  PPC_vsrab = 0x10000304,
  PPC_vsrad = 0x100003C4,
  PPC_vsrah = 0x10000344,
  PPC_vsraw = 0x10000384,
  PPC_vsrb = 0x10000204,
  PPC_vsrd = 0x100006C4,
  PPC_vsrh = 0x10000244,
  PPC_vsro = 0x1000044C,
  PPC_vsrw = 0x10000284,
  PPC_vpkshss = 0x1000018E,
  PPC_vpkshus = 0x1000010E,
  PPC_vpkswss = 0x100001CE,
  PPC_vpkswus = 0x1000014E,
  PPC_vupkhsb = 0x1000020E,
  PPC_vupkhsh = 0x1000024E,
  PPC_vupkhsw = 0x1000064E,
  PPC_vupklsb = 0x1000028E,
  PPC_vupklsh = 0x100002CE,
  PPC_vupklsw = 0x100006CE,
  PPC_vsububm = 0x10000400,
  PPC_vsubuhm = 0x10000440,
  PPC_vsubuwm = 0x10000480,
  PPC_vsubudm = 0x100004C0,
  PPC_vsububs = 0x10000600,
  PPC_vsubuhs = 0x10000640,
  PPC_vsubsbs = 0x10000700,
  PPC_vsubshs = 0x10000740,
  PPC_xscvdpspn = 0xF000042C,
  PPC_xscvspdpn = 0xF000052C,
  // POWER9 (ISA 3.0) scalar FP16 conversions, XX2-form. The UIM
  // disambiguator is baked into the constant (xscvdphp=17, xscvhpdp=16).
  // Encodings cross-checked against binutils with `.machine power9`.
  PPC_xscvdphp = 0xF011056C,
  PPC_xscvhpdp = 0xF010056C,
  // POWER9 (ISA 3.0) scalar VSX extract biased exponent, XX2-form.
  // XT.dword[0] = (zero || biased_exp_11bit), XT.dword[1] = 0. XO=347
  // (shares XO with xscv{dp,hp}{hp,dp} — disambiguated by bits 16-20=0).
  // Encoding cross-checked against binutils with `.machine power9`.
  PPC_xsxexpdp = 0xF000056C,
  // POWER9 (ISA 3.0) scalar FP16 load/store, X-form (opcode 31).
  // lxsihzx zero-extends; stxsihx writes 16 bits from VSR dword 0
  // word 1's low halfword.
  PPC_lxsihzx = 0x7C00065A,
  PPC_stxsihx = 0x7C00075A,
  // POWER9 scalar VSX max/min with Java/JavaScript semantics — handles
  // ±0 and NaN identically to Math.max/Math.min in ECMA-262 (covers
  // 19 corner cases against the JS shell).
  // XX3-form, primary opcode 60, XO=144 (max) / XO=152 (min).
  PPC_xsmaxjdp = 0xF0000480,
  PPC_xsminjdp = 0xF00004C0,
  PPC_xxbrd = 0xF017076C,
  PPC_xvabsdp = 0xF0000764,
  PPC_xvabssp = 0xF0000664,
  PPC_xvadddp = 0xF0000300,
  PPC_xvaddsp = 0xF0000200,
  PPC_xvcmpeqdp = 0xF0000318,
  PPC_xvcmpeqsp = 0xF0000218,
  PPC_xvcmpgedp = 0xF0000398,
  PPC_xvcmpgesp = 0xF0000298,
  PPC_xvcmpgtdp = 0xF0000358,
  PPC_xvcmpgtsp = 0xF0000258,
  PPC_xvcvdpsp = 0xF0000624,
  PPC_xvcvdpsxws = 0xF0000360,
  PPC_xvcvdpuxws = 0xF0000320,
  PPC_xvcvspdp = 0xF0000724,
  PPC_xvcvspsxws = 0xF0000260,
  PPC_xvcvspuxws = 0xF0000220,
  PPC_xvcvsxwdp = 0xF00003E0,
  PPC_xvcvsxwsp = 0xF00002E0,
  PPC_xvcvuxwdp = 0xF00003A0,
  PPC_xvcvuxwsp = 0xF00002A0,
  PPC_xvdivdp = 0xF00003C0,
  PPC_xvdivsp = 0xF00002C0,
  PPC_xvmaddadp = 0xF0000308,
  PPC_xvmaddasp = 0xF0000208,
  PPC_xvmaxdp = 0xF0000700,
  PPC_xvmaxsp = 0xF0000600,
  PPC_xvmindp = 0xF0000740,
  PPC_xvminsp = 0xF0000640,
  PPC_xvmuldp = 0xF0000380,
  PPC_xvmulsp = 0xF0000280,
  PPC_xvnegdp = 0xF00007E4,
  PPC_xvnmsubadp = 0xF0000788,
  PPC_xvnmsubasp = 0xF0000688,
  PPC_xvnegsp = 0xF00006E4,
  PPC_xvrdpic = 0xF00003AC,
  PPC_xvrdpim = 0xF00003E4,
  PPC_xvrdpip = 0xF00003A4,
  PPC_xvrdpiz = 0xF0000364,
  PPC_xvrspic = 0xF00002AC,
  PPC_xvrspim = 0xF00002E4,
  PPC_xvrspip = 0xF00002A4,
  PPC_xvrspiz = 0xF0000264,
  PPC_xvsqrtdp = 0xF000032C,
  PPC_xvsqrtsp = 0xF000022C,
  PPC_xvsubdp = 0xF0000340,
  PPC_xvsubsp = 0xF0000240,
  PPC_xxextractuw = 0xF0000294,
  PPC_xxinsertw = 0xF00002D4,
  PPC_xxland = 0xF0000410,
  PPC_xxlandc = 0xF0000450,
  PPC_xxlnor = 0xF0000510,
  PPC_xxlor = 0xF0000490,
  PPC_xxlxor = 0xF00004D0,
  PPC_xxpermdi = 0xF0000050,
  PPC_xxsel = 0xF0000030,
  PPC_xxspltib = 0xF00002D0,  // POWER9 (ISA 3.0): XX1-form, no Rc
  PPC_xxspltw = 0xF0000290,

  // Simplified mnemonics.
  PPC_mr = PPC_or_,
  PPC_not = PPC_nor,
  PPC_nop = PPC_ori,
  PPC_lwsync = PPC_sync | (1 << 21),

  PPC_MAJOR_OPCODE_MASK = 0xFC000000
};

static const uint32_t NopInst = (uint32_t)PPC_nop;
static const uint32_t PPC_STANZA_LENGTH = 16;

class Instruction;
class InstReg;
class InstImm;
class BOffImm16;
class JOffImm26;

// PPC64 base instruction type: a single 32-bit word.
class Instruction {
 protected:
  uint32_t data;

 public:
  explicit Instruction(uint32_t data_) : data(data_) {}
  explicit Instruction(PPCOpcodes op) : data((uint32_t)op) {}

  uint32_t encode() const { return data; }

  void makeNop() { data = NopInst; }
  void makeOp_mtctr(Register r) {
    data = PPC_mtspr | ((uint32_t)r.code()) << 21 | PPC_SPR(9);
  }
  void makeOp_bctr(uint32_t linkBit = 0) { data = PPC_bctr | linkBit; }

  void setData(uint32_t data) { this->data = data; }

  const Instruction& operator=(const Instruction& src) {
    data = src.data;
    return *this;
  }

  uint32_t extractBit(uint32_t bit) const { return (encode() >> bit) & 1; }
  uint32_t extractBitField(uint32_t hi, uint32_t lo) const {
    return (encode() >> lo) & ((2 << (hi - lo)) - 1);
  }

  uint32_t extractOpcode() const { return data & PPC_MAJOR_OPCODE_MASK; }
  bool isOpcode(uint32_t op) const {
    return extractOpcode() == (op & PPC_MAJOR_OPCODE_MASK);
  }

  uint32_t extractRT() const {
    return extractBitField(RTShift + RTBits - 1, RTShift);
  }
  uint32_t extractRA() const {
    return extractBitField(RAShift + RABits - 1, RAShift);
  }
  uint32_t extractRB() const {
    return extractBitField(RBShift + RBBits - 1, RBShift);
  }
  uint32_t extractImm16() const { return data & Imm16Mask; }

  Instruction* next() { return this + 1; }

  const uint32_t* raw() const { return &data; }
  uint32_t size() const { return 4; }
};

static_assert(sizeof(Instruction) == 4);

class InstNOP : public Instruction {
 public:
  InstNOP() : Instruction(NopInst) {}
};

// Register-register-register instruction (X-form and XO-form).
class InstReg : public Instruction {
 public:
  explicit InstReg(PPCOpcodes op) : Instruction(op) {}
  InstReg(PPCOpcodes op, Register rt, Register ra, Register rb)
      : Instruction((uint32_t)op | RT(rt) | RA(ra) | RB(rb)) {}
  InstReg(PPCOpcodes op, FloatRegister frt, FloatRegister fra,
          FloatRegister frb)
      : Instruction((uint32_t)op | RT(frt) | RA(fra) | RB(frb)) {}

  void setRT(Register r) { data = (data & ~RTMask) | RT(r); }
  void setRA(Register r) { data = (data & ~RAMask) | RA(r); }
  void setRB(Register r) { data = (data & ~RBMask) | RB(r); }

  void setImm16(uint32_t imm) {
    data = (data & 0xFFFF0000) | (imm & Imm16Mask);
  }
  uint32_t extractImm16Value() const { return data & Imm16Mask; }
};

// Register-immediate instruction (D-form).
// Bits 21-25 hold RT (loads, addi) or RS (stores, ori). Both encode identically
// since RT and RS occupy the same field; the caller simply passes the right
// register.
class InstImm : public Instruction {
 public:
  explicit InstImm(PPCOpcodes op) : Instruction(op) {}
  InstImm(PPCOpcodes op, Register rt, Register ra, uint32_t imm16)
      : Instruction((uint32_t)op | RT(rt) | RA(ra) | (imm16 & Imm16Mask)) {}

  void setRT(Register r) { data = (data & ~RTMask) | RT(r); }
  void setRA(Register r) { data = (data & ~RAMask) | RA(r); }

  void setImm16(uint32_t imm) {
    data = (data & 0xFFFF0000) | (imm & Imm16Mask);
  }
  void setLowerReg(Register rl) {
    data = (data & 0xFFE0FFFF) | ((uint32_t)rl.code() << 16);
  }
  uint32_t extractImm16Value() const { return data & Imm16Mask; }

  // Extract the TrapTag from a tagged trap instruction (tw).
  // Defined in Assembler-ppc64.cpp. Returns a TrapTag value as uint8_t
  // because Assembler::TrapTag is not yet defined at this point in the header.
  uint8_t traptag();
};

// A BOffImm16 is a 16-bit signed branch offset for conditional branches
// (bc-form instructions).  The offset is stored in bits 2..15 and is
// 4-byte aligned, giving a range of +/-32 KB.
class BOffImm16 {
  int32_t data;

 public:
  uint32_t encode() const {
    MOZ_ASSERT(!isInvalid());
    return static_cast<uint32_t>(data) & 0xFFFC;
  }
  int32_t decode() const {
    MOZ_ASSERT(!isInvalid());
    return data;
  }

  explicit BOffImm16(int offset) : data(offset) {
    MOZ_ASSERT((offset & 0x3) == 0);
    MOZ_ASSERT(IsInRange(offset));
  }
  static bool IsInRange(int offset) {
    return offset >= -32768 && offset <= 32764;
  }

  static const int32_t INVALID = 0x00020000;
  BOffImm16() : data(INVALID) {}

  bool isInvalid() const { return data == INVALID; }

  Instruction* getDest(Instruction* src) const;

  explicit BOffImm16(InstImm inst);
};

// A JOffImm26 is a 26-bit signed branch offset for unconditional branches
// (b/bl instructions).  Bits 2..25 encode the offset, 4-byte aligned,
// giving a range of +/-32 MB.
class JOffImm26 {
  int32_t data;

 public:
  uint32_t encode() const {
    MOZ_ASSERT(!isInvalid());
    return static_cast<uint32_t>(data) & 0x03FFFFFC;
  }
  int32_t decode() const {
    MOZ_ASSERT(!isInvalid());
    return data;
  }

  explicit JOffImm26(int offset) : data(offset) {
    MOZ_ASSERT((offset & 0x3) == 0);
    MOZ_ASSERT(IsInRange(offset));
  }
  static bool IsInRange(int offset) {
    return offset >= -33554432 && offset <= 33554428;
  }

  static const int32_t INVALID = 0x20000000;
  JOffImm26() : data(INVALID) {}

  bool isInvalid() const { return data == INVALID; }

  Instruction* getDest(Instruction* src) const;
};

// A 16-bit immediate value used in D-form instructions.
class Imm16 {
  int32_t value;

 public:
  Imm16();
  explicit Imm16(uint32_t imm) : value(imm) {}
  uint32_t encode() const { return static_cast<uint32_t>(value) & 0xffff; }
  int32_t decodeSigned() const { return value; }
  uint32_t decodeUnsigned() const { return value; }
  static bool IsInSignedRange(int32_t imm) {
    return imm >= INT16_MIN && imm <= INT16_MAX;
  }
  static bool IsInUnsignedRange(uint32_t imm) { return imm <= UINT16_MAX; }
  static Imm16 Lower(Imm32 imm) { return Imm16(imm.value & 0xffff); }
  static Imm16 Upper(Imm32 imm) { return Imm16((imm.value >> 16) & 0xffff); }
};

class Imm8 {
  uint8_t value;

 public:
  Imm8();
  explicit Imm8(uint32_t imm) : value(imm) {}
  uint32_t encode(uint32_t shift) const { return value << shift; }
  int32_t decodeSigned() const { return value; }
  uint32_t decodeUnsigned() const { return value; }
  static bool IsInSignedRange(int32_t imm) {
    return imm >= INT8_MIN && imm <= INT8_MAX;
  }
  static bool IsInUnsignedRange(uint32_t imm) { return imm <= UINT8_MAX; }
  static Imm8 Lower(Imm16 imm) { return Imm8(imm.decodeSigned() & 0xff); }
  static Imm8 Upper(Imm16 imm) {
    return Imm8((imm.decodeSigned() >> 8) & 0xff);
  }
};

class Operand {
 public:
  enum Tag { REG, FREG, MEM };

 private:
  Tag tag : 3;
  uint32_t reg : 5;
  int32_t offset;

 public:
  MOZ_IMPLICIT Operand(Register reg_) : tag(REG), reg(reg_.code()) {}

  explicit Operand(FloatRegister freg) : tag(FREG), reg(freg.code()) {}

  Operand(Register base, Imm32 off)
      : tag(MEM), reg(base.code()), offset(off.value) {}

  Operand(Register base, int32_t off)
      : tag(MEM), reg(base.code()), offset(off) {}

  explicit Operand(const Address& addr)
      : tag(MEM), reg(addr.base.code()), offset(addr.offset) {}

  Tag getTag() const { return tag; }

  Register toReg() const {
    MOZ_ASSERT(tag == REG);
    return Register::FromCode(reg);
  }

  FloatRegister toFReg() const {
    MOZ_ASSERT(tag == FREG);
    return FloatRegister::FromCode(reg);
  }

  void toAddr(Register* r, Imm32* dest) const {
    MOZ_ASSERT(tag == MEM);
    *r = Register::FromCode(reg);
    *dest = Imm32(offset);
  }
  Address toAddress() const {
    MOZ_ASSERT(tag == MEM);
    return Address(Register::FromCode(reg), offset);
  }
  int32_t disp() const {
    MOZ_ASSERT(tag == MEM);
    return offset;
  }

  int32_t base() const {
    MOZ_ASSERT(tag == MEM);
    return reg;
  }
  Register baseReg() const {
    MOZ_ASSERT(tag == MEM);
    return Register::FromCode(reg);
  }
};

// Bug 2034064 collapsed the per-buffer compile-time configuration of
// AssemblerBufferWithConstantPools into AssemblerBufferSettings, and reduced
// the runtime ctor to (poolMaxOffset, nopFill). instBufferAlign and the
// NumShortBranchRanges template arg were dropped: PPC64 previously passed
// instBufferAlign=8 (unused on this backend; pool entries are 4-byte aligned)
// and NumShortBranchRanges=0.
using PPCBuffer = js::jit::AssemblerBufferWithConstantPools<
    Instruction, Assembler,
    js::jit::AssemblerBufferSettings{
        .instSize = 4,
        .guardSize = 1,
        .headerSize = 1,
        .pcBias = 0,
        .alignFillInst = NopInst,
        .nopFillInst = NopInst,
    }>;

// Inherits executableCopy() and appendRawCode() from
// AssemblerBufferWithConstantPools, which assert pool is flushed.
class PPCBufferWithExecutableCopy : public PPCBuffer {
 public:
  PPCBufferWithExecutableCopy(size_t poolMaxOffset, unsigned nopFill)
      : PPCBuffer(poolMaxOffset, nopFill) {}
};

class Assembler : public AssemblerShared {
 public:
  // Trap tags encoded in the low bits of a trap word.
  // FreeBSD and others may use r1 in their trap word, so bit 0 is avoided.
  enum TrapTag {
    BTag = 2,
    BCTag = 4,
    CallTag = 6,
    DebugTag0 = 10,
    DebugTag1 = 12,
    DebugTag2 = 14
  };

  // Pool load types encoded in bits 21-22 of pool hint words.
  // Used by InsertIndexIntoTag / PatchConstantPoolLoad.
  enum PoolLoadType {
    PoolLoadFPR64 = 1,    // lfd fD, offset(rBase)
    PoolLoadSimd128 = 2,  // addi rBase, rBase, offset; lxvx vsD, 0, rBase
    PoolLoadFPR32 = 3     // lfs fD, offset(rBase) — auto-expands to double
  };

  enum BranchBits {
    BranchOnClear = 0x04,
    BranchOnSet = 0x0c,
    BranchOptionMask = 0x0f,
    BranchOptionInvert = 0x08
  };

  // PPC condition encoding. The top nybble is the offset to the CR field
  // (the x in BIF*4+x), and the bottom is the BO field.
  // Synthetic flags sit in the MSB and are masked off before use.
  enum Condition {
    ConditionUnsigned = 0x100,
    ConditionUnsignedHandled = 0x2ff,
    ConditionZero = 0x400,
    ConditionOnlyXER = 0x200,
    ConditionXERCA = 0x23c,
    ConditionXERNCA = 0x234,
    ConditionXEROV = 0x21c,

    Equal = 0x2c,
    NotEqual = 0x24,
    GreaterThan = 0x1c,
    GreaterThanOrEqual = 0x04,
    LessThan = 0x0c,
    LessThanOrEqual = 0x14,

    Above = GreaterThan | ConditionUnsigned,
    AboveOrEqual = GreaterThanOrEqual | ConditionUnsigned,
    Below = LessThan | ConditionUnsigned,
    BelowOrEqual = LessThanOrEqual | ConditionUnsigned,

    Signed = LessThan | ConditionZero,
    NotSigned = GreaterThanOrEqual | ConditionZero,
    Zero = Equal | ConditionZero,
    NonZero = NotEqual | ConditionZero,

    Overflow = ConditionXEROV,
    NotOverflow = ConditionOnlyXER | LessThanOrEqual,
    CarrySet = ConditionXERCA,
    CarryClear = ConditionXERNCA,

    Always = 0x1f,
    SOBit = 0x3c,
    NSOBit = 0x34
  };

  enum DoubleCondition {
    DoubleConditionUnordered = 0x100,
    DoubleOrdered = 0x34,
    DoubleEqual = 0x2c,
    DoubleNotEqual = 0x24,
    DoubleGreaterThan = 0x1c,
    DoubleGreaterThanOrEqual = 0x04,
    DoubleLessThan = 0x0c,
    DoubleLessThanOrEqual = 0x14,
    DoubleUnordered = 0x3c,
    DoubleEqualOrUnordered = DoubleEqual | DoubleConditionUnordered,
    DoubleNotEqualOrUnordered = DoubleNotEqual | DoubleConditionUnordered,
    DoubleGreaterThanOrUnordered = DoubleGreaterThan | DoubleConditionUnordered,
    DoubleGreaterThanOrEqualOrUnordered =
        DoubleGreaterThanOrEqual | DoubleConditionUnordered,
    DoubleLessThanOrUnordered = DoubleLessThan | DoubleConditionUnordered,
    DoubleLessThanOrEqualOrUnordered =
        DoubleLessThanOrEqual | DoubleConditionUnordered,
  };

  enum JumpOrCall { BranchIsJump, BranchIsCall };

  enum LinkBit {
    DontLinkB = 0,
    LinkB = 1,
  };

  enum LikelyBit {
    NotLikelyB = 0,
    LikelyB = 1,
  };

  enum BranchAddressType {
    RelativeBranch = 0,
    AbsoluteBranch = 2,
  };

  enum FloatFormat { SingleFloat, DoubleFloat };
  enum FloatTestKind { TestForTrue, TestForFalse };

  BufferOffset nextOffset() { return m_buffer.nextOffset(); }

 protected:
  Instruction* editSrc(BufferOffset bo) {
    if (!bo.assigned()) {
      // Under OOM, writeInst may return an unassigned BufferOffset.
      // Return a dummy writable area so callers (WriteLoad64Instructions)
      // can proceed harmlessly; the compilation will be discarded.
      static uint32_t oomDummy_[8];
      return (Instruction*)oomDummy_;
    }
    return m_buffer.getInst(bo);
  }

  struct RelativePatch {
    BufferOffset offset;
    void* target;
    RelocationKind kind;

    RelativePatch(BufferOffset offset, void* target, RelocationKind kind)
        : offset(offset), target(target), kind(kind) {}
  };

  js::Vector<RelativePatch, 8, SystemAllocPolicy> jumps_;

  CompactBufferWriter jumpRelocations_;
  CompactBufferWriter dataRelocations_;

  PPCBufferWithExecutableCopy m_buffer;

#ifdef JS_JITSPEW
  Sprinter* printer;
#endif

 public:
  // Which absolute bit number does a CR + Condition pair refer to?
  static uint8_t crBit(CRegisterID cr, Condition cond) {
    return (cr << 2) + ((cond & 0xf0) >> 4);
  }
  static uint8_t crBit(CRegisterID cr, DoubleCondition cond) {
    return (cr << 2) + ((cond & 0xf0) >> 4);
  }

  Assembler()
      : m_buffer(/* poolMaxOffset */ 8192, /* nopFill */ 0),
#ifdef JS_JITSPEW
        printer(nullptr),
#endif
        isFinished(false),
        scratch_register_list_((1 << Registers::r11) | (1 << Registers::r12)) {
  }

  void setUnlimitedBuffer() { m_buffer.setUnlimited(); }

  // Constant pool callbacks required by AssemblerBufferWithConstantPools.
  static void InsertIndexIntoTag(uint8_t* load, uint32_t index);
  static bool PatchConstantPoolLoad(void* loadAddr, void* constPoolAddr);
  static void WritePoolGuard(BufferOffset branch, Instruction* inst,
                             BufferOffset dest);
  static void WritePoolHeader(uint8_t* start, js::jit::Pool* p, bool isNatural);
  static void PatchShortRangeBranchToVeneer(PPCBuffer*, unsigned rangeIdx,
                                            BufferOffset deadline,
                                            BufferOffset veneer);

  static Condition InvertCondition(Condition cond);
  static DoubleCondition InvertCondition(DoubleCondition cond);

  void writeRelocation(BufferOffset src) {
    jumpRelocations_.writeUnsigned(src.getOffset());
  }

  void writeDataRelocation(ImmGCPtr ptr) {
    if (ptr.value) {
      if (gc::IsInsideNursery(ptr.value)) {
        embedsNurseryPointers_ = true;
      }
      dataRelocations_.writeUnsigned(nextOffset().getOffset());
    }
  }
  void writeDataRelocation(BufferOffset bo, ImmGCPtr ptr) {
    if (ptr.value) {
      if (gc::IsInsideNursery(ptr.value)) {
        embedsNurseryPointers_ = true;
      }
      dataRelocations_.writeUnsigned(bo.getOffset());
    }
  }

  void assertNoGCThings() const {
#ifdef DEBUG
    MOZ_ASSERT(dataRelocations_.length() == 0);
    for (auto& j : jumps_) {
      MOZ_ASSERT(j.kind == RelocationKind::HARDCODED);
    }
#endif
  }

  bool oom() const;

  void setPrinter(Sprinter* sp) {
#ifdef JS_JITSPEW
    printer = sp;
#endif
  }

#ifdef JS_JITSPEW
  inline void spew(const char* fmt, ...) MOZ_FORMAT_PRINTF(2, 3) {
    if (MOZ_UNLIKELY(printer || JitSpewEnabled(JitSpew_Codegen))) {
      va_list va;
      va_start(va, fmt);
      spewVA(fmt, va);
      va_end(va);
    }
  }
  MOZ_COLD void spewVA(const char* fmt, va_list va) MOZ_FORMAT_PRINTF(2, 0) {
    char buf[200];
    int i = VsprintfLiteral(buf, fmt, va);
    if (i > -1) {
      if (printer) {
        printer->printf("%s\n", buf);
      }
      js::jit::JitSpew(js::jit::JitSpew_Codegen, "%s", buf);
    }
  }
#else
  MOZ_ALWAYS_INLINE void spew(const char* fmt, ...) MOZ_FORMAT_PRINTF(2, 3) {}
#endif

  Register getStackPointer() const { return StackPointer; }

 protected:
  bool isFinished;

 public:
  static uintptr_t GetPointer(uint8_t*);
  void flush() {
    MOZ_ASSERT(!isFinished);
    m_buffer.flushPool();
  }
  // Inhibit pool flushes for the next maxInst instructions. Mirrors the
  // ARM/ARM64 wrappers; lets shared code (e.g. WasmFrameIter epilogues
  // that need static byte distances between currentOffset() captures)
  // fence a small instruction window without reaching into m_buffer.
  void enterNoPool(size_t maxInst) { m_buffer.enterNoPool(maxInst); }
  void leaveNoPool() { m_buffer.leaveNoPool(); }
  void finish();
  bool appendRawCode(const uint8_t* code, size_t numBytes);
  bool reserve(size_t size);
  bool swapBuffer(wasm::Bytes& bytes);
  void executableCopy(void* buffer);
  void copyJumpRelocationTable(uint8_t* dest);
  void copyDataRelocationTable(uint8_t* dest);

  size_t size() const;
  size_t jumpRelocationTableBytes() const;
  size_t dataRelocationTableBytes() const;
  size_t bytesNeeded() const;

  BufferOffset writeInst(uint32_t x, uint32_t* dest = nullptr);
  static void WriteInstStatic(uint32_t x, uint32_t* dest);

 public:
  BufferOffset haltingAlign(int alignment);
  BufferOffset nopAlign(int alignment);
  BufferOffset as_nop();

  // --- Instruction emission (declarations only, implemented in later commits)

  // Branch instructions.
  uint16_t computeConditionCode(Condition op, CRegisterID cr = cr0);
  uint16_t computeConditionCode(DoubleCondition cond, CRegisterID cr = cr0);
  BufferOffset as_b(JOffImm26 off, BranchAddressType bat = RelativeBranch,
                    LinkBit lb = DontLinkB);
  BufferOffset as_b(int32_t off, BranchAddressType bat = RelativeBranch,
                    LinkBit lb = DontLinkB);
  BufferOffset as_blr(LinkBit lb = DontLinkB);
  BufferOffset as_bctr(LinkBit lb = DontLinkB);
  BufferOffset as_bc(BOffImm16 off, Condition cond, CRegisterID cr = cr0,
                     LikelyBit lkb = NotLikelyB, LinkBit lb = DontLinkB);
  BufferOffset as_bc(int16_t off, Condition cond, CRegisterID cr = cr0,
                     LikelyBit lkb = NotLikelyB, LinkBit lb = DontLinkB);
  BufferOffset as_bc(BOffImm16 off, DoubleCondition cond, CRegisterID cr = cr0,
                     LikelyBit lkb = NotLikelyB, LinkBit lb = DontLinkB);
  BufferOffset as_bc(int16_t off, DoubleCondition cond, CRegisterID cr = cr0,
                     LikelyBit lkb = NotLikelyB, LinkBit lb = DontLinkB);
  BufferOffset as_bcctr(Condition cond, CRegisterID cr = cr0,
                        LikelyBit lkb = NotLikelyB, LinkBit lb = DontLinkB);
  BufferOffset as_bcctr(DoubleCondition cond, CRegisterID cr = cr0,
                        LikelyBit lkb = NotLikelyB, LinkBit lb = DontLinkB);
  BufferOffset as_bc(int16_t off, uint16_t op, LikelyBit lkb = NotLikelyB,
                     LinkBit lb = DontLinkB);
  BufferOffset as_bcctr(uint16_t op, LikelyBit lkb = NotLikelyB,
                        LinkBit lb = DontLinkB);

  // SPR operations.
  BufferOffset as_mtspr(SPRegisterID spr, Register ra);
  BufferOffset as_mfspr(Register rd, SPRegisterID spr);

  // CR operations.
  BufferOffset as_crand(uint8_t t, uint8_t a, uint8_t b);
  BufferOffset as_crandc(uint8_t t, uint8_t a, uint8_t b);
  BufferOffset as_cror(uint8_t t, uint8_t a, uint8_t b);
  BufferOffset as_crorc(uint8_t t, uint8_t a, uint8_t b);
  BufferOffset as_crxor(uint8_t t, uint8_t a, uint8_t b);
  BufferOffset as_mtcrf(uint32_t mask, Register rs);
  BufferOffset as_mfocrf(Register rd, CRegisterID crfs);
  BufferOffset as_mcrxrx(CRegisterID crt);

  // Compare instructions.
  BufferOffset as_cmpd(CRegisterID cr, Register ra, Register rb);
  BufferOffset as_cmpdi(CRegisterID cr, Register ra, int16_t im);
  BufferOffset as_cmpld(CRegisterID cr, Register ra, Register rb);
  BufferOffset as_cmpldi(CRegisterID cr, Register ra, int16_t im);
  BufferOffset as_cmpw(CRegisterID cr, Register ra, Register rb);
  BufferOffset as_cmpwi(CRegisterID cr, Register ra, int16_t im);
  BufferOffset as_cmplw(CRegisterID cr, Register ra, Register rb);
  BufferOffset as_cmplwi(CRegisterID cr, Register ra, int16_t im);
  BufferOffset as_cmpd(Register ra, Register rb);
  BufferOffset as_cmpdi(Register ra, int16_t im);
  BufferOffset as_cmpld(Register ra, Register rb);
  BufferOffset as_cmpldi(Register ra, int16_t im);
  BufferOffset as_cmpw(Register ra, Register rb);
  BufferOffset as_cmpwi(Register ra, int16_t im);
  BufferOffset as_cmplw(Register ra, Register rb);
  BufferOffset as_cmplwi(Register ra, int16_t im);

  // ALU (three-register).
  BufferOffset as_add(Register rd, Register ra, Register rb);
  BufferOffset as_addc(Register rd, Register ra, Register rb);
  BufferOffset as_adde(Register rd, Register ra, Register rb);
  BufferOffset as_subf(Register rd, Register ra, Register rb);
  BufferOffset as_subfc(Register rd, Register ra, Register rb);
  BufferOffset as_subfe(Register rd, Register ra, Register rb);
  BufferOffset as_neg(Register rd, Register rs);

  BufferOffset as_mulld(Register rd, Register ra, Register rb);
  BufferOffset as_mulhd(Register rd, Register ra, Register rb);
  BufferOffset as_mulhdu(Register rd, Register ra, Register rb);
  BufferOffset as_mulldo(Register rd, Register ra, Register rb);
  BufferOffset as_mullw(Register rd, Register ra, Register rb);
  BufferOffset as_mulhwu(Register rd, Register ra, Register rb);

  BufferOffset as_divd(Register rd, Register ra, Register rb);
  BufferOffset as_divdu(Register rd, Register ra, Register rb);
  BufferOffset as_divw(Register rd, Register ra, Register rb);
  BufferOffset as_divwu(Register rd, Register ra, Register rb);
  // POWER9 modulo.
  BufferOffset as_modsd(Register rd, Register ra, Register rb);
  BufferOffset as_modsw(Register rd, Register ra, Register rb);
  BufferOffset as_modud(Register rd, Register ra, Register rb);
  BufferOffset as_moduw(Register rd, Register ra, Register rb);

  // ALU immediate.
  BufferOffset as_addi(Register rd, Register ra, int16_t im,
                       bool actually_li = false);
  BufferOffset as_addis(Register rd, Register ra, int16_t im,
                        bool actually_lis = false);
  BufferOffset as_mulli(Register rd, Register ra, int16_t im);
  BufferOffset as_subfic(Register rd, Register ra, int16_t im);

  // ALU unary/extended.
  BufferOffset as_cntlzw(Register rd, Register ra);
  BufferOffset as_cntlzd(Register rd, Register ra);
  BufferOffset as_cnttzd(Register rd, Register ra);
  BufferOffset as_cnttzw(Register rd, Register ra);
  BufferOffset as_popcntd(Register ra, Register rs);
  BufferOffset as_popcntw(Register ra, Register rs);
  // POWER10 byte-reverse doubleword: ra = bswap64(rs). 1 insn replacing the
  // POWER9 mtvsrd / xxbrd / mfvsrd round-trip in byteSwap64.
  BufferOffset as_brd(Register ra, Register rs);
  // POWER10 byte-reverse each halfword (4 halfwords) / each word (2 words)
  // in the 64-bit doubleword. The wasm/asm caller usually masks or
  // sign-extends the low halfword/word afterwards.
  BufferOffset as_brh(Register ra, Register rs);
  BufferOffset as_brw(Register ra, Register rs);

  // Bit operations (logical, three-register).
  BufferOffset as_and_(Register rd, Register rs, Register rb);
  BufferOffset as_and__rc(Register rd, Register rs, Register rb);
  BufferOffset as_nor(Register rd, Register rs, Register rb);
  BufferOffset as_or_(Register rd, Register rs, Register rb);
  BufferOffset as_xor_(Register rd, Register rs, Register rb);
  BufferOffset as_slw(Register rd, Register rs, Register rb);
  BufferOffset as_srw(Register rd, Register rs, Register rb);
  BufferOffset as_sraw(Register rd, Register rs, Register rb);
  BufferOffset as_sld(Register rd, Register rs, Register rb);
  BufferOffset as_srd(Register rd, Register rs, Register rb);
  BufferOffset as_srad(Register rd, Register rs, Register rb);

  // Bit operations (logical, immediate).
  BufferOffset as_ori(Register rd, Register ra, uint16_t im);
  BufferOffset as_oris(Register rd, Register ra, uint16_t im);
  BufferOffset as_xori(Register rd, Register ra, uint16_t im);
  BufferOffset as_xoris(Register rd, Register ra, uint16_t im);
  BufferOffset as_andi_rc(Register rd, Register ra, uint16_t im);

  // Sign extension.
  BufferOffset as_extsb(Register rd, Register rs);
  BufferOffset as_extsh(Register rd, Register rs);
  BufferOffset as_extsw(Register rd, Register rs);
  BufferOffset as_extsw_rc(Register rd, Register rs);

  // Shift/rotate with immediates.
  BufferOffset as_srawi(Register id, Register rs, uint8_t n);
  BufferOffset as_sradi(Register rd, Register rs, int n);
  BufferOffset as_rldcl(Register ra, Register rs, Register rb, uint8_t mb);
  BufferOffset as_rldicl(Register ra, Register rs, uint8_t sh, uint8_t mb);
  BufferOffset as_rldicl_rc(Register ra, Register rs, uint8_t sh, uint8_t mb);
  BufferOffset as_rldicr(Register ra, Register rs, uint8_t sh, uint8_t mb);
  BufferOffset as_rldicr_rc(Register ra, Register rs, uint8_t sh, uint8_t mb);
  BufferOffset as_rlwinm(Register rd, Register rs, uint8_t sh, uint8_t mb,
                         uint8_t me);
  BufferOffset as_rlwinm_rc(Register rd, Register rs, uint8_t sh, uint8_t mb,
                            uint8_t me);
  BufferOffset as_rlwimi(Register rd, Register rs, uint8_t sh, uint8_t mb,
                         uint8_t me);
  BufferOffset as_rldimi(Register rd, Register rs, uint8_t sh, uint8_t mb);
  BufferOffset as_rlwnm(Register rd, Register rs, Register rb, uint8_t mb,
                        uint8_t me);

  // Integer loads (D-form).
  BufferOffset as_lbz(Register rd, Register rb, int16_t off);
  BufferOffset as_lha(Register rd, Register rb, int16_t off);
  BufferOffset as_lhz(Register rd, Register rb, int16_t off);
  BufferOffset as_lwa(Register rd, Register rb, int16_t off);
  BufferOffset as_lwz(Register rd, Register rb, int16_t off);
  BufferOffset as_ld(Register rd, Register rb, int16_t off);

  // Integer stores (D-form).
  BufferOffset as_stb(Register rd, Register rb, int16_t off);
  BufferOffset as_sth(Register rd, Register rb, int16_t off);
  BufferOffset as_stw(Register rd, Register rb, int16_t off);
  BufferOffset as_std(Register rd, Register rb, int16_t off);
  BufferOffset as_stdu(Register rd, Register rb, int16_t off);

  // Integer loads/stores (X-form, indexed).
  BufferOffset as_lbzx(Register rd, Register ra, Register rb);
  BufferOffset as_lhax(Register rd, Register ra, Register rb);
  BufferOffset as_lhzx(Register rd, Register ra, Register rb);
  BufferOffset as_lwzx(Register rd, Register ra, Register rb);
  // X-form sign-extending word load. Single-insn equivalent of lwzx + extsw.
  BufferOffset as_lwax(Register rd, Register ra, Register rb);
  BufferOffset as_lwarx(Register rd, Register ra, Register rb);
  BufferOffset as_lbarx(Register rd, Register ra, Register rb);
  BufferOffset as_lharx(Register rd, Register ra, Register rb);
  BufferOffset as_ldx(Register rd, Register ra, Register rb);
  BufferOffset as_ldarx(Register rd, Register ra, Register rb);
  BufferOffset as_stbx(Register rd, Register ra, Register rb);
  BufferOffset as_stbcx(Register rd, Register ra, Register rb);
  BufferOffset as_stwx(Register rd, Register ra, Register rb);
  BufferOffset as_stwbrx(Register rd, Register ra, Register rb);
  BufferOffset as_sthx(Register rd, Register ra, Register rb);
  BufferOffset as_sthcx(Register rd, Register ra, Register rb);
  BufferOffset as_stdx(Register rd, Register ra, Register rb);
  BufferOffset as_stdcx(Register rd, Register ra, Register rb);
  BufferOffset as_stwcx(Register rd, Register ra, Register rb);

  // Integer select.
  // POWER10 (ISA 3.1). Set RT = 1/0 based on a CR bit.
  BufferOffset as_setbc(Register rt, uint16_t bc, CRegisterID cr);
  BufferOffset as_setbcr(Register rt, uint16_t bc, CRegisterID cr);
  BufferOffset as_isel(Register rt, Register ra, Register rb, uint16_t rc,
                       CRegisterID cr = cr0);
  BufferOffset as_isel0(Register rt, Register ra, Register rb, uint16_t rc,
                        CRegisterID cr = cr0);

  // FP compare.
  BufferOffset as_fcmpu(CRegisterID cr, FloatRegister ra, FloatRegister rb);
  BufferOffset as_fcmpu(FloatRegister ra, FloatRegister rb);

  // FP arithmetic (two-source).
  BufferOffset as_fadd(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fadds(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fsub(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fsubs(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fdiv(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fdivs(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fmul(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fmuls(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  BufferOffset as_fcpsgn(FloatRegister rd, FloatRegister ra, FloatRegister rc);
  // FP unary.
  BufferOffset as_fabs(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fneg(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fmr(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fsqrt(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fsqrts(FloatRegister rd, FloatRegister rs);
  BufferOffset as_frsp(FloatRegister rd, FloatRegister rs);

  // FP conversions.
  BufferOffset as_fcfid(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fcfids(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fcfidu(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fcfidus(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fctid(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fctidz(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fctiduz(FloatRegister rd, FloatRegister rs);
  BufferOffset as_fctiwz(FloatRegister rd, FloatRegister rs);

  // FP rounding.
  BufferOffset as_frim(FloatRegister rd, FloatRegister rs);
  BufferOffset as_frip(FloatRegister rd, FloatRegister rs);
  BufferOffset as_friz(FloatRegister rd, FloatRegister rs);

  // FP loads (D-form).
  BufferOffset as_lfd(FloatRegister rd, Register rb, int16_t off);
  BufferOffset as_lfs(FloatRegister rd, Register rb, int16_t off);

  // FP stores (D-form).
  BufferOffset as_stfd(FloatRegister rd, Register rb, int16_t off);
  BufferOffset as_stfs(FloatRegister rd, Register rb, int16_t off);
  BufferOffset as_stfdu(FloatRegister rd, Register rb, int16_t off);
  BufferOffset as_stfsu(FloatRegister rd, Register rb, int16_t off);

  // FP loads/stores (X-form, indexed).
  BufferOffset as_lfdx(FloatRegister rd, Register ra, Register rb);
  BufferOffset as_lfsx(FloatRegister rd, Register ra, Register rb);
  BufferOffset as_lfiwax(FloatRegister rd, Register ra, Register rb);
  BufferOffset as_stfdx(FloatRegister rd, Register ra, Register rb);
  BufferOffset as_stfsx(FloatRegister rd, Register ra, Register rb);

  // FPSCR operations.
  BufferOffset as_mtfsb0(uint8_t bt);
  BufferOffset as_mcrfs(CRegisterID bf, uint8_t bfa);

  // VSX (FPR-only subset).
  BufferOffset as_mfvsrd(Register ra, FloatRegister xs);
  BufferOffset as_mtvsrd(FloatRegister xs, Register ra);
  // POWER8+ (ISA 2.07). Sign-extending move of RA's low 32 bits to FPR.
  BufferOffset as_mtvsrwa(FloatRegister xs, Register ra);
  BufferOffset as_mtvsrwz(FloatRegister xs, Register ra);
  BufferOffset as_mtvsrws(FloatRegister xs, Register ra);
  BufferOffset as_xxbrd(FloatRegister xt, FloatRegister xb);
  // POWER9 scalar VSX max/min with Java/JavaScript semantics (matches
  // ECMA-262 Math.max / Math.min). Operate on FPR-space (encoding 0..31).
  BufferOffset as_xsmaxjdp(FloatRegister xt, FloatRegister xa,
                           FloatRegister xb);
  BufferOffset as_xsminjdp(FloatRegister xt, FloatRegister xa,
                           FloatRegister xb);
  BufferOffset as_xscvdpspn(FloatRegister xt, FloatRegister xb);
  BufferOffset as_xscvspdpn(FloatRegister xt, FloatRegister xb);
  // POWER9 (ISA 3.0) scalar FP16 conversions.
  BufferOffset as_xscvdphp(FloatRegister xt, FloatRegister xb);
  BufferOffset as_xscvhpdp(FloatRegister xt, FloatRegister xb);
  // POWER9 (ISA 3.0) scalar extract biased exponent.
  BufferOffset as_xsxexpdp(FloatRegister xt, FloatRegister xb);
  // POWER9 (ISA 3.0) scalar FP16 load/store, X-form indexed.
  BufferOffset as_lxsihzx(FloatRegister xt, Register ra, Register rb);
  BufferOffset as_stxsihx(FloatRegister xs, Register ra, Register rb);

  // VSX SIMD load/store (X-form, indexed).
  BufferOffset as_lxvx(FloatRegister xt, Register ra, Register rb);
  BufferOffset as_stxvx(FloatRegister xs, Register ra, Register rb);
  BufferOffset as_lxvd2x(FloatRegister xt, Register ra, Register rb);
  BufferOffset as_stxvd2x(FloatRegister xs, Register ra, Register rb);

  // VMX SIMD load/store (X-form, indexed). Take a raw VR number (0-31)
  // because VR20-VR31 are outside the FloatRegister encoding (which only
  // covers VSR0-31 = f0-f31). Used by the JIT trampoline to save/restore
  // the ELFv2 callee-saved VR20-VR31. EA is force-aligned to 16 bytes
  // (low 4 bits of the address are ignored), so the slot's alignment
  // matters for layout but not for trap avoidance.
  BufferOffset as_lvx(uint8_t vrt, Register ra, Register rb);
  BufferOffset as_stvx(uint8_t vrs, Register ra, Register rb);

  // VSX SIMD register operations (XX3-form / XX1-form / XX2-form).
  BufferOffset as_xxlor(FloatRegister xt, FloatRegister xa, FloatRegister xb);

  // VSX bitwise operations (XX3-form).
  BufferOffset as_xxland(FloatRegister xt, FloatRegister xa, FloatRegister xb);
  BufferOffset as_xxlxor(FloatRegister xt, FloatRegister xa, FloatRegister xb);
  BufferOffset as_xxlnor(FloatRegister xt, FloatRegister xa, FloatRegister xb);
  BufferOffset as_xxlandc(FloatRegister xt, FloatRegister xa, FloatRegister xb);
  BufferOffset as_xxsel(FloatRegister xt, FloatRegister xa, FloatRegister xb,
                        FloatRegister xc);

  // VMX integer arithmetic (VR0-31 = VSR32-63 only).
  // Callers must ensure operands are in VR space.
  BufferOffset as_vaddubm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vadduhm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vadduwm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vaddudm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsububm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsubuhm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsubuwm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsubudm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vaddsbs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vaddshs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vaddubs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vadduhs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsubsbs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsubshs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsububs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsubuhs(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vminsb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vminsh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vminsw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxsb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxsh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxsw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxsd(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vminub(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vminuh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vminuw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxub(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxuh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmaxuw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  // POWER9 (ISA 3.0): per-lane integer negate.
  BufferOffset as_vnegw(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vnegd(uint8_t vrt, uint8_t vrb);
  // POWER9 (ISA 3.0): addpcis rT, D.  Computes rT = (CIA + 4) + (D << 16).
  // D is a 16-bit signed immediate; DX-form splits D across three instruction
  // fields (d0[16..25] ∥ d1[11..15] ∥ d2[31]).  No LR clobber, no RAS hazard.
  BufferOffset as_addpcis(Register rt, int16_t d);
  // POWER10 (ISA 3.1) prefixed instructions. Each emits 8 bytes (prefix +
  // suffix) with a single nop inserted before iff the prefix would
  // straddle a 64-byte block. Caller must guarantee HasPOWER10().
  // imm34 is signed 34-bit; R=true selects PC-relative form (RA must be r0).
  // Returns the offset of the prefix word.
  BufferOffset as_paddi(Register rt, Register ra, int64_t imm34, bool R);
  BufferOffset as_pld(Register rt, Register ra, int64_t imm34, bool R);
  BufferOffset as_plxv(uint8_t xt, Register ra, int64_t imm34, bool R);
  // FP-target prefixed loads: plfd/plfs are MLS (Type=2) with suffix
  // opcodes 50 and 48. plfs widens single → double in the FPR
  // (matches non-prefixed lfs semantics).
  BufferOffset as_plfd(FloatRegister frt, Register ra, int64_t imm34,
                       bool R);
  BufferOffset as_plfs(FloatRegister frt, Register ra, int64_t imm34,
                       bool R);
  // Prefixed-store counterparts. Same prefix shape; suffix opcodes are
  // the D-form variants of std/stxv/stfd/stfs (61, 27, 54, 52).
  BufferOffset as_pstd(Register rs, Register ra, int64_t imm34, bool R);
  BufferOffset as_pstxv(uint8_t xs, Register ra, int64_t imm34, bool R);
  BufferOffset as_pstfd(FloatRegister frs, Register ra, int64_t imm34,
                        bool R);
  BufferOffset as_pstfs(FloatRegister frs, Register ra, int64_t imm34,
                        bool R);

 private:
  // Emit a nop before a prefixed instruction iff the prefix would otherwise
  // start at offset 60 (mod 64) and the suffix would land in the next block.
  void ensurePrefixedAlignment();

 public:
  BufferOffset as_vavgub(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vavguh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmuluwm(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulld(uint8_t vrt, uint8_t vra, uint8_t vrb);
  // VMX shift (VR0-31 only).
  BufferOffset as_vslb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vslh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vslw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsld(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrd(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrab(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrah(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsraw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsrad(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vslo(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vsro(uint8_t vrt, uint8_t vra, uint8_t vrb);

  // VMX integer compare (VR0-31 only).
  BufferOffset as_vcmpequb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpequh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpequw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpequd(uint8_t vrt, uint8_t vra, uint8_t vrb);
  // Record forms set CR6: LT = all-true, EQ = none-true.
  BufferOffset as_vcmpequb_rc(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpequh_rc(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpequw_rc(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpequd_rc(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtsb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtsh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtsw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtsd(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtub(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtuh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtuw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpgtud(uint8_t vrt, uint8_t vra, uint8_t vrb);
  // POWER9 (ISA 3.0). NotEqual compare; no doubleword variant.
  BufferOffset as_vcmpneb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpneh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vcmpnew(uint8_t vrt, uint8_t vra, uint8_t vrb);

  // VSX float compare (XX3-form, VSR0-63).
  BufferOffset as_xvcmpeqsp(FloatRegister xt, FloatRegister xa,
                            FloatRegister xb);
  BufferOffset as_xvcmpgtsp(FloatRegister xt, FloatRegister xa,
                            FloatRegister xb);
  BufferOffset as_xvcmpgesp(FloatRegister xt, FloatRegister xa,
                            FloatRegister xb);
  BufferOffset as_xvcmpeqdp(FloatRegister xt, FloatRegister xa,
                            FloatRegister xb);
  BufferOffset as_xvcmpgtdp(FloatRegister xt, FloatRegister xa,
                            FloatRegister xb);
  BufferOffset as_xvcmpgedp(FloatRegister xt, FloatRegister xa,
                            FloatRegister xb);

  // VSX float arithmetic (XX3-form binary, XX2-form unary).
#define DECL_VSX_BIN(op) \
  BufferOffset as_##op(FloatRegister xt, FloatRegister xa, FloatRegister xb);
  DECL_VSX_BIN(xvaddsp)
  DECL_VSX_BIN(xvadddp) DECL_VSX_BIN(xvsubsp) DECL_VSX_BIN(
      xvsubdp) DECL_VSX_BIN(xvmulsp) DECL_VSX_BIN(xvmuldp) DECL_VSX_BIN(xvdivsp)
      DECL_VSX_BIN(xvdivdp) DECL_VSX_BIN(xvminsp) DECL_VSX_BIN(
          xvmindp) DECL_VSX_BIN(xvmaxsp) DECL_VSX_BIN(xvmaxdp)
          DECL_VSX_BIN(xvmaddasp) DECL_VSX_BIN(xvmaddadp) DECL_VSX_BIN(
              xvnmsubasp) DECL_VSX_BIN(xvnmsubadp)
#undef DECL_VSX_BIN
#define DECL_VSX_UN(op) \
  BufferOffset as_##op(FloatRegister xt, FloatRegister xb);
              DECL_VSX_UN(xvabssp) DECL_VSX_UN(xvabsdp) DECL_VSX_UN(xvnegsp)
                  DECL_VSX_UN(xvnegdp) DECL_VSX_UN(xvsqrtsp) DECL_VSX_UN(
                      xvsqrtdp) DECL_VSX_UN(xvrspip) DECL_VSX_UN(xvrdpip)
                      DECL_VSX_UN(xvrspim) DECL_VSX_UN(xvrdpim) DECL_VSX_UN(
                          xvrspiz) DECL_VSX_UN(xvrdpiz) DECL_VSX_UN(xvrspic)
                          DECL_VSX_UN(xvrdpic) DECL_VSX_UN(xvcvsxwsp)
                              DECL_VSX_UN(xvcvuxwsp) DECL_VSX_UN(xvcvsxwdp)
                                  DECL_VSX_UN(xvcvuxwdp) DECL_VSX_UN(xvcvspsxws)
                                      DECL_VSX_UN(xvcvspuxws)
                                          DECL_VSX_UN(xvcvdpsxws)
                                              DECL_VSX_UN(xvcvdpuxws)
                                                  DECL_VSX_UN(xvcvdpsp)
                                                      DECL_VSX_UN(xvcvspdp)
#undef DECL_VSX_UN

      // VMX widen/narrow/merge/pack (VR0-31 only).
      BufferOffset as_vupkhsb(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vupklsb(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vupkhsh(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vupklsh(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vupkhsw(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vupklsw(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vpkshss(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vpkswss(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vpkshus(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vpkswus(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmrghb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmrghh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmrghw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmrglb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmrglh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmrglw(uint8_t vrt, uint8_t vra, uint8_t vrb);

  // VMX extended multiply (VR0-31 only).
  BufferOffset as_vmulesb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulosb(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmuleub(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmuloub(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulesh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulosh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmuleuh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulouh(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulesw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulosw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmuleuw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vmulouw(uint8_t vrt, uint8_t vra, uint8_t vrb);
  BufferOffset as_vpopcntb(uint8_t vrt, uint8_t vrb);
  BufferOffset as_vperm(uint8_t vrt, uint8_t vra, uint8_t vrb, uint8_t vrc);
  // POWER8+ (ISA 2.07). VX-form bit-permute. See PPC_vbpermq comment.
  BufferOffset as_vbpermq(uint8_t vrt, uint8_t vra, uint8_t vrb);
  // POWER10 (ISA 3.1) Vector Extract Mask. RT is a GPR.
  BufferOffset as_vextractbm(Register rt, FloatRegister vrb);
  BufferOffset as_vextracthm(Register rt, FloatRegister vrb);
  BufferOffset as_vextractwm(Register rt, FloatRegister vrb);
  BufferOffset as_vextractdm(Register rt, FloatRegister vrb);
  // POWER10 (ISA 3.1) Vector Insert from GPR at immediate byte offset.
  // UIM range: vinsw 0..12, vinsd 0..8 (caller must enforce).
  BufferOffset as_vinsw(FloatRegister vrt, Register rb, uint8_t uim);
  BufferOffset as_vinsd(FloatRegister vrt, Register rb, uint8_t uim);
  // POWER10 (ISA 3.1) Vector Insert byte / halfword from GPR with the
  // byte position supplied by another GPR (RA & 0xF for vinsbrx,
  // RA & 0xE for vinshrx). "rx" = right-indexed = LE-natural.
  BufferOffset as_vinsbrx(FloatRegister vrt, Register ra, Register rb);
  BufferOffset as_vinshrx(FloatRegister vrt, Register ra, Register rb);
  // POWER9 (ISA 3.0) Vector Insert byte / halfword from VR at immediate
  // byte position. UIM range: vinsertb 0..15, vinserth 0..14
  // (caller must enforce; vinserth UIM is in bytes, even-aligned).
  BufferOffset as_vinsertb(FloatRegister vrt, FloatRegister vrb, uint8_t uim);
  BufferOffset as_vinserth(FloatRegister vrt, FloatRegister vrb, uint8_t uim);
  // POWER9 (ISA 3.0) Vector Extract byte / halfword from VR at immediate
  // BE byte position. UIM range: vextractub 0..15, vextractuh 0..14
  // (caller must enforce; vextractuh UIM is in bytes, even-aligned). The
  // extracted byte/halfword lands at BE byte 7 of VRT, with the rest
  // zeroed — so a subsequent mfvsrd reads it as the low byte/halfword
  // of the GPR with implicit zero-extension.
  BufferOffset as_vextractub(FloatRegister vrt, FloatRegister vrb, uint8_t uim);
  BufferOffset as_vextractuh(FloatRegister vrt, FloatRegister vrb, uint8_t uim);
  // VX-form with 5-bit signed immediate splat: each lane of VRT is
  // set to sign_extend(SIMM5) (range [-16, 15]) at byte/halfword/word granularity.
  BufferOffset as_vspltisb(uint8_t vrt, int8_t simm5);
  BufferOffset as_vspltish(uint8_t vrt, int8_t simm5);
  BufferOffset as_vspltisw(uint8_t vrt, int8_t simm5);

  // VA-form ternary VMX instructions.
  BufferOffset as_vmladduhm(uint8_t vrt, uint8_t vra, uint8_t vrb, uint8_t vrc);
  BufferOffset as_vmhraddshs(uint8_t vrt, uint8_t vra, uint8_t vrb,
                             uint8_t vrc);
  BufferOffset as_vmsumshm(uint8_t vrt, uint8_t vra, uint8_t vrb, uint8_t vrc);
  BufferOffset as_vmsumuhm(uint8_t vrt, uint8_t vra, uint8_t vrb, uint8_t vrc);
  BufferOffset as_xxpermdi(FloatRegister xt, FloatRegister xa, FloatRegister xb,
                           uint8_t dm);
  BufferOffset as_xxspltw(FloatRegister xt, FloatRegister xb, uint8_t uim);
  // POWER9 (ISA 3.0). Splat 8-bit immediate to all 16 bytes of an FPR-encoded
  // VSR (TX bit forced 0). XX1-form, no Rc.
  BufferOffset as_xxspltib(FloatRegister xt, uint8_t imm8);
  BufferOffset as_xxinsertw(FloatRegister xt, FloatRegister xb, uint8_t uim);
  BufferOffset as_xxextractuw(FloatRegister xt, FloatRegister xb, uint8_t uim);
  BufferOffset as_mtvsrdd(FloatRegister xt, Register ra, Register rb);
  BufferOffset as_mfvsrld(Register rt, FloatRegister xs);

  // VMX vector operations.
  BufferOffset as_vspltb(FloatRegister vrt, FloatRegister vrb, uint8_t uim);
  BufferOffset as_vsplth(FloatRegister vrt, FloatRegister vrb, uint8_t uim);
  BufferOffset as_vsldoi(FloatRegister vrt, FloatRegister vra,
                         FloatRegister vrb, uint8_t shb);

  // Barrier and sync instructions.
  BufferOffset as_lwsync();
  BufferOffset as_sync();
  BufferOffset as_isync();

  // Convenience pseudo-instructions.
  BufferOffset xs_trap();
  BufferOffset xs_trap_tagged(TrapTag tag);
  BufferOffset xs_mr(Register rd, Register ra);
  BufferOffset xs_mtctr(Register ra);
  BufferOffset xs_mtlr(Register ra);
  BufferOffset xs_mflr(Register rd);
  BufferOffset xs_mtcr(Register rs);
  BufferOffset xs_mfxer(Register ra);
  BufferOffset xs_mtxer(Register ra);
  BufferOffset xs_li(Register rd, int16_t im);
  BufferOffset xs_lis(Register rd, int16_t im);
  BufferOffset x_subi(Register rd, Register ra, int16_t im);
  BufferOffset x_not(Register rd, Register ra);
  BufferOffset x_slwi(Register rd, Register rs, int n);
  BufferOffset x_sldi(Register rd, Register rs, int n);
  BufferOffset x_srwi(Register rd, Register rs, int n);
  BufferOffset x_srdi(Register rd, Register rs, int n);
  BufferOffset x_insertbits0_15(Register rd, Register rs);
  BufferOffset x_bit_value(Register rd, Register rs, unsigned bit);
  BufferOffset x_sr_mulli(Register rd, Register ra, int16_t im);

  // --- Label operations.
  void bind(Label* label) { bind(label, nextOffset()); }
  void bind(Label* label, BufferOffset boff);
  void bind(InstImm* inst, uintptr_t branch, uintptr_t target);
  void bind(CodeLabel* label) { label->target()->bind(currentOffset()); }
  uint32_t currentOffset() { return nextOffset().getOffset(); }
  void retarget(Label* label, Label* target);
  void call(Label* label);
  void call(void* target);

  void as_break(uint32_t code);

  // --- Static capability queries.
  static bool SupportsFloatingPoint() { return true; }
  static bool SupportsWasmSimd() { return true; }
  static bool SupportsUnalignedAccesses() { return true; }
  static bool SupportsFastUnalignedFPAccesses() { return true; }
  // POWER9 has scalar FP16 hardware (xscvdphp/xscvhpdp); POWER8 doesn't.
  // Runtime-gate like x86's SupportsFloat32To16 (which keys off F16C).
  static bool SupportsFloat64To16() { return HasPOWER9(); }
  static bool SupportsFloat32To16() { return HasPOWER9(); }
  static bool HasRoundInstruction(RoundingMode mode) {
    // PPC64 has friz (trunc), frip (ceil), frim (floor), which are all correct.
    // frin (round-to-nearest) does NOT implement proper IEEE banker's rounding
    // (ties to even), so NearestTiesToEven is not supported.
    return mode == RoundingMode::TowardsZero || mode == RoundingMode::Up ||
           mode == RoundingMode::Down;
  }

 protected:
  InstImm invertBranch(InstImm branch, BOffImm16 skipOffset);
  void addPendingJump(BufferOffset src, ImmPtr target, RelocationKind kind) {
    enoughMemory_ &= jumps_.append(RelativePatch(src, target.value, kind));
    if (kind == RelocationKind::JITCODE) {
      writeRelocation(src);
    }
  }
  void addLongJump(BufferOffset src, BufferOffset dst) {
    CodeLabel cl;
    cl.patchAt()->bind(src.getOffset());
    cl.target()->bind(dst.getOffset());
    cl.setLinkMode(CodeLabel::JumpImmediate);
    addCodeLabel(std::move(cl));
  }

 public:
  void flushBuffer() { m_buffer.flushPool(); }
  void comment(const char* msg) { spew("; %s", msg); }
  static uint32_t NopSize() { return 4; }

  // --- Static patching API.
  static uint64_t ExtractLoad64Value(Instruction* inst0);
  static void UpdateLoad64Value(Instruction* inst0, uint64_t value);
  static void WriteLoad64Instructions(Instruction* inst0, Register reg,
                                      uint64_t value);

  static void PatchWrite_Imm32(CodeLocationLabel label, Imm32 imm);
  static uint8_t* NextInstruction(uint8_t* instruction,
                                  uint32_t* count = nullptr);
  static void ToggleToJmp(CodeLocationLabel inst_);
  static void ToggleToCmp(CodeLocationLabel inst_);

  void verifyHeapAccessDisassembly(uint32_t begin, uint32_t end,
                                   const Disassembler::HeapAccess& ha) {}

  // --- Public patching API (required by shared code).
  static void Bind(uint8_t* rawCode, const CodeLabel& label);
  void processCodeLabels(uint8_t* rawCode);

  static void TraceJumpRelocations(JSTracer* trc, JitCode* code,
                                   CompactBufferReader& reader);
  static void TraceDataRelocations(JSTracer* trc, JitCode* code,
                                   CompactBufferReader& reader);

  void executableCopy(uint8_t* buffer);

  static uint32_t PatchWrite_NearCallSize();
  static void PatchWrite_NearCall(CodeLocationLabel start,
                                  CodeLocationLabel toCall);
  static void PatchDataWithValueCheck(CodeLocationLabel label, ImmPtr newValue,
                                      ImmPtr expectedValue);
  static void PatchDataWithValueCheck(CodeLocationLabel label,
                                      PatchedImmPtr newValue,
                                      PatchedImmPtr expectedValue);
  static void ToggleCall(CodeLocationLabel inst_, bool enabled);

 private:
  GeneralRegisterSet scratch_register_list_;

 public:
  GeneralRegisterSet* GetScratchRegisterList() {
    return &scratch_register_list_;
  }
};  // Assembler

inline bool IsUnaligned(const wasm::MemoryAccessDesc& access) {
  if (!access.align()) {
    return false;
  }
  return access.align() < access.byteSize();
}

}  // namespace jit
}  // namespace js

// Whether an Imm32 fits in an unsigned 16-bit immediate.
#define PPC_IMM_OK_U(x) (MOZ_LIKELY(((x).value & 0xffff0000) == 0))

// Whether an Imm32 fits in a signed 16-bit immediate.
#define PPC_IMM_OK_S(x)                        \
  (MOZ_LIKELY(((x).value & 0xffff8000) == 0 || \
              ((x).value & 0xffff8000) == 0xffff8000))

// Whether the offset part of an Address fits in a signed 16-bit immediate.
#define PPC_OFFS_OK(x)                          \
  (MOZ_LIKELY(((x).offset & 0xffff8000) == 0 || \
              ((x).offset & 0xffff8000) == 0xffff8000))

// Same test but checking a bit ahead (for paired loads).
#define PPC_OFFS_INCR_OK(x, incr)                          \
  (MOZ_LIKELY((((x).offset + (incr)) & 0xffff8000) == 0 || \
              (((x).offset + (incr)) & 0xffff8000) == 0xffff8000))

#endif /* jit_ppc64_Assembler_ppc64_h */
