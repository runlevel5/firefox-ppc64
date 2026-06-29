/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_SharedICRegisters_ppc64_h
#define jit_ppc64_SharedICRegisters_ppc64_h

#include "jit/ppc64/Assembler-ppc64.h"
#include "jit/Registers.h"
#include "jit/RegisterSets.h"

namespace js {
namespace jit {

// ValueOperands R0, R1, and R2.
// R0 == JSReturnReg, and R2 uses registers not preserved across calls. R1 value
// should be preserved across calls.
static constexpr ValueOperand R0(r5);
static constexpr ValueOperand R1(r15);
static constexpr ValueOperand R2(r4);

// ICTailCallReg and ICStubReg.
// On PPC64, LR is not a GPR, so ICTailCallReg must be a normal GPR.
// PPC64 ELFv2 has no volatile non-arg GPRs (r3-r10 are all arg regs), so
// using an arg register risks clobbering by ABI calls with enough arguments.
// We use callee-saved registers instead, matching MIPS64/RISC-V strategy.
// These are excluded from BaselineICAvailableGeneralRegs.
static constexpr Register ICTailCallReg = r27;
static constexpr Register ICStubReg = r26;

// FloatReg0 must be equal to ReturnFloatReg.
static constexpr FloatRegister FloatReg0 = {FloatRegisters::f1,
                                            FloatRegisters::Double};
static constexpr FloatRegister FloatReg1 = {FloatRegisters::f2,
                                            FloatRegisters::Double};
static constexpr FloatRegister FloatReg2 = {FloatRegisters::f3,
                                            FloatRegisters::Double};
static constexpr FloatRegister FloatReg3 = {FloatRegisters::f4,
                                            FloatRegisters::Double};

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_SharedICRegisters_ppc64_h */
